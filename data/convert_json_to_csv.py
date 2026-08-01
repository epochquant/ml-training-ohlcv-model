#!/usr/bin/env python3
"""
EpochQuant — JSON → CSV Pipeline CLI
=====================================
Run interactively:   python data/convert_json_to_csv.py
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from repo root (one level up from data/)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------------------
# ANSI colour helpers (no extra dependencies)
# ---------------------------------------------------------------------------
_NO_COLOR = not sys.stdout.isatty()  # disable colours when piped/redirected

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def cyan(t):    return _c("96", t)
def green(t):   return _c("92", t)
def yellow(t):  return _c("93", t)
def red(t):     return _c("91", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)


# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Print a labelled prompt and return the user's input (or the default)."""
    default_hint = f"  {dim(f'[default: {default}]')}" if default else ""
    print(f"  {cyan('›')} {label}{default_hint}")
    try:
        if secret:
            import getpass
            value = getpass.getpass("    > ").strip()
        else:
            value = input("    > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        _abort()
    return value if value else default


def _confirm(question: str, default: bool = True) -> bool:
    """Ask a yes/no question and return True/False."""
    hint = "[Y/n]" if default else "[y/N]"
    print(f"  {cyan('›')} {question}  {dim(hint)}")
    try:
        raw = input("    > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        _abort()
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return default


def _abort():
    print(f"\n{yellow('  Aborted.')} No files were modified.\n")
    sys.exit(0)


def _section(n: int, total: int, title: str):
    print(f"\n{bold(f'[{n}/{total}]')} {bold(title)}")


def _banner():
    width = 52
    border = "═" * width
    print(f"\n{cyan('╔' + border + '╗')}")
    print(f"{cyan('║')}{bold('  EpochQuant — JSON → CSV Pipeline CLI'):^{width + 8}}{cyan('║')}")
    print(f"{cyan('╚' + border + '╝')}\n")


def _summary_line(label: str, value: str):
    print(f"    {dim(label + ':'): <20} {value}")


# ---------------------------------------------------------------------------
# Core pipeline functions (unchanged logic)
# ---------------------------------------------------------------------------

def process_single_json(file_path):
    """Parses a single JSON file and returns a formatted DataFrame."""
    with open(file_path, 'r') as file:
        raw_data = json.load(file)

    df = pd.DataFrame(raw_data)

    # 1. Convert string prices and volumes to floats
    target_cols = ['open', 'high', 'low', 'close', 'volume']
    df[target_cols] = df[target_cols].astype(float)

    # 2. Convert Unix milliseconds to readable datetime format
    df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')

    # 3. Filter down to strictly what KronosPredictor requires
    result = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()
    result['symbol'] = file_path.stem

    # 4. Zero out volume field (uncomment to enable)
    ## result['volume'] = 0

    return result


def _split_single_symbol(
    df,
    price_jump_threshold=0.10,
    min_segment_len=512,
    illiquid_threshold=None,
    stagnant_threshold=None,
    volume_near_zero_val=1e-8
):
    """
    Splits a DataFrame into valid continuous sub-segments based on:
    - Price Jump Threshold (Theta_jump)
    - Illiquid Periods (consecutive bars with zero or near-zero volume)
    - Stagnant Periods (consecutive bars with constant closing price)
    And discards short segments.
    """
    df = df.sort_values(by='timestamps').reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df

    # We will compute split_flags across all rows
    split_flags = pd.Series(False, index=df.index)

    # 1. Price Jump: |open_t / close_{t-1} - 1| > price_jump_threshold
    if n > 1 and price_jump_threshold is not None:
        prev_close = df['close'].shift(1)
        relative_jump = (df['open'] / prev_close - 1).abs()
        jump_mask = (relative_jump > price_jump_threshold) & (df.index > 0)
        split_flags = split_flags | jump_mask

    # 2. Illiquid Periods: split if >= illiquid_threshold consecutive bars have volume <= volume_near_zero_val
    if illiquid_threshold is not None and illiquid_threshold > 0:
        is_inactive = (df['volume'] <= volume_near_zero_val).values
        inactive_run = 0
        for i in range(n):
            if is_inactive[i]:
                inactive_run += 1
                if inactive_run >= illiquid_threshold:
                    split_flags.iloc[i] = True
            else:
                inactive_run = 0

    # 3. Stagnant Periods: split if >= stagnant_threshold consecutive bars have constant closing price
    if stagnant_threshold is not None and stagnant_threshold > 0:
        constant_close_run = 1
        for i in range(1, n):
            if df['close'].iloc[i] == df['close'].iloc[i-1]:
                constant_close_run += 1
                if constant_close_run >= stagnant_threshold:
                    split_flags.iloc[i] = True
            else:
                constant_close_run = 1

    # Assign group IDs based on cumulative sum of breaks
    df['segment_id'] = split_flags.cumsum()

    # Filter out segments that do not meet the minimum length
    valid_segments = []
    for seg_id, group in df.groupby('segment_id'):
        if len(group) >= min_segment_len:
            valid_segments.append(group.drop(columns=['segment_id']))

    if not valid_segments:
        return pd.DataFrame(columns=[c for c in df.columns if c != 'segment_id'])

    return pd.concat(valid_segments, ignore_index=True)


def split_on_structural_breaks(
    df,
    price_jump_threshold=0.10,
    min_segment_len=512,
    illiquid_threshold=None,
    stagnant_threshold=None,
    volume_near_zero_val=1e-8
):
    if 'symbol' in df.columns:
        valid_segments = []
        for sym, group in df.groupby('symbol'):
            res = _split_single_symbol(
                group, 
                price_jump_threshold, 
                min_segment_len, 
                illiquid_threshold, 
                stagnant_threshold, 
                volume_near_zero_val
            )
            valid_segments.append(res)
        if not valid_segments:
            return pd.DataFrame(columns=[c for c in df.columns if c != 'segment_id'])
        return pd.concat(valid_segments, ignore_index=True)
    else:
        return _split_single_symbol(
            df,
            price_jump_threshold,
            min_segment_len,
            illiquid_threshold,
            stagnant_threshold,
            volume_near_zero_val
        )


def build_kronos_dataset_from_folder(
    target_folder,
    output_csv_path,
    price_jump_threshold=0.10,
    min_segment_len=512,
    illiquid_threshold=None,
    stagnant_threshold=None
):
    """
    Builds a unified time-series dataset from a target path (which can be a single file
    or a folder of JSON/CSV files).
    """
    path_obj = Path(target_folder)
    
    json_files = []
    csv_files = []
    
    if path_obj.is_file():
        if path_obj.suffix.lower() == '.json':
            json_files = [path_obj]
        elif path_obj.suffix.lower() == '.csv':
            csv_files = [path_obj]
        else:
            print(red(f"  Error: Unsupported file format '{path_obj.suffix}'"))
            return False
    elif path_obj.is_dir():
        json_files = list(path_obj.rglob('*.json'))
        csv_files = list(path_obj.rglob('*.csv'))
    else:
        print(red(f"  Error: Path does not exist or is invalid: {target_folder}"))
        return False

    if not json_files and not csv_files:
        print(red(f"  Error: No JSON or CSV files found at '{target_folder}'"))
        return False

    if json_files:
        print(f"  Found {green(str(len(json_files)))} JSON file(s).")
    if csv_files:
        print(f"  Found {green(str(len(csv_files)))} CSV file(s).")

    dataframes = []

    # Process JSON files
    for json_file in json_files:
        try:
            df = process_single_json(json_file)
            dataframes.append(df)
            print(f"  {green('[+]')} {json_file.name}  {dim(f'({len(df)} rows)')}")
        except Exception as e:
            print(f"  {red('[-]')} Failed to process {json_file.name}: {e}")

    # Process CSV files
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'timestamps' not in df.columns and 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'timestamps'}, inplace=True)
            
            # Keep only required columns
            target_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
            for col in target_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df[target_cols].copy()
            df['symbol'] = csv_file.stem
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            dataframes.append(df)
            print(f"  {green('[+]')} {csv_file.name}  {dim(f'({len(df)} rows)')}")
        except Exception as e:
            print(f"  {red('[-]')} Failed to process {csv_file.name}: {e}")

    if not dataframes:
        print(red("  Error: No valid data could be extracted from the files."))
        return False

    # Combine all individual DataFrames into one master DataFrame
    print(f"\n  {dim('Merging data...')}")
    master_df = pd.concat(dataframes, ignore_index=True)

    # CRITICAL: Sort chronologically to ensure the time-series sequence is intact
    if 'symbol' in master_df.columns:
        master_df = master_df.sort_values(by=['symbol', 'timestamps']).reset_index(drop=True)
        # CRITICAL: Drop overlapping candles that might exist between file exports
        initial_row_count = len(master_df)
        master_df = master_df.drop_duplicates(subset=['symbol', 'timestamps'], keep='last')
    else:
        master_df = master_df.sort_values(by='timestamps').reset_index(drop=True)
        # CRITICAL: Drop overlapping candles that might exist between file exports
        initial_row_count = len(master_df)
        master_df = master_df.drop_duplicates(subset=['timestamps'], keep='last')
    duplicates_removed = initial_row_count - len(master_df)

    if duplicates_removed > 0:
        print(f"  {yellow('!')} Removed {duplicates_removed} duplicate timestamps.")

    # CRITICAL: Filter out segments using the segmentation thresholds
    print(f"  {dim('Applying structural break segmentation filter...')}")
    initial_len = len(master_df)
    master_df = split_on_structural_breaks(
        master_df,
        price_jump_threshold=price_jump_threshold,
        min_segment_len=min_segment_len,
        illiquid_threshold=illiquid_threshold,
        stagnant_threshold=stagnant_threshold
    )
    filtered_out = initial_len - len(master_df)
    if filtered_out > 0:
        print(f"  {yellow('!')} Filtered out {filtered_out} rows (structural breaks / short segments).")

    # Ensure output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Export the final clean dataset
    master_df.to_csv(output_csv_path, index=False)
    print(f"\n  {green('[OK]')} Exported {bold(str(len(master_df)))} rows -> {cyan(output_csv_path)}")
    return True


def upload_to_gcs(
    local_file_path: str,
    bucket_name: str,
    key_file: str,
    dest_prefix: str,
) -> None:
    """
    Uploads a local CSV file to a GCS bucket.

    Args:
        local_file_path : Local path to the CSV file.
        bucket_name     : Target GCS bucket name.
        key_file        : Path to a Service Account JSON key file, or '' for ADC.
        dest_prefix     : Subfolder inside the bucket (e.g. 'training-data').
    """
    from google.cloud import storage  # imported here — only needed for GCS uploads

    local_path = Path(local_file_path)
    blob_name  = f"{dest_prefix.strip('/')}/{local_path.name}" if dest_prefix else local_path.name
    gcs_uri    = f"gs://{bucket_name}/{blob_name}"

    print(f"\n  {dim('Uploading')} {cyan(local_path.name)} → {cyan(gcs_uri)} ...")

    try:
        if key_file:
            client = storage.Client.from_service_account_json(key_file)
        else:
            # Fall back to Application Default Credentials (ADC)
            client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        print(f"  {green('✔')} Uploaded successfully → {cyan(gcs_uri)}")
    except Exception as e:
        print(f"\n  {red('✘ GCS upload failed:')} {e}")
        raise  # Fatal: exit with non-zero code


# ---------------------------------------------------------------------------
# Interactive CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    TOTAL_STEPS = 5

    _banner()

    # -- Env-var defaults (shown as hints) --
    env_bucket  = os.getenv("GCS_BUCKET_NAME", "epochquant-training")
    env_prefix  = os.getenv("GCS_DESTINATION_PREFIX", "training-data")
    env_key     = os.getenv("GCS_KEY_FILE", "").strip()

    # ── Step 1: Input folder ────────────────────────────────────────────────
    _section(1, TOTAL_STEPS, "Input JSON folder")
    input_folder = _prompt(
        "Path to the folder containing OHLCV JSON files:",
        default="./input_ohlcv/BNBUSDT/1m",
    )

    # ── Step 2: Output CSV path ─────────────────────────────────────────────
    _section(2, TOTAL_STEPS, "Output CSV path")
    output_csv = _prompt(
        "Destination path for the generated CSV file:",
        default="./output_csv/bnbusdt_1m_kronos_data.csv",
    )

    # ── Step 3: Upload to GCS? ──────────────────────────────────────────────
    _section(3, TOTAL_STEPS, "GCS upload")
    upload = _confirm("Upload the CSV to a Google Cloud Storage bucket?", default=True)

    bucket_name = env_bucket
    key_file    = env_key
    dest_prefix = env_prefix

    if upload:
        # ── Step 4: Bucket name ─────────────────────────────────────────────
        _section(4, TOTAL_STEPS, "GCS bucket name")
        bucket_name = _prompt(
            "Bucket name:",
            default=env_bucket,
        )

        # ── Step 5: Credentials ─────────────────────────────────────────────
        _section(5, TOTAL_STEPS, "GCS credentials")
        print(f"  {dim('Leave blank to use Application Default Credentials (ADC).')}")
        key_file = _prompt(
            "Path to Service Account JSON key file:",
            default=env_key if env_key else "",
        )

        dest_prefix = _prompt(
            "Destination folder prefix inside the bucket:",
            default=env_prefix,
        )
    else:
        print(f"\n  {dim('Skipping GCS upload.')}")

    # ── Confirmation summary ────────────────────────────────────────────────
    blob_name = (
        f"{dest_prefix.strip('/')}/{Path(output_csv).name}" if dest_prefix else Path(output_csv).name
    )
    print(f"\n{'─' * 56}")
    print(bold("  Run summary"))
    print(f"{'─' * 56}")
    _summary_line("Input folder", cyan(input_folder))
    _summary_line("Output CSV",   cyan(output_csv))
    if upload:
        _summary_line("GCS target", cyan(f"gs://{bucket_name}/{blob_name}"))
        _summary_line("Credentials", cyan(key_file) if key_file else dim("Application Default Credentials"))
    else:
        _summary_line("GCS upload", yellow("skipped"))
    print(f"{'─' * 56}\n")

    if not _confirm("Proceed with the pipeline?", default=True):
        _abort()

    # ── Execute ─────────────────────────────────────────────────────────────
    print(f"\n{bold('  ── Processing JSON files ──')}\n")
    success = build_kronos_dataset_from_folder(
        target_folder=input_folder,
        output_csv_path=output_csv,
    )

    if not success:
        print(red("\n  Pipeline aborted — no output produced.\n"))
        sys.exit(1)

    if upload:
        print(f"\n{bold('  ── Uploading to GCS ──')}")
        upload_to_gcs(
            local_file_path=output_csv,
            bucket_name=bucket_name,
            key_file=key_file,
            dest_prefix=dest_prefix,
        )

    print(f"\n{green(bold('  ✔ Pipeline complete.'))} Local copy kept at {cyan(output_csv)}\n")


if __name__ == "__main__":
    main()