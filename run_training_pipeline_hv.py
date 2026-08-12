#!/usr/bin/env python3
"""
EpochQuant — High-Volatility Training Pipeline CLI
====================================================
Entrypoint for the `high-volatility-model` variant. Mirrors
run_training_pipeline.py's CLI shape, but always runs the `_hv` training
scripts and namespaces GCS artifacts under `high-volatility/`.
Run interactively:   python run_training_pipeline_hv.py
"""

import sys
import os
import subprocess
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MODEL_VARIANT = "high-volatility-model"

# ANSI colour helpers
_NO_COLOR = not sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"

def cyan(t):    return _c("96", t)
def green(t):   return _c("92", t)
def yellow(t):  return _c("93", t)
def red(t):     return _c("91", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)

def _prompt(label: str, default: str = "") -> str:
    default_hint = f"  {dim(f'[default: {default}]')}" if default else ""
    print(f"  {cyan('>')} {label}{default_hint}")
    try:
        value = input("    > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        _abort()
    return value if value else default

def _confirm(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    print(f"  {cyan('>')} {question}  {dim(hint)}")
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
    print(f"\n{yellow('  Aborted.')} Pipeline execution cancelled.\n")
    sys.exit(0)

def _section(n: int, total: int, title: str):
    print(f"\n{bold(f'[{n}/{total}]')} {bold(title)}")

def _banner():
    width = 52
    border = "=" * width
    print(f"\n{cyan('+' + border + '+')}")
    print(f"{cyan('|')}{bold('  EpochQuant — High-Volatility Training Pipeline CLI'):^{width + 8}}{cyan('|')}")
    print(f"{cyan('+' + border + '+')}\n")

def _summary_line(label: str, value: str):
    print(f"    {dim(label + ':'): <20} {value}")

def run_torchrun(module_name: str, num_gpus: str):
    print(f"\n{bold(f'  -- Starting {module_name} --')}\n")
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "-m",
        module_name
    ]
    print(f"  {dim('Executing:')} {cyan(' '.join(cmd))}\n")

    env = os.environ.copy()
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\n  {green('[OK]')} {module_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n  {red('[FAIL]')} {module_name} failed with exit code {e.returncode}.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"\n  {red('[FAIL]')} 'torchrun' command not found. Ensure PyTorch is installed and in your PATH.")
        sys.exit(1)

def run_shell_command(cmd: str):
    print(f"  {dim('Executing:')} {cyan(cmd)}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  {green('[OK]')} Command succeeded.")
    except subprocess.CalledProcessError as e:
        print(f"  {red('[FAIL]')} Command failed with exit code {e.returncode}.")

def download_from_gcs(gcs_uri: str, local_path: str) -> bool:
    """Download a file from GCS using google-cloud-storage Python SDK with CLI fallback."""
    try:
        from google.cloud import storage
        clean_uri = gcs_uri.replace("gs://", "")
        bucket_name, blob_name = clean_uri.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"  {green('[OK]')} [GCS SDK] Downloaded {gcs_uri} -> {local_path}")
        return True
    except Exception as e:
        print(f"  {yellow('[NOTICE]')} [GCS SDK] SDK download ({e}). Attempting CLI fallback...")
        for cmd in [f"gcloud storage cp \"{gcs_uri}\" \"{local_path}\"", f"gsutil cp \"{gcs_uri}\" \"{local_path}\""]:
            try:
                subprocess.run(cmd, shell=True, check=True)
                return True
            except Exception:
                continue
        return False

def upload_to_gcs(local_path: str, gcs_uri: str) -> bool:
    """Upload a file to GCS using google-cloud-storage Python SDK with CLI fallback."""
    try:
        from google.cloud import storage
        clean_uri = gcs_uri.replace("gs://", "")
        bucket_name, blob_name = clean_uri.split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"  {green('[OK]')} [GCS SDK] Uploaded {local_path} -> {gcs_uri}")
        return True
    except Exception as e:
        print(f"  {yellow('[NOTICE]')} [GCS SDK] SDK upload ({e}). Attempting CLI fallback...")
        for cmd in [f"gcloud storage cp \"{local_path}\" \"{gcs_uri}\"", f"gsutil cp \"{local_path}\" \"{gcs_uri}\""]:
            try:
                subprocess.run(cmd, shell=True, check=True)
                return True
            except Exception:
                continue
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EpochQuant — High-Volatility Training Pipeline CLI")
    parser.add_argument("--symbol", type=str, help="Symbol being trained")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--dataset-gs-uri", type=str, help="GCS URI to download the dataset from")
    args, _ = parser.parse_known_args()

    TOTAL_STEPS = 4
    dataset_path = ""

    if args.non_interactive:
        symbol = args.symbol.strip().lower() if args.symbol else "dogeusdt"

        if args.dataset_gs_uri:
            print(f"\n[Cloud Execution] Downloading dataset from {args.dataset_gs_uri}...")
            local_csv = "./downloaded_dataset_hv.csv"
            if download_from_gcs(args.dataset_gs_uri, local_csv):
                dataset_path = local_csv
                print(f"[Cloud Execution] Successfully downloaded to {dataset_path}")
            else:
                print(red("\n  [FAIL] Failed to download dataset from GCS. Exiting."))
                sys.exit(1)
        else:
            dataset_path = f"./output_csv/{symbol}_1m_kronos_data.csv"
    else:
        _banner()
        _section(1, TOTAL_STEPS, "Symbol Configuration")
        symbol = _prompt("Enter high-volatility symbol being trained (e.g. dogeusdt, pepeusdt):", default="dogeusdt").strip().lower()

        _section(2, TOTAL_STEPS, "Dataset")
        env_bucket = os.getenv("GCS_BUCKET_NAME", "epochquant-training")
        default_gcs_path = f"gs://{env_bucket}/high-volatility/training-data/{symbol}_1m.csv"
        dataset_path = _prompt("Path to the input CSV file (local path or gs:// URI):", default=default_gcs_path)

    if args.non_interactive:
        num_gpus = "1"
        run_tokenizer = True
        run_predictor = True
    else:
        _section(3, TOTAL_STEPS, "Hardware configuration")
        num_gpus = _prompt("Number of GPUs to use for DDP training:", default="1")
        if not num_gpus.isdigit() or int(num_gpus) < 1:
            print(red("  Invalid number of GPUs. Must be an integer >= 1."))
            _abort()

        _section(4, TOTAL_STEPS, "Training Stages")
        run_tokenizer = _confirm("Execute high-volatility tokenizer fine-tuning?", default=True)
        run_predictor = _confirm("Execute high-volatility predictor fine-tuning?", default=True)

        if not run_tokenizer and not run_predictor:
            print(yellow("\n  No training steps selected. Exiting.\n"))
            sys.exit(0)

        print(f"\n{'─' * 56}")
        print(bold("  High-Volatility Training Run Summary"))
        print(f"{'─' * 56}")
        _summary_line("Model Variant", cyan(MODEL_VARIANT))
        _summary_line("Symbol", cyan(symbol.upper()))
        _summary_line("Dataset Path", cyan(dataset_path))
        _summary_line("Num GPUs", cyan(num_gpus))
        _summary_line("Train Tokenizer", green("Yes") if run_tokenizer else yellow("No"))
        _summary_line("Train Predictor", green("Yes") if run_predictor else yellow("No"))
        print(f"{'─' * 56}\n")

        if not _confirm("Proceed with high-volatility training pipeline?", default=True):
            _abort()

    os.environ["DATASET_PATH"] = dataset_path
    os.environ["SYMBOL"] = symbol
    os.environ["MODEL_VARIANT"] = MODEL_VARIANT
    os.environ["PYTHONPATH"] = os.getcwd()

    print(f"\n{bold('  -- Running Pre-execution Commands --')}")
    run_shell_command("pip install -r requirements.txt")
    run_shell_command("pip install pandas")
    run_shell_command("pip install comet_ml")

    if run_tokenizer:
        run_torchrun("training.train_tokenizer_hv", num_gpus)

    if run_predictor:
        run_torchrun("training.train_predictor_hv", num_gpus)

    print(f"\n{bold('  -- Running Post-execution Commands --')}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_upper = symbol.upper()
    zip_filename = f"my_trained_kronos_hv_{symbol_upper}_{timestamp}.zip"

    run_shell_command(f"zip -r {zip_filename} ./output_models_hv")
    env_bucket = os.getenv("GCS_BUCKET_NAME", "epochquant-training")
    if not env_bucket.startswith("gs://"):
        env_bucket = f"gs://{env_bucket}"
    upload_to_gcs(zip_filename, f"{env_bucket}/high-volatility/models/{zip_filename}")

    print(f"\n{green(bold('  [OK] All selected high-volatility training pipelines complete.'))}\n")

if __name__ == "__main__":
    main()
