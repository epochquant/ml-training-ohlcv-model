import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def compute_indicators_hv(df: pd.DataFrame) -> pd.DataFrame:
    """High-volatility counterpart of src.data.label_short_signals.compute_indicators.

    Identical feature set, but additionally exposes ATR as a percentage of
    price (`atr_pct`) so that label_top_reversals_hv can threshold drop/upside
    relative to the asset's own recent volatility instead of a flat percentage.
    src/data/label_short_signals.py::compute_indicators itself is untouched.
    """
    df = df.copy()

    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[col].astype(float)

    if 'volume' not in df.columns:
        df['volume'] = 0.0
    else:
        df['volume'] = df['volume'].astype(float)

    if 'takerBuyBaseAssetVolume' in df.columns and df['volume'].sum() > 0:
        df['takerBuyBaseAssetVolume'] = df['takerBuyBaseAssetVolume'].astype(float)
        df['taker_ratio'] = df['takerBuyBaseAssetVolume'] / (df['volume'] + 1e-8)
        df['taker_ratio'] = df['taker_ratio'].clip(0.0, 1.0)
    else:
        df['taker_ratio'] = 0.5

    vol_ema = df['volume'].ewm(span=20, adjust=False).mean()
    df['vol_ratio'] = np.log1p(df['volume'] / (vol_ema + 1e-5))

    body_max = np.maximum(df['open'], df['close'])
    candle_range = df['high'] - df['low'] + 1e-5
    df['upper_wick_ratio'] = (df['high'] - body_max) / candle_range

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = tr.ewm(span=14, adjust=False).mean()

    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    df['overextension'] = (df['close'] - ema50) / (atr + 1e-5)
    df['atr_pct'] = (atr / (df['close'].abs() + 1e-8)) * 100.0

    return df


def label_top_reversals_hv(df: pd.DataFrame,
                            lookaround: int = 12,
                            k_drop: float = 1.5,
                            k_upside: float = 0.4) -> pd.DataFrame:
    """ATR-relative counterpart of label_short_signals.label_top_reversals.

    Same local-maximum + forward-drawdown logic, but the drop/upside
    thresholds scale with each candle's own `atr_pct` (k_drop/k_upside are
    multiples of ATR%) instead of a flat percentage -- a 5% drop is noise on
    a memecoin with 8% ATR but a real signal on an asset with 1% ATR.
    Requires `atr_pct` from compute_indicators_hv.
    """
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    max_drawdown = np.zeros(n, dtype=float)

    highs = df['high'].values
    closes = df['close'].values
    atr_pct = df['atr_pct'].values

    for i in range(lookaround, n - lookaround):
        current_high = highs[i]

        window_highs = highs[i - lookaround: i + lookaround + 1]
        if current_high < np.max(window_highs):
            continue

        future_highs = highs[i + 1: i + lookaround + 1]
        future_closes = closes[i + 1: i + lookaround + 1]

        max_future_high = np.max(future_highs)
        min_future_close = np.min(future_closes)

        drop_pct = (current_high - min_future_close) / current_high * 100.0
        upside_pct = (max_future_high - current_high) / current_high * 100.0

        max_drawdown[i] = drop_pct

        candle_atr_pct = max(atr_pct[i], 1e-5)
        min_drop_pct = k_drop * candle_atr_pct
        max_upside_pct = k_upside * candle_atr_pct

        if drop_pct >= min_drop_pct and upside_pct <= max_upside_pct:
            labels[i] = 1

    df['label'] = labels
    df['future_max_drop_pct'] = max_drawdown
    return df


def process_single_file_hv(file_path: Path, lookaround: int = 12, k_drop: float = 1.5, k_upside: float = 0.4) -> pd.DataFrame:
    """Loads a single JSON or CSV candle file, computes indicators and labels tops (ATR-relative)."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    if 'openTime' in df.columns:
        df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')
    elif 'timestamp' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamp'])
    elif 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'])

    req_cols = ['open', 'high', 'low', 'close']
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {file_path}")

    df = df.sort_values('timestamps').reset_index(drop=True)
    df = compute_indicators_hv(df)
    df = label_top_reversals_hv(df, lookaround=lookaround, k_drop=k_drop, k_upside=k_upside)
    return df


def process_dataset_hv(input_path: str, output_csv: str, k_drop: float = 1.5, k_upside: float = 0.4):
    """Processes a single file or a directory of JSON/CSV candle files and merges them (ATR-relative labels)."""
    input_path = Path(input_path)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.rglob('*.json')) + list(input_path.rglob('*.csv'))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Found {len(files)} candle data file(s). Processing (ATR-relative, k_drop={k_drop}, k_upside={k_upside})...")

    all_dfs = []
    total_signals = 0

    for f in files:
        try:
            df = process_single_file_hv(f, k_drop=k_drop, k_upside=k_upside)
            num_signals = df['label'].sum()
            total_signals += num_signals
            print(f"  [+] {f.name}: {len(df)} rows | {num_signals} Top Reversal Signals")
            all_dfs.append(df)
        except Exception as e:
            print(f"  [!] Error processing {f.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No valid data frames were processed.")

    master_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Processed {len(master_df)} total candles.")
    print(f"Total Top Reversal Signals (Label = 1): {total_signals} ({total_signals / len(master_df) * 100:.2f}%)")
    print(f"Dataset exported to: {output_csv}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ATR-relative labeled dataset for the high-volatility Short Reversal Classifier.")
    parser.add_argument('--input', type=str, required=True, help="Path to input JSON/CSV file or directory.")
    parser.add_argument('--output', type=str, default="./output_csv_hv/_changeit_master_short_labeled_hv.csv", help="Path to output CSV.")
    parser.add_argument('--k_drop', type=float, default=1.5, help="Minimum drop as a multiple of ATR%% for a top reversal label (default: 1.5).")
    parser.add_argument('--k_upside', type=float, default=0.4, help="Maximum allowed upside as a multiple of ATR%% before the drop (default: 0.4).")

    args = parser.parse_args()
    process_dataset_hv(args.input, args.output, k_drop=args.k_drop, k_upside=args.k_upside)
