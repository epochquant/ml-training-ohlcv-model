import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical and microstructural features for top reversal detection:
    - Taker Buy Volume Ratio (buying vs selling volume exhaustion)
    - Relative Volume Ratio (volume relative to 20-period EMA)
    - Upper Wick Ratio (selling pressure wick at high)
    - Price Overextension relative to 50-period EMA normalized by ATR
    """
    df = df.copy()

    # Ensure required numerical columns
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[col] = df[col].astype(float)

    if 'volume' not in df.columns:
        df['volume'] = 0.0
    else:
        df['volume'] = df['volume'].astype(float)

    # 1. Taker Buy Volume Ratio
    if 'takerBuyBaseAssetVolume' in df.columns and df['volume'].sum() > 0:
        df['takerBuyBaseAssetVolume'] = df['takerBuyBaseAssetVolume'].astype(float)
        df['taker_ratio'] = df['takerBuyBaseAssetVolume'] / (df['volume'] + 1e-8)
        df['taker_ratio'] = df['taker_ratio'].clip(0.0, 1.0)
    else:
        df['taker_ratio'] = 0.5  # Neutral default if missing

    # 2. Relative Volume Ratio (Volume / EMA_20(Volume))
    vol_ema = df['volume'].ewm(span=20, adjust=False).mean()
    df['vol_ratio'] = np.log1p(df['volume'] / (vol_ema + 1e-5))

    # 3. Upper Wick Ratio: (High - max(Open, Close)) / (High - Low + 1e-5)
    body_max = np.maximum(df['open'], df['close'])
    candle_range = df['high'] - df['low'] + 1e-5
    df['upper_wick_ratio'] = (df['high'] - body_max) / candle_range

    # 4. Average True Range (ATR 14) and Price Overextension
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = tr.ewm(span=14, adjust=False).mean()

    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    df['overextension'] = (df['close'] - ema50) / (atr + 1e-5)

    return df


def label_top_reversals(df: pd.DataFrame, 
                        lookaround: int = 12, 
                        min_drop_pct: float = 3.0, 
                        max_upside_pct: float = 0.8) -> pd.DataFrame:
    """
    Labels candles with 1 (Short Reversal Signal) or 0 (Normal Candle).
    
    A candle at index i is labeled 1 if:
    1. High at i is the maximum high in local window [i - lookaround, i + lookaround].
    2. Within the next `lookaround` candles, price drops by at least `min_drop_pct` below High[i].
    3. Maximum upside price move before the drop is <= `max_upside_pct` above High[i].
    """
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    max_drawdown = np.zeros(n, dtype=float)

    highs = df['high'].values
    closes = df['close'].values

    for i in range(lookaround, n - lookaround):
        current_high = highs[i]
        
        # Check 1: Is current high the local maximum?
        window_highs = highs[i - lookaround : i + lookaround + 1]
        if current_high < np.max(window_highs):
            continue

        # Forward window for evaluation
        future_highs = highs[i + 1 : i + lookaround + 1]
        future_closes = closes[i + 1 : i + lookaround + 1]

        max_future_high = np.max(future_highs)
        min_future_close = np.min(future_closes)

        # Drawdown calculation relative to peak high
        drop_pct = (current_high - min_future_close) / current_high * 100.0
        upside_pct = (max_future_high - current_high) / current_high * 100.0

        max_drawdown[i] = drop_pct

        # Check 2 & 3: Significant drop with low upside risk
        if drop_pct >= min_drop_pct and upside_pct <= max_upside_pct:
            labels[i] = 1

    df['label'] = labels
    df['future_max_drop_pct'] = max_drawdown
    return df


def process_single_file(file_path: Path, lookaround: int = 12, min_drop_pct: float = 3.0) -> pd.DataFrame:
    """Loads a single JSON or CSV candle file, computes indicators and labels tops."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Standardize column names
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
    df = compute_indicators(df)
    df = label_top_reversals(df, lookaround=lookaround, min_drop_pct=min_drop_pct)
    return df


def process_dataset(input_path: str, output_csv: str, min_drop_pct: float = 3.0):
    """Processes a single file or a directory of JSON/CSV candle files and merges them."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob('*.json')) + list(input_path.glob('*.csv'))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"Found {len(files)} candle data file(s). Processing...")

    all_dfs = []
    total_signals = 0

    for f in files:
        try:
            df = process_single_file(f, min_drop_pct=min_drop_pct)
            num_signals = df['label'].sum()
            total_signals += num_signals
            print(f"  [+] {f.name}: {len(df)} rows | {num_signals} Top Reversal Signals")
            all_dfs.append(df)
        except Exception as e:
            print(f"  [!] Error processing {f.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No valid data frames were processed.")

    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    master_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Processed {len(master_df)} total candles.")
    print(f"Total Top Reversal Signals (Label = 1): {total_signals} ({total_signals / len(master_df) * 100:.2f}%)")
    print(f"Dataset exported to: {output_csv}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate labeled dataset for Kronos Short Reversal Classifier.")
    parser.add_argument('--input', type=str, required=True, help="Path to input JSON/CSV file or directory.")
    parser.add_argument('--output', type=str, default="./dataset/kronos_short_labeled.csv", help="Path to output CSV.")
    parser.add_argument('--min_drop', type=float, default=3.0, help="Minimum percentage drop for top reversal label (default: 3.0%).")

    args = parser.parse_args()
    process_dataset(args.input, args.output, min_drop_pct=args.min_drop)
