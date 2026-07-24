import os
import json
import glob
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# GCS Loader
# ---------------------------------------------------------------------------

def load_from_gcs(gcs_path: str, gcs_project: str = None) -> pd.DataFrame:
    """
    Load a CSV directly from a GCS bucket.

    Requires: pip install gcsfs

    Args:
        gcs_path:    Full GCS URI, e.g. "gs://epochquant-training/processed/bnbusdt_1m.csv"
        gcs_project: GCP project ID (optional, needed if ADC does not resolve it).

    Returns:
        Preprocessed DataFrame with columns:
        [timestamps, open, high, low, close, volume, amount]
    """
    try:
        import gcsfs  # noqa: F401 — triggers gcsfs registration with fsspec
    except ImportError:
        raise ImportError(
            "gcsfs is required for GCS access.\n"
            "Install with: pip install gcsfs google-cloud-storage"
        )

    print(f"[GCS] Loading: {gcs_path}")
    storage_options = {"project": gcs_project} if gcs_project else {}
    df = pd.read_csv(gcs_path, storage_options=storage_options)

    if "timestamps" not in df.columns and "timestamp" in df.columns:
        df.rename(columns={"timestamp": "timestamps"}, inplace=True)

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return preprocess_ohlcv(df)


# ---------------------------------------------------------------------------
# Local Loaders
# ---------------------------------------------------------------------------

def load_from_json_folder(folder_path: str) -> pd.DataFrame:
    """
    Read all *.json files in folder_path (Binance-style kline arrays).
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return pd.DataFrame()

    all_data = []
    for file_path in json_files:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                all_data.extend(json.load(f))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    if "openTime" in df.columns:
        df["timestamps"] = pd.to_datetime(df["openTime"], unit="ms")
    elif "timestamp" in df.columns:
        df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")

    if "quoteAssetVolume" in df.columns:
        df["amount"] = df["quoteAssetVolume"]
    elif "takerBuyQuoteAssetVolume" in df.columns and "amount" not in df.columns:
        df["amount"] = df["takerBuyQuoteAssetVolume"]

    return preprocess_ohlcv(df)


def load_from_csv(file_path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV file."""
    if not os.path.exists(file_path):
        print(f"CSV file not found: {file_path}")
        return pd.DataFrame()

    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    for alias in ("timestamp", "date"):
        if "timestamps" not in df.columns and alias in df.columns:
            df.rename(columns={alias: "timestamps"}, inplace=True)

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return preprocess_ohlcv(df)


# ---------------------------------------------------------------------------
# Smart Dispatcher
# ---------------------------------------------------------------------------

def load_dataset(path: str, gcs_project: str = None) -> pd.DataFrame:
    """
    Auto-detect and load from GCS URI, local CSV, or local JSON folder.

    Args:
        path:        "gs://epochquant-training/processed/bnbusdt_1m.csv"
                     OR  "./data/raw/bnbusdt/"
                     OR  "./data/processed/bnbusdt_1m.csv"
        gcs_project: Optional GCP project ID (for GCS paths).

    Returns:
        Preprocessed DataFrame.
    """
    if path.startswith("gs://"):
        return load_from_gcs(path, gcs_project=gcs_project)
    elif os.path.isdir(path):
        return load_from_json_folder(path)
    elif path.endswith(".csv"):
        return load_from_csv(path)
    else:
        raise ValueError(f"Cannot determine data source type for: {path}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, sorts, and validates OHLCV data.
    Returns standard columns: [timestamps, open, high, low, close, volume, amount]
    """
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = df.sort_values("timestamps").reset_index(drop=True)
    df = df.ffill().bfill()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from data.")

    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].copy()


def chunk_data(df: pd.DataFrame, sequence_length: int, step: int = None):
    """Split a DataFrame into contiguous chunks of sequence_length."""
    if step is None:
        step = sequence_length
    return [
        df.iloc[i: i + sequence_length].copy().reset_index(drop=True)
        for i in range(0, len(df) - sequence_length + 1, step)
    ]
