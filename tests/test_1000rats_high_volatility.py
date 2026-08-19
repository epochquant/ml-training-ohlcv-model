import json
import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.high_volatility.normalization import normalize_window, denormalize_continuation
from model.high_volatility.predictor import HighVolatilityPredictor, safe_calc_time_stamps
from model.high_volatility.model import HighVolatilityKronos, load_hv_kronos_from_base
from model.high_volatility.tokenizer import HighVolatilityTokenizer
from model.high_volatility.regime import compute_regime_vector


def _load_1000rats_3m_data():
    """Helper to load and resample 1000RATSUSDT data."""
    raw_path = Path(r"C:\00 - GITHUB\volume-usdt-batch\candles\historical_futures\1000RATSUSDT\1m\2026-08-15_2026-08-19.json")
    if not raw_path.exists():
        # Synthetic fallback if external path is not mounted in some test environment
        np.random.seed(42)
        n = 1000
        base_ts = pd.date_range("2026-08-18 10:00:00", periods=n, freq="3min", tz="UTC")
        prices = 0.040 + np.cumsum(np.random.randn(n) * 0.0001)
        # simulate breakout around index 800
        prices[800:] += 0.010
        highs = prices + 0.0005
        lows = prices - 0.0005
        opens = prices - 0.0001
        closes = prices
        vols = np.random.uniform(10000, 500000, n)
        df_3m = pd.DataFrame({
            "datetime": base_ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "quoteAssetVolume": vols * closes,
        })
        target_ts = base_ts[810]
        return df_3m, target_ts

    with open(raw_path, "r") as f:
        raw = json.load(f)
    df_1m = pd.DataFrame(raw)
    if "openTime" in df_1m.columns:
        df_1m["datetime"] = pd.to_datetime(df_1m["openTime"], unit="ms", utc=True)
    elif "timestamp" in df_1m.columns:
        df_1m["datetime"] = pd.to_datetime(df_1m["timestamp"], utc=True)

    df_1m = df_1m.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "quoteAssetVolume", "takerBuyBaseAssetVolume", "takerBuyQuoteAssetVolume"]:
        if col in df_1m.columns:
            df_1m[col] = df_1m[col].astype(float)

    df_1m = df_1m.set_index("datetime")
    df_3m = df_1m.resample("3min", closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quoteAssetVolume": "sum",
        "takerBuyBaseAssetVolume": "sum",
        "takerBuyQuoteAssetVolume": "sum"
    }).dropna().reset_index()

    target_ts = pd.Timestamp("2026-08-18 14:09:00", tz="UTC")
    return df_3m, target_ts


def test_1000rats_gap_elimination_and_geometry():
    """Verify that 1000RATSUSDT at 14:09 has zero unphysical price gap and valid geometry."""
    df_3m, target_ts = _load_1000rats_3m_data()
    idx = df_3m[df_3m["datetime"] <= target_ts].index[-1]
    window = df_3m.iloc[idx - 399 : idx + 1].copy().reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "volume", "quoteAssetVolume"]
    x_raw = window[feature_cols].values.astype(np.float64)
    last_close = x_raw[-1, 3]

    # Normalize
    x_norm, stats = normalize_window(x_raw, mode="logreturn", clip=5.0, soft_clip=True)

    # Simulate 15-step continuation
    np.random.seed(42)
    dummy_pred_norm = np.random.randn(15, len(feature_cols)) * 0.5
    denorm = denormalize_continuation(dummy_pred_norm, stats)

    assert denorm.shape == (15, len(feature_cols))
    assert not np.isnan(denorm).any()

    # 1. Step 0 Open Gap is eliminated (< 1% vs last close, avoiding previous +6%/-9% jump)
    step0_open = denorm[0, 0]
    gap_pct = abs(step0_open - last_close) / last_close * 100.0
    assert gap_pct < 1.0, f"Step 0 open gap too large: {gap_pct:.2f}%"

    # 2. Strict Candlestick Geometry: High >= max(Open, Close) and Low <= min(Open, Close)
    for i in range(15):
        o, h, l, c = denorm[i, 0], denorm[i, 1], denorm[i, 2], denorm[i, 3]
        assert h >= max(o, c) - 1e-7, f"Step {i}: High {h} < max(Open {o}, Close {c})"
        assert l <= min(o, c) + 1e-7, f"Step {i}: Low {l} > min(Open {o}, Close {c})"

    # 3. Inter-candle contiguity: Open[i] == Close[i-1] for i >= 1
    assert np.allclose(denorm[1:, 0], denorm[:-1, 3], rtol=1e-5), "Candles are not contiguous"


def test_breakout_multi_timestamp_continuity():
    """Verify gap elimination across all volatile breakout timestamps (14:00 to 14:15)."""
    df_3m, _ = _load_1000rats_3m_data()
    feature_cols = ["open", "high", "low", "close", "volume", "quoteAssetVolume"]

    test_timestamps = [
        "2026-08-18 14:00:00",
        "2026-08-18 14:03:00",
        "2026-08-18 14:06:00",
        "2026-08-18 14:09:00",
        "2026-08-18 14:12:00",
    ]

    for t_str in test_timestamps:
        target_ts = pd.Timestamp(t_str, tz="UTC")
        matches = df_3m[df_3m["datetime"] <= target_ts]
        if matches.empty:
            continue
        idx = matches.index[-1]
        window = df_3m.iloc[max(0, idx - 399) : idx + 1].copy().reset_index(drop=True)
        x_raw = window[feature_cols].values.astype(np.float64)
        last_close = x_raw[-1, 3]

        _, stats = normalize_window(x_raw, mode="logreturn")
        pred_zeros = np.zeros((5, len(feature_cols)))
        denorm = denormalize_continuation(pred_zeros, stats)

        open_gap = abs(denorm[0, 0] - last_close) / last_close * 100.0
        assert open_gap < 1.0, f"At {t_str}, open gap was {open_gap:.2f}%"


def test_safe_calc_time_stamps_and_datetime_index():
    """Verify safe_calc_time_stamps works seamlessly with both pd.DatetimeIndex and pd.Series."""
    dt_idx = pd.date_range("2026-08-18 14:09:00", periods=5, freq="3min", tz="UTC")
    df_res = safe_calc_time_stamps(dt_idx)
    assert isinstance(df_res, pd.DataFrame)
    assert "minute" in df_res.columns
    assert "hour" in df_res.columns
    assert len(df_res) == 5

    dt_series = pd.Series(dt_idx)
    df_res_series = safe_calc_time_stamps(dt_series)
    assert len(df_res_series) == 5
    assert df_res_series["minute"].tolist() == df_res["minute"].tolist()
