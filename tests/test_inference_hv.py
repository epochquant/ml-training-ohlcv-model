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

from src.inference.predict_time_series_hv import TimeSeriesPredictorHV
from src.inference.fastapi_time_series_router import (
    time_series_router,
    TimeSeriesPredictionRequest,
    CandleInput
)


def _generate_synthetic_candles(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    base_ts = pd.date_range("2026-08-18 10:00:00", periods=n, freq="3min", tz="UTC")
    prices = 0.050 + np.cumsum(np.random.randn(n) * 0.0002)
    # simulate a breakout
    prices[-10:] += 0.003
    highs = prices + 0.0003
    lows = prices - 0.0003
    opens = prices - 0.0001
    closes = prices
    vols = np.random.uniform(10000, 50000, n)
    return pd.DataFrame({
        "datetime": base_ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })


def test_time_series_predictor_dataframe():
    df = _generate_synthetic_candles(60)
    last_close = float(df.iloc[-1]['close'])

    predictor = TimeSeriesPredictorHV(
        predictor_path="NeoQuasar/Kronos-base",
        tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
        normalization_mode="logreturn",
        soft_clip=True,
        use_regime=True
    )

    pred_len = 5
    res = predictor.predict_dataframe(df, pred_len=pred_len, freq="3min", sample_count=5)

    assert isinstance(res, pd.DataFrame)
    assert len(res) == pred_len
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in res.columns

    first_open = float(res.iloc[0]['open'])
    gap_pct = abs(first_open - last_close) / last_close * 100.0
    assert gap_pct < 1.0, f"Gap too large: {gap_pct:.2f}%"

    # Check OHLC bounds
    for i in range(pred_len):
        o, h, l, c = res.iloc[i]['open'], res.iloc[i]['high'], res.iloc[i]['low'], res.iloc[i]['close']
        assert h >= max(o, c) - 1e-7
        assert l <= min(o, c) + 1e-7


def test_fastapi_time_series_request_schema():
    df = _generate_synthetic_candles(40)
    candles = [
        CandleInput(
            openTime=int(row['datetime'].timestamp() * 1000),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            takerBuyBaseAssetVolume=float(row['volume'] * 0.5)
        )
        for _, row in df.iterrows()
    ]

    req = TimeSeriesPredictionRequest(
        symbol="1000RATSUSDT",
        candles=candles,
        timeframe="3min",
        pred_len=5,
        sample_count=5
    )

    assert req.symbol == "1000RATSUSDT"
    assert len(req.candles) == 40
    assert req.timeframe == "3min"
    assert req.pred_len == 5
