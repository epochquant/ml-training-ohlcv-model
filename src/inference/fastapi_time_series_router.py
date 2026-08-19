"""
FastAPI Router for Kronos High-Volatility Time Series Forecasting.

Copy this file to your FastAPI inference repository (e.g. ml-inference-ohlcv)
and register the router in your main FastAPI application:

    from fastapi import FastAPI
    from fastapi_time_series_router import time_series_router

    app = FastAPI()
    app.include_router(time_series_router, prefix="/api/v1")
"""

from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from src.inference.predict_time_series_hv import TimeSeriesPredictorHV

time_series_router = APIRouter(tags=["Kronos Time Series Forecast"])

# Global Predictor Instance (Lazy loaded)
_ts_predictor: Optional[TimeSeriesPredictorHV] = None


def get_time_series_predictor() -> TimeSeriesPredictorHV:
    global _ts_predictor
    if _ts_predictor is None:
        _ts_predictor = TimeSeriesPredictorHV(
            predictor_path="NeoQuasar/Kronos-base",
            tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
            normalization_mode="logreturn",
            soft_clip=True,
            use_regime=True,
            aggregation_mode="regime_aligned"
        )
    return _ts_predictor


class CandleInput(BaseModel):
    openTime: Optional[int] = None
    timestamps: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float
    takerBuyBaseAssetVolume: Optional[float] = 0.0


class PredictedCandle(BaseModel):
    openTime: int
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeSeriesPredictionRequest(BaseModel):
    symbol: str = Field(..., example="1000RATSUSDT")
    candles: List[CandleInput] = Field(..., description="List of historical candles (at least 60 rows, ideally 400)")
    timeframe: str = Field(default="3min", description="Candle timeframe (e.g. 1min, 3min, 5min, 15min)")
    pred_len: int = Field(default=10, ge=1, le=60, description="Number of future candles to forecast")
    sample_count: int = Field(default=20, ge=1, le=50, description="Ensemble sample rollout count")


class TimeSeriesPredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    latest_close: float
    forecasted_close: float
    variance_pct: float
    variance_display: str
    trend_direction: str
    gap_pct: float
    forecasted_candles: List[PredictedCandle]


@time_series_router.post("/predict-time-series", response_model=TimeSeriesPredictionResponse, status_code=status.HTTP_200_OK)
@time_series_router.post("/predict-chart", response_model=TimeSeriesPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_time_series(payload: TimeSeriesPredictionRequest):
    """
    Evaluates historical candles and returns continuous, gap-free future candle forecast
    specifically optimized for volatile symbols and breakouts.
    """
    if len(payload.candles) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"At least 30 historical candles are required. Received {len(payload.candles)}."
        )

    try:
        raw_data = [candle.dict() for candle in payload.candles]
        df = pd.DataFrame(raw_data)

        if 'openTime' in df.columns and df['openTime'].notnull().any():
            df['datetime'] = pd.to_datetime(df['openTime'], unit='ms', utc=True)
        elif 'timestamps' in df.columns and df['timestamps'].notnull().any():
            df['datetime'] = pd.to_datetime(df['timestamps'], utc=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        predictor = get_time_series_predictor()
        pred_df = predictor.predict_dataframe(
            df=df,
            pred_len=payload.pred_len,
            freq=payload.timeframe,
            sample_count=payload.sample_count
        )

        last_close = float(df.iloc[-1]['close'])
        first_open = float(pred_df.iloc[0]['open'])
        final_close = float(pred_df.iloc[-1]['close'])
        gap_pct = (first_open - last_close) / last_close * 100.0
        variance_pct = (final_close - last_close) / last_close * 100.0

        if variance_pct > 0.5:
            trend_dir = "BULLISH"
        elif variance_pct < -0.5:
            trend_dir = "BEARISH"
        else:
            trend_dir = "NEUTRAL"

        forecasted_list = []
        for dt_idx, row in pred_df.iterrows():
            ts_ms = int(dt_idx.timestamp() * 1000) if hasattr(dt_idx, 'timestamp') else int(pd.Timestamp(dt_idx).timestamp() * 1000)
            forecasted_list.append(PredictedCandle(
                openTime=ts_ms,
                datetime=str(dt_idx),
                open=round(float(row['open']), 8),
                high=round(float(row['high']), 8),
                low=round(float(row['low']), 8),
                close=round(float(row['close']), 8),
                volume=round(float(row.get('volume', 0.0)), 2)
            ))

        return TimeSeriesPredictionResponse(
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            latest_close=last_close,
            forecasted_close=final_close,
            variance_pct=round(variance_pct, 2),
            variance_display=f"{variance_pct:+.2f}%",
            trend_direction=trend_dir,
            gap_pct=round(gap_pct, 2),
            forecasted_candles=forecasted_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
