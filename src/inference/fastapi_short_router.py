"""
FastAPI Router for Kronos Short Reversal Signal Classifier.

Copy this file to your FastAPI inference repository (e.g. ml-inference-ohlcv)
and register the router in your main FastAPI application:

    from fastapi import FastAPI
    from fastapi_short_router import short_router

    app = FastAPI()
    app.include_router(short_router, prefix="/api/v1")
"""

from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from src.inference.predict_short_signal import ShortSignalPredictor

short_router = APIRouter(tags=["Kronos Short Reversal Detector"])

# Global Predictor Instance (Lazy loaded)
_predictor: Optional[ShortSignalPredictor] = None


def get_predictor() -> ShortSignalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ShortSignalPredictor(
            model_path="./output_models/kronos_short_classifier.pt",
            pretrained_kronos="NeoQuasar/Kronos-base",
            pretrained_tokenizer="NeoQuasar/Kronos-Tokenizer-base"
        )
    return _predictor


class CandleInput(BaseModel):
    openTime: Optional[int] = None
    timestamps: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: float
    takerBuyBaseAssetVolume: Optional[float] = 0.0


class ShortPredictionRequest(BaseModel):
    symbol: str = Field(..., example="GIGGLEUSDT")
    threshold: float = Field(default=0.80, ge=0.5, le=0.99, description="Confidence threshold")
    candles: List[CandleInput] = Field(..., description="List of historical 5m candles (at least 400 rows)")


class ShortPredictionResponse(BaseModel):
    symbol: str
    short_probability: float
    short_probability_pct: str
    is_short_signal: bool
    signal_action: str
    threshold_used: float
    latest_close: float
    latest_high: float
    suggested_take_profit: str
    suggested_stop_loss: str


@short_router.post("/predict-short-signal", response_model=ShortPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_short_signal(payload: ShortPredictionRequest):
    """
    Evaluates historical 5m candles and predicts if the current candle is a blow-off top ready to drop.
    """
    if len(payload.candles) < 400:
        raise HTTPException(
            status_code=422,
            detail=f"At least 400 candles are required for inference. Received {len(payload.candles)}."
        )

    try:
        raw_data = [candle.dict() for candle in payload.candles]
        df = pd.DataFrame(raw_data)

        if 'openTime' in df.columns and df['openTime'].notnull().any():
            df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')

        predictor = get_predictor()
        result = predictor.predict_dataframe(df, threshold=payload.threshold)
        result["symbol"] = payload.symbol

        return ShortPredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
