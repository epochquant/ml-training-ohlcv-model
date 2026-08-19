import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Union, List, Dict, Optional

# Ensure repository root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.high_volatility.model import HighVolatilityKronos, load_hv_kronos_from_base
from model.high_volatility.tokenizer import HighVolatilityTokenizer
from model.high_volatility.predictor import HighVolatilityPredictor


class TimeSeriesPredictorHV:
    """Production inference utility for High-Volatility Time Series Forecasting.

    Drop-in predictor that uses HighVolatilityPredictor with logreturn normalization
    and single-anchor continuation, eliminating initial candle gaps (+6%/-9%) and
    avoiding false mean-reversion crashes on breakout cryptocurrency symbols.
    """

    def __init__(self,
                 predictor_path: str = "NeoQuasar/Kronos-base",
                 tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
                 device: Optional[str] = None,
                 max_context: int = 512,
                 normalization_mode: str = "logreturn",
                 soft_clip: bool = True,
                 use_regime: bool = True,
                 n_regime_features: int = 8,
                 aggregation_mode: str = "regime_aligned"):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_context = max_context
        self.normalization_mode = normalization_mode
        self.soft_clip = soft_clip
        self.use_regime = use_regime
        self.n_regime_features = n_regime_features
        self.aggregation_mode = aggregation_mode

        # Load tokenizer and model
        print(f"[TimeSeriesPredictorHV] Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = HighVolatilityTokenizer.from_pretrained(tokenizer_path).to(self.device)

        print(f"[TimeSeriesPredictorHV] Loading model from: {predictor_path}")
        if os.path.exists(predictor_path) and os.path.isdir(predictor_path):
            self.model = HighVolatilityKronos.from_pretrained(predictor_path, n_regime_features=n_regime_features).to(self.device)
        else:
            self.model = load_hv_kronos_from_base(predictor_path, n_regime_features=n_regime_features).to(self.device)

        self.predictor = HighVolatilityPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_context=self.max_context,
            normalization_mode=self.normalization_mode,
            soft_clip=self.soft_clip,
            use_regime=self.use_regime,
            n_regime_features=self.n_regime_features,
            aggregation_mode=self.aggregation_mode
        )
        print(f"[TimeSeriesPredictorHV] Ready on device: {self.device} (Mode: {self.normalization_mode})")

    def predict_dataframe(self,
                          df: pd.DataFrame,
                          pred_len: int = 10,
                          freq: str = "3min",
                          sample_count: int = 20,
                          temperature: float = 1.0,
                          top_p: float = 0.90,
                          return_quantiles: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Predict future candle path from a DataFrame of historical candles.

        Args:
            df: Historical candles DataFrame containing 'open', 'high', 'low', 'close', 'volume',
                and a timestamp column ('datetime', 'timestamps', or 'openTime').
            pred_len: Number of future candles to forecast.
            freq: Timeframe frequency string (e.g. '1min', '3min', '5min', '15min').
            sample_count: Number of ensemble autoregressive rollout trajectories.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p threshold.
            return_quantiles: If True, returns dict of DataFrames with quantile bounds ('p10', 'p50', 'p90').

        Returns:
            pd.DataFrame or dict of DataFrames with predicted future candles.
        """
        df = df.copy()

        # Standardize timestamp column
        ts_col = None
        for candidate in ['datetime', 'timestamps', 'timestamp', 'openTime']:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            # Fallback: create synthetic timestamp series
            df['datetime'] = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=len(df), freq=freq)
            ts_col = 'datetime'
        elif ts_col == 'openTime' and pd.api.types.is_numeric_dtype(df[ts_col]):
            df['datetime'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
            ts_col = 'datetime'
        else:
            df['datetime'] = pd.to_datetime(df[ts_col], utc=True)
            ts_col = 'datetime'

        df = df.sort_values(ts_col).reset_index(drop=True)

        # Slice to max context if larger
        lookback = min(len(df), self.max_context)
        df_window = df.tail(lookback).copy().reset_index(drop=True)

        x_timestamp = pd.Series(df_window['datetime'])
        last_time = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(pd.date_range(start=last_time + pd.Timedelta(freq), periods=pred_len, freq=freq))

        result = self.predictor.predict(
            df=df_window,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
            return_quantiles=return_quantiles,
            quantiles=(0.1, 0.5, 0.9)
        )

        return result


def main():
    parser = argparse.ArgumentParser(description="Predict High-Volatility Time Series Candles")
    parser.add_argument('--file', type=str, required=True, help="Path to JSON or CSV candle file")
    parser.add_argument('--timeframe', type=str, default="3min", help="Candle timeframe (default: 3min)")
    parser.add_argument('--pred_len', type=int, default=10, help="Number of future candles to forecast (default: 10)")
    parser.add_argument('--sample_count', type=int, default=10, help="Ensemble sample count (default: 10)")
    parser.add_argument('--model_path', type=str, default="NeoQuasar/Kronos-base", help="Model checkpoint path")
    parser.add_argument('--tokenizer_path', type=str, default="NeoQuasar/Kronos-Tokenizer-base", help="Tokenizer checkpoint path")
    args = parser.parse_args()

    predictor = TimeSeriesPredictorHV(
        predictor_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )

    if args.file.endswith('.json'):
        with open(args.file, 'r') as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        if 'openTime' in df.columns:
            df['datetime'] = pd.to_datetime(df['openTime'], unit='ms', utc=True)
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = df[c].astype(float)
            df = df.set_index('datetime').resample(args.timeframe, closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
    else:
        df = pd.read_csv(args.file)

    preds = predictor.predict_dataframe(df, pred_len=args.pred_len, freq=args.timeframe, sample_count=args.sample_count)

    print("\n" + "=" * 60)
    print(f"HIGH-VOLATILITY TIME SERIES FORECAST ({args.timeframe}, {args.pred_len} steps)")
    print("=" * 60)
    print(preds[['open', 'high', 'low', 'close', 'volume']])
    last_close = float(df.iloc[-1]['close'])
    first_open = float(preds.iloc[0]['open'])
    first_close = float(preds.iloc[0]['close'])
    final_close = float(preds.iloc[-1]['close'])
    gap_pct = (first_open - last_close) / last_close * 100.0
    var_pct = (final_close - last_close) / last_close * 100.0
    print("-" * 60)
    print(f"Last Historical Close: {last_close:.6f}")
    print(f"First Predicted Open : {first_open:.6f} (Gap: {gap_pct:+.2f}%)")
    print(f"First Predicted Close: {first_close:.6f}")
    print(f"Final Predicted Close: {final_close:.6f} (Variance: {var_pct:+.2f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
