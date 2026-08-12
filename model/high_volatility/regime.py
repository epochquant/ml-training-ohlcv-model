import numpy as np
import torch
import torch.nn as nn


class RegimeEmbedding(nn.Module):
    """Additive volatility-regime embedding for HighVolatilityKronos (Area B / B1).

    Projects a small continuous feature vector (e.g. ATR%, overextension,
    volume-ratio) to d_model and adds it into the token embedding stream,
    the same way TemporalEmbedding (model/module.py) adds calendar features.

    Both the linear projection's weight and bias are zero-initialized, so
    immediately after warm-starting from a pretrained Kronos checkpoint this
    module contributes exactly zero -- HighVolatilityKronos is numerically
    identical to the base Kronos until fine-tuned.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.proj = nn.Linear(n_features, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, regime: torch.Tensor) -> torch.Tensor:
        """regime: [batch, seq_len, n_features] -> [batch, seq_len, d_model]"""
        return self.proj(regime)


def _ewm_mean(arr: np.ndarray, span: float) -> np.ndarray:
    """Minimal recursive EWM mean (adjust=False), matching pandas' ewm(span=...).mean()."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_regime_vector(x: np.ndarray, price_idx=(0, 1, 2, 3), vol_idx: int = 4,
                           atr_span: float = 14.0, trend_span: float = 50.0,
                           vol_span: float = 20.0, eps: float = 1e-8) -> np.ndarray:
    """Compute a fixed-size [atr_pct, overextension, vol_ratio] regime vector from a
    raw OHLCV window, using the same formulas as
    src/data/label_short_signals_hv.py::compute_indicators_hv, but operating
    directly on a numpy window (no pandas) so it can run inside the
    autoregressive inference loop.

    x: np.ndarray [seq_len, n_features], columns ordered so that price_idx
       indexes (open, high, low, close) and vol_idx indexes volume -- this
       matches KronosPredictor's `price_cols + [vol_col, amt_vol]` convention.
    """
    x = np.asarray(x, dtype=np.float64)
    _, h_idx, l_idx, c_idx = price_idx
    high, low, close = x[:, h_idx], x[:, l_idx], x[:, c_idx]

    prev_close = np.concatenate([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _ewm_mean(tr, span=atr_span)
    atr_pct = atr[-1] / (close[-1] + eps)

    ema_trend = _ewm_mean(close, span=min(trend_span, len(close)))
    overextension = (close[-1] - ema_trend[-1]) / (atr[-1] + eps)

    volume = x[:, vol_idx]
    ema_vol = _ewm_mean(volume, span=min(vol_span, len(volume)))
    vol_ratio = np.log1p(volume[-1] / (ema_vol[-1] + eps))

    return np.array([atr_pct, overextension, vol_ratio], dtype=np.float32)
