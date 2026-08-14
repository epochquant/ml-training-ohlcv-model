import numpy as np
import torch
import torch.nn as nn


class RegimeEmbedding(nn.Module):
    """Additive volatility-regime embedding for HighVolatilityKronos (Area B / B1).

    Projects a multi-scale continuous feature vector (e.g. ATR%, short & macro overextension,
    multi-timeframe returns, volume-ratio, taker-ratio, wick rejection) to d_model and adds
    it into the token embedding stream, the same way TemporalEmbedding adds calendar features.

    Uses a 2-layer MLP with LayerNorm and SiLU, and a zero-initialized final projection
    so immediately after warm-starting from a pretrained Kronos checkpoint this module
    contributes exactly zero -- HighVolatilityKronos is numerically identical to the base
    Kronos until fine-tuned.
    """

    def __init__(self, n_features: int, d_model: int, hidden_dim: int = None):
        super().__init__()
        self.n_features = n_features
        hidden = hidden_dim or max(d_model // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, regime: torch.Tensor) -> torch.Tensor:
        """regime: [batch, seq_len, n_features] -> [batch, seq_len, d_model]"""
        return self.mlp(regime)


def _ewm_mean(arr: np.ndarray, span: float) -> np.ndarray:
    """Minimal recursive EWM mean (adjust=False), matching pandas' ewm(span=...).mean()."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_regime_vector(x: np.ndarray, price_idx=(0, 1, 2, 3), vol_idx: int = 4,
                           taker_idx: int = None, atr_span: float = 14.0,
                           trend_span: float = 50.0, macro_trend_span: float = 200.0,
                           vol_span: float = 20.0, eps: float = 1e-8,
                           n_features: int = 8) -> np.ndarray:
    """Compute a multi-scale regime vector from a raw OHLCV window.

    By default (n_features=8), computes:
      1. atr_pct: Short-term volatility magnitude relative to price.
      2. overextension_short: (close - EMA50) / ATR (short-term momentum/overextension).
      3. overextension_macro: (close - EMA200) / ATR (macro trend/support alignment).
      4. ret_60m: 1-hour percentage return (intermediate momentum).
      5. ret_240m: 4-hour percentage return (macro trend direction).
      6. vol_ratio: log1p(volume / EMA20_volume) (volume surge intensity).
      7. taker_ratio: Taker buy volume ratio (buyer aggression / orderflow).
      8. lower_wick_ratio: (min(open, close) - low) / (high - low) (rejection of lower prices).

    If n_features=3 (legacy mode), returns [atr_pct, overextension_short, vol_ratio].
    """
    x = np.asarray(x, dtype=np.float64)
    o_idx, h_idx, l_idx, c_idx = price_idx
    open_p, high, low, close = x[:, o_idx], x[:, h_idx], x[:, l_idx], x[:, c_idx]

    prev_close = np.concatenate([close[:1], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _ewm_mean(tr, span=atr_span)
    atr_pct = atr[-1] / (close[-1] + eps)

    ema_trend = _ewm_mean(close, span=min(trend_span, len(close)))
    overextension_short = (close[-1] - ema_trend[-1]) / (atr[-1] + eps)

    volume = x[:, vol_idx]
    ema_vol = _ewm_mean(volume, span=min(vol_span, len(volume)))
    vol_ratio = np.log1p(volume[-1] / (ema_vol[-1] + eps))

    if n_features == 3:
        return np.array([atr_pct, overextension_short, vol_ratio], dtype=np.float32)

    # Macro EMA (e.g. 200)
    ema_macro = _ewm_mean(close, span=min(macro_trend_span, len(close)))
    overextension_macro = (close[-1] - ema_macro[-1]) / (atr[-1] + eps)

    # Multi-timeframe returns (60m and 240m)
    idx_60 = max(0, len(close) - 60)
    ret_60m = (close[-1] - close[idx_60]) / (close[idx_60] + eps)

    idx_240 = max(0, len(close) - 240)
    ret_240m = (close[-1] - close[idx_240]) / (close[idx_240] + eps)

    # Taker ratio
    if taker_idx is not None and taker_idx < x.shape[1]:
        taker_vol = x[:, taker_idx]
        taker_ratio = float(np.clip(taker_vol[-1] / (volume[-1] + eps), 0.0, 1.0))
    else:
        taker_ratio = 0.5

    # Lower wick ratio: absorption of selling pressure
    candle_range = high[-1] - low[-1] + eps
    body_min = np.minimum(open_p[-1], close[-1])
    lower_wick_ratio = float(np.clip((body_min - low[-1]) / candle_range, 0.0, 1.0))

    return np.array([
        atr_pct,
        overextension_short,
        overextension_macro,
        ret_60m,
        ret_240m,
        vol_ratio,
        taker_ratio,
        lower_wick_ratio,
    ], dtype=np.float32)
