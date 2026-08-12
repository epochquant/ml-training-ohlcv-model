import numpy as np

VALID_MODES = ("logreturn", "robust")


def normalize_window(x: np.ndarray, mode: str = "logreturn", clip: float = 5.0,
                      soft_clip: bool = True, eps: float = 1e-8):
    """Normalize a raw OHLCV(+amount) window for the high-volatility variant (Area A).

    Replaces the base pipeline's raw-price z-score + hard clip
    (model/kronos.py::KronosPredictor.predict, training/dataset.py::QlibDataset)
    with either:
      - "logreturn": per-column log-returns, z-scored by the window's own return
        mean/std. Scale-invariant -- a 50% move reads the same on a sub-cent
        memecoin and on BTC.
      - "robust": per-column (x - median) / (MAD * 1.4826), resistant to the
        single extreme-outlier candle that would dominate a plain mean/std.

    soft_clip=True applies clip * tanh(z / clip) instead of a hard clip, so
    tail candles keep a (compressed) distinct value rather than all saturating
    to the same +-clip constant.

    Returns (x_norm, stats) -- stats is required by denormalize_continuation
    to invert model output back to price levels.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown normalization mode '{mode}', expected one of {VALID_MODES}")

    x = np.asarray(x, dtype=np.float64)
    stats = {"mode": mode, "clip": float(clip), "soft_clip": bool(soft_clip), "last_raw": x[-1].copy()}

    if mode == "logreturn":
        prev = np.concatenate([x[:1], x[:-1]], axis=0)
        safe_x = np.clip(x, eps, None)
        safe_prev = np.clip(prev, eps, None)
        returns = np.log(safe_x / safe_prev)
        ret_mean = returns.mean(axis=0)
        ret_std = returns.std(axis=0)
        stats["ret_mean"] = ret_mean
        stats["ret_std"] = ret_std
        z = (returns - ret_mean) / (ret_std + eps)
    else:  # robust
        median = np.median(x, axis=0)
        mad = np.median(np.abs(x - median), axis=0)
        stats["median"] = median
        stats["mad"] = mad
        z = (x - median) / (mad * 1.4826 + eps)

    if soft_clip:
        z = clip * np.tanh(z / clip)
    else:
        z = np.clip(z, -clip, clip)

    return z.astype(np.float32), stats


def denormalize_continuation(pred_norm: np.ndarray, stats: dict, eps: float = 1e-8) -> np.ndarray:
    """Invert normalize_window's transform for a *continuation* (the predicted
    rows that follow the historical window used to compute `stats`).

    For mode="logreturn", each predicted row is treated as the (normalized)
    log-return relative to the previous step, and prices are reconstructed via
    cumulative sum from `stats["last_raw"]` (the last historical raw row).
    """
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    mode = stats["mode"]
    clip = stats["clip"]
    soft_clip = stats["soft_clip"]

    if soft_clip:
        z = clip * np.arctanh(np.clip(pred_norm / clip, -0.999999, 0.999999))
    else:
        z = pred_norm

    if mode == "logreturn":
        returns = z * (stats["ret_std"] + eps) + stats["ret_mean"]
        log_anchor = np.log(np.clip(stats["last_raw"], eps, None))
        log_prices = np.cumsum(returns, axis=0) + log_anchor
        prices = np.exp(log_prices)
    else:  # robust
        prices = z * (stats["mad"] * 1.4826 + eps) + stats["median"]

    return prices.astype(np.float32)
