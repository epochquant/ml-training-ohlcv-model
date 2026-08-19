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
    log-return relative to the previous step, and prices are reconstructed with:
      - Step 0 open anchored continuously to last_close (Open[0] ≈ Close[-1]),
        eliminating artificial price gaps from decoupled column histories.
      - Step-to-step continuity: Open[i] = Close[i-1] for all i >= 1.
      - Physical candlestick invariants: High[i] >= max(Open[i], Close[i])
        and Low[i] <= min(Open[i], Close[i]).
      - Non-negative bounds for volume and amount channels.
    """
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    mode = stats["mode"]
    clip = stats["clip"]
    soft_clip = stats["soft_clip"]

    if soft_clip:
        z = clip * np.arctanh(np.clip(pred_norm / clip, -0.999999, 0.999999))
    else:
        z = pred_norm

    n_rows, n_cols = z.shape
    prices = np.zeros_like(z, dtype=np.float64)

    if mode == "logreturn":
        ret_std = stats["ret_std"] + eps
        ret_mean = stats["ret_mean"]
        returns = z * ret_std + ret_mean
        last_raw = stats["last_raw"]

        if n_cols >= 4:
            # Candlestick-aware single-anchor reconstruction for OHLC channels
            last_close = float(np.clip(last_raw[3], eps, None))
            curr_close = last_close

            for i in range(n_rows):
                if i == 0:
                    r_open = returns[0, 0]
                    # Bounded opening return relative to previous close
                    open_p = curr_close * np.exp(np.clip(r_open, -0.05, 0.05))
                else:
                    open_p = curr_close

                # Close price from close return
                r_close = returns[i, 3]
                close_p = open_p * np.exp(r_close)

                # High and Low relative to candle body bounds
                r_high = returns[i, 1]
                r_low = returns[i, 2]

                base_high = max(open_p, close_p)
                high_p = max(base_high, base_high * np.exp(abs(r_high)))

                base_low = min(open_p, close_p)
                low_p = min(base_low, base_low * np.exp(-abs(r_low)))

                prices[i, 0] = open_p
                prices[i, 1] = high_p
                prices[i, 2] = low_p
                prices[i, 3] = close_p
                curr_close = close_p

            # Reconstruct extra non-price columns (volume, amount, etc.) via cumulative returns
            if n_cols > 4:
                for c in range(4, n_cols):
                    log_anchor = np.log(np.clip(last_raw[c], eps, None))
                    prices[:, c] = np.maximum(0.0, np.exp(np.cumsum(returns[:, c], axis=0) + log_anchor))
        else:
            log_anchor = np.log(np.clip(last_raw, eps, None))
            log_prices = np.cumsum(returns, axis=0) + log_anchor
            prices = np.exp(log_prices)
    else:  # robust
        mad = stats["mad"] * 1.4826 + eps
        median = stats["median"]
        raw_prices = z * mad + median
        prices = raw_prices.copy()

        # Enforce OHLC physical constraints if >= 4 columns
        if n_cols >= 4:
            for i in range(n_rows):
                o, h, l, c = prices[i, 0], prices[i, 1], prices[i, 2], prices[i, 3]
                prices[i, 1] = max(h, o, c)
                prices[i, 2] = min(l, o, c)
            if n_cols > 4:
                prices[:, 4:] = np.maximum(0.0, prices[:, 4:])

    return prices.astype(np.float32)
