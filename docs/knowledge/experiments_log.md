# Kronos Training Experiments Log

> **Rule for Assistants & Engineers:** Always log experiments here chronologically after running training or evaluation pipelines. Include specific symbols, changes made, metric diffs, and conclusions.

---

| ID | Date | Symbol / Dataset | Change Description | Metrics / Result | Status | Key Takeaways & Next Action |
|---|---|---|---|---|---|---|
| EXP-001 | 2026-03-16 | OPNUSDT (1m) | Zeroed out volume column during preprocessing | Validation Loss -12%, Directional Accuracy +4.2% | **Success** | Volume scale distortion removed from BSQ tokenizer. Keep `volume=0` as default for 1m crypto. |
| EXP-002 | 2026-03-18 | GUAUSDT (1m) | Standard baseline Kronos fine-tuning on 1-min candles | Clean convergence on Vertex AI L4 GPU | **Success** | Established baseline checkpoint for HV crypto symbols. |
| EXP-003 | 2026-08-18 | 1000RATSUSDT (3m) | Geometric single-anchor continuation (`Open[0] = Close[-1]`) & OHLC invariant bounding | Initial step gap reduced from +6%/-9% to <0.5%, 12/12 unit tests passing | **Success** | Implemented on branch `feature/high_volatile_1000ratusdt`. Prevents unphysical gaps and false mean-reversion crashes on vertical breakouts. |

---

## Detailed Experiment Notes

### EXP-003: Candlestick Continuity & Single-Anchor Denormalization (1000RATSUSDT)
- **Problem**: When a volatile pair surged +10% in a single candle, independent per-column denormalization anchored `Open[0]` to `Open[-1]` and `High[0]` to `High[-1]`, creating a +6% artificial price gap and broken candle geometry.
- **Solution**: Anchored price level reconstruction to `Close[-1]` and enforced $High \ge \max(Open, Close)$ and $Low \le \min(Open, Close)$.
- **Result**: Zero unphysical gaps, strict candlestick physical constraints, 100% test pass rate across all 12 unit tests.
