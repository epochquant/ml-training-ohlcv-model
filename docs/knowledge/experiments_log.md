# Kronos Training Experiments Log

> **Rule for Assistants & Engineers:** Always log experiments here chronologically after running training or evaluation pipelines. Include specific symbols, changes made, metric diffs, and conclusions.

---

| ID | Date | Symbol / Dataset | Change Description | Metrics / Result | Status | Key Takeaways & Next Action |
|---|---|---|---|---|---|---|
| EXP-001 | 2026-03-16 | OPNUSDT (1m) | Zeroed out volume column during preprocessing | Validation Loss -12%, Directional Accuracy +4.2% | **Success** | Volume scale distortion removed from BSQ tokenizer. Keep `volume=0` as default for 1m crypto. |
| EXP-002 | 2026-03-18 | GUAUSDT (1m) | Standard baseline Kronos fine-tuning on 1-min candles | Clean convergence on Vertex AI L4 GPU | **Success** | Established baseline checkpoint for HV crypto symbols. |

---

## Detailed Experiment Notes

### EXP-001: Volume Zeroing on High-Volatility Pairs
- **Hypothesis**: The discrete token quantization in Kronos fails when volume changes by orders of magnitude in volatile pairs.
- **Outcome**: Model achieved higher directional accuracy on out-of-sample data when trained purely on OHLC structural ratios.
- **Reference**: `FINE-TUNNING.md#Phase-0`
