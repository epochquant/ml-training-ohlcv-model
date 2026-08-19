# Kronos Foundational Time-Series Model: Core Knowledge

## Overview
**Kronos** is an autoregressive foundational time-series model designed for multi-resolution financial series (OHLCV). It treats time series as language-like discrete token sequences.

---

## Key Components

### 1. Tokenizer & Quantization (BSQ)
- **Mechanism**: Continuous OHLC data is transformed via Binary Spherical Quantization (BSQ) into hierarchical discrete tokens.
- **Scale Sensitivity**: Because discrete bins map relative price relations, improper cross-window scaling or unbounded volume spikes distort token assignments.

### 2. Autoregressive Temporal Backbone
- **Context Length**: Standard lookback window (e.g. 512 context candles) predicting forward sequence (e.g. 64 horizon candles).
- **Inductive Biases**:
  - Temporal causality is strictly preserved.
  - Candlestick structural geometry ($High \ge \max(Open, Close)$ and $Low \le \min(Open, Close)$) is enforced or implicitly learned.

### 3. Normalization Invariants
- Normalization within each window must preserve the geometric relationship between Open, High, Low, and Close.
- Independent per-feature min-max scaling across individual OHLC columns is **forbidden** because it destroys candle ratios.

---

## Best Practices for Enhancements
1. **Never break OHLC consistency** when normalizing. Always scale all 4 price components by the same window base reference (e.g. window initial open or window mean/std).
2. **Tokenizer Alignment**: Any pre-processing transformation must be invertible during inference/prediction decoding.
