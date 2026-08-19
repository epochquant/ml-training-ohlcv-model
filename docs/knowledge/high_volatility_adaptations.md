# High-Volatility Cryptocurrency Adaptations

## Domain Challenges in High-Volatility Symbols
High-volatility crypto assets (e.g. meme tokens, newly listed perpetuals, low-liquidity coins) exhibit properties that break standard financial ML assumptions:
1. **Flash Spikes & Fat Tails**: 10-50% price moves within 1 to 5 minutes.
2. **Asymmetric Downside**: Cascading long liquidations create vertical drops ("waterfalls").
3. **Volume Distortion**: Volume surges by orders of magnitude (100x-1000x) during breakouts, overwhelming standard normalization.

---

## Applied Enhancements & Rationale

### 1. Volume Zeroing (`volume = 0`)
- **What**: During dataset conversion (`convertJsonToDataset.py`), volume values are set to `0`.
- **Why**: Extreme volume spikes in crypto distort the BSQ discrete tokenizer. Training strictly on normalized OHLC price geometry yielded lower validation loss and better directional stability.

### 2. Short / Downside Regime Classification
- **What**: Auxiliary classification layer in `model/high_volatility/short_classifier.py`.
- **Why**: Volatile crypto markets dump significantly faster than they pump due to leverage liquidations. Separating downside regime probability from continuous price autoregression improves risk-adjusted prediction metrics.

### 3. Pipeline Configurations
- Vertex AI job launch utilities in `launch_high_volatility_job.py` and `run_training_pipeline_hv.py` isolate hyperparameter configs tailored to volatile assets (learning rate damping, gradient clipping).
