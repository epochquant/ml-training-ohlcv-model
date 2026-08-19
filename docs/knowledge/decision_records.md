# Architecture Decision Records (ADRs)

Records foundational architectural decisions, rationale, and tradeoffs to ensure consistency across all AI assistant interactions.

---

## ADR-001: Bypass Qlib and Use Direct CSV Preprocessing Pipeline
- **Date**: 2026-03-16
- **Status**: Accepted
- **Context**: Kronos natively references Microsoft Qlib for data loading, which adds heavy dependencies, installation friction, and complex schema requirements.
- **Decision**: Implemented standalone Binance JSON-to-CSV converter (`convertJsonToDataset.py`) and direct CSV dataset loaders.
- **Consequence**: Dramatically simplified local and GCP Vertex AI container training loops without sacrificing throughput.

---

## ADR-002: Dual Architecture for High Volatility (Backbone + Downside Classifier)
- **Date**: 2026-03-18
- **Status**: Accepted
- **Context**: Extreme crypto volatility exhibits sharp asymmetric liquidation cascades.
- **Decision**: Complement continuous autoregressive price token generation with dedicated downside regime classifier (`model/high_volatility/short_classifier.py`).
- **Consequence**: Downstream trading systems can filter out false breakout signals during high-probability cascade regimes.
