# Universal AI Assistant Guidelines & Knowledge Index

This repository trains and fine-tunes the **Kronos** foundational time-series model for predicting high-volatility cryptocurrency symbols.

## Mandatory Assistant Protocol

All AI coding assistants (Antigravity IDE, Claude Code, GitHub Copilot, Cursor, etc.) MUST adhere to the following workflow:

### 1. Ingest Context Before Action
- **Active Task & Status**: Read [SCRATCHPAD.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/SCRATCHPAD.md) to understand current state, hypothesis, and open items.
- **Experiment History**: Check [experiments_log.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/experiments_log.md) before proposing hyperparameter changes or new architectures to avoid repeating failed attempts.
- **Model Mechanics**: Consult [kronos_architecture.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/kronos_architecture.md) and [high_volatility_adaptations.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/high_volatility_adaptations.md) to preserve model invariants (discrete tokenization, normalization rules, volume treatment).

### 2. Preserve & Update Knowledge
- **After Task Completion**: Update [SCRATCHPAD.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/SCRATCHPAD.md) with completed items and updated next steps.
- **After Running Experiments**: Append a row with quantitative/qualitative findings to [experiments_log.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/experiments_log.md).
- **After Architectural Decisions**: Add an entry in [decision_records.md](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/decision_records.md).

## Quick Reference Links
- [AI Assistants & Scratchpad Guide](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/AI_ASSISTANTS_GUIDE.md)
- [Active Scratchpad](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/SCRATCHPAD.md)
- [Kronos Architecture Knowledge](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/kronos_architecture.md)
- [High Volatility Adaptations](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/high_volatility_adaptations.md)
- [Experiments Log](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/experiments_log.md)
- [Decision Records (ADR)](file:///c:/00%20-%20GITHUB/ml-training-ohlcv-model/docs/knowledge/decision_records.md)
