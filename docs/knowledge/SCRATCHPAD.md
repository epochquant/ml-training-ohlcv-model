# Kronos High-Volatility Training Scratchpad (Active)

> **Instructions for AI Assistants:** Keep this document lean (<100 lines). Update after completing tasks or shifting experiment focus. Move completed findings to `experiments_log.md`.

---

## 1. Current Objective
- Setup knowledge base and memory system across AI assistants (Antigravity IDE, Claude Code, GitHub Copilot).
- Track iterative enhancements of the **Kronos** foundation model for high-volatility cryptocurrency symbols.

## 2. Current State & Active Files
- **Branch / Environment**: Main / Local Windows + Vertex AI GPU pipeline.
- **Active Focus**: High volatility model architecture (`model/high_volatility/`), data preparation, and training launch scripts (`launch_high_volatility_job.py`, `run_training_pipeline_hv.py`).

## 3. Pending Tasks & Next Steps
- [x] Create multi-assistant persistent memory structure in `docs/knowledge/`.
- [x] Generate `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`, and `.github/copilot-instructions.md`.
- [ ] Document foundational Kronos architecture specifics in `kronos_architecture.md`.
- [ ] Document high-volatility adjustments (volume zeroing, short classification) in `high_volatility_adaptations.md`.
- [ ] Benchmark latest training run against baseline.

## 4. Current Hypotheses & Open Questions
- **Volume Handling**: Zeroing volume prevents scale mismatch in discrete BSQ tokens, but does auxiliary feature concatenation re-introduce useful signal without disrupting tokenization?
- **Extreme Spikes**: How do rolling normalization boundaries behave during multi-sigma flash crashes on 1m timeframe?
