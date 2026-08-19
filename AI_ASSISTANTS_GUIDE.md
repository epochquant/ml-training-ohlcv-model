# AI Assistants & Knowledge Management Guide

This guide describes how to manage context, active scratchpads, model knowledge, and experiment logs across **Antigravity IDE**, **Claude Code**, and **GitHub Copilot** within this repository.

---

## 1. Core Architecture: 3-Tier Persistent Memory

AI assistant conversations are inherently volatile and context-bounded. To maintain continuity and accelerate iterative enhancements of the **Kronos** foundational time-series model for high-volatility cryptocurrencies, this repository utilizes a 3-tier memory system located in `docs/knowledge/`:

```
ml-training-ohlcv-model/
├── AGENTS.md                                # Universal assistant contract & context index
├── AI_ASSISTANTS_GUIDE.md                   # This comprehensive reference guide
├── CLAUDE.md                               # Claude Code instruction entrypoint
├── GEMINI.md                               # Antigravity IDE instruction entrypoint
├── .github/
│   └── copilot-instructions.md             # GitHub Copilot instruction entrypoint
└── docs/
    └── knowledge/
        ├── SCRATCHPAD.md                    # Tier 1: Active working memory & handoff
        ├── kronos_architecture.md           # Tier 2: Foundational model architecture & tokenization
        ├── high_volatility_adaptations.md   # Tier 2: Adaptations for high-volatility crypto symbols
        ├── experiments_log.md               # Tier 3: Chronological log of runs, metrics & outcomes
        └── decision_records.md              # Tier 3: Architectural Decision Records (ADRs)
```

---

## 2. Memory Tier Details

### Tier 1: Active Working Memory (`docs/knowledge/SCRATCHPAD.md`)
- **Purpose**: Tracks what is being actively worked on right now (current hypothesis, active scripts, next steps, blockers).
- **Lifecycle**: Short-lived, high-frequency updates. Keep under ~100 lines.
- **Rule**: When completing a task or starting a new experiment, move findings to `experiments_log.md` and refresh `SCRATCHPAD.md`.

### Tier 2: Foundational & Domain Knowledge (`docs/knowledge/kronos_architecture.md`, `high_volatility_adaptations.md`)
- **Purpose**: Prevents assistants from hallucinating model mechanics or introducing invalid assumptions (e.g. discrete tokenization requirements, lookback/prediction horizon limits, volume handling, scaling constraints).
- **Lifecycle**: Updated when new foundational features or domain rules are discovered.

### Tier 3: Experiment & Decision History (`docs/knowledge/experiments_log.md`, `decision_records.md`)
- **Purpose**: Records past training runs, datasets tested, hyperparameters, and explicit metric diffs (RMSE, Sharpe, Directional Accuracy), alongside Architecture Decision Records (ADRs).
- **Benefit**: Prevents any AI assistant from proposing or repeating past failed approaches.

---

## 3. How Each Assistant Uses This System

| Assistant | Entrypoint File | Automatic Ingestion Behavior |
|---|---|---|
| **Antigravity IDE** | `GEMINI.md` | Auto-loaded at session start; reads `AGENTS.md` and checks `SCRATCHPAD.md`. |
| **Claude Code** | `CLAUDE.md` | Reads instructions on startup; references `@AGENTS.md` and `@docs/knowledge/SCRATCHPAD.md`. |
| **GitHub Copilot** | `.github/copilot-instructions.md` | Loaded automatically for code completion and chat context in VS Code / IDE. |

---

## 4. Best Practices for Developers

1. **Starting a Session**:
   In any assistant chat, start with a kick-off prompt such as:
   > *"Review `@docs/knowledge/SCRATCHPAD.md` and `@docs/knowledge/experiments_log.md`. Let's continue working on [current task]."*

2. **Ending a Session**:
   Before closing a chat session, ask the assistant:
   > *"Update `@docs/knowledge/SCRATCHPAD.md` with our latest findings, next steps, and log any experiment results in `@docs/knowledge/experiments_log.md`."*

3. **Git Integration**:
   Always commit `docs/knowledge/*` changes together with the corresponding code changes. This permanently links code revisions to the reasoning and empirical results behind them.
