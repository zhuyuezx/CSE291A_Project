# Task Plan: Self-Evolving Game Agent

**Goal:** Build a hybrid RL+LLM system where Python MCTS starts vanilla and an LLM periodically injects auto-discovered heuristic tools (as Python code plugins), enabling MCTS to tackle hard turn-based games without human-designed heuristics.

**Project:** CSE 291A — AI Agents, UCSD Winter 2026
**Team:** Team 09
**Target games:** Connect Four → Quoridor → Chess (via OpenSpiel)

---

## Architecture (3 Subsystems)

| Subsystem | Description |
|-----------|-------------|
| A: MCTS Engine | Custom Python MCTS with 6 dynamic tool hook points; OpenSpiel for game logic; OpenSpiel MCTSBot as vanilla baseline |
| B: Tool Creation Pipeline | Yunjue-inspired LLM pipeline: analyze game traces → spec → codegen → validate → A/B test → promote |
| C: Cross-Game Tool Manager | Persists tools across games; handles transfer; deduplication/merging |

**Tech Stack:** Python 3.11+, pyspiel (OpenSpiel), OpenAI-compatible LLM API (configurable via conf.yaml), PyYAML, pytest

**Code Location:** New code goes under `CSE291A_Project/` (the existing git repo) or a fresh `src/` tree in the code root.

---

## Phase 1: MCTS Engine + Tool Hooks + Hand-Written Tools on Connect Four

| Task | Description | Status |
|------|-------------|--------|
| 1 | Project scaffolding: pyproject.toml, conf.yaml, src/__init__.py packages, tool_pool dirs, metadata.json | pending |
| 2 | MCTSNode: UCT, PUCT, backpropagation | pending |
| 3 | Tool interface definitions: ToolType enum, ToolMeta dataclass, validate_tool_meta(), load_tool_from_file() | pending |
| 4 | ToolRegistry: register, unregister, get_tools, load_from_directory, list_all | pending |
| 5 | OpenSpiel GameAdapter: thin wrapper for game creation, state management, clone, apply_action, returns | pending |
| 6 | Core MCTSEngine (vanilla): select, expand, simulate (random rollout), backpropagate; search() and search_with_policy() | pending |
| 7 | Hand-written Connect Four tools: center_column_bias (state_evaluator), threat_detector (action_filter), greedy_rollout (rollout_policy) | pending |
| 8 | Integration test: MCTS+tools vs vanilla MCTS on Connect Four | pending |

## Phase 2: LLM Tool Generation Pipeline

| Task | Description | Status |
|------|-------------|--------|
| 9 | LLM Client: configurable per conf.yaml; OpenAI-compatible; role-based prompting | pending |
| 10 | Game Trace Recorder: record states, actions, outcomes; select informative traces (losses, close games) | pending |
| 11 | Prompt Templates: trace_analysis.md, code_generation.md, tool_validation.md, tool_merge.md | pending |
| 12 | Tool Validator: syntax (AST), runtime (100 states), timeout, LLM self-healing (up to 3 retries) | pending |
| 13 | Tool Generator: full LLM pipeline — analyze → spec → codegen → validate → A/B test → promote | pending |
| 14 | Training Loop with Plateau Detection: rolling window win rate, trigger evolution at stall/regression | pending |
| 15 | Tool Pool Manager: cross-game persistence, metadata.json, global/ promotion | pending |
| 16 | Evaluation Framework: win rate vs baselines, sample efficiency curves, cross-game transfer metrics | pending |
| 17 | Main Entry Point (main.py): CLI for training, evaluation, tool generation | pending |

## Phase 3: Cross-Game Transfer (Quoridor → Chess)

| Task | Description | Status |
|------|-------------|--------|
| 18 | Quoridor validation: run all tools on Quoridor, A/B test, promote winners to global/ | pending |
| 19 | Chess validation: same process for Chess | pending |

## Phase 4: Full Evaluation Suite

| Task | Description | Status |
|------|-------------|--------|
| 20 | Evaluation scripts: win rate tables (100/500/1000/5000 sims), sample efficiency curves, cross-game transfer speed | pending |

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MCTS implementation | Custom Python (not OpenSpiel MCTSBot) | Allows dynamic tool hooks; MCTSBot used as vanilla baseline only |
| Tool interface | `__TOOL_META__` dict + `run()` fn in each .py file | Follows Yunjue convention; dynamically loadable |
| Tool composition | action_filters intersect; evaluators average; rollout_policies weighted-sample | Handles multiple tools per hook cleanly |
| LLM model config | conf.yaml with separate TRACE_ANALYZER/CODE_GENERATOR/TOOL_VALIDATOR sections | Model-agnostic; swap cheaply |
| Plateau detection | Rolling window N=50, trigger if <2% improvement over 2 consecutive windows OR >5% drop | Avoids premature triggering |
| A/B testing | 50 games with vs 50 without new tool | Balance between speed and statistical reliability |
| Cross-game tool dir | tool_pool/global/ for tools valid on 2+ games; game-specific dirs otherwise | Clean separation |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| poppler not installed (PDF read failed) | 1 | Installed via `brew install poppler`; PDF content extracted via pdftotext |
