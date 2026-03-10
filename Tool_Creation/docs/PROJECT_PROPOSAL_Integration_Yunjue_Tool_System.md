# Project Proposal: Integrating Yunjue-Style Self-Evolving Tool System into CSE291A Tool_Creation

**Document version:** 1.1  
**Date:** March 8, 2026  
**Scope:** Tool_Creation project (MCTS + LLM heuristic optimization)

---

## 1. Executive Summary

This proposal outlines the integration of **Yunjue Agent**–style self-evolving tool generation and management into the **CSE291A Tool_Creation** project. The goal is to augment the existing MCTS+LLM pipeline with a **global toolset**, **multiagent tool critique and processing** (integrator, merger, aggregator), and **high-level reasoning over past and current best tools** so that heuristics can be built, maintained, synthesized, and upgraded in a principled way—while **preserving the current Tool_Creation architecture** (LLM/, MCTS_tools/, orchestrator/, same data flow). New work will **keep the original architecture runnable**: all new components must respect existing input/output contracts (e.g., ToolManager response format and phase signatures) so that tool calling does not fail. The design also emphasizes the **potential to reason over current best tools** and **innovate** better heuristics, including an **optional external LLM API** dedicated to strategic reasoning (separate from code-generation) whose output guides the next heuristic draft.

**Invariant — maintaining the current architecture:** All integration work **must** keep the existing Tool_Creation pipeline runnable as-is. New components are **additive and optional**; with every new feature disabled, behavior must be **identical** to the current system. No existing directories, APIs, or data flow are replaced; new code sits around the current flow and respects its input/output contracts.

---

## 2. Context and Motivation

### 2.1 Current Tool_Creation Architecture (CSE291A)

- **Purpose:** Use an LLM to generate and evolve **exploration heuristics as tools** that optimize the MCTS process for puzzle games (e.g., Sokoban, Rush Hour).
- **Flow:** MCTS trace files + current tool source → **PromptBuilder** → **LLMQuerier** → **ToolManager** (parse, validate, install) → **Optimizer** (orchestrator entry). Optional 3-step pipeline: analysis → draft → critique.
- **Orchestrator:** **OptimizationRunner** + **Evaluator** run an iterative loop: pick level/phase, play, run optimizer, accept/reject against per-level baselines, track mastery.
- **Project improvement report:** The report *Project improvement report.pdf* (workspace root: `Project improvement report.pdf`) — “Integrating Yunjue-Agent-Style Tool Evolution into a MCTS + LLM Heuristic System” — is the primary source for the gaps and proposed integration. The following bullets align with its *Challenges in the MCTS + LLM Tool Project* and *Proposed Integration* sections.
- **Gaps (from project improvement report and codebase analysis):**
  - **No global toolset:** Tools are per-phase, single-file; no registry or versioned history of all heuristics.
  - **No synthesis/upgrade path:** No clustering of similar heuristics, no merging of complementary ideas, no explicit “evolve better tool from past tools” step.
  - **Limited high-level reasoning:** The LLM sees traces and current tool source but not a structured summary of *why* past tools succeeded or failed, or how to combine strengths of multiple candidates.
  - **Tool section capacity:** Building, maintaining, synthesizing, and upgrading tools is underdeveloped compared to the strong MCTS and evaluation pipeline.

### 2.2 Yunjue Agent Strengths (to Integrate)

- **Global toolset and evolution loop:** Tools are clustered (semantic similarity), merged (LLM or naive), and consolidated into a shared pool; evolution is driven by task feedback.
- **Multiagent roles:**
  - **Integrator:** Synthesizes execution results into a final answer (in Yunjue: QA; in 291A: “which tool or combination to use / final heuristic choice”).
  - **Merger:** Combines multiple tools (or code variants) into a single, correct implementation (see `tool_merge.md`, `merge_tools` in `evolve.py`).
  - **Aggregator / cluster:** Groups tools by fundamental action (see `tool_cluster.md`, `cluster_tools`), then merges or keeps best representative.
- **Tool-first evolution:** Binary feedback (run success/failure) and structured prompts (cluster → merge → enhance) support stable accumulation and reuse of capabilities.
- **High-level reasoning:** Clustering and merge prompts force the system to reason about *what* each tool does and *when* to merge vs. keep separate.

---

## 3. Design Principles (Invariants)

**Overarching principle:** The **current architecture is preserved**. Existing modules, data flow, and tool-calling behavior remain the default; new capabilities are layered on via feature flags and must not break or replace the core pipeline.

1. **Preserve Tool_Creation architecture:** Keep existing directories and roles:
   - `mcts/` — MCTS engine, games, trace logging
   - `MCTS_tools/` — phase tools (selection, expansion, simulation, backpropagation, hyperparams)
   - `LLM/` — prompt building, LLM querying, tool manager, optimizer
   - `orchestrator/` — Evaluator, OptimizationRunner
2. **Same data flow for “single iteration”:** Trace files + tool source → prompt → LLM → parse/validate/install → smoke test. New components sit *around* this flow (before/after/alongside), not replace it.
3. **MCTS-phase semantics stay:** All tools remain phase-scoped (selection, expansion, simulation, backpropagation, hyperparams) with existing signatures and `tool_config.json`-driven loading.
4. **Game-agnostic orchestrator:** Configuration continues to live in `MCTS_tools/hyperparams/` and `MCTS_tools/training_logic/`; no hardcoding of Sokoban-only logic in the new tool-system layer.

### 3.2 Compatibility and Tool-Calling Guarantees (Original Architecture Must Keep Running)

All new components **must preserve the existing tool-calling contract** so that the original pipeline and the MCTS engine continue to work without failure.

- **Input/output contracts:**
  - **ToolManager** expects a single structured LLM response (ACTION, FILE_NAME, FUNCTION_NAME, DESCRIPTION, one ```python block). Any new component that produces “tool code” (e.g., merger, aggregator-suggested draft) must output in this format or be passed through the same `parse_response` → `validate` → `install` path. No alternate response schema should bypass validation.
  - **Function signatures:** `ToolManager.EXPECTED_SIGNATURES` and `_PARAM_ALIASES` define the exact parameter names and order per phase (e.g., simulation: `state`, `perspective_player`, `max_depth`). Every installed tool—whether from the existing Optimizer or from the new merger—must validate against these; otherwise `verify_loadable` and the engine’s `set_tool(phase, fn)` will receive a callable with the wrong signature and runtime calls will fail.
  - **Engine interface:** The MCTS engine expects `set_tool(phase, fn)` with a **callable**; `get_tool_source()` returns `dict[str, str]` (phase → source code string). The registry and any “tool selection” logic must only pass through **paths** or **callables** that were produced by the existing load path (e.g., `ToolManager.install` + dynamic import), not new ad-hoc types.
- **Recommendation:** New modules (tool_merge, tool_cluster, tool_aggregator) should consume and produce only **existing** types: e.g., `dict` for parsed response, `Path` for installed file, `str` for phase name. Avoid introducing new wrapper classes for “tool” unless they are internal and never cross the boundary into `Optimizer.run()` or `engine.set_tool()`.
- **Testing:** The existing test suite (e.g., `test_mcts_engine.py` for set_tool/get_tool/load_tool, `test_llm_querier.py`, `test_prompt_builder.py`) must remain passing. New code should add tests that assert: merged/aggregator-assisted outputs still parse and validate, and that the engine can load and run the resulting function without signature errors.

---

## 4. Proposed New Components (Within Existing Architecture)

### 4.1 Global Tool Registry (New: `LLM/tool_registry.py`)

- **Role:** Maintain a **global view** of MCTS heuristic tools across phases and iterations.
- **Contents (per phase):**
  - **Current active tool:** Path + function name + short description (as today).
  - **History:** For each phase, a list of *installed* tools (path, description, iteration, optional performance summary).
  - **Metadata:** Phase, game, iteration index, composite score / solve rate when adopted (from Evaluator).
- **Interface:**
  - `register(phase, path, description, iteration, metrics?)` — called by Optimizer/ToolManager after successful install.
  - `get_history(phase, last_k?)` — return recent tool metadata (for “past tools” context in prompts).
  - `get_all_phase_tools(phase)` — list of (path, description, metrics) for clustering/merge input.
- **Persistence:** Optional JSON/SQLite under `LLM/registry/` or alongside `results/` so that “high-level reasoning over past tools” can be grounded in real history.
- **Integration point:** Called from `Optimizer.run()` after install and from OptimizationRunner when recording an iteration result.

### 4.2 Tool Clustering (New: `LLM/tool_cluster.py`)

- **Role:** Group heuristics **within a phase** that implement the “same fundamental action” (e.g., multiple simulation strategies that are semantically similar).
- **Adapted from Yunjue:** Reuse the *logic* of `tool_cluster.md` and `cluster_tools()` but with MCTS-phase-specific schema:
  - Input: list of `{name, description, source_code_or_path}` for a single phase (e.g., all simulation tools in history).
  - Output: `consolidated_tool_clusters` — list of clusters; each cluster has `suggested_master_tool_name` and `tool_names` (file stems or IDs).
- **Prompts:** New prompt template under `LLM/prompts/` (or `game_infos/`-style) that encodes MCTS phase semantics (e.g., “simulation: state, perspective_player, max_depth”) and the “same fundamental action” rule.
- **When to run:** Optionally at end of every N iterations, or when the number of tools in a phase exceeds a threshold. Controlled by a flag in hyperparams or a new `TOOL_EVOLUTION` config section.

### 4.3 Tool Merger (New: `LLM/tool_merge.py`)

- **Role:** Given a **cluster** of tools (same phase), produce a single merged implementation that preserves correctness and avoids redundant logic.
- **Adapted from Yunjue:** Same idea as `merge_tools()` and `tool_merge.md`, but:
  - Input: list of tool *source code* strings (and names) for one phase; all conform to the same phase signature.
  - Output: one Python file (or code string) that ToolManager can validate and install.
- **Constraints:** Must preserve `EXPECTED_SIGNATURES` for that phase; no external APIs (unlike Yunjue’s web/proxy clauses). Prompt template should stress MCTS semantics and game-agnostic heuristic logic.
- **Integration:** After clustering, for each cluster of size > 1, call merger → validate with ToolManager → install as a new “merged” tool and register it in the global registry; optionally archive or mark old tools as superseded.

### 4.4 Aggregator / Critique Synthesizer (New: `LLM/tool_aggregator.py`)

- **Role:** **High-level reasoning** over past tools and current run: summarize *why* certain heuristics worked or failed, and produce a short “strategic summary” for the next Optimizer run.
- **Inputs:**
  - Tool registry history for the target phase (last K tools + their metrics).
  - Recent iteration results (level, composite, solve_rate, description).
  - Optional: last analysis/draft/critique text from the 3-step pipeline.
- **Output:** A structured summary string to be injected into `additional_context` for the next Optimizer call, e.g.:
  - “Past best tools for simulation favored X; avoid Y because it led to low solve_rate on level5; consider combining idea A (from tool T1) with idea B (from tool T2).”
- **Implementation:** One LLM call with a dedicated prompt (new template) that asks for “synthesis of past tool performance and recommendations.” Can use the same LLMQuerier and a simple non-code response format (no code block required).

### 4.5 High-Level Reasoning Over Current Best Tools to Innovate Better Heuristics

A central goal of the integration is to enable **reasoning over the current best tools** so the system can **innovate** better heuristics rather than only iterating locally on the last candidate.

- **Capability:** Use the registry’s “best” tools (e.g., by composite score or solve rate per level) plus their descriptions and performance summaries as input to a **dedicated reasoning step**. This step answers: What made the best tools work? What patterns should the next heuristic adopt or avoid? How could ideas from different top tools be combined or generalized?
- **Optional external LLM API for reasoning:** The reasoning step can be implemented as an **optional call to an external LLM API** (separate from the existing TritonAI/code-generation endpoint if desired). For example:
  - **Config:** e.g. `REASONING_LLM_ENABLED`, `REASONING_LLM_API_URL`, `REASONING_LLM_MODEL` so that “strategic reasoning” can use a different model or endpoint (e.g., a stronger or cheaper model tuned for analysis).
  - **Flow:** Before building the Optimizer prompt, call the reasoning API with: current best tools per phase (from registry), their metrics, and recent failure modes. The API returns a short “innovation memo” (no code)—e.g., “Emphasize distance-to-goal in simulation; avoid heavy branching in expansion.” This memo is then passed into `additional_context` for the main Optimizer, so the code-generation LLM receives explicit high-level guidance.
- **Potential:** This creates a clear path from “what worked best so far” to “what to try next,” improving the chance of discovering better heuristics than single-step trace-only prompting. The aggregator (Section 4.4) can consume this memo or be the same component that calls the optional reasoning API; either way, the pipeline stays compatible (reasoning is optional, output is plain text for context).

### 4.6 Integrator (Adapted Role)

- **Role in Yunjue:** Consumes execution findings and produces final answer.  
- **Role in 291A:** Do **not** replace the current “play game → trace → optimize → evaluate” flow. Instead, introduce an **integrator** in the *tool-selection* sense:
  - **Option A (lightweight):** After the optimizer produces a candidate tool, the “integrator” is simply the existing Evaluator + accept/reject logic (i.e., “integrate” = adopt or reject based on composite score).
  - **Option B (richer):** A small “tool selection” step that, given the current level and registry history, suggests *which* existing tool (or merged tool) to use for the next play, or to seed the Optimizer’s context. This can be a single LLM call: “Given level L and these past tools and scores, which tool or strategy should we try next?” Output: tool ID or short strategy note → passed into `additional_context`.
- **Recommendation:** Start with Option A; add Option B as a second phase if time allows. Both keep the same architecture (orchestrator still runs the loop; integrator is an optional step inside the loop).

---

## 5. Data Flow (Updated, Same Architecture)

- **Single optimization iteration (unchanged at core):**
  1. Orchestrator picks level and phase.
  2. Play with current tool → trace.
  3. Build prompt (with optional **aggregator** summary in `additional_context`).
  4. Optimizer.run(trace, tool_list, state_factory, additional_context) → 3-step LLM → parse → validate → install → smoke test.
  5. **New:** Register installed tool in **global tool registry** (phase, path, description, iteration, metrics when available).
  6. Evaluator evaluates new tool; accept/reject; update baselines and mastery.
  7. **New (periodic):** If clustering/merge is enabled and triggered (e.g., every N iters or when registry size > K), run **tool_cluster** → **tool_merge** for phases with multiple tools → install merged tools and update registry.

- **High-level reasoning:** Before step 3, optionally call **aggregator** with registry history + recent results → get summary → pass as `additional_context` into step 4. This gives the LLM “memory” of past tools and performance without changing the rest of the pipeline.

---

## 6. File and Module Layout (Additions Only)

```
Tool_Creation/
├── LLM/
│   ├── ... (existing: prompt_builder.py, llm_querier.py, tool_manager.py, optimizer.py)
│   ├── tool_registry.py      # NEW: global toolset state + history
│   ├── tool_cluster.py       # NEW: cluster tools by semantic similarity (per phase)
│   ├── tool_merge.py         # NEW: merge cluster into one implementation
│   ├── tool_aggregator.py    # NEW: synthesize past-tool reasoning for prompts
│   ├── prompts/             # NEW (optional): templates for cluster, merge, aggregator
│   │   ├── tool_cluster_mcts.md
│   │   ├── tool_merge_mcts.md
│   │   └── tool_aggregator_mcts.md
│   └── registry/             # NEW (optional): persisted registry storage
├── orchestrator/
│   ├── ... (existing: runner.py, evaluator.py)
│   └── (runner.py extended to call registry, optional cluster/merge/aggregator)
└── MCTS_tools/
    └── (unchanged; merged tools still land in <phase>/*.py)
```

No changes to `mcts/`, `MCTS_tools/` structure, or to the existing `Optimizer`/`PromptBuilder`/`LLMQuerier`/`ToolManager` APIs beyond passing through `additional_context` and registering results.

---

## 7. Configuration and Feature Flags

- **Hyperparams or new config:** Add optional keys, e.g. in `default_hyperparams.py` or a small `tool_evolution_config.py`:
  - `ENABLE_TOOL_REGISTRY`: bool (default True after integration).
  - `ENABLE_AGGREGATOR`: bool (default False initially).
  - `ENABLE_CLUSTER_MERGE`: bool (default False initially).
  - `CLUSTER_MERGE_INTERVAL`: int (e.g., every 5 iterations).
  - `REGISTRY_HISTORY_LEN`: int (e.g., last 10 tools per phase for aggregator).
  - **High-level reasoning (optional external LLM):** `REASONING_LLM_ENABLED` (bool), `REASONING_LLM_API_URL`, `REASONING_LLM_MODEL` (or reuse existing API_KEYS/OPENAI_BASE_URL with a different model name) so that the “innovate from best tools” step can call an external LLM API for reasoning-only output.
- This allows the same codebase to run with “current behavior only” (all new features off) or with incremental adoption (registry only → registry + aggregator → full cluster/merge).

---

## 8. Implementation Phases

| Phase | Deliverable | Risk |
|-------|-------------|------|
| **1** | `tool_registry.py` + wiring in Optimizer and OptimizationRunner to register and persist tool history | Low |
| **2** | `tool_aggregator.py` + prompt; call before Optimizer.run() and pass summary as `additional_context` | Low |
| **3** | `tool_cluster.py` + MCTS-phase-specific cluster prompt; run periodically from runner | Medium (prompt design) |
| **4** | `tool_merge.py` + MCTS-phase-specific merge prompt; integrate with cluster output and ToolManager | Medium (signature preservation) |
| **5** | Optional “integrator” tool-selection step (Option B) and end-to-end tests with Sokoban | Low–Medium |

---

## 9. Success Criteria

- **Architecture preservation (required):** All existing tests (e.g., `test_llm_querier.py`, `test_prompt_builder.py`, `test_mcts_engine.py`) still pass; no removal or breaking changes to existing APIs. The current Tool_Creation architecture remains the single source of truth for the core loop.
- **Behavioral parity:** With all new features disabled, the pipeline must behave **identically** to the current system. No regression in default runs.
- **New capability:** With registry + aggregator enabled, the LLM receives concise “past tool reasoning” in the prompt and can produce heuristics that explicitly build on or avoid past choices.
- **High-level innovation:** With optional reasoning API enabled, the system can use an external LLM to reason over current best tools and produce an “innovation memo” that guides the next heuristic generation, improving the potential to discover better heuristics than trace-only iteration.
- **Evolution:** With cluster + merge enabled, multiple variants of a phase (e.g., several simulation heuristics) can be merged into a single, signature-compliant tool and used in subsequent iterations.

---

## 10. References

- **CSE291A Tool_Creation:** `Tool_Creation/README.md`, `LLM/architecture.md`, `orchestrator/runner.py`, `LLM/optimizer.py`, `LLM/tool_manager.py`.
- **Yunjue Agent:** `Yunjue-Agent-Game/README.md`, `evolve.py` (cluster_tools, merge_tools, optimize_tools), `src/core/nodes.py` (integrator_node, manager_node, tool_developer_node), `src/prompts/templates/tool_cluster.md`, `tool_merge.md`.
- **Project improvement report:** `Project improvement report.pdf` (repo root) — “Integrating Yunjue-Agent-Style Tool Evolution into a MCTS + LLM Heuristic System.” Defines challenges (unstructured heuristic accumulation, limited reasoning over game traces, absence of multi-agent pipeline), the Heuristic Manager / Developer / Trace Analyzer / Integrator / Absorbing Mechanism (Aggregator + Merger), and implementation considerations for Cursor’s agent. This proposal implements that vision within the existing Tool_Creation architecture. Mapping: report’s *Trace Analyzer* → our aggregator + high-level reasoning (§4.4–4.5); *Heuristic Manager* → optional integrator / tool-selection (§4.6); *Absorbing Mechanism* → tool_cluster + tool_merge (§4.2–4.3); *Heuristic Developer* remains the existing Optimizer + PromptBuilder pipeline.

---

*This proposal **maintains the current Tool_Creation architecture** as the invariant: the existing pipeline stays runnable and unchanged when new features are off. On top of that, a global toolset and Yunjue-style multiagent tool processing (integrator, merger, aggregator) are added so that the 291 project can reason about and evolve better MCTS heuristics over time.*
