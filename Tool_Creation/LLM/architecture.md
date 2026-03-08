# LLM Subsystem Architecture

The `llm/` package is a **self-contained LLM optimization pipeline** for improving MCTS heuristic functions. It is completely **decoupled from gameplay** — it only consumes trace files and tool source code, and produces installable Python files.

---

## Component Overview

```
llm/
├── __init__.py          — public API exports
├── game_infos/          — static game rule descriptions (*.txt)
│   ├── sokoban.txt
│   └── rush_hour.txt
├── drafts/              — saved raw prompt files (*.txt, auto-generated)
├── prompt_builder.py    — assembles structured prompts
├── llm_querier.py       — sends prompts to LLM API, extracts code
├── tool_manager.py      — parses/validates/installs LLM-generated tools
└── optimizer.py         — top-level orchestrator tying all 3 together
```

---

## Data Flow

```
[MCTS trace files (JSON)]  +  [tool source code]
           │
           ▼
      PromptBuilder
  (assembles structured prompt)
           │
           ▼
       LLMQuerier
  (queries OpenAI-compatible API)
           │
           ▼
     LLM Response (text)
           │
      ┌────┴─────┐
      │          │
  extract     parse_response
  code block  (headers + code)
      │
      ▼
   validate()
   (syntax, function presence, signature)
      │
      ▼
  ToolManager.install()
  → writes to MCTS_tools/<phase>/<file>.py
      │
      ▼
  smoke_test (load + call function)
  → optional LLM repair loop on failure
```

---

## 1. `PromptBuilder` — Prompt Assembly

The entry point for constructing prompts. Initialized with `game` (e.g. `"sokoban"`) and `target_phase` (one of `selection`, `expansion`, `simulation`, `backpropagation`, `hyperparams`).

**Three prompt-building strategies:**

| Method | Purpose |
|---|---|
| `build(...)` | Single-step legacy prompt (analysis + codegen combined) |
| `build_analysis_prompt(...)` | Step 1 of 2/3: asks for written analysis, no code |
| `build_generation_prompt(analysis, ...)` | Step 2 of 2/3: generate code from prior analysis |
| `build_critique_prompt(analysis, draft_code, ...)` | Step 3 of 3: review draft code, output final version |

**Each prompt is assembled from ordered sections:**
1. System instruction (role + 70/30 rule: incremental vs paradigm-shift)
2. Game rules (loaded from `game_infos/<game>.txt`)
3. All 4 MCTS tool sources (optional, with `◀ TARGET` marker on the target phase)
4. Target heuristic code (the one being improved)
5. Gameplay traces (formatted from MCTS JSON records)
6. Optional additional context (e.g. iteration history)
7. Task instruction (structured output format enforcement)

**Output format enforced via prompt:**
```
ACTION: modify
FILE_NAME: improved_simulation.py
FUNCTION_NAME: improved_simulation
DESCRIPTION: <one-line>
```python
<complete function code>
```
```

---

## 2. `LLMQuerier` — API Communication

Wraps an **OpenAI-compatible async client** (e.g. UCSD's TritonAI endpoint).

**Configuration** (loaded from `.env` in `Tool_Creation/`):
- `API_KEYS` — comma-separated, randomly selected per request
- `OPENAI_BASE_URL` — defaults to `https://tritonai-api.ucsd.edu`
- `MODEL_NAME` — defaults to `api-gpt-oss-120b`

**Key methods:**

| Method | Description |
|---|---|
| `query(prompt)` | Single synchronous request; returns result dict |
| `query_batch(prompts)` | Parallel requests with semaphore + round-robin key distribution |
| `query_two_step(analysis_prompt, gen_fn)` | Two sequential calls: analysis then codegen |
| `query_three_step(analysis_prompt, gen_fn, critique_fn)` | Three sequential calls: analysis → draft → critique |

**Jupyter compatibility:** Uses a `ThreadPoolExecutor` workaround when a running event loop is detected (IPykernel), so `asyncio.run()` never conflicts.

**Every result dict contains:**
```python
{
  "response": str,        # full LLM text
  "code": str | None,     # extracted ```python block
  "validation": {...},    # valid bool + error
  "parsed": {...},        # structured header fields
  "model": str,
  "elapsed_seconds": float,
  "status": "success" | "error",
}
```

---

## 3. `ToolManager` — Parse, Validate, Install

Handles the lifecycle from LLM text → working `.py` file on disk.

**`parse_response(text)`** — extracts `ACTION`, `FILE_NAME`, `FUNCTION_NAME`, `DESCRIPTION`, and the `` ```python `` block using regex.

**`validate(parsed, phase)`** — checks:
1. Required header fields present
2. `ACTION` is `create` or `modify`
3. `FILE_NAME` ends in `.py`, only `[a-z0-9_]` chars
4. Code parses without `SyntaxError`
5. Declared `FUNCTION_NAME` exists in the code
6. Function signature matches expected params per phase (from `EXPECTED_SIGNATURES`)

**Expected signatures:**
```python
"selection":       ["root", "exploration_weight"]
"expansion":       ["node"]
"simulation":      ["state", "perspective_player", "max_depth"]
"backpropagation": ["node", "reward"]
"hyperparams":     []   # get_hyperparams()
```

**`install(parsed, phase)`** — writes file to `MCTS_tools/<phase>/<filename>.py` with a docstring header (description + timestamp). Refuses to overwrite unless `overwrite=True`.

**`verify_loadable(filepath, func_name)`** — uses `importlib` to confirm the file imports and the named function is callable at runtime.

---

## 4. `Optimizer` — Orchestrator

The single public entry point for the pipeline. Instantiate once per optimization iteration and call `run()`.

```python
opt = Optimizer(game="sokoban", target_phase="simulation", three_step=True)
result = opt.run(
    record_files=["path/to/trace.json"],
    tool_list=engine.get_tool_source(),   # dict[phase -> source str]
    state_factory=lambda: game.new_initial_state(),
    additional_context="Previous best score: 0.72",
)
```

**Pipeline modes (controlled by constructor flags):**

| Mode | Flags | LLM calls | Steps |
|---|---|---|---|
| Single-step (legacy) | `two_step=False` | 1 | Build → Query → Parse → Install → Smoke |
| Two-step (default) | `two_step=True` | 2 | Analysis → Codegen → Parse → Install → Smoke |
| Three-step | `three_step=True` | 3 | Analysis → Draft → Critique → Parse → Install → Smoke |

**Smoke test + repair loop:**
- After install, the function is dynamically loaded and called with synthetic test args derived from `EXPECTED_SIGNATURES`.
- If it raises, a targeted **repair prompt** (including the traceback and real `GameState` API) is sent back to the LLM.
- Controlled by `max_repair_attempts` (default 1).

**Return dict:**
```python
{
  "llm_result": {...},      # raw LLM response(s)
  "parsed": {...},          # parsed response
  "installed_path": Path,   # where tool was written
  "smoke_test": bool,       # did it pass?
  "function": callable,     # loaded function (or None)
  "error": str | None,
}
```

---

## Key Design Decisions

- **Decoupled from MCTS engine**: `Optimizer` never imports the game or MCTS engine. All context is passed in as strings/callables.
- **Structured output protocol**: The LLM is forced via prompt to produce parseable headers + a single code block. This makes parsing robust without relying on LLM "cooperation."
- **70/30 heuristic strategy**: The system prompt explicitly instructs the LLM to prefer incremental improvement (~70%) over full rewrites (~30%), nudging toward stable evolution.
- **Multi-key load balancing**: Batch queries distribute over multiple API keys in round-robin to stay under per-key rate limits.
- **Lazy initialization**: `Optimizer` defers creating `PromptBuilder`, `LLMQuerier`, and `ToolManager` to first use, keeping construction cheap.
