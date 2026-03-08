# Brainstorming: LLM Debug Mode for Prompt/Response Logging

**Date:** 2026-03-08
**Feature:** Add debug mode to `LLM/llm_querier.py` (and possibly `LLM/optimizer.py`) that logs full input/output prompts for debugging LLM API issues.
**Status:** In progress — Q&A phase

---

## Project Context

### Relevant Files

- `LLM/llm_querier.py` — Core API client. Uses `AsyncOpenAI` to send prompts. Has `query()`, `query_batch()`, `query_two_step()`, `query_three_step()`. No prompt logging today.
- `LLM/optimizer.py` — Orchestrates the pipeline. Calls `LLMQuerier` methods. Has `verbose` flag for progress messages but no raw prompt/response logging.
- `LLM/prompt_builder.py` — Builds prompts (not yet read in detail).
- `LLM/tool_manager.py` — Parses/validates/installs generated code.

### Current Pain Points (inferred)
- No way to see what exact prompt was sent to the API.
- No way to see what raw response came back.
- Hard to diagnose failures (bad code extraction, wrong function name, API errors).

---

## Q&A Log

### Q1
> Where should debug logs be written — to files on disk, printed to stdout/stderr, or both?

**Answer:** Write to files on disk only. Logs must be **per-session**: all prompt I/O from a single `optimizer.run()` call goes into the same folder so the full sequence of steps can be analyzed together.

**Key implication:** A "session" = one call to `Optimizer.run()`. Each session gets its own timestamped directory, e.g. `LLM/debug_logs/2026-03-08_143022/`. Inside, each LLM call gets its own file (step1, step2, repair attempts, etc.).

---

### Q2
> What format should each log file be in?

**Answer:** Markdown (`.md`) — human-readable with structured headers: metadata at top (model, elapsed, step name, timestamp), then `## Prompt` and `## Response` sections.

**Key implication:** Each LLM call produces one `.md` file inside the session folder. A reader can open any file and immediately see context + full prompt + full response without parsing JSON.

---

### Q3
> How should debug mode be activated?

**Answer:** `debug` parameter on `LLMQuerier.__init__()`, defaulting to the value of an env variable `LLM_DEBUG` (e.g. `LLM_DEBUG=1` in `.env` sets `debug=True` by default). Caller can still override explicitly.

**Key implication:**
- `LLMQuerier(debug=True)` or `LLM_DEBUG=1` in `.env` both enable logging.
- `Optimizer` does NOT need a `debug` flag — it just constructs `LLMQuerier()` as usual and the env var handles it.
- Callers who want explicit control can pass `LLMQuerier(debug=True/False)` to override the env default.

---

### Q4
> What should the session folder naming convention be?

**Answer:** Game + phase + timestamp: `LLM/debug_logs/sokoban_simulation_20260308_143022/`

**Key implication:** The session folder name immediately tells you which game/phase was being optimized and when. Since `LLMQuerier` doesn't know the game/phase directly, it will need to accept optional `session_tag` (or `game`+`phase`) parameters to construct the folder name — or `Optimizer` passes a tag when constructing `LLMQuerier`.

---

### Q5
> What should individual log files inside a session be named?

**Answer:** Step name only: `step1_analysis.md`, `step2_generation.md`, `repair_1.md`

**Key implication:** Simple and readable. The step name is passed in by the caller (querier's `query()` method gets a `step_name` argument). Repair attempts get `repair_1.md`, `repair_2.md`, etc.

---

### Q6
> What metadata should appear at the top of each `.md` log file?

**Answer:** All fields: timestamp, model name, elapsed time, step name, status, validation result, token count (if available from API response).

**Key implication:** The MD file header will be a YAML-style or table block with all fields, followed by `## Prompt` and `## Response` sections. Token count pulled from `response.usage` if the API returns it.

---

### Q7
> Should there be a session-level summary/index file in the folder?

**Answer:** Yes — auto-generated at the end of the session (option A).

**How it works:** No API call or key usage involved. The `LLMQuerier` accumulates a list of step records in memory (step name, timestamp, status, elapsed, validation result, token count) as each query completes. When the session is "finalized" (explicitly via a `finalize_session()` call, or implicitly on `__del__`/context manager exit), it writes `index.md` with a summary table.

**Example `index.md` structure:**
```
# Session: sokoban_simulation_20260308_143022

| Step              | Status  | Elapsed | Tokens | Validation |
|-------------------|---------|---------|--------|------------|
| step1_analysis    | success | 4.2s    | 1024   | N/A        |
| step2_generation  | success | 6.1s    | 2048   | valid       |
| repair_1          | success | 3.8s    | 900    | valid       |

Total elapsed: 14.1s
```

---

### Q8
> How should the session be "closed" to trigger writing `index.md`?

**Answer:** Automatic — rewrite `index.md` after every query call. Always up to date, no explicit close needed.

**Key implication:** After each `_query_async()` completes, the querier appends to its in-memory step list and rewrites the full `index.md`. Simple, zero-risk of missing the final write. Slight overhead of rewriting the index file each time is negligible (it's tiny text).

---

### Q9
> Should the root directory for debug logs be configurable?

**Answer:** Fixed to `LLM/debug_logs/` — no configuration needed.

**Key implication:** `_DEBUG_DIR = Path(__file__).resolve().parent / "debug_logs"` hardcoded in `llm_querier.py`. Simple and consistent.

---

### Q10
> How does `LLMQuerier` get the game/phase tag for the session folder name?

**Answer:** Add `session_tag: str | None = None` to `LLMQuerier.__init__()`. Defaults to `"session"` if not provided. `Optimizer` passes `f"{self.game}_{self.target_phase}"` when constructing its querier.

**Key implication:**
- `LLMQuerier()` alone → folder: `LLM/debug_logs/session_20260308_143022/`
- `LLMQuerier(session_tag="sokoban_simulation")` → folder: `LLM/debug_logs/sokoban_simulation_20260308_143022/`
- `Optimizer` lazy-property for `querier` changes to: `LLMQuerier(session_tag=f"{self.game}_{self.target_phase}")`

---

### Q11
> Should `query()` / `query_two_step()` / `query_three_step()` accept a `step_name` argument for the log filename?

**Answer:** Both A and C — higher-level methods (`query_two_step`, `query_three_step`) name their internal calls automatically. Bare `query()` accepts an optional `step_name: str | None = None`, falling back to `query_N` (auto-incremented counter) if not provided.

**Key implication:**
- `query_two_step()` internally calls `query(..., step_name="step1_analysis")` and `query(..., step_name="step2_generation")`
- `query_three_step()` uses `step1_analysis`, `step2_generation`, `step3_critique`
- Repair calls in `Optimizer._repair()` pass `step_name=f"repair_{attempt+1}"`
- Standalone `query()` with no name → `query_1.md`, `query_2.md`, etc.

---

### Q12
> Any other edge cases or requirements to cover?

**Answer:** All of the above (D):
- `debug=False` → zero file I/O, no overhead
- Step name collision → auto-suffix: `step2_generation_2.md`, etc.
- Session folder creation failure → fail silently, disable debug for session, print one warning

---

## Design (APPROVED)

### Approach: `DebugLogger` helper class inside `llm_querier.py`

---

### Section 1: `DebugLogger` class

Private class inside `llm_querier.py`.

**Responsibilities:**
- On init: create session folder `LLM/debug_logs/{session_tag}_{timestamp}/`
- On folder creation failure: set `self.active = False`, print one warning, no crash
- Track: `step_records[]`, `used_names` set (for collision detection), `call_counter`

**Methods:**
- `log(step_name, prompt, response_text, metadata)` — resolve name collisions (`step2_generation_2` etc.), write `{step_name}.md`, append to `step_records`, rewrite `index.md`
- `_write_index()` — write `index.md` table of all steps so far

**Metadata fields:** `status`, `elapsed_seconds`, `model`, `validation`, `token_count` (from `response.usage.total_tokens` if API returns it, else `None`)

---

### Section 2: Changes to `LLMQuerier`

**New `__init__` parameters:**
```python
session_tag: str | None = None   # passed to DebugLogger; defaults to "session"
debug: bool | None = None        # None → read LLM_DEBUG env var ("1" = True)
```

**Init logic:**
```python
_debug = debug if debug is not None else os.getenv("LLM_DEBUG", "0") == "1"
self._logger = DebugLogger(session_tag or "session") if _debug else None
self._call_counter = 0
```

**`query()` new parameter:** `step_name: str | None = None`
- If None → auto-name `query_{N}` using `self._call_counter`
- After async call resolves → `self._logger.log(...)` if logger active

**`query_two_step()` internal step names:**
- Step 1 → `step_name="step1_analysis"`
- Step 2 → `step_name="step2_generation"`

**`query_three_step()` internal step names:**
- Step 1 → `step_name="step1_analysis"`
- Step 2 → `step_name="step2_generation"`
- Step 3 → `step_name="step3_critique"`

**`debug=False` path:** `self._logger` is `None` — all `if self._logger:` guards are no-ops, zero I/O overhead.

---

### Section 3: Changes to `Optimizer`

**`querier` lazy property:** pass `session_tag`:
```python
self._querier = LLMQuerier(session_tag=f"{self.game}_{self.target_phase}")
```

**`_repair()` method:** add `step_name` to repair query call:
```python
result = self.querier.query(
    repair_prompt,
    required_func_name=func_name,
    step_name=f"repair_{attempt + 1}",
)
```

No other `Optimizer` changes. `Optimizer` remains unaware of `debug` — controlled entirely by env var or direct `LLMQuerier` instantiation.

---

### Section 4: Log file formats

**Per-step file** (`LLM/debug_logs/{session}/step1_analysis.md`):
```markdown
# step1_analysis

| Field      | Value                          |
|------------|--------------------------------|
| Timestamp  | 2026-03-08 14:30:22            |
| Model      | api-gpt-oss-120b               |
| Elapsed    | 4.21s                          |
| Status     | success                        |
| Tokens     | 1024                           |
| Validation | valid — default_simulation     |

---

## Prompt

<full prompt text>

---

## Response

<full LLM response text>
```

**Session index** (`LLM/debug_logs/{session}/index.md`):
```markdown
# Session: sokoban_simulation_20260308_143022

| Step              | Status  | Elapsed | Tokens | Validation                  |
|-------------------|---------|---------|--------|-----------------------------|
| step1_analysis    | success | 4.21s   | 1024   | N/A                         |
| step2_generation  | success | 6.14s   | 2048   | valid — default_simulation  |
| repair_1          | success | 3.82s   | 900    | valid — default_simulation  |

**Total elapsed:** 14.17s
```
Rewritten after every query call (no explicit close needed).

---

### Edge Cases
- `debug=False` → `self._logger = None`, zero overhead
- Step name collision → auto-suffix: `step2_generation_2.md`
- Session folder creation failure → `self._logger.active = False`, one warning printed, no crash

---

### Files Changed
| File | Change |
|------|--------|
| `LLM/llm_querier.py` | Add `DebugLogger` class; add `debug`, `session_tag`, `step_name` params; wire logging into `_query_async` and multi-step methods |
| `LLM/optimizer.py` | Pass `session_tag` in `querier` property; pass `step_name` in `_repair()` |
| `.env` | Document new `LLM_DEBUG=1` variable |


