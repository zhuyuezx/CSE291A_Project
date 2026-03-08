# LLM Debug Mode

Debug mode logs every LLM prompt and response to timestamped Markdown files, making it easy to inspect what the optimizer sent and received.

---

## Quick Start

Set the environment variable in `.env`:

```bash
LLM_DEBUG=1
```

Or pass it programmatically:

```python
from LLM.llm_querier import LLMQuerier

querier = LLMQuerier(api_keys=["..."], debug=True, session_tag="sokoban_simulation")
result = querier.query("your prompt", step_name="step1_analysis")
```

---

## Output Structure

Each debug session creates a timestamped folder under `LLM/debug_logs/`:

```
LLM/debug_logs/
├── iter1_level4_simulation_20260308_143022/
│   ├── index.md              # summary table of all calls
│   ├── step1_analysis.md     # prompt + response for step 1
│   ├── step2_generation.md   # prompt + response for step 2
│   └── repair_1.md           # only if a repair was triggered
├── iter2_level5_simulation_20260308_143055/
│   ├── index.md
│   ├── step1_analysis.md
│   └── step2_generation.md
└── ...
```

### Per-step files

Each `.md` file contains a metadata table followed by the full prompt and response:

```markdown
# step1_analysis

| Field      | Value            |
|------------|------------------|
| Timestamp  | 2026-03-08 14:30 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 3.45s            |
| Status     | success          |
| Tokens     | 1523             |
| Validation | valid            |

---

## Prompt
<full prompt text>

---

## Response
<full LLM response>
```

### Index file

`index.md` is rewritten after each call and summarizes the session:

| Step                      | Status  | Elapsed | Tokens | Validation |
|---------------------------|---------|---------|--------|------------|
| step1_analysis            | success | 3.45s   | 1523   | valid      |
| step2_generation          | success | 4.12s   | 2041   | valid      |

---

## How It Works

### Activation

Debug mode activates when **either**:

- `LLM_DEBUG=1` is set in the environment / `.env` file, **or**
- `debug=True` is passed to `LLMQuerier()`

The `debug` constructor parameter takes precedence over the env var. When neither is set, debug logging is off (no overhead).

### Session Tags

The `session_tag` parameter controls the folder name prefix. When running through the `Optimizer` → `OptimizationRunner` pipeline, each `opt.run()` call automatically creates a **separate debug folder** tagged with iteration, level, and phase (e.g. `iter2_level5_simulation`). This means each level's complete loop (analysis → code generation → repair) lives in its own folder.

### Step Names

| Context | Step names |
|---------|-----------|
| `query_two_step()` | `step1_analysis`, `step2_generation` |
| `query_three_step()` | `step1_analysis`, `step2_generation`, `step3_critique` |
| `query()` with explicit name | whatever you pass as `step_name` |
| `query()` without name | auto-numbered `query_0`, `query_1`, ... |
| Repair loop (via `Optimizer`) | `repair_1`, `repair_2`, ... |

If the same step name is logged twice, a suffix is appended (`step1_analysis_2`).

### Error Handling

- If the debug folder cannot be created, logging silently disables (`logger.active = False`) and the pipeline continues normally.
- API errors are still logged with the error text in place of the response.

---

## API Reference

### `DebugLogger`

Internal class in `LLM/llm_querier.py`. Not typically used directly.

```python
DebugLogger(session_tag="session", debug_root=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_tag` | `str` | `"session"` | Prefix for the session folder |
| `debug_root` | `Path \| None` | `LLM/debug_logs/` | Root directory for all sessions |

### `LLMQuerier` (new parameters)

```python
LLMQuerier(..., debug=None, session_tag=None, _debug_root=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `debug` | `bool \| None` | `None` | Enable debug logging. Falls back to `LLM_DEBUG` env var |
| `session_tag` | `str \| None` | `None` | Passed to `DebugLogger`. Defaults to `"session"` |
| `_debug_root` | `Path \| None` | `None` | Override debug directory (for testing) |

### `LLMQuerier.new_session()`

```python
querier.new_session(session_tag)
```

Starts a fresh debug session with its own folder and resets the call counter. No-op when debug is disabled. Called automatically by `Optimizer.run()`.

### `query()` (new parameter)

```python
querier.query(prompt, required_func_name=None, step_name=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_name` | `str \| None` | `None` | Name for the debug log file. Auto-numbered if omitted |
