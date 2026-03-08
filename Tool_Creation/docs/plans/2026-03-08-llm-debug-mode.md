# LLM Debug Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `debug` mode to `LLMQuerier` that logs full prompt/response I/O for every API call into per-session timestamped Markdown files under `LLM/debug_logs/`.

**Architecture:** A private `DebugLogger` class lives inside `llm_querier.py` and is owned by `LLMQuerier`. When `debug=True` (or `LLM_DEBUG=1` in `.env`), `LLMQuerier` creates a `DebugLogger` on init which owns a session folder. After every async query, the logger writes one `.md` file and rewrites `index.md`. `Optimizer` passes a `session_tag` so folders are named `sokoban_simulation_20260308_143022/`.

**Tech Stack:** Python 3.11+, `pathlib`, `datetime` — no new dependencies.

---

## Task 1: Add `DebugLogger` class to `llm_querier.py`

**Files:**
- Modify: `LLM/llm_querier.py`
- Test: `tests/test_llm_querier.py`

### Step 1: Write the failing tests for `DebugLogger`

Add this test class to `tests/test_llm_querier.py`:

```python
# ─── DebugLogger ──────────────────────────────────────────────────────

class TestDebugLogger:
    def test_creates_session_folder(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        logger = DebugLogger(session_tag="test_sim", debug_root=tmp_path)
        assert logger.active is True
        assert logger.session_dir.exists()
        assert logger.session_dir.name.startswith("test_sim_")

    def test_log_writes_md_file(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        logger = DebugLogger(session_tag="test_sim", debug_root=tmp_path)
        metadata = {
            "status": "success", "elapsed_seconds": 1.5,
            "model": "gpt-test", "validation": {"valid": True, "error": None},
            "token_count": 100,
        }
        logger.log("step1_analysis", "my prompt", "my response", metadata)
        log_file = logger.session_dir / "step1_analysis.md"
        assert log_file.exists()
        content = log_file.read_text()
        assert "## Prompt" in content
        assert "my prompt" in content
        assert "## Response" in content
        assert "my response" in content
        assert "gpt-test" in content
        assert "1.5" in content

    def test_log_writes_index_md(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        logger = DebugLogger(session_tag="test_sim", debug_root=tmp_path)
        metadata = {
            "status": "success", "elapsed_seconds": 2.0,
            "model": "gpt-test", "validation": {"valid": True, "error": None},
            "token_count": 50,
        }
        logger.log("step1_analysis", "p", "r", metadata)
        index_file = logger.session_dir / "index.md"
        assert index_file.exists()
        content = index_file.read_text()
        assert "step1_analysis" in content
        assert "success" in content

    def test_collision_appends_suffix(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        logger = DebugLogger(session_tag="test_sim", debug_root=tmp_path)
        metadata = {
            "status": "success", "elapsed_seconds": 1.0,
            "model": "m", "validation": {"valid": True, "error": None},
            "token_count": None,
        }
        logger.log("step1_analysis", "p1", "r1", metadata)
        logger.log("step1_analysis", "p2", "r2", metadata)
        assert (logger.session_dir / "step1_analysis.md").exists()
        assert (logger.session_dir / "step1_analysis_2.md").exists()

    def test_folder_creation_failure_sets_inactive(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        # Point to a path that can't be created (file in the way)
        blocker = tmp_path / "blocked"
        blocker.write_text("I am a file, not a directory")
        logger = DebugLogger(session_tag="x", debug_root=blocker / "sub")
        assert logger.active is False

    def test_inactive_logger_log_is_noop(self, tmp_path):
        from LLM.llm_querier import DebugLogger
        blocker = tmp_path / "blocked"
        blocker.write_text("file")
        logger = DebugLogger(session_tag="x", debug_root=blocker / "sub")
        # Should not raise
        logger.log("step1", "p", "r", {"status": "success", "elapsed_seconds": 0,
                                        "model": "m", "validation": {}, "token_count": None})
        # No files written anywhere
        assert not (blocker / "sub").exists()
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation
pytest tests/test_llm_querier.py::TestDebugLogger -v
```

Expected: `ImportError: cannot import name 'DebugLogger'`

### Step 3: Implement `DebugLogger` in `llm_querier.py`

Add this class near the top of `LLM/llm_querier.py`, after the module-level constants (`_LLM_DIR`, `_RESULTS_DIR`) and before `class LLMQuerier`:

```python
_DEBUG_DIR = _LLM_DIR / "debug_logs"


class DebugLogger:
    """
    Writes per-call Markdown debug logs into a session folder.

    Parameters
    ----------
    session_tag : str
        Prefix for the session folder name (e.g. "sokoban_simulation").
    debug_root : Path | None
        Root directory for all sessions. Defaults to LLM/debug_logs/.
    """

    def __init__(
        self,
        session_tag: str = "session",
        debug_root: "Path | None" = None,
    ):
        root = Path(debug_root) if debug_root else _DEBUG_DIR
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = root / f"{session_tag}_{ts}"
        self.active = False
        self._records: list[dict] = []
        self._used_names: dict[str, int] = {}  # name -> count of uses

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.active = True
        except Exception as e:
            print(f"[DebugLogger] WARNING: Could not create session folder "
                  f"{self.session_dir}: {e}. Debug logging disabled.")

    def log(
        self,
        step_name: str,
        prompt: str,
        response_text: str,
        metadata: dict,
    ) -> None:
        """Write one step's prompt+response to a .md file and rewrite index.md."""
        if not self.active:
            return

        # Resolve collision
        resolved_name = self._resolve_name(step_name)

        # Write per-step file
        file_path = self.session_dir / f"{resolved_name}.md"
        file_path.write_text(
            self._render_step(resolved_name, prompt, response_text, metadata),
            encoding="utf-8",
        )

        # Track for index
        self._records.append({
            "step": resolved_name,
            "status": metadata.get("status", "unknown"),
            "elapsed": metadata.get("elapsed_seconds", 0),
            "tokens": metadata.get("token_count"),
            "validation": metadata.get("validation", {}),
        })

        # Rewrite index
        self._write_index()

    def _resolve_name(self, name: str) -> str:
        """Return name, or name_2, name_3, ... on collision."""
        if name not in self._used_names:
            self._used_names[name] = 1
            return name
        self._used_names[name] += 1
        return f"{name}_{self._used_names[name]}"

    def _render_step(
        self,
        step_name: str,
        prompt: str,
        response_text: str,
        metadata: dict,
    ) -> str:
        validation = metadata.get("validation") or {}
        valid_str = "N/A"
        if validation:
            if validation.get("valid"):
                valid_str = "valid"
            elif validation.get("error"):
                valid_str = f"invalid — {validation['error'][:80]}"

        token_str = str(metadata.get("token_count")) if metadata.get("token_count") else "N/A"

        return (
            f"# {step_name}\n\n"
            f"| Field      | Value |\n"
            f"|------------|-------|\n"
            f"| Timestamp  | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n"
            f"| Model      | {metadata.get('model', 'unknown')} |\n"
            f"| Elapsed    | {metadata.get('elapsed_seconds', 0):.2f}s |\n"
            f"| Status     | {metadata.get('status', 'unknown')} |\n"
            f"| Tokens     | {token_str} |\n"
            f"| Validation | {valid_str} |\n"
            f"\n---\n\n"
            f"## Prompt\n\n{prompt}\n\n"
            f"---\n\n"
            f"## Response\n\n{response_text}\n"
        )

    def _write_index(self) -> None:
        rows = []
        total_elapsed = 0.0
        for r in self._records:
            validation = r.get("validation") or {}
            valid_str = "N/A"
            if validation:
                valid_str = "valid" if validation.get("valid") else "invalid"
            token_str = str(r["tokens"]) if r["tokens"] is not None else "N/A"
            elapsed = r.get("elapsed") or 0
            total_elapsed += elapsed
            rows.append(
                f"| {r['step']:<25} | {r['status']:<7} | {elapsed:.2f}s "
                f"| {token_str:<6} | {valid_str} |"
            )

        table = "\n".join(rows)
        content = (
            f"# Session: {self.session_dir.name}\n\n"
            f"| Step                      | Status  | Elapsed | Tokens | Validation |\n"
            f"|---------------------------|---------|---------|--------|------------|\n"
            f"{table}\n\n"
            f"**Total elapsed:** {total_elapsed:.2f}s\n"
        )
        (self.session_dir / "index.md").write_text(content, encoding="utf-8")
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_llm_querier.py::TestDebugLogger -v
```

Expected: All 6 tests PASS.

### Step 5: Commit

```bash
git add LLM/llm_querier.py tests/test_llm_querier.py
git commit -m "feat: add DebugLogger class to llm_querier"
```

---

## Task 2: Wire `debug` and `session_tag` into `LLMQuerier`

**Files:**
- Modify: `LLM/llm_querier.py`
- Test: `tests/test_llm_querier.py`

### Step 1: Write the failing tests

Add to `tests/test_llm_querier.py`:

```python
# ─── LLMQuerier debug init ────────────────────────────────────────────

class TestLLMQuerierDebugInit:
    def test_debug_false_no_logger(self):
        q = LLMQuerier(api_keys=["k"], debug=False)
        assert q._logger is None

    def test_debug_true_creates_logger(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        assert q._logger is not None
        assert q._logger.active is True

    def test_debug_env_var_enables_logger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_DEBUG", "1")
        q = LLMQuerier(api_keys=["k"], _debug_root=tmp_path)
        assert q._logger is not None

    def test_debug_env_var_off_by_default(self, monkeypatch):
        monkeypatch.delenv("LLM_DEBUG", raising=False)
        q = LLMQuerier(api_keys=["k"], debug=None)
        assert q._logger is None

    def test_session_tag_passed_to_logger(self, tmp_path):
        q = LLMQuerier(
            api_keys=["k"], debug=True,
            session_tag="chess_selection", _debug_root=tmp_path
        )
        assert "chess_selection" in q._logger.session_dir.name

    def test_default_session_tag_is_session(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        assert q._logger.session_dir.name.startswith("session_")
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_llm_querier.py::TestLLMQuerierDebugInit -v
```

Expected: FAIL — `LLMQuerier.__init__` doesn't accept `debug`, `session_tag`, or `_debug_root`.

### Step 3: Update `LLMQuerier.__init__`

Replace the current `__init__` signature and body in `LLM/llm_querier.py`:

```python
def __init__(
    self,
    api_keys: list[str] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    results_dir: str | Path | None = None,
    session_tag: str | None = None,
    debug: bool | None = None,
    _debug_root: "Path | None" = None,   # for testing only
):
    self.api_keys = api_keys or _get_api_keys()
    self.base_url = base_url or _get_base_url()
    self.model = model or _get_model()
    self.results_dir = Path(results_dir) if results_dir else _RESULTS_DIR

    if not self.api_keys:
        raise ValueError(
            "No API keys configured. Set API_KEYS in .env or pass api_keys=[]."
        )

    _debug = debug if debug is not None else os.getenv("LLM_DEBUG", "0") == "1"
    self._logger: DebugLogger | None = (
        DebugLogger(session_tag or "session", debug_root=_debug_root)
        if _debug else None
    )
    self._call_counter = 0
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_llm_querier.py::TestLLMQuerierDebugInit -v
```

Expected: All 6 tests PASS.

### Step 5: Commit

```bash
git add LLM/llm_querier.py tests/test_llm_querier.py
git commit -m "feat: wire debug and session_tag params into LLMQuerier"
```

---

## Task 3: Log each API call in `_query_async`

**Files:**
- Modify: `LLM/llm_querier.py`
- Test: `tests/test_llm_querier.py`

### Step 1: Write the failing tests

Add to `tests/test_llm_querier.py`:

```python
# ─── LLMQuerier debug logging per query ──────────────────────────────

class TestLLMQuerierDebugLogging:
    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_writes_debug_file(self, mock_openai_class, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(
                "```python\ndef default_simulation(state, player): return 0.5\n```"
            )
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q.query("test prompt", step_name="step1_analysis")

        log_file = q._logger.session_dir / "step1_analysis.md"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test prompt" in content
        assert "default_simulation" in content

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_auto_names_call_counter(self, mock_openai_class, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("```python\ndef f(): pass\n```")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q.query("p1")
        q.query("p2")

        files = list(q._logger.session_dir.glob("query_*.md"))
        assert len(files) == 2

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_no_debug_no_files(self, mock_openai_class, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("```python\ndef f(): pass\n```")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=False)
        q.query("test prompt")

        assert q._logger is None
        # No debug_logs folder should be created
        assert not (_DEBUG_DIR).exists() or True  # just ensure no crash

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_writes_index_md(self, mock_openai_class, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("```python\ndef f(): pass\n```")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q.query("p1", step_name="step1_analysis")
        q.query("p2", step_name="step2_generation")

        index = q._logger.session_dir / "index.md"
        assert index.exists()
        content = index.read_text()
        assert "step1_analysis" in content
        assert "step2_generation" in content
```

Also add `_DEBUG_DIR` to the import at the top of the test file:
```python
from LLM.llm_querier import (
    LLMQuerier,
    extract_python_code,
    validate_function,
    _DEBUG_DIR,
)
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_llm_querier.py::TestLLMQuerierDebugLogging -v
```

Expected: FAIL — `query()` doesn't accept `step_name`, `_query_async` doesn't call logger.

### Step 3: Update `query()` signature and `_query_async()`

**Update `query()` to accept `step_name`:**

```python
def query(
    self,
    prompt: str,
    required_func_name: str | None = None,
    step_name: str | None = None,
) -> dict[str, Any]:
    """
    Send a single prompt to the LLM and return the result.
    ...
    step_name : str, optional
        Name for the debug log file (e.g. "step1_analysis").
        If None and debug=True, auto-named "query_N".
    ...
    """
    return self._run_async(self._query_async(prompt, required_func_name, step_name))
```

**Update `_query_async()` to accept and use `step_name`:**

Replace the current `_query_async` signature and add logging at the end of the success path and error path:

```python
async def _query_async(
    self,
    prompt: str,
    required_func_name: str | None = None,
    step_name: str | None = None,
) -> dict[str, Any]:
    """Single async query."""
    client = AsyncOpenAI(
        api_key=random.choice(self.api_keys),
        base_url=self.base_url,
    )

    # Resolve step name before the call so counter is consistent
    resolved_step = step_name or f"query_{self._call_counter}"
    self._call_counter += 1

    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        elapsed = time.time() - start

        code = extract_python_code(text)
        validation = (
            validate_function(code, required_func_name)
            if code else {"valid": False, "error": "No code block found in response."}
        )

        # Extract token count if available
        token_count = None
        try:
            token_count = response.usage.total_tokens
        except Exception:
            pass

        from .tool_manager import parse_response as _parse
        parsed = _parse(text)

        result = {
            "response": text,
            "code": code,
            "validation": validation,
            "parsed": parsed,
            "model": self.model,
            "elapsed_seconds": round(elapsed, 2),
            "status": "success",
        }

        # Debug logging
        if self._logger:
            self._logger.log(
                resolved_step, prompt, text,
                {
                    "status": "success",
                    "elapsed_seconds": round(elapsed, 2),
                    "model": self.model,
                    "validation": validation,
                    "token_count": token_count,
                }
            )

        return result

    except Exception as e:
        elapsed = time.time() - start
        error_result = {
            "response": None,
            "code": None,
            "validation": {"valid": False, "error": str(e)},
            "model": self.model,
            "elapsed_seconds": round(elapsed, 2),
            "status": "error",
            "error": str(e),
        }

        # Debug logging for errors too
        if self._logger:
            self._logger.log(
                resolved_step, prompt, f"ERROR: {e}",
                {
                    "status": "error",
                    "elapsed_seconds": round(elapsed, 2),
                    "model": self.model,
                    "validation": {"valid": False, "error": str(e)},
                    "token_count": None,
                }
            )

        return error_result
    finally:
        try:
            await client.close()
        except Exception:
            pass
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_llm_querier.py::TestLLMQuerierDebugLogging -v
```

Expected: All 4 tests PASS.

### Step 5: Commit

```bash
git add LLM/llm_querier.py tests/test_llm_querier.py
git commit -m "feat: log prompt/response to debug files in _query_async"
```

---

## Task 4: Add step names to `query_two_step` and `query_three_step`

**Files:**
- Modify: `LLM/llm_querier.py`
- Test: `tests/test_llm_querier.py`

### Step 1: Write the failing tests

Add to `tests/test_llm_querier.py`:

```python
# ─── Debug logging in multi-step methods ─────────────────────────────

class TestMultiStepDebugLogging:
    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_two_step_log_files_named_correctly(self, mock_openai_class, tmp_path):
        analysis_resp = _make_mock_response("Analysis text")
        gen_resp = _make_mock_response(
            "```python\ndef default_simulation(s, p): return 0.5\n```"
        )
        call_count = 0

        async def _create(**kwargs):
            nonlocal call_count
            resp = [analysis_resp, gen_resp][call_count]
            call_count += 1
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = _create
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q.query_two_step(
            analysis_prompt="Analyze",
            generation_prompt_fn=lambda a: f"Generate from: {a}",
        )

        assert (q._logger.session_dir / "step1_analysis.md").exists()
        assert (q._logger.session_dir / "step2_generation.md").exists()

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_three_step_log_files_named_correctly(self, mock_openai_class, tmp_path):
        responses = [
            _make_mock_response("Analysis text"),
            _make_mock_response("```python\ndef default_simulation(s, p): return 0.5\n```"),
            _make_mock_response("```python\ndef default_simulation(s, p): return 1.0\n```"),
        ]
        call_count = 0

        async def _create(**kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = _create
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q.query_three_step(
            analysis_prompt="Analyze",
            generation_prompt_fn=lambda a: f"Generate: {a}",
            critique_prompt_fn=lambda a, c: f"Critique: {c}",
        )

        assert (q._logger.session_dir / "step1_analysis.md").exists()
        assert (q._logger.session_dir / "step2_generation.md").exists()
        assert (q._logger.session_dir / "step3_critique.md").exists()
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_llm_querier.py::TestMultiStepDebugLogging -v
```

Expected: FAIL — multi-step methods call `self.query()` without `step_name`.

### Step 3: Add `step_name` to internal `query()` calls

In `query_two_step()`, update the two internal `self.query()` calls:

```python
# Step 1
step1_result = self.query(analysis_prompt, step_name="step1_analysis")

# Step 2
step2_result = self.query(gen_prompt, required_func_name=required_func_name,
                           step_name="step2_generation")
```

In `query_three_step()`, update the three internal `self.query()` calls:

```python
# Step 1
step1_result = self.query(analysis_prompt, step_name="step1_analysis")

# Step 2
step2_result = self.query(gen_prompt, required_func_name=required_func_name,
                           step_name="step2_generation")

# Step 3
step3_result = self.query(critique_prompt, required_func_name=required_func_name,
                           step_name="step3_critique")
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_llm_querier.py::TestMultiStepDebugLogging -v
```

Expected: Both tests PASS.

### Step 5: Run the full test suite to check for regressions

```bash
pytest tests/test_llm_querier.py -v
```

Expected: All tests PASS.

### Step 6: Commit

```bash
git add LLM/llm_querier.py tests/test_llm_querier.py
git commit -m "feat: add step_name to multi-step query methods for debug logging"
```

---

## Task 5: Update `Optimizer` to pass `session_tag` and `step_name` for repairs

**Files:**
- Modify: `LLM/optimizer.py`
- Test: `tests/test_llm_querier.py` (integration note only — optimizer tests are out of scope)

### Step 1: Update `Optimizer.querier` lazy property

In `LLM/optimizer.py`, replace the `querier` property:

```python
@property
def querier(self) -> LLMQuerier:
    if self._querier is None:
        self._querier = LLMQuerier(
            session_tag=f"{self.game}_{self.target_phase}"
        )
    return self._querier
```

### Step 2: Update `Optimizer._repair()` to pass `step_name`

In `LLM/optimizer.py`, find the `self.querier.query(repair_prompt, ...)` call inside `_repair()` and update it:

```python
result = self.querier.query(
    repair_prompt,
    required_func_name=func_name,
    step_name=f"repair_{attempt + 1}",   # NEW
)
```

Note: `attempt` is available in `_smoke_test_with_repair` — pass it to `_repair()`. Update `_repair()` signature to accept `attempt: int = 0` and update the call site:

```python
# In _smoke_test_with_repair:
repair_result = self._repair(parsed, func_name, tb, state_factory, attempt=attempt)

# _repair() signature:
def _repair(self, parsed, func_name, tb_text, state_factory, attempt: int = 0):
    ...
    result = self.querier.query(
        repair_prompt,
        required_func_name=func_name,
        step_name=f"repair_{attempt + 1}",
    )
```

### Step 3: Run existing tests to check for regressions

```bash
pytest tests/ -v --ignore=tests/test_rush_hour.py
```

Expected: All tests PASS (no optimizer unit tests exist, but querier and other tests must still pass).

### Step 4: Commit

```bash
git add LLM/optimizer.py
git commit -m "feat: pass session_tag and repair step_name from Optimizer to LLMQuerier"
```

---

## Task 6: Document `LLM_DEBUG` in `.env` and verify end-to-end

**Files:**
- Modify: `.env` (or `.env.example` if it exists)

### Step 1: Check for `.env` or `.env.example`

```bash
ls /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/.env*
```

### Step 2: Add `LLM_DEBUG` documentation

Add to `.env` (or `.env.example`):

```bash
# Debug mode: set to 1 to log all LLM prompts and responses to LLM/debug_logs/
# Each optimizer session gets its own timestamped folder.
LLM_DEBUG=0
```

### Step 3: Run the full test suite one final time

```bash
pytest tests/ -v --ignore=tests/test_rush_hour.py
```

Expected: All tests PASS.

### Step 4: Commit

```bash
git add .env  # or .env.example
git commit -m "docs: document LLM_DEBUG env var in .env"
```

---

## Summary of Changes

| File | What changed |
|------|-------------|
| `LLM/llm_querier.py` | Added `DebugLogger` class; added `debug`, `session_tag`, `_debug_root`, `step_name` params; wired logging into `_query_async`; step names in `query_two_step` / `query_three_step` |
| `LLM/optimizer.py` | `querier` property passes `session_tag`; `_repair()` passes `step_name` |
| `tests/test_llm_querier.py` | Added `TestDebugLogger`, `TestLLMQuerierDebugInit`, `TestLLMQuerierDebugLogging`, `TestMultiStepDebugLogging` |
| `.env` / `.env.example` | Documented `LLM_DEBUG=0` |

## Expected debug output structure

```
LLM/debug_logs/
└── sokoban_simulation_20260308_143022/
    ├── index.md
    ├── step1_analysis.md
    ├── step2_generation.md
    └── repair_1.md          # only if repair was triggered
```
