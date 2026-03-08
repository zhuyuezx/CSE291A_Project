"""Tests for LLM.llm_querier module."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Allow running even without openai installed
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLM.llm_querier import (
    LLMQuerier,
    extract_python_code,
    validate_function,
    _DEBUG_DIR,
)


# ─── extract_python_code ─────────────────────────────────────────────

class TestExtractPythonCode:
    def test_basic_code_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    return 42\n```\nDone."
        assert extract_python_code(text) == "def foo():\n    return 42"

    def test_multiline_code_block(self):
        text = (
            "```python\n"
            "import math\n\n"
            "def compute(x):\n"
            "    return math.sqrt(x)\n"
            "```"
        )
        code = extract_python_code(text)
        assert "import math" in code
        assert "def compute(x):" in code

    def test_no_code_block_returns_none(self):
        assert extract_python_code("No code here.") is None

    def test_empty_code_block(self):
        text = "```python\n```"
        result = extract_python_code(text)
        assert result is None or result == ""

    def test_only_first_block_extracted(self):
        text = (
            "```python\ndef first(): pass\n```\n"
            "```python\ndef second(): pass\n```"
        )
        code = extract_python_code(text)
        assert "first" in code
        assert "second" not in code

    def test_non_python_block_ignored(self):
        text = "```javascript\nconsole.log('hi')\n```"
        assert extract_python_code(text) is None

    def test_code_with_decorators(self):
        text = '```python\n@staticmethod\ndef bar(x, y):\n    """Docstring."""\n    return x + y\n```'
        code = extract_python_code(text)
        assert "@staticmethod" in code
        assert "def bar(x, y):" in code

    def test_preserves_indentation(self):
        text = "```python\ndef f():\n    if True:\n        x = 1\n        return x\n```"
        code = extract_python_code(text)
        lines = code.split("\n")
        assert lines[2].startswith("        ")


# ─── validate_function ───────────────────────────────────────────────

class TestValidateFunction:
    def test_valid_function(self):
        code = "def my_func(x):\n    return x + 1"
        result = validate_function(code)
        assert result["valid"] is True
        assert result["error"] is None

    def test_syntax_error(self):
        code = "def broken(\n    return"
        result = validate_function(code)
        assert result["valid"] is False
        assert "SyntaxError" in result["error"]

    def test_no_function_definition(self):
        code = "x = 42\ny = x + 1"
        result = validate_function(code)
        assert result["valid"] is False
        assert "No function definition" in result["error"]

    def test_required_name_present(self):
        code = "def simulate(node, game):\n    pass"
        result = validate_function(code, required_name="simulate")
        assert result["valid"] is True

    def test_required_name_missing(self):
        code = "def other_func():\n    pass"
        result = validate_function(code, required_name="simulate")
        assert result["valid"] is False
        assert "simulate" in result["error"]

    def test_multiple_functions(self):
        code = "def helper():\n    pass\ndef main_func():\n    pass"
        result = validate_function(code, required_name="main_func")
        assert result["valid"] is True

    def test_class_method_not_counted(self):
        code = "class Foo:\n    def method(self):\n        pass"
        result = validate_function(code, required_name="method")
        assert result["valid"] is True

    def test_complex_valid_code(self):
        code = (
            "import math\n\n"
            "def simulate(node, game, max_depth=50):\n"
            "    state = game.clone()\n"
            "    for _ in range(max_depth):\n"
            "        if state.is_terminal():\n"
            "            break\n"
            "        action = state.get_actions()[0]\n"
            "        state.apply(action)\n"
            "    return state.reward()\n"
        )
        result = validate_function(code, required_name="simulate")
        assert result["valid"] is True


# ─── LLMQuerier init ─────────────────────────────────────────────────

class TestLLMQuerierInit:
    def test_init_with_explicit_keys(self):
        q = LLMQuerier(api_keys=["key1", "key2"])
        assert q.api_keys == ["key1", "key2"]
        assert q.model == "api-gpt-oss-120b"

    def test_init_custom_model(self):
        q = LLMQuerier(api_keys=["k"], model="gpt-4")
        assert q.model == "gpt-4"

    def test_init_custom_base_url(self):
        q = LLMQuerier(api_keys=["k"], base_url="https://example.com")
        assert q.base_url == "https://example.com"

    def test_init_no_keys_raises(self):
        with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
            with pytest.raises(ValueError, match="No API keys"):
                LLMQuerier(api_keys=[])

    def test_init_custom_results_dir(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], results_dir=tmp_path / "out")
        assert q.results_dir == tmp_path / "out"


# ─── query (mocked) ──────────────────────────────────────────────────

def _make_mock_response(content: str):
    """Create a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMQuerierQuery:
    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_success(self, mock_openai_class):
        # Setup mock
        llm_response_text = (
            "Here is an improved simulate:\n"
            "```python\n"
            "def simulate(node, game):\n"
            "    return game.reward()\n"
            "```\n"
        )
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(llm_response_text)
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["test-key"])
        result = q.query("test prompt", required_func_name="simulate")

        assert result["status"] == "success"
        assert result["code"] is not None
        assert "def simulate" in result["code"]
        assert result["validation"]["valid"] is True
        assert result["elapsed_seconds"] >= 0

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_no_code_block(self, mock_openai_class):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("I couldn't generate code.")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["test-key"])
        result = q.query("test prompt")

        assert result["status"] == "success"
        assert result["code"] is None
        assert result["validation"]["valid"] is False

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_api_error(self, mock_openai_class):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["test-key"])
        result = q.query("test prompt")

        assert result["status"] == "error"
        assert "rate limit" in result["error"]
        assert result["code"] is None

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_invalid_syntax_in_code(self, mock_openai_class):
        bad_code = "```python\ndef broken(\n    return x\n```"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(bad_code)
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["test-key"])
        result = q.query("test prompt")

        assert result["status"] == "success"
        assert result["code"] is not None
        assert result["validation"]["valid"] is False
        assert "SyntaxError" in result["validation"]["error"]

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_query_wrong_function_name(self, mock_openai_class):
        code = "```python\ndef wrong_name(x):\n    return x\n```"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(code)
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["test-key"])
        result = q.query("test prompt", required_func_name="simulate")

        assert result["status"] == "success"
        assert result["validation"]["valid"] is False
        assert "simulate" in result["validation"]["error"]


# ─── query_batch (mocked) ────────────────────────────────────────────

class TestLLMQuerierBatch:
    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_batch_multiple_prompts(self, mock_openai_class):
        responses = [
            _make_mock_response("```python\ndef f1(): pass\n```"),
            _make_mock_response("```python\ndef f2(): pass\n```"),
            _make_mock_response("No code here."),
        ]
        call_count = 0

        async def _create(**kwargs):
            nonlocal call_count
            resp = responses[call_count % len(responses)]
            call_count += 1
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = _create
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k1", "k2"])
        results = q.query_batch(["p1", "p2", "p3"])

        assert len(results) == 3
        assert results[0]["code"] is not None
        assert results[1]["code"] is not None
        assert results[2]["code"] is None

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_batch_round_robin_keys(self, mock_openai_class):
        """Verify that multiple clients are created for multiple keys."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("```python\ndef f(): pass\n```")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k1", "k2", "k3"])
        results = q.query_batch(["p1", "p2", "p3"])

        # Three clients should have been created (one per key)
        assert mock_openai_class.call_count == 3
        assert len(results) == 3


# ─── save ─────────────────────────────────────────────────────────────

class TestLLMQuerierSave:
    def test_save_auto_filename(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], results_dir=tmp_path)
        result = {
            "response": "test",
            "code": "def f(): pass",
            "status": "success",
        }
        saved_path = q.save(result)
        assert saved_path.exists()
        data = json.loads(saved_path.read_text())
        assert data["code"] == "def f(): pass"

    def test_save_explicit_path(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], results_dir=tmp_path)
        target = tmp_path / "sub" / "result.json"
        q.save({"test": True}, filepath=target)
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["test"] is True

    def test_save_creates_dirs(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], results_dir=tmp_path / "deep" / "nested")
        result = {"status": "success"}
        saved = q.save(result)
        assert saved.parent.exists()
        assert saved.exists()


# ─── Integration: extract + validate pipeline ────────────────────────

class TestExtractValidatePipeline:
    def test_full_pipeline_valid(self):
        response = (
            "I've improved the simulation function:\n\n"
            "```python\n"
            "import random\n\n"
            "def simulate(node, game, max_depth=100):\n"
            '    """Improved simulation with heuristic."""\n'
            "    state = game.clone()\n"
            "    depth = 0\n"
            "    while not state.is_terminal() and depth < max_depth:\n"
            "        actions = state.get_actions()\n"
            "        action = random.choice(actions)\n"
            "        state.apply(action)\n"
            "        depth += 1\n"
            "    return state.reward()\n"
            "```\n\n"
            "This version uses a depth limit."
        )
        code = extract_python_code(response)
        assert code is not None
        result = validate_function(code, required_name="simulate")
        assert result["valid"] is True

    def test_full_pipeline_invalid(self):
        response = "Here:\n```python\nclass NotAFunction:\n    pass\n```"
        code = extract_python_code(response)
        assert code is not None
        result = validate_function(code, required_name="simulate")
        assert result["valid"] is False


# ─── query_three_step (mocked) ───────────────────────────────────────

class TestLLMQuerierThreeStep:
    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_three_step_success(self, mock_openai_class):
        """Three-step pipeline: analysis → draft → critique+finalize."""
        analysis_resp = _make_mock_response("ANALYSIS: The heuristic is too simple.")
        draft_resp = _make_mock_response(
            "ACTION: modify\nFILE_NAME: simulation.py\nFUNCTION_NAME: default_simulation\n"
            "```python\ndef default_simulation(state, player, max_depth=50):\n"
            "    return 0.5\n```"
        )
        final_resp = _make_mock_response(
            "CRITIQUE:\n- Draft always returns 0.5\n\n"
            "ACTION: modify\nFILE_NAME: simulation.py\nFUNCTION_NAME: default_simulation\n"
            "```python\ndef default_simulation(state, player, max_depth=50):\n"
            "    return state.returns()[player]\n```"
        )
        call_count = 0
        responses = [analysis_resp, draft_resp, final_resp]

        async def _create(**kwargs):
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = _create
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k1"])
        result = q.query_three_step(
            analysis_prompt="Analyze this",
            generation_prompt_fn=lambda a: f"Generate code based on: {a}",
            critique_prompt_fn=lambda a, c: f"Critique: {c}",
            required_func_name="default_simulation",
        )

        assert result["status"] == "success"
        assert result["step1_analysis"] == "ANALYSIS: The heuristic is too simple."
        assert result["step2_draft_code"] is not None
        assert "default_simulation" in result["step2_draft_code"]
        assert result["code"] is not None
        assert "state.returns()" in result["code"]
        assert result["elapsed_seconds"] >= 0
        assert call_count == 3  # 3 separate LLM calls

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_three_step_step1_error(self, mock_openai_class):
        """If step 1 fails, the pipeline stops early."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API down")
        )
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k1"])
        result = q.query_three_step(
            analysis_prompt="Analyze",
            generation_prompt_fn=lambda a: f"Gen: {a}",
            critique_prompt_fn=lambda a, c: f"Crit: {c}",
        )

        assert result["status"] == "error"
        assert result["step1_analysis"] is None

    @patch("LLM.llm_querier.AsyncOpenAI")
    def test_three_step_step2_error(self, mock_openai_class):
        """If step 2 fails, pipeline stops before critique."""
        analysis_resp = _make_mock_response("Analysis text")
        call_count = 0

        async def _create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return analysis_resp
            raise Exception("Generation failed")

        mock_client = MagicMock()
        mock_client.chat.completions.create = _create
        mock_openai_class.return_value = mock_client

        q = LLMQuerier(api_keys=["k1"])
        result = q.query_three_step(
            analysis_prompt="Analyze",
            generation_prompt_fn=lambda a: f"Gen: {a}",
            critique_prompt_fn=lambda a, c: f"Crit: {c}",
        )

        assert result["status"] == "error"
        assert result["step1_analysis"] == "Analysis text"


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

    def test_new_session_creates_separate_folder(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        first_dir = q._logger.session_dir
        q.new_session("level5_simulation")
        second_dir = q._logger.session_dir
        assert first_dir != second_dir
        assert "level5_simulation" in second_dir.name
        assert second_dir.exists()

    def test_new_session_resets_call_counter(self, tmp_path):
        q = LLMQuerier(api_keys=["k"], debug=True, _debug_root=tmp_path)
        q._call_counter = 5
        q.new_session("fresh")
        assert q._call_counter == 0

    def test_new_session_noop_when_debug_off(self):
        q = LLMQuerier(api_keys=["k"], debug=False)
        q.new_session("anything")
        assert q._logger is None


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
