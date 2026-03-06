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
