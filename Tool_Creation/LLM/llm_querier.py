"""
LLM Querier for MCTS heuristic improvement.

Sends assembled prompts to an OpenAI-compatible LLM API and extracts
the improved heuristic function code from the response.

This module adapts the multi-API batch template for the MCTS pipeline:
  1. Takes a prompt string (from PromptBuilder)
  2. Sends it to the LLM via async OpenAI client
  3. Extracts the Python code block from the response
  4. Optionally validates the extracted function
  5. Saves raw response + extracted code

Configuration is loaded from .env:
    API_KEYS       — comma-separated API keys
    OPENAI_BASE_URL — API base URL
    MODEL_NAME     — model identifier

Usage::

    querier = LLMQuerier()
    result = querier.query(prompt_text)
    print(result["code"])        # extracted Python code
    print(result["response"])    # full LLM response
    querier.save(result, "LLM/results/sim_v1.json")

    # Batch mode (multiple prompts):
    results = querier.query_batch([prompt1, prompt2, prompt3])
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    # Walk up to find .env in Tool_Creation/
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc,assignment]


# ── Configuration from environment ───────────────────────────────────
def _get_api_keys() -> list[str]:
    raw = os.getenv("API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys

def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")

def _get_model() -> str:
    return os.getenv("MODEL_NAME", "api-gpt-oss-120b")


_LLM_DIR = Path(__file__).resolve().parent
_RESULTS_DIR = _LLM_DIR / "results"


# ── Code extraction ──────────────────────────────────────────────────

def extract_python_code(response: str) -> str | None:
    """
    Extract the first ```python ... ``` code block from an LLM response.

    Returns None if no code block is found.
    """
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_function(code: str, required_name: str | None = None) -> dict[str, Any]:
    """
    Basic validation of extracted Python code.

    Checks:
      - Parses without SyntaxError
      - Contains at least one function definition
      - If required_name is given, that function exists

    Returns dict with 'valid' bool and 'error' message if invalid.
    """
    try:
        import ast
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "error": f"SyntaxError: {e}"}

    func_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    if not func_names:
        return {"valid": False, "error": "No function definition found in code."}

    if required_name and required_name not in func_names:
        return {
            "valid": False,
            "error": f"Expected function '{required_name}', found: {func_names}",
        }

    return {"valid": True, "error": None}


# ── LLM Querier ──────────────────────────────────────────────────────

class LLMQuerier:
    """
    Query an OpenAI-compatible LLM with MCTS improvement prompts.

    Parameters
    ----------
    api_keys : list[str] | None
        API keys. Defaults to API_KEYS from .env.
    base_url : str | None
        API base URL. Defaults to OPENAI_BASE_URL from .env.
    model : str | None
        Model name. Defaults to MODEL_NAME from .env.
    results_dir : str | Path | None
        Directory to save results. Defaults to LLM/results/.
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        base_url: str | None = None,
        model: str | None = None,
        results_dir: str | Path | None = None,
    ):
        self.api_keys = api_keys or _get_api_keys()
        self.base_url = base_url or _get_base_url()
        self.model = model or _get_model()
        self.results_dir = Path(results_dir) if results_dir else _RESULTS_DIR

        if not self.api_keys:
            raise ValueError(
                "No API keys configured. Set API_KEYS in .env or pass api_keys=[]."
            )

    def query(
        self,
        prompt: str,
        required_func_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a single prompt to the LLM and return the result.

        Parameters
        ----------
        prompt : str
            The full prompt text (from PromptBuilder).
        required_func_name : str, optional
            If given, validate that the extracted code defines this function.

        Returns
        -------
        dict with keys:
            response     — full LLM text response
            code         — extracted Python code (or None)
            validation   — dict with 'valid' and 'error'
            model        — model used
            elapsed_seconds — time taken
            status       — 'success' or 'error'
        """
        return asyncio.run(self._query_async(prompt, required_func_name))

    def query_batch(
        self,
        prompts: list[str],
        required_func_name: str | None = None,
        concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Send multiple prompts in parallel.

        Parameters
        ----------
        prompts : list[str]
            List of prompt strings.
        required_func_name : str, optional
            Expected function name for validation.
        concurrency : int, optional
            Max concurrent requests. Defaults to len(api_keys).

        Returns
        -------
        List of result dicts (same format as query()).
        """
        return asyncio.run(
            self._query_batch_async(prompts, required_func_name, concurrency)
        )

    def save(
        self,
        result: dict[str, Any],
        filepath: str | Path | None = None,
    ) -> Path:
        """
        Save a query result to a JSON file.

        Returns the path the file was saved to.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / f"llm_result_{ts}.json"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        return filepath

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _query_async(
        self,
        prompt: str,
        required_func_name: str | None = None,
    ) -> dict[str, Any]:
        """Single async query."""
        client = AsyncOpenAI(
            api_key=self.api_keys[0],
            base_url=self.base_url,
        )

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

            return {
                "response": text,
                "code": code,
                "validation": validation,
                "model": self.model,
                "elapsed_seconds": round(elapsed, 2),
                "status": "success",
            }

        except Exception as e:
            elapsed = time.time() - start
            return {
                "response": None,
                "code": None,
                "validation": {"valid": False, "error": str(e)},
                "model": self.model,
                "elapsed_seconds": round(elapsed, 2),
                "status": "error",
                "error": str(e),
            }

    async def _query_batch_async(
        self,
        prompts: list[str],
        required_func_name: str | None = None,
        concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        """Send multiple prompts in parallel with round-robin key distribution."""
        from itertools import cycle

        max_concurrent = concurrency or len(self.api_keys)
        semaphore = asyncio.Semaphore(max_concurrent)

        clients = [
            AsyncOpenAI(api_key=key, base_url=self.base_url)
            for key in self.api_keys
        ]
        key_cycle = cycle(range(len(clients)))

        async def _single(prompt: str, client: AsyncOpenAI) -> dict:
            async with semaphore:
                start = time.time()
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.choices[0].message.content or ""
                    elapsed = time.time() - start
                    code = extract_python_code(text)
                    validation = (
                        validate_function(code, required_func_name)
                        if code else {"valid": False, "error": "No code block found."}
                    )
                    return {
                        "response": text,
                        "code": code,
                        "validation": validation,
                        "model": self.model,
                        "elapsed_seconds": round(elapsed, 2),
                        "status": "success",
                    }
                except Exception as e:
                    elapsed = time.time() - start
                    return {
                        "response": None,
                        "code": None,
                        "validation": {"valid": False, "error": str(e)},
                        "model": self.model,
                        "elapsed_seconds": round(elapsed, 2),
                        "status": "error",
                        "error": str(e),
                    }

        tasks = [
            _single(p, clients[next(key_cycle)])
            for p in prompts
        ]
        return list(await asyncio.gather(*tasks))
