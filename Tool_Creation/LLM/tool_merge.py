"""
Tool Merger — merge a cluster of MCTS heuristic tools into one.

Given a cluster (list of tool source-code strings for the same MCTS
phase), this module asks an LLM to produce a single merged
implementation that:
  1. Preserves the correct function signature for the phase.
  2. Combines the best ideas from all variants.
  3. Returns code that passes ``ToolManager.validate()``.

Adapted from Yunjue Agent's ``tool_merge.md`` / ``merge_tools()``,
specialised for MCTS heuristics.

Usage::

    from LLM.tool_merge import merge_tools
    merged_code = merge_tools(
        tool_sources=[("sim_v1", src1), ("sim_v2", src2)],
        phase="simulation",
        querier=querier,
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .tool_manager import ToolManager, EXPECTED_SIGNATURES, validate


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_merge_prompt() -> str:
    path = _PROMPTS_DIR / "tool_merge_mcts.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def merge_tools(
    tool_sources: list[tuple[str, str]],
    phase: str = "simulation",
    querier: Any | None = None,
    suggested_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Merge multiple tool source-code variants into one.

    Parameters
    ----------
    tool_sources : list of (name, source_code)
        The tools to merge.  All belong to the same MCTS *phase*.
    phase : str
        MCTS phase (determines expected function signature).
    querier : LLMQuerier | None
        If None, returns the shortest source (naive merge).
    suggested_name : str | None
        Suggested merged function / file name stem.

    Returns
    -------
    dict with keys ``code``, ``function_name``, ``description``,
    ``file_name`` — compatible with ``ToolManager.install()`` — or
    *None* if the merge failed.
    """
    if not tool_sources:
        return None
    if len(tool_sources) == 1:
        name, src = tool_sources[0]
        return {
            "action": "modify",
            "file_name": f"{_safe_stem(suggested_name or name)}.py",
            "function_name": _infer_function_name(phase),
            "description": f"Single tool (no merge needed): {name}",
            "code": src,
            "parse_errors": [],
        }

    if querier is not None:
        result = _llm_merge(tool_sources, phase, querier, suggested_name)
        if result is not None:
            return result

    return _naive_merge(tool_sources, phase, suggested_name)


def _llm_merge(
    tool_sources: list[tuple[str, str]],
    phase: str,
    querier: Any,
    suggested_name: str | None,
) -> dict[str, Any] | None:
    """Ask LLM to merge, parse response through ToolManager contract."""
    template = _load_merge_prompt()
    if not template:
        template = _default_merge_prompt()

    snippets = "\n".join(
        f"=== Tool '{name}' ===\n```python\n{src.rstrip()}\n```"
        for name, src in tool_sources
    )
    expected_sig = EXPECTED_SIGNATURES.get(phase, [])
    sig_str = ", ".join(expected_sig) if expected_sig else "(no params)"
    func_name = _infer_function_name(phase)
    fname = _safe_stem(suggested_name or func_name)

    prompt = (
        template
        .replace("{{PHASE}}", phase)
        .replace("{{TOOL_SNIPPETS}}", snippets)
        .replace("{{EXPECTED_SIGNATURE}}", sig_str)
        .replace("{{FUNCTION_NAME}}", func_name)
        .replace("{{FILE_NAME}}", f"{fname}.py")
    )

    try:
        result = querier.query(prompt, step_name="tool_merge")
        response_text = result.get("response", "")

        mgr = ToolManager()
        parsed = mgr.parse_response(response_text)
        # Ensure correct function name even if LLM deviates
        if not parsed.get("function_name"):
            parsed["function_name"] = func_name
        if not parsed.get("file_name"):
            parsed["file_name"] = f"{fname}.py"
        if not parsed.get("action"):
            parsed["action"] = "modify"
        # Strip parse errors about missing optional headers
        parsed["parse_errors"] = [
            e for e in parsed.get("parse_errors", [])
            if "Missing ACTION" not in e
            and "Missing FILE_NAME" not in e
            and "Missing FUNCTION_NAME" not in e
        ]

        validation = validate(parsed, phase=phase)
        if validation["valid"]:
            return parsed
        return None
    except Exception:
        return None


def _naive_merge(
    tool_sources: list[tuple[str, str]],
    phase: str,
    suggested_name: str | None,
) -> dict[str, Any]:
    """Pick the shortest source as a simple fallback."""
    name, src = min(tool_sources, key=lambda t: len(t[1]))
    func_name = _infer_function_name(phase)
    fname = _safe_stem(suggested_name or func_name)
    return {
        "action": "modify",
        "file_name": f"{fname}.py",
        "function_name": func_name,
        "description": f"Naive merge (kept shortest variant: {name})",
        "code": src,
        "parse_errors": [],
    }


def _infer_function_name(phase: str) -> str:
    if phase == "hyperparams":
        return "get_hyperparams"
    return f"default_{phase}"


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
    return stem or "merged_tool"


def _default_merge_prompt() -> str:
    return (
        "You are an expert Python engineer specialising in MCTS heuristics.\n\n"
        "MCTS phase: {{PHASE}}\n"
        "Required function signature: def {{FUNCTION_NAME}}({{EXPECTED_SIGNATURE}})\n\n"
        "Below are multiple heuristic implementations for the same phase.\n"
        "Merge them into ONE function that combines the best ideas while\n"
        "preserving correctness.  Use only the Python standard library.\n\n"
        "{{TOOL_SNIPPETS}}\n\n"
        "Output EXACTLY:\n\n"
        "ACTION: modify\n"
        "FILE_NAME: {{FILE_NAME}}\n"
        "FUNCTION_NAME: {{FUNCTION_NAME}}\n"
        "DESCRIPTION: <one-line summary of the merged heuristic>\n"
        "```python\n<complete merged function>\n```\n\n"
        "Rules:\n"
        "- Keep the exact function signature shown above.\n"
        "- The code must be standalone (standard library only).\n"
        "- Combine complementary ideas; remove redundant logic.\n"
        "- Preserve the best-performing heuristic aspects."
    )
