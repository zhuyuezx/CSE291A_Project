"""
Tool Clustering — group MCTS heuristic tools by semantic similarity.

Given a list of tool metadata (name, description, optional source snippet)
for a single MCTS phase, this module asks an LLM to cluster tools that
implement the "same fundamental action" so they can later be merged.

Adapted from Yunjue Agent's ``tool_cluster.md`` / ``cluster_tools()`` but
specialised for MCTS-phase heuristics:
  - Inputs are per-phase (all share the same function signature).
  - Clustering criteria use heuristic intent rather than API-action verbs.
  - Falls back to a naive name-based clustering when the LLM is unavailable.

Usage::

    from LLM.tool_cluster import cluster_tools

    tools = registry.get_all_phase_tools("simulation")
    clusters = cluster_tools(tools, phase="simulation", querier=querier)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_cluster_prompt() -> str:
    path = _PROMPTS_DIR / "tool_cluster_mcts.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def cluster_tools(
    tool_meta_list: list[dict[str, Any]],
    phase: str = "simulation",
    querier: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Cluster a list of tool metadata dicts for *phase*.

    Parameters
    ----------
    tool_meta_list : list of dict
        Each dict should have at least ``name`` and ``description``.
        Optional: ``source_snippet``, ``metrics``.
    phase : str
        MCTS phase (for context in the prompt).
    querier : LLMQuerier | None
        If provided, use the LLM to cluster; otherwise fall back to
        naive name-based clustering.

    Returns
    -------
    list of dict, each with keys:
        suggested_master_tool_name : str
        tool_names : list[str]
    """
    if len(tool_meta_list) <= 1:
        if tool_meta_list:
            return [{
                "suggested_master_tool_name": tool_meta_list[0]["name"],
                "tool_names": [tool_meta_list[0]["name"]],
            }]
        return []

    if querier is not None:
        result = _llm_cluster(tool_meta_list, phase, querier)
        if result is not None:
            return result

    return _naive_cluster(tool_meta_list)


def _llm_cluster(
    tool_meta_list: list[dict[str, Any]],
    phase: str,
    querier: Any,
) -> list[dict[str, Any]] | None:
    """Use an LLM to cluster tools.  Returns None on failure."""
    template = _load_cluster_prompt()
    if not template:
        template = _default_cluster_prompt()

    tool_descriptions = "\n".join(
        f"- Name: '{t['name']}', Description: '{t.get('description', '')}'"
        for t in tool_meta_list
    )
    prompt = template.replace("{{PHASE}}", phase).replace(
        "{{TOOL_LIST}}", tool_descriptions
    )

    try:
        result = querier.query(prompt, step_name="tool_cluster")
        response_text = result.get("response", "")
        return _parse_cluster_response(response_text, tool_meta_list)
    except Exception:
        return None


def _parse_cluster_response(
    text: str,
    tool_meta_list: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Extract cluster JSON from LLM response text."""
    json_match = re.search(r"\{[\s\S]*\"consolidated_tool_clusters\"[\s\S]*\}", text)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group(0))
        clusters = data.get("consolidated_tool_clusters", [])
        if not clusters:
            return None
        # Validate that every input tool appears in exactly one cluster
        all_names = {t["name"] for t in tool_meta_list}
        clustered = set()
        for c in clusters:
            for n in c.get("tool_names", []):
                clustered.add(n)
        if not clustered.issubset(all_names):
            return None
        return [
            {
                "suggested_master_tool_name": c.get("suggested_master_tool_name", c["tool_names"][0]),
                "tool_names": c["tool_names"],
            }
            for c in clusters
        ]
    except (json.JSONDecodeError, KeyError):
        return None


def _naive_cluster(tool_meta_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group tools by normalised name stem (strip trailing _NN suffixes).
    """
    clusters: dict[str, list[str]] = {}
    for t in tool_meta_list:
        name = t["name"]
        base = re.sub(r"(_\d+)+$", "", name)
        clusters.setdefault(base, []).append(name)
    return [
        {"suggested_master_tool_name": base, "tool_names": names}
        for base, names in clusters.items()
    ]


def _default_cluster_prompt() -> str:
    return (
        "You are an expert at analysing MCTS heuristic functions.\n\n"
        "MCTS phase: {{PHASE}}\n\n"
        "Below is a list of heuristic tool functions for this phase.\n"
        "Cluster tools that implement the **same fundamental strategy**\n"
        "(e.g. two variants of Manhattan-distance simulation).\n"
        "Tools with different strategies stay in separate clusters.\n\n"
        "{{TOOL_LIST}}\n\n"
        "Output a single JSON object:\n"
        "```json\n"
        '{\n  "consolidated_tool_clusters": [\n'
        '    {"suggested_master_tool_name": "...", "tool_names": [...]}\n'
        "  ]\n}\n```\n"
        "Every input tool must appear in exactly one cluster."
    )
