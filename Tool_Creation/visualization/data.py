"""
Data and helpers for visualization.

This module supports both legacy hardcoded benchmark values and dynamic
loading from current pipeline outputs:
- mcts/records/*.json traces
- mcts/records/optimization_summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


# ── Legacy fallback data (kept for backward compatibility) ──────────────────

SOKOBAN_LEVELS: List[str] = [f"level{i}" for i in range(1, 11)]

SOKOBAN_BASELINE_SOLVE: List[float] = [
    1.0, 1.0, 0.667, 1.0, 0.667, 0.0, 0.0, 0.0, 0.0, 0.0
]
SOKOBAN_OPTIMIZED_SOLVE: List[float] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
]
SOKOBAN_BASELINE_STEPS: List[float] = [
    1.0, 3.0, 146.0, 8.0, 144.7, 200.0, 200.0, 200.0, 200.0, 200.0
]
SOKOBAN_OPTIMIZED_STEPS: List[float] = [
    1.0, 3.0, 34.0, 8.7, 103.0, 200.0, 200.0, 200.0, 57.0, 200.0
]

RUSH_HOUR_LEVELS: List[str] = [
    "easy1",
    "easy2",
    "easy3",
    "medium1",
    "medium2",
    "hard1",
    "hard2",
    "hard3",
]

SLIDING_PUZZLE_BASELINE_SOLVE = 0.40
SLIDING_PUZZLE_IMPROVED_SOLVE = 0.533
CONNECT_FOUR_BASELINE_WIN_RATE = 0.80
CONNECT_FOUR_IMPROVED_WIN_RATE = 0.9667


def compute_sokoban_hard_levels_summary() -> Dict[str, float]:
    indices = [2, 4, 8]  # level3, level5, level9
    base = [SOKOBAN_BASELINE_SOLVE[i] for i in indices]
    opt = [SOKOBAN_OPTIMIZED_SOLVE[i] for i in indices]
    return {
        "baseline_solve": sum(base) / len(base),
        "optimized_solve": sum(opt) / len(opt),
    }


# ── Iteration records ────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration: int
    level: str
    composite: float
    solve_rate: float
    avg_returns: float
    adopted: bool
    is_best: bool


ITERATION_DATA: List[IterationRecord] = [
    IterationRecord(1, "level3", 1.0, 1.0, 1.0, True, True),
    IterationRecord(2, "level1", 1.0, 1.0, 1.0, True, False),
    IterationRecord(3, "level5", 0.9, 1.0, 0.8, True, True),
    IterationRecord(4, "level4", 0.85, 1.0, 0.7, False, False),
    IterationRecord(5, "level9", 1.0, 1.0, 1.0, True, True),
]


# ── Trace JSON helpers ───────────────────────────────────────────────────────

def load_trace(trace_path: str | Path) -> Dict[str, Any]:
    path = Path(trace_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_traces(records_dir: str | Path) -> List[Path]:
    return sorted(Path(records_dir).glob("*.json"))


def load_optimization_summary(summary_path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(summary_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_game_and_level(game_label: str) -> Tuple[str, str]:
    text = (game_label or "").strip()
    if "(" in text and text.endswith(")"):
        game_part, level_part = text[:-1].split("(", 1)
        return game_part.strip(), level_part.strip()
    return text, text


def aggregate_traces(
    records_dir: str | Path,
    game_filter: Optional[str] = None,
) -> Dict[str, Any]:
    traces = list_traces(records_dir)
    per_level_raw: Dict[str, Dict[str, List[float]]] = {}
    detected_game = ""

    for trace_file in traces:
        try:
            trace = load_trace(trace_file)
        except Exception:
            continue

        metadata = trace.get("metadata", {}) or {}
        outcome = trace.get("outcome", {}) or {}
        game_label = str(metadata.get("game", ""))
        game_name, level_name = _parse_game_and_level(game_label)
        if game_filter and game_filter.lower() not in game_name.lower():
            continue
        if not detected_game and game_name:
            detected_game = game_name

        solved = 1.0 if bool(outcome.get("solved")) else 0.0
        steps = float(outcome.get("steps", 0.0))
        returns = outcome.get("returns", [0.0])
        ret0 = float(returns[0]) if isinstance(returns, list) and returns else float(returns or 0.0)

        bucket = per_level_raw.setdefault(
            level_name,
            {"solve": [], "steps": [], "returns": []},
        )
        bucket["solve"].append(solved)
        bucket["steps"].append(steps)
        bucket["returns"].append(ret0)

    per_level: Dict[str, Dict[str, float]] = {}
    for level, vals in per_level_raw.items():
        n = max(1, len(vals["solve"]))
        per_level[level] = {
            "runs": float(n),
            "solve_rate": sum(vals["solve"]) / n,
            "avg_steps": sum(vals["steps"]) / n,
            "avg_returns": sum(vals["returns"]) / n,
        }

    return {
        "game_name": detected_game,
        "levels": sorted(per_level.keys()),
        "per_level": per_level,
        "total_runs": int(sum(int(v["runs"]) for v in per_level.values())),
    }


def load_iteration_records(summary: Optional[Dict[str, Any]]) -> List[IterationRecord]:
    if not summary:
        return list(ITERATION_DATA)
    out: List[IterationRecord] = []
    for item in summary.get("all_results", []) or []:
        out.append(
            IterationRecord(
                iteration=int(item.get("iteration", 0)),
                level=str(item.get("level", "unknown")),
                composite=float(item.get("composite", 0.0)),
                solve_rate=float(item.get("solve_rate", 0.0)),
                avg_returns=float(item.get("avg_returns", 0.0)),
                adopted=bool(item.get("adopted", False)),
                is_best=bool(item.get("is_best", False)),
            )
        )
    return out if out else list(ITERATION_DATA)


def level_comparison_from_summary(
    summary: Optional[Dict[str, Any]],
    levels: List[str],
) -> Optional[Dict[str, List[float]]]:
    if not summary:
        return None

    baselines = summary.get("level_baselines", {}) or {}
    all_results = summary.get("all_results", []) or []
    by_level_best: Dict[str, Dict[str, float]] = {}

    for record in all_results:
        level = str(record.get("level", ""))
        if not level:
            continue
        comp = float(record.get("composite", 0.0))
        prev = by_level_best.get(level)
        if prev is None or comp > prev.get("composite", -1e9):
            by_level_best[level] = {
                "composite": comp,
                "solve_rate": float(record.get("solve_rate", 0.0)),
                "avg_steps": float(record.get("avg_steps", 0.0)),
            }

    baseline_solve: List[float] = []
    baseline_steps: List[float] = []
    optimized_solve: List[float] = []
    optimized_steps: List[float] = []

    for level in levels:
        base = baselines.get(level, {}) or {}
        best = by_level_best.get(level, {}) or {}
        b_solve = float(base.get("solve_rate", 0.0))
        b_steps = float(base.get("avg_steps", 0.0))
        o_solve = float(best.get("solve_rate", b_solve))
        o_steps = float(best.get("avg_steps", b_steps))
        baseline_solve.append(b_solve)
        baseline_steps.append(b_steps)
        optimized_solve.append(o_solve)
        optimized_steps.append(o_steps)

    return {
        "baseline_solve": baseline_solve,
        "baseline_steps": baseline_steps,
        "optimized_solve": optimized_solve,
        "optimized_steps": optimized_steps,
    }


def extract_children_stats_from_move(
    trace: Dict[str, Any],
    move_index: int,
) -> Optional[Dict[str, Dict[str, float]]]:
    moves = trace.get("moves", [])
    if not (0 <= move_index < len(moves)):
        return None
    move = moves[move_index]
    stats = move.get("children_stats") or {}
    normalized: Dict[str, Dict[str, float]] = {}
    for action, info in stats.items():
        normalized[action] = {
            "visits": float(info.get("visits", 0)),
            "value": float(info.get("value", 0.0)),
            "avg_value": float(info.get("avg_value", 0.0)),
        }
    return normalized

