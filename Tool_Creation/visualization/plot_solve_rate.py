"""
Grouped bar charts for baseline vs optimized solve rates.

Primary figure: Sokoban levels 1–10 baseline vs optimized solve rate.
Secondary helper: cross-game summary (Sliding Puzzle, Sokoban, Connect Four).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from . import data


def _resolve_solve_data(
    levels: List[str],
    game_name: str,
    summary_path: Optional[str],
    records_dir: Optional[str],
) -> Tuple[List[float], List[float]]:
    summary = data.load_optimization_summary(summary_path) if summary_path else None
    from_summary = data.level_comparison_from_summary(summary, levels)
    if from_summary:
        return from_summary["baseline_solve"], from_summary["optimized_solve"]
    if records_dir:
        agg = data.aggregate_traces(records_dir, game_filter=game_name)
        per = agg.get("per_level", {})
        solve = [float((per.get(lv, {}) or {}).get("solve_rate", 0.0)) for lv in levels]
        return solve, solve
    if "rush" in game_name.lower():
        return [0.0 for _ in levels], [0.0 for _ in levels]
    return data.SOKOBAN_BASELINE_SOLVE[: len(levels)], data.SOKOBAN_OPTIMIZED_SOLVE[: len(levels)]


def plot_solve_rate(
    save_path: Optional[str] = None,
    show: bool = True,
    levels: Optional[List[str]] = None,
    game_name: str = "Sokoban",
    summary_path: Optional[str] = None,
    records_dir: Optional[str] = None,
) -> None:
    level_names = levels or data.SOKOBAN_LEVELS
    baseline, optimized = _resolve_solve_data(level_names, game_name, summary_path, records_dir)

    x = range(len(level_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_base = ax.bar(
        [i - width / 2 for i in x],
        [s * 100 for s in baseline],
        width,
        label="Baseline",
        color="#1f77b4",
    )
    bars_opt = ax.bar(
        [i + width / 2 for i in x],
        [s * 100 for s in optimized],
        width,
        label="Optimized",
        color="#2ca02c",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(level_names)
    ax.set_ylabel("Solve rate (%)")
    ax.set_xlabel("Level")
    ax.set_ylim(0, 110)
    ax.set_title(f"{game_name} solve rate: baseline vs LLM-optimized")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate delta where there is an improvement.
    for i, (b, o, bar) in enumerate(zip(baseline, optimized, bars_opt)):
        delta = o - b
        if delta <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"+{delta * 100:.0f}pp",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cross_game_summary(
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a high-level cross-game comparison of baseline vs optimized performance
    for:
      - Sliding Puzzle
      - Sokoban (hard levels 3, 5, 9 averaged)
      - Connect Four
    """
    sok_summary = data.compute_sokoban_hard_levels_summary()

    games = ["SlidingPuzzle", "Sokoban(3,5,9)", "ConnectFour"]
    baseline = [
        data.SLIDING_PUZZLE_BASELINE_SOLVE * 100,
        sok_summary["baseline_solve"] * 100,
        data.CONNECT_FOUR_BASELINE_WIN_RATE * 100,
    ]
    optimized = [
        data.SLIDING_PUZZLE_IMPROVED_SOLVE * 100,
        sok_summary["optimized_solve"] * 100,
        data.CONNECT_FOUR_IMPROVED_WIN_RATE * 100,
    ]

    x = range(len(games))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        label="Baseline",
        color="#1f77b4",
    )
    ax.bar(
        [i + width / 2 for i in x],
        optimized,
        width,
        label="Optimized",
        color="#2ca02c",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(games, rotation=15)
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Cross-game improvement from LLM-optimized heuristics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # Simple manual smoke test when run directly.
    plot_solve_rate()

