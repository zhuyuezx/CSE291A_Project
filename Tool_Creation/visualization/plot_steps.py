"""
Steps-to-solve comparison for Sokoban levels.

Plots average number of steps taken to solve each level, baseline vs
LLM-optimized heuristic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from . import data


def _resolve_steps_data(
    levels: List[str],
    game_name: str,
    summary_path: Optional[str],
    records_dir: Optional[str],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    summary = data.load_optimization_summary(summary_path) if summary_path else None
    from_summary = data.level_comparison_from_summary(summary, levels)
    if from_summary:
        return (
            from_summary["baseline_steps"],
            from_summary["optimized_steps"],
            from_summary["baseline_solve"],
            from_summary["optimized_solve"],
        )

    if records_dir:
        agg = data.aggregate_traces(records_dir, game_filter=game_name)
        per = agg.get("per_level", {})
        steps = [float((per.get(lv, {}) or {}).get("avg_steps", 0.0)) for lv in levels]
        solve = [float((per.get(lv, {}) or {}).get("solve_rate", 0.0)) for lv in levels]
        return steps, steps, solve, solve

    return (
        data.SOKOBAN_BASELINE_STEPS[: len(levels)],
        data.SOKOBAN_OPTIMIZED_STEPS[: len(levels)],
        data.SOKOBAN_BASELINE_SOLVE[: len(levels)],
        data.SOKOBAN_OPTIMIZED_SOLVE[: len(levels)],
    )


def plot_steps_comparison(
    save_path: Optional[str] = None,
    show: bool = True,
    levels: Optional[List[str]] = None,
    game_name: str = "Sokoban",
    summary_path: Optional[str] = None,
    records_dir: Optional[str] = None,
) -> None:
    level_names = levels or data.SOKOBAN_LEVELS
    baseline, optimized, baseline_solve, optimized_solve = _resolve_steps_data(
        level_names, game_name, summary_path, records_dir
    )

    # Only keep levels that are solved by at least one configuration.
    filtered_indices = [
        i
        for i, (b, o) in enumerate(zip(baseline_solve, optimized_solve))
        if b > 0.0 or o > 0.0
    ]
    level_names = [level_names[i] for i in filtered_indices]
    baseline = [baseline[i] for i in filtered_indices]
    optimized = [optimized[i] for i in filtered_indices]

    x = range(len(level_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    bars_base = ax.bar(
        [i - width / 2 for i in x],
        baseline,
        width,
        label="Baseline",
        color="#1f77b4",
    )
    bars_opt = ax.bar(
        [i + width / 2 for i in x],
        optimized,
        width,
        label="Optimized",
        color="#ff7f0e",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(level_names)
    ax.set_ylabel("Average steps to solve")
    ax.set_xlabel("Level")
    ax.set_title(f"{game_name} steps to solve: baseline vs LLM-optimized")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    # Annotate relative speed-up.
    for bar_base, bar_opt in zip(bars_base, bars_opt):
        b_height = bar_base.get_height()
        o_height = bar_opt.get_height()
        if b_height > 0 and o_height < b_height:
            speedup = b_height / o_height
            ax.text(
                bar_opt.get_x() + bar_opt.get_width() / 2,
                o_height + 3,
                f"{speedup:.1f}× faster",
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


if __name__ == "__main__":
    plot_steps_comparison()

