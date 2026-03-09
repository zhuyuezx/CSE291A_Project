"""
Optimization progress over iterations.

Line chart of composite score per iteration, with markers indicating
whether the candidate heuristic was adopted and/or became the best for
that level so far.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt

from . import data


def plot_optimization_progress(
    save_path: Optional[str] = None,
    show: bool = True,
    summary_path: Optional[str] = None,
) -> None:
    """
    Plot composite score over iterations using optimization summary data.
    """
    summary = data.load_optimization_summary(summary_path) if summary_path else None
    records = data.load_iteration_records(summary)
    if not records:
        return

    iterations = [r.iteration for r in records]
    composites = [r.composite for r in records]

    # Assign a distinct color per level.
    level_colors: Dict[str, str] = {}
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for rec in records:
        if rec.level not in level_colors:
            level_colors[rec.level] = palette[len(level_colors) % len(palette)]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot running best composite as a dashed line.
    running_best = []
    best_so_far = 0.0
    for c in composites:
        if c > best_so_far:
            best_so_far = c
        running_best.append(best_so_far)
    ax.plot(
        iterations,
        running_best,
        linestyle="--",
        color="#999999",
        label="Running best (any level)",
    )

    # Scatter individual iterations with markers by status.
    for rec in records:
        color = level_colors[rec.level]
        if rec.is_best:
            marker = "*"
            size = 120
            edge_color = "black"
        elif rec.adopted:
            marker = "o"
            size = 60
            edge_color = "black"
        else:
            marker = "x"
            size = 60
            edge_color = None
        scatter_kwargs = {
            "x": rec.iteration,
            "y": rec.composite,
            "color": color,
            "marker": marker,
            "s": size,
            "zorder": 3,
        }
        if edge_color is not None:
            scatter_kwargs["edgecolors"] = edge_color
            scatter_kwargs["linewidths"] = 0.5
        ax.scatter(
            **scatter_kwargs,
        )
        # Light text label with level name.
        ax.text(
            rec.iteration,
            rec.composite + 0.02,
            rec.level,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Composite score")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Optimization progress over iterations")
    ax.grid(axis="both", linestyle="--", alpha=0.3)

    # Build a legend manually for marker types.
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="*", color="w", label="New best for level", markerfacecolor="#333333", markersize=10, markeredgecolor="black"),
        Line2D([0], [0], marker="o", color="w", label="Accepted (not best)", markerfacecolor="#333333", markersize=8, markeredgecolor="black"),
        Line2D([0], [0], marker="x", color="w", label="Rejected", markeredgecolor="#333333", markersize=8),
        Line2D([0], [0], linestyle="--", color="#999999", label="Running best (any level)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_optimization_progress()

