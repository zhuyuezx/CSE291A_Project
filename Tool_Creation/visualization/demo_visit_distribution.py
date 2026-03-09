"""
Interactive demo: MCTS root search behavior across all Sokoban levels.

Modes:
- Interactive (default): matplotlib window with Prev/Next buttons to
  browse levels 1-10.  Each page shows baseline vs optimized visit
  distribution (top row) and average-value distribution (bottom row).
- Static export: save one PNG per level to visualization/output/.
  Used by run_all.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from .config import get_game_config
from .plot_visit_distribution import (
    _action_label,
    _sort_action_keys,
    extract_root_action_stats,
)

ITERATIONS = 200
MAX_ROLLOUT_DEPTH = 200
MAX_STEPS = 200


def _ensure_imports():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _instantiate_game(game_class, level_name: str, constructor_kwargs: Dict):
    try:
        return game_class(level_name, **constructor_kwargs)
    except TypeError:
        return game_class(**constructor_kwargs)


def _precompute_all_levels(root: Path):
    """Run MCTS search for every configured level with both engines."""
    from mcts import MCTSEngine

    cfg = get_game_config()
    levels = cfg["levels"]
    game_class = cfg["game_class"]
    constructor_kwargs = cfg["constructor_kwargs"]
    hp = cfg.get("hyperparams", {})
    iterations = int(hp.get("iterations", ITERATIONS))
    max_rollout_depth = int(hp.get("max_rollout_depth", MAX_ROLLOUT_DEPTH))
    tool_path = cfg.get("optimized_tool_path")
    results: List[Dict] = []

    for level_name in levels:
        game = _instantiate_game(game_class, level_name, constructor_kwargs)
        state = game.new_initial_state()

        engine_base = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
        engine_opt = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
        if tool_path is not None:
            engine_opt.load_tool("simulation", tool_path)

        root_base, best_base = engine_base._search_internal(state)
        root_opt, best_opt = engine_opt._search_internal(state.clone())

        stats_base = extract_root_action_stats(root_base)
        stats_opt = extract_root_action_stats(root_opt)

        board_str = str(state)

        results.append({
            "level": level_name,
            "game_name": game.name(),
            "board": board_str,
            "stats_base": stats_base,
            "stats_opt": stats_opt,
            "best_base": str(best_base),
            "best_opt": str(best_opt),
        })

    return results


def _draw_level(fig, axes, level_data: Dict) -> None:
    """Redraw all four subplots for a single level."""
    for ax in axes.flat:
        ax.clear()

    map_base = {str(s["action"]): s for s in level_data["stats_base"]}
    map_opt = {str(s["action"]): s for s in level_data["stats_opt"]}
    actions = _sort_action_keys(list(set(map_base.keys()) | set(map_opt.keys())))
    labels = [_action_label(a, level_data["game_name"]) for a in actions]
    x = range(len(actions))

    visits_base = [float(map_base.get(a, {"visits": 0})["visits"]) for a in actions]
    visits_opt = [float(map_opt.get(a, {"visits": 0})["visits"]) for a in actions]
    avg_base = [float(map_base.get(a, {"avg_value": 0})["avg_value"]) for a in actions]
    avg_opt = [float(map_opt.get(a, {"avg_value": 0})["avg_value"]) for a in actions]

    best_b = level_data["best_base"]
    best_o = level_data["best_opt"]

    def bar_colors(action_keys, best, default_color):
        return ["#d62728" if a == best else default_color for a in action_keys]

    # Row 0: visits
    axes[0, 0].bar(x, visits_base, color=bar_colors(actions, best_b, "#1f77b4"))
    axes[0, 0].set_title("Baseline — visits")
    axes[0, 0].set_ylabel("Visits")
    axes[0, 0].set_xticks(list(x)); axes[0, 0].set_xticklabels(labels)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[0, 1].bar(x, visits_opt, color=bar_colors(actions, best_o, "#2ca02c"))
    axes[0, 1].set_title("Optimized — visits")
    axes[0, 1].set_xticks(list(x)); axes[0, 1].set_xticklabels(labels)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)

    # Row 1: average value
    axes[1, 0].bar(x, avg_base, color=bar_colors(actions, best_b, "#1f77b4"))
    axes[1, 0].set_title("Baseline — avg value")
    axes[1, 0].set_ylabel("Avg value")
    axes[1, 0].set_xticks(list(x)); axes[1, 0].set_xticklabels(labels)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1, 1].bar(x, avg_opt, color=bar_colors(actions, best_o, "#2ca02c"))
    axes[1, 1].set_title("Optimized — avg value")
    axes[1, 1].set_xticks(list(x)); axes[1, 1].set_xticklabels(labels)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        f"MCTS root search — {level_data['game_name']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.canvas.draw_idle()


def interactive(start_level: int = 0) -> None:
    """Launch an interactive matplotlib window with Prev/Next buttons."""
    root = _ensure_imports()
    print("Pre-computing MCTS search for all configured levels (this may take a moment)...")
    all_data = _precompute_all_levels(root)
    print("Done. Launching interactive viewer.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(bottom=0.13, hspace=0.45, wspace=0.25)

    state = {"idx": start_level}

    def _refresh():
        _draw_level(fig, axes, all_data[state["idx"]])

    def on_prev(_event):
        state["idx"] = (state["idx"] - 1) % len(all_data)
        _refresh()

    def on_next(_event):
        state["idx"] = (state["idx"] + 1) % len(all_data)
        _refresh()

    ax_prev = fig.add_axes([0.30, 0.02, 0.15, 0.05])
    ax_next = fig.add_axes([0.55, 0.02, 0.15, 0.05])
    btn_prev = Button(ax_prev, "< Prev level")
    btn_next = Button(ax_next, "Next level >")
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    _refresh()
    plt.show()


def export_all_levels(output_dir: Optional[str] = None) -> None:
    """Save a static 2x2 PNG for every level to output_dir."""
    root = _ensure_imports()
    if output_dir is None:
        out = root / "visualization" / "output"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dir = out / "mcts_root"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Pre-computing MCTS search for all configured levels...")
    all_data = _precompute_all_levels(root)

    for ld in all_data:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.subplots_adjust(hspace=0.45, wspace=0.25)
        _draw_level(fig, axes, ld)
        path = save_dir / f"mcts_root_{ld['level']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.name}")

    print(f"Exported {len(all_data)} PNGs to {save_dir}")


def main(
    save_visits_path: str | None = None,
    save_avg_value_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Backward-compatible entry point used by run_all.py.

    When save paths are given it exports static PNGs for all levels.
    When show=True it also launches the interactive viewer.
    """
    root = _ensure_imports()
    out_dir = root / "visualization" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_all_levels(str(out_dir))

    if show:
        interactive()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive MCTS visit distribution viewer for all Sokoban levels.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export static PNGs for all levels (no interactive window).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for exported PNGs.",
    )
    args = parser.parse_args()

    if args.export:
        export_all_levels(args.output_dir)
    else:
        interactive()
