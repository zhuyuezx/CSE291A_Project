"""
Principal Variation (best-path) visualization for Sokoban MCTS.

Shows the current best path from the root by repeatedly taking the most-visited
child:

root -> a1 -> a2 -> ... -> ak -> ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .config import get_game_config
from .plot_visit_distribution import _action_label

MAX_STEPS = 200
ITERATIONS = 200
MAX_ROLLOUT_DEPTH = 200


def _extract_principal_variation(root, game_name: str, max_depth: int = 10) -> List[Dict]:
    """
    Follow the most-visited child repeatedly from root.
    Returns per-step action, visits, and avg value.
    """
    pv: List[Dict] = []
    node = root
    for _ in range(max_depth):
        children = getattr(node, "children", {})
        if not children:
            break
        # Principal variation: greedy by visits from current node.
        best_child = max(children.values(), key=lambda c: getattr(c, "visits", 0))
        action = getattr(best_child, "parent_action", None)
        action_str = str(action) if action is not None else "?"
        visits = getattr(best_child, "visits", 0) or 0
        value = getattr(best_child, "value", 0.0) or 0.0
        avg = value / visits if visits else 0.0
        pv.append(
            {
                "action": _action_label(action_str, game_name),
                "visits": visits,
                "avg_value": avg,
            }
        )
        node = best_child
    return pv


def _draw_pv_nodes(
    ax,
    pv: List[Dict],
    title: str,
    root_color: str = "#1f77b4",
    path_color: str = "#6f6f6f",
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    n = len(pv)
    denom = max(n + 2, 3)
    y_root = 1.0 - (1.0 / denom)
    x_center = 0.35

    root = Circle((x_center, y_root), 0.045, facecolor=root_color, edgecolor="black", linewidth=0.8, zorder=2)
    ax.add_patch(root)
    ax.text(x_center + 0.07, y_root, "root", fontsize=9, ha="left", va="center")

    prev_x, prev_y = x_center, y_root
    for i, step in enumerate(pv):
        y = 1.0 - ((i + 2.0) / denom)
        # Slight zig-zag so edges/labels are easier to read.
        x = x_center + (0.06 if i % 2 == 0 else -0.06)
        ax.plot([prev_x, x], [prev_y, y], color="#444444", lw=1.0, zorder=1)

        node = Circle((x, y), 0.040, facecolor=path_color, edgecolor="black", linewidth=0.8, zorder=2)
        ax.add_patch(node)
        label = f"{step['action']}  (v={step['visits']}, q={step['avg_value']:.2f})"
        ax.text(x + 0.07, y, label, fontsize=8, ha="left", va="center")
        prev_x, prev_y = x, y

    if n > 0:
        y_ellipsis = max(0.03, prev_y - (1.0 / denom))
        ax.plot([prev_x, prev_x], [prev_y, y_ellipsis], color="#666666", lw=0.8, zorder=1)
        ax.text(prev_x - 0.015, y_ellipsis, "...", fontsize=12, color="#666666", ha="center", va="center")


def plot_principal_variation(
    pv_baseline: List[Dict],
    pv_optimized: List[Dict],
    game_name: str,
    level_name: str,
    max_depth: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    _draw_pv_nodes(
        ax_left,
        pv_baseline,
        "Baseline PV (best path by visits)",
        root_color="#1f77b4",
        path_color="#5f7fa3",
    )
    _draw_pv_nodes(
        ax_right,
        pv_optimized,
        "Optimized PV (best path by visits)",
        root_color="#2ca02c",
        path_color="#6fa36f",
    )

    fig.suptitle(
        f"Principal Variation — {game_name} ({level_name}) | top {max_depth}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_principal_variation_demo(
    level_name: Optional[str] = None,
    max_depth: int = 10,
    output_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    import sys

    root_path = Path(__file__).resolve().parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from mcts import MCTSEngine
    cfg = get_game_config()
    game_class = cfg["game_class"]
    constructor_kwargs = cfg["constructor_kwargs"]
    levels = cfg["levels"]
    hp = cfg.get("hyperparams", {})
    iterations = int(hp.get("iterations", ITERATIONS))
    max_rollout_depth = int(hp.get("max_rollout_depth", MAX_ROLLOUT_DEPTH))
    tool_path = cfg.get("optimized_tool_path")

    if level_name is None:
        level_name = levels[0] if levels else "level1"

    try:
        game = game_class(level_name, **constructor_kwargs)
    except TypeError:
        game = game_class(**constructor_kwargs)
    state = game.new_initial_state()

    engine_base = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
    engine_opt = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
    if tool_path is not None:
        engine_opt.load_tool("simulation", tool_path)

    root_base, _ = engine_base._search_internal(state)
    root_opt, _ = engine_opt._search_internal(state.clone())

    pv_b = _extract_principal_variation(root_base, game.name(), max_depth=max_depth)
    pv_o = _extract_principal_variation(root_opt, game.name(), max_depth=max_depth)

    out_root = Path(output_dir) if output_dir else root_path / "visualization" / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / "principal_variation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"principal_variation_{level_name}.png"

    plot_principal_variation(
        pv_baseline=pv_b,
        pv_optimized=pv_o,
        game_name=game.name(),
        level_name=level_name,
        max_depth=max_depth,
        save_path=str(path),
        show=show,
    )
    print(f"Saved: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot MCTS principal variation (best path).")
    parser.add_argument("--level", default=None, help="Level name (uses active game config).")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum PV depth to show.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    args = parser.parse_args()

    run_principal_variation_demo(
        level_name=args.level,
        max_depth=args.max_depth,
        show=args.show,
    )
