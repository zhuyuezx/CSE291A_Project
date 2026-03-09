"""
MCTS tree snapshot visualization.

Draws the actual search tree (root + children + optional grandchildren)
for a given state. Side-by-side: baseline tree (left) vs optimized tree (right).

Supports:
- Single-level snapshot (plot_mcts_tree_snapshot / run_mcts_tree_demo).
- Interactive viewer: all 10 levels with Prev/Next (interactive_tree).
- Export all levels to PNG (export_tree_all_levels).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button

from .config import get_game_config
from .plot_visit_distribution import _action_label

MAX_DEPTH_DEFAULT = 3
ITERATIONS = 200
MAX_ROLLOUT_DEPTH = 200
MAX_STEPS = 200


class _EllipsisNode:
    """Placeholder node used to indicate the shown tree is truncated."""

    def __init__(self):
        self.visits = 0
        self.value = 0.0
        self.children = {}
        self.is_ellipsis = True


def _collect_tree(
    root,
    max_depth: int,
    game_name: str,
    best_action_at_root: Optional[str] = None,
) -> Tuple[
    List[Tuple[Any, int, Optional[str], Optional[int]]],
    List[Tuple[int, int, str]],
    List[int],
]:
    """
    BFS from root. Returns:
    - nodes: list of (node, depth, action_from_parent, parent_idx), root has parent_idx=None, action=None
    - edges: list of (parent_idx, child_idx, action_label)

    When best_action_at_root is set, at depth 2 we only include children of the
    root child that corresponds to that action (best branch only).
    """
    nodes: List[Tuple[Any, int, Optional[str], Optional[int]]] = []
    edges: List[Tuple[int, int, str]] = []
    chosen_indices: List[int] = []
    node_to_idx: Dict[int, int] = {}
    root_id = id(root)
    node_to_idx[root_id] = 0
    nodes.append((root, 0, None, None))

    def _best_child_action(node_obj) -> Optional[str]:
        children = getattr(node_obj, "children", {}) or {}
        if not children:
            return None
        best_child = max(children.values(), key=lambda c: getattr(c, "visits", 0))
        return str(getattr(best_child, "parent_action", None))

    # Queue stores (node_idx, depth, on_principal_path).
    queue: List[Tuple[int, int, bool]] = [(0, 0, True)]
    while queue:
        node_idx, depth, on_principal_path = queue.pop(0)
        if depth >= max_depth:
            continue
        node, _, _action_from_parent, _parent_idx = nodes[node_idx]
        # Only expand nodes that lie on the principal branch (except root).
        if depth > 0 and not on_principal_path:
            continue
        chosen_action = _best_child_action(node)
        if depth == 0 and best_action_at_root is not None:
            chosen_action = best_action_at_root
        for action, child in node.children.items():
            child_id = id(child)
            if child_id not in node_to_idx:
                child_idx = len(nodes)
                node_to_idx[child_id] = child_idx
                action_str = str(action)
                nodes.append((child, depth + 1, action_str, node_idx))
                edges.append((node_idx, child_idx, _action_label(action_str, game_name)))
                child_on_principal = on_principal_path and (chosen_action is not None) and (action_str == chosen_action)
                if child_on_principal:
                    chosen_indices.append(child_idx)
                queue.append((child_idx, depth + 1, child_on_principal))
    # Add a continuation marker layer below the displayed frontier.
    # This makes it explicit the search tree does not terminate here.
    frontier = list(nodes)
    for idx, (node, depth, _action_from_parent, _parent_idx) in enumerate(frontier):
        if depth != max_depth:
            continue
        child_count = len(getattr(node, "children", {}) or {})
        if child_count <= 0:
            continue
        ellipsis_idx = len(nodes)
        nodes.append((_EllipsisNode(), depth + 1, "...", idx))
        edges.append((idx, ellipsis_idx, "..."))

    return nodes, edges, chosen_indices


def _value_to_lightness(value: float) -> float:
    """Map avg value (typically in [-1,1] or [0,1]) to 0=light, 1=dark for coloring."""
    # Clip to [-1, 1] then map to [0, 1]: -1 -> 0 (light), +1 -> 1 (dark)
    v = max(-1.0, min(1.0, value))
    return (v + 1.0) / 2.0


def _color_from_base_and_value(base_hex: str, value: float) -> str:
    """Darker = higher value. Interpolate from light (0.85) to base; t=0 -> light, t=1 -> base (dark)."""
    t = _value_to_lightness(value)
    r, g, b = to_rgb(base_hex)
    light = 0.85
    r = (1.0 - t) * light + t * r
    g = (1.0 - t) * light + t * g
    b = (1.0 - t) * light + t * b
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _layout_tree(
    nodes: List[Tuple[Any, int, Optional[str], Optional[int]]],
    edges: List[Tuple[int, int, str]],
) -> Dict[int, Tuple[float, float]]:
    """Assign (x, y) to each node index. Root at top center; each level spread horizontally."""
    pos: Dict[int, Tuple[float, float]] = {}
    by_depth: Dict[int, List[int]] = {}
    for i, (_, depth, _, _) in enumerate(nodes):
        by_depth.setdefault(depth, []).append(i)
    max_depth = max(by_depth.keys()) if by_depth else 0
    for depth, indices in sorted(by_depth.items()):
        n = len(indices)
        y = 1.0 - depth * 0.28
        for k, idx in enumerate(indices):
            x = (k + 0.5) / n if n > 0 else 0.5
            pos[idx] = (x, y)
    return pos


def _draw_tree_on_ax(
    ax,
    nodes: List[Tuple[Any, int, Optional[str], Optional[int]]],
    edges: List[Tuple[int, int, str]],
    pos: Dict[int, Tuple[float, float]],
    title: str,
    node_color: str = "#1f77b4",
    highlight_action: Optional[str] = None,
    chosen_indices: Optional[List[int]] = None,
) -> None:
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    # Edges
    for pi, ci, action_label in edges:
        if pi not in pos or ci not in pos:
            continue
        x0, y0 = pos[pi]
        x1, y1 = pos[ci]
        ax.plot([x0, x1], [y0, y1], color="#333333", lw=1, zorder=0)
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mid_x, mid_y, action_label, fontsize=7, ha="center", va="center", color="#555")

    # Node size by visits, color by value (darker = higher). Chosen = red.
    max_visits = max((getattr(n, "visits", 0) or 0 for n, *_ in nodes), default=1)
    radius_scale = 0.03 / max(math.sqrt(max_visits), 1e-6)
    radius_min, radius_max = 0.012, 0.045
    chosen_set = set(chosen_indices or [])
    for i, (node, depth, action_from_parent, _) in enumerate(nodes):
        if i not in pos:
            continue
        x, y = pos[i]
        is_ellipsis = bool(getattr(node, "is_ellipsis", False))
        if is_ellipsis:
            circle = plt.Circle((x, y), 0.012, color="#bdbdbd", ec="#666666", lw=0.6, zorder=2)
            ax.add_patch(circle)
            ax.text(x + 0.02, y, "...", fontsize=8, ha="left", va="center", zorder=3, color="#555555")
            continue
        visits = getattr(node, "visits", 0) or 0
        value = getattr(node, "value", 0.0)
        avg_val = value / visits if visits else 0.0
        radius = max(radius_min, min(radius_max, 0.012 + radius_scale * math.sqrt(visits)))
        is_chosen = i in chosen_set or (
            depth == 1 and highlight_action is not None and action_from_parent == highlight_action
        )
        if is_chosen:
            color = "#d62728"
        else:
            color = _color_from_base_and_value(node_color, avg_val)
        circle = plt.Circle((x, y), radius, color=color, ec="black", lw=0.8, zorder=2)
        ax.add_patch(circle)
        label = f"v:{visits}\n{avg_val:.2f}" if depth > 0 else f"root\nv:{visits}"
        # Place label to the right of the node
        ax.text(x + radius + 0.02, y, label, fontsize=6, ha="left", va="center", zorder=3)


def plot_mcts_tree_snapshot(
    game,
    state,
    engine_baseline,
    engine_optimized,
    game_name: str,
    max_depth: int = MAX_DEPTH_DEFAULT,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Same state: run baseline and optimized MCTS, then draw left (baseline)
    and right (optimized) tree snapshots.

    max_depth: 1 = root + children; 2 = + grandchildren; 3 = one more choice layer.
    """
    root_base, best_base = engine_baseline._search_internal(state)
    root_opt, best_opt = engine_optimized._search_internal(state.clone())

    best_action_base = str(best_base) if best_base is not None else None
    best_action_opt = str(best_opt) if best_opt is not None else None
    nodes_b, edges_b, chosen_b = _collect_tree(root_base, max_depth, game_name, best_action_at_root=best_action_base)
    nodes_o, edges_o, chosen_o = _collect_tree(root_opt, max_depth, game_name, best_action_at_root=best_action_opt)

    pos_b = _layout_tree(nodes_b, edges_b)
    pos_o = _layout_tree(nodes_o, edges_o)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)

    _draw_tree_on_ax(
        ax_left,
        nodes_b,
        edges_b,
        pos_b,
        "Baseline MCTS",
        node_color="#1f77b4",
        highlight_action=str(best_base),
        chosen_indices=chosen_b,
    )
    _draw_tree_on_ax(
        ax_right,
        nodes_o,
        edges_o,
        pos_o,
        "Optimized MCTS",
        node_color="#2ca02c",
        highlight_action=str(best_opt),
        chosen_indices=chosen_o,
    )
    fig.suptitle(f"MCTS tree snapshot — {game_name} (same state)", fontsize=12, fontweight="bold")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _precompute_tree_all_levels(root_path: Path, max_depth: int = MAX_DEPTH_DEFAULT) -> List[Dict]:
    """Run MCTS for all configured levels with both engines."""
    import sys
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    from mcts import MCTSEngine
    cfg = get_game_config()
    levels = cfg["levels"]
    game_class = cfg["game_class"]
    constructor_kwargs = cfg["constructor_kwargs"]
    hp = cfg.get("hyperparams", {})
    iterations = int(hp.get("iterations", ITERATIONS))
    max_rollout_depth = int(hp.get("max_rollout_depth", MAX_ROLLOUT_DEPTH))
    tool_path = cfg.get("optimized_tool_path")

    def _make_game(level_name: str):
        try:
            return game_class(level_name, **constructor_kwargs)
        except TypeError:
            return game_class(**constructor_kwargs)

    results: List[Dict] = []

    for level_name in levels:
        game = _make_game(level_name)
        state = game.new_initial_state()
        engine_base = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
        engine_opt = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
        if tool_path is not None:
            engine_opt.load_tool("simulation", tool_path)

        root_base, best_base = engine_base._search_internal(state)
        root_opt, best_opt = engine_opt._search_internal(state.clone())

        best_act_b = str(best_base) if best_base is not None else None
        best_act_o = str(best_opt) if best_opt is not None else None
        nodes_b, edges_b, chosen_b = _collect_tree(root_base, max_depth, game.name(), best_action_at_root=best_act_b)
        nodes_o, edges_o, chosen_o = _collect_tree(root_opt, max_depth, game.name(), best_action_at_root=best_act_o)
        pos_b = _layout_tree(nodes_b, edges_b)
        pos_o = _layout_tree(nodes_o, edges_o)

        results.append({
            "level": level_name,
            "game_name": game.name(),
            "nodes_b": nodes_b,
            "edges_b": edges_b,
            "pos_b": pos_b,
            "best_base": str(best_base),
            "chosen_b": chosen_b,
            "nodes_o": nodes_o,
            "edges_o": edges_o,
            "pos_o": pos_o,
            "best_opt": str(best_opt),
            "chosen_o": chosen_o,
        })
    return results


def _draw_tree_level(
    fig,
    ax_left,
    ax_right,
    level_data: Dict,
) -> None:
    """Redraw left and right tree axes for one level."""
    ax_left.clear()
    ax_right.clear()
    _draw_tree_on_ax(
        ax_left,
        level_data["nodes_b"],
        level_data["edges_b"],
        level_data["pos_b"],
        "Baseline MCTS",
        node_color="#1f77b4",
        highlight_action=level_data["best_base"],
        chosen_indices=level_data.get("chosen_b"),
    )
    _draw_tree_on_ax(
        ax_right,
        level_data["nodes_o"],
        level_data["edges_o"],
        level_data["pos_o"],
        "Optimized MCTS",
        node_color="#2ca02c",
        highlight_action=level_data["best_opt"],
        chosen_indices=level_data.get("chosen_o"),
    )
    fig.suptitle(
        f"MCTS tree snapshot — {level_data['game_name']} (same state)",
        fontsize=12,
        fontweight="bold",
    )
    fig.canvas.draw_idle()


def interactive_tree(
    start_level: int = 0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    precomputed_data: Optional[List[Dict]] = None,
) -> None:
    """Launch interactive matplotlib window: Prev/Next across configured levels."""
    import sys
    root_path = Path(__file__).resolve().parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    if precomputed_data is None:
        print("Pre-computing MCTS trees for all configured levels (this may take a moment)...")
        all_data = _precompute_tree_all_levels(root_path, max_depth=max_depth)
        print("Done. Launching interactive tree viewer.")
    else:
        all_data = precomputed_data

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.12)

    state = {"idx": start_level}

    def _refresh():
        _draw_tree_level(fig, ax_left, ax_right, all_data[state["idx"]])

    def on_prev(_event):
        state["idx"] = (state["idx"] - 1) % len(all_data)
        _refresh()

    def on_next(_event):
        state["idx"] = (state["idx"] + 1) % len(all_data)
        _refresh()

    ax_prev = fig.add_axes([0.32, 0.02, 0.15, 0.05])
    ax_next = fig.add_axes([0.52, 0.02, 0.15, 0.05])
    btn_prev = Button(ax_prev, "< Prev level")
    btn_next = Button(ax_next, "Next level >")
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    # Keep references so buttons are not garbage-collected (callbacks would stop working)
    fig._btn_prev, fig._btn_next = btn_prev, btn_next

    _refresh()
    plt.show()


def export_tree_all_levels(
    output_dir: Optional[str] = None,
    max_depth: int = MAX_DEPTH_DEFAULT,
    precomputed_data: Optional[List[Dict]] = None,
) -> None:
    """Save one mcts_tree_<level>.png per configured level."""
    import sys
    root_path = Path(__file__).resolve().parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    out = Path(output_dir) if output_dir else root_path / "visualization" / "output"
    out.mkdir(parents=True, exist_ok=True)
    save_dir = out / "mcts_tree"
    save_dir.mkdir(parents=True, exist_ok=True)

    if precomputed_data is None:
        print("Pre-computing MCTS trees for all configured levels...")
        all_data = _precompute_tree_all_levels(root_path, max_depth=max_depth)
    else:
        all_data = precomputed_data

    for ld in all_data:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.05)
        _draw_tree_level(fig, ax_left, ax_right, ld)
        path = save_dir / f"mcts_tree_{ld['level']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.name}")
    print(f"Exported {len(all_data)} tree PNGs to {save_dir}")


def run_mcts_tree_demo(
    level_name: Optional[str] = None,
    max_depth: int = MAX_DEPTH_DEFAULT,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Single-level snapshot (used by run_all.py when show=False), or
    export all levels + interactive viewer (when show=True).
    """
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

    def _make_game(level: str):
        try:
            return game_class(level, **constructor_kwargs)
        except TypeError:
            return game_class(**constructor_kwargs)

    if show:
        # Interactive: precompute once, export all level PNGs, then Prev/Next viewer
        out_dir = root_path / "visualization" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        print("Pre-computing MCTS trees for all configured levels...")
        data = _precompute_tree_all_levels(root_path, max_depth=max_depth)
        export_tree_all_levels(str(out_dir), max_depth=max_depth, precomputed_data=data)
        interactive_tree(start_level=0, max_depth=max_depth, precomputed_data=data)
        return

    # Single-level export for run_all (no interactive window)
    game = _make_game(level_name)
    state = game.new_initial_state()
    engine_base = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
    engine_opt = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_rollout_depth)
    if tool_path is not None:
        engine_opt.load_tool("simulation", tool_path)

    if save_path is None:
        out_dir = root_path / "visualization" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / "mcts_tree_snapshot.png")

    plot_mcts_tree_snapshot(
        game=game,
        state=state,
        engine_baseline=engine_base,
        engine_optimized=engine_opt,
        game_name=game.name(),
        max_depth=max_depth,
        save_path=save_path,
        show=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCTS tree snapshot (interactive: all 10 levels with Prev/Next).")
    parser.add_argument("--export", action="store_true", help="Export PNGs for all levels only (no window).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for --export.")
    args = parser.parse_args()
    if args.export:
        export_tree_all_levels(args.output_dir)
    else:
        interactive_tree()
