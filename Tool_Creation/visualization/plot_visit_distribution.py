"""
MCTS search-behavior visualizations.

This module focuses on root-node action statistics after search:
- visits
- total value
- average value

It supports:
1) Trace-based plots from `children_stats` in JSON logs
2) Live plots from root.children after `_search_internal(...)`
3) Baseline-vs-optimized side-by-side comparisons on the same state
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from . import data


ActionStats = Dict[str, float | str]


def _action_label(action_str: str, game_name: str) -> str:
    """
    Convert an action string into a human-friendly label.
    Supports Sokoban directional actions; falls back to the raw string.
    """
    if "Sokoban" in game_name:
        # Sokoban actions are 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        try:
            a = int(action_str)
        except ValueError:
            return action_str
        mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        return mapping.get(a, action_str)
    if "RushHour" in game_name or "rush_hour" in game_name.lower():
        try:
            parsed = ast.literal_eval(action_str)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                piece_idx, delta = int(parsed[0]), int(parsed[1])
                piece_label = chr(ord("A") + piece_idx)
                sign = "+" if delta >= 0 else ""
                return f"{piece_label}{sign}{delta}"
        except Exception:
            return action_str
    return action_str


def _sort_action_keys(keys: List[str]) -> List[str]:
    return sorted(
        keys,
        key=lambda k: float(k) if k.replace(".", "", 1).isdigit() else k,
    )


def extract_root_action_stats(root) -> List[ActionStats]:
    """
    Build per-action stats from a root MCTSNode.

    Returns list of dicts with:
    - action
    - visits
    - value
    - avg_value
    """
    stats: List[ActionStats] = []
    for action, child in root.children.items():
        visits = float(child.visits)
        value = float(child.value)
        avg_value = value / visits if visits > 0 else 0.0
        stats.append(
            {
                "action": str(action),
                "visits": visits,
                "value": value,
                "avg_value": avg_value,
            }
        )
    stats.sort(key=lambda d: float(d["action"]) if str(d["action"]).replace(".", "", 1).isdigit() else str(d["action"]))
    return stats


def _plot_single_metric_bar(
    stats: List[ActionStats],
    metric: str,
    game_name: str,
    chosen_action: Optional[str],
    title: str,
    ylabel: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    actions = [str(s["action"]) for s in stats]
    values = [float(s[metric]) for s in stats]
    labels = [_action_label(a, game_name) for a in actions]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = []
    for action in actions:
        if chosen_action is not None and action == chosen_action:
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")

    ax.bar(range(len(actions)), values, color=colors)
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Action")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_visits_from_trace(
    trace_path: str | Path,
    move_index: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot visit distribution for a single move from a trace JSON file.
    """
    trace = data.load_trace(trace_path)
    stats = data.extract_children_stats_from_move(trace, move_index)
    if not stats:
        return

    game_name = str(trace.get("metadata", {}).get("game", "Unknown"))
    chosen = None
    moves = trace.get("moves", [])
    if 0 <= move_index < len(moves):
        chosen = str(moves[move_index].get("action_chosen"))
    actions = _sort_action_keys(list(stats.keys()))
    root_stats: List[ActionStats] = []
    for action in actions:
        info = stats[action]
        root_stats.append(
            {
                "action": action,
                "visits": float(info["visits"]),
                "value": float(info["value"]),
                "avg_value": float(info["avg_value"]),
            }
        )

    plot_root_visit_distribution(
        stats=root_stats,
        game_name=game_name,
        chosen_action=chosen,
        title=f"Visit distribution (move {move_index + 1}) — {game_name}",
        save_path=save_path,
        show=show,
    )


def plot_avg_value_from_trace(
    trace_path: str | Path,
    move_index: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot average value per action for a move from a trace JSON file.
    """
    trace = data.load_trace(trace_path)
    stats = data.extract_children_stats_from_move(trace, move_index)
    if not stats:
        return

    game_name = str(trace.get("metadata", {}).get("game", "Unknown"))
    chosen = None
    moves = trace.get("moves", [])
    if 0 <= move_index < len(moves):
        chosen = str(moves[move_index].get("action_chosen"))

    actions = _sort_action_keys(list(stats.keys()))
    root_stats: List[ActionStats] = []
    for action in actions:
        info = stats[action]
        root_stats.append(
            {
                "action": action,
                "visits": float(info["visits"]),
                "value": float(info["value"]),
                "avg_value": float(info["avg_value"]),
            }
        )

    plot_root_avg_value_distribution(
        stats=root_stats,
        game_name=game_name,
        chosen_action=chosen,
        title=f"Average value per action (move {move_index + 1}) — {game_name}",
        save_path=save_path,
        show=show,
    )


def plot_root_visit_distribution(
    stats: List[ActionStats],
    game_name: str,
    chosen_action: Optional[str] = None,
    title: str = "Root visit distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot visit counts from root-action stats (live or trace-derived).
    """
    _plot_single_metric_bar(
        stats=stats,
        metric="visits",
        game_name=game_name,
        chosen_action=chosen_action,
        title=title,
        ylabel="Visits",
        save_path=save_path,
        show=show,
    )


def plot_root_avg_value_distribution(
    stats: List[ActionStats],
    game_name: str,
    chosen_action: Optional[str] = None,
    title: str = "Root average value distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot average value per action from root-action stats.
    """
    _plot_single_metric_bar(
        stats=stats,
        metric="avg_value",
        game_name=game_name,
        chosen_action=chosen_action,
        title=title,
        ylabel="Average value",
        save_path=save_path,
        show=show,
    )


def plot_visits_live(
    game,
    state,
    engine_baseline,
    engine_optimized,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Run MCTS search with baseline and optimized engines on the same
    state, and plot their root visit distributions side by side.

    Args:
        game:           Game instance (used only for name).
        state:          GameState to search from.
        engine_baseline: MCTSEngine with default tools.
        engine_optimized: MCTSEngine with optimized tools loaded.
    """
    # Baseline
    root_base, best_action_base = engine_baseline._search_internal(state)
    stats_base = extract_root_action_stats(root_base)

    # Optimized (work on a cloned state to keep them independent)
    root_opt, best_action_opt = engine_optimized._search_internal(state.clone())
    stats_opt = extract_root_action_stats(root_opt)

    map_base = {str(s["action"]): s for s in stats_base}
    map_opt = {str(s["action"]): s for s in stats_opt}
    actions = _sort_action_keys(list(set(map_base.keys()) | set(map_opt.keys())))
    labels = [_action_label(a, game.name()) for a in actions]

    visits_base = [float(map_base.get(a, {"visits": 0.0})["visits"]) for a in actions]
    visits_opt = [float(map_opt.get(a, {"visits": 0.0})["visits"]) for a in actions]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Baseline subplot
    colors_base = []
    for a in actions:
        if str(best_action_base) == a:
            colors_base.append("#d62728")
        else:
            colors_base.append("#1f77b4")
    axes[0].bar(range(len(actions)), visits_base, color=colors_base)
    axes[0].set_title("Baseline")
    axes[0].set_xticks(range(len(actions)))
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].set_ylabel("Visits")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Optimized subplot
    colors_opt = []
    for a in actions:
        if str(best_action_opt) == a:
            colors_opt.append("#d62728")
        else:
            colors_opt.append("#2ca02c")
    axes[1].bar(range(len(actions)), visits_opt, color=colors_opt)
    axes[1].set_title("Optimized")
    axes[1].set_xticks(range(len(actions)))
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"Root visit distribution — {game.name()}")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mcts_root_comparison(
    game,
    state,
    engine_baseline,
    engine_optimized,
    save_visits_path: Optional[str] = None,
    save_avg_value_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate two practical baseline-vs-optimized charts on the same state:
    1) visit count per action
    2) average value per action
    """
    root_base, best_action_base = engine_baseline._search_internal(state)
    root_opt, best_action_opt = engine_optimized._search_internal(state.clone())

    stats_base = extract_root_action_stats(root_base)
    stats_opt = extract_root_action_stats(root_opt)
    map_base = {str(s["action"]): s for s in stats_base}
    map_opt = {str(s["action"]): s for s in stats_opt}
    actions = _sort_action_keys(list(set(map_base.keys()) | set(map_opt.keys())))
    labels = [_action_label(a, game.name()) for a in actions]

    visits_base = [float(map_base.get(a, {"visits": 0.0})["visits"]) for a in actions]
    visits_opt = [float(map_opt.get(a, {"visits": 0.0})["visits"]) for a in actions]
    avg_base = [float(map_base.get(a, {"avg_value": 0.0})["avg_value"]) for a in actions]
    avg_opt = [float(map_opt.get(a, {"avg_value": 0.0})["avg_value"]) for a in actions]

    def _draw_pair(values_left, values_right, ylabel, title, left_color, right_color, left_choice, right_choice, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        left_colors = ["#d62728" if str(left_choice) == a else left_color for a in actions]
        right_colors = ["#d62728" if str(right_choice) == a else right_color for a in actions]

        axes[0].bar(range(len(actions)), values_left, color=left_colors)
        axes[0].set_title("Baseline")
        axes[0].set_xticks(range(len(actions)))
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel(ylabel)
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        axes[1].bar(range(len(actions)), values_right, color=right_colors)
        axes[1].set_title("Optimized")
        axes[1].set_xticks(range(len(actions)))
        axes[1].set_xticklabels(labels)
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)

        fig.suptitle(title)
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    _draw_pair(
        values_left=visits_base,
        values_right=visits_opt,
        ylabel="Visits",
        title=f"Root visit distribution — {game.name()}",
        left_color="#1f77b4",
        right_color="#2ca02c",
        left_choice=best_action_base,
        right_choice=best_action_opt,
        save_path=save_visits_path,
    )
    _draw_pair(
        values_left=avg_base,
        values_right=avg_opt,
        ylabel="Average value",
        title=f"Root average value per action — {game.name()}",
        left_color="#1f77b4",
        right_color="#2ca02c",
        left_choice=best_action_base,
        right_choice=best_action_opt,
        save_path=save_avg_value_path,
    )


