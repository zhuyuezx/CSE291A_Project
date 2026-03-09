"""
Game trajectory visualization (PNG timelines + GIF animations).

Works with the active game configured in MCTS_tools/hyperparams.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .config import get_game_config, get_game_config_for
from .plot_visit_distribution import _action_label
from .renderers import draw_state

MAX_STEPS = 200
DEFAULT_ITERATIONS = 150
DEFAULT_MAX_ROLLOUT_DEPTH = 200
MAX_TIMELINE_SEGMENTS = 5


def _terminal_reason(state) -> str:
    if hasattr(state, "_is_solved") and getattr(state, "_is_solved")():
        return "solved"
    if getattr(state, "steps", 0) >= getattr(state, "max_steps", 0):
        return "max_steps"
    if hasattr(state, "_is_deadlocked") and getattr(state, "_is_deadlocked")():
        return "deadlock"
    return "stopped"


def rollout_with_mcts_policy(game, engine, max_moves: Optional[int] = None) -> Dict:
    state = game.new_initial_state()
    states = [state.clone()]
    actions: List = []

    cap = max_moves
    if cap is None:
        cap = getattr(state, "max_steps", None)
    if cap is None:
        cap = getattr(state, "max_moves", MAX_STEPS)

    while not state.is_terminal() and len(actions) < int(cap):
        action = engine.search(state)
        if action is None:
            break
        actions.append(action)
        state.apply_action(action)
        states.append(state.clone())

    solved = bool(state.returns()[0] > 0.0)
    return {
        "states": states,
        "actions": actions,
        "steps": len(actions),
        "solved": solved,
        "terminal_reason": _terminal_reason(state),
    }


def _timeline_frame_indices(num_states: int) -> List[int]:
    final = max(0, num_states - 1)
    if final == 0:
        return [0]

    raw_step = final / MAX_TIMELINE_SEGMENTS
    exponent = 10 ** int(math.floor(math.log10(max(raw_step, 1.0))))
    fraction = raw_step / exponent
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    step = int(max(1, nice_fraction * exponent))

    idxs = list(range(0, final + 1, step))
    if idxs[-1] != final:
        idxs.append(final)
    return idxs


def _format_action(game_name: str, action) -> str:
    return _action_label(str(action), game_name)


def _parse_game_level_label(label: str) -> tuple[str, str]:
    raw = str(label or "")
    if "(" in raw and ")" in raw:
        game_part, rest = raw.split("(", 1)
        level = rest.split(")", 1)[0].strip()
        game = game_part.strip()
        return game, level
    return raw.strip(), ""


def _normalize_game_key(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _resolve_action_from_trace(state, action_text: str):
    target = str(action_text).strip()
    legal = list(state.legal_actions())
    for action in legal:
        if str(action) == target:
            return action

    parsed = None
    if target.lstrip("-").isdigit():
        parsed = int(target)
    elif target.startswith("(") and target.endswith(")"):
        try:
            parsed = ast.literal_eval(target)
        except (SyntaxError, ValueError):
            parsed = None
    if parsed is not None:
        for action in legal:
            if action == parsed:
                return action
    return None


def _rollout_from_trace(trace: Dict, game) -> Dict:
    state = game.new_initial_state()
    states = [state.clone()]
    actions: List = []

    for move in trace.get("moves", []) or []:
        action = _resolve_action_from_trace(state, str(move.get("action_chosen", "")))
        if action is None:
            break
        actions.append(action)
        state.apply_action(action)
        states.append(state.clone())

    outcome = trace.get("outcome", {}) or {}
    solved = bool(outcome.get("solved", bool(state.returns()[0] > 0.0)))
    return {
        "states": states,
        "actions": actions,
        "steps": len(actions),
        "solved": solved,
        "terminal_reason": _terminal_reason(state),
    }


def _trace_is_baseline(metadata: Dict) -> bool:
    """
    Classify trace as baseline vs optimized from metadata.tools only (visualization-only).
    Path-agnostic: use the last path segment (filename) for each of the four phases.
    - Baseline: all four tools end with default_<phase>.py (e.g. default_simulation.py).
    - Optimized: any tool is "(set programmatically)" or has a different last segment.
    """
    tools = metadata.get("tools") or {}
    default_names = {
        "selection": "default_selection.py",
        "expansion": "default_expansion.py",
        "simulation": "default_simulation.py",
        "backpropagation": "default_backpropagation.py",
    }
    for phase, default_name in default_names.items():
        raw = str(tools.get(phase, "")).strip()
        if raw.lower() == "(set programmatically)":
            return False
        # Last term = filename (path-agnostic across machines)
        last_term = raw.replace("\\", "/").split("/")[-1] if "/" in raw else raw
        if last_term != default_name:
            return False
    return True


def _load_rollouts_from_latest_records(
    records_dir: Path,
    game_name: str,
    level_name: str,
    make_game,
) -> tuple[Optional[Dict], Optional[Dict], Optional[Path], Optional[Path]]:
    """Returns (base_rollout, opt_rollout, base_json_path, opt_json_path)."""
    if not records_dir.exists():
        return None, None, None, None

    game_key = _normalize_game_key(game_name)
    target_level = str(level_name)
    traces = sorted(records_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    latest_base: Optional[Dict] = None
    latest_opt: Optional[Dict] = None
    base_path: Optional[Path] = None
    opt_path: Optional[Path] = None

    for path in traces:
        if path.name == "optimization_summary.json":
            continue
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = trace.get("metadata", {}) or {}
        moves = trace.get("moves", []) or []
        if not metadata or not moves:
            continue
        parsed_game, parsed_level = _parse_game_level_label(str(metadata.get("game", "")))
        if _normalize_game_key(parsed_game) != game_key or parsed_level != target_level:
            continue

        rollout = _rollout_from_trace(trace, make_game(target_level))
        if _trace_is_baseline(metadata):
            if latest_base is None:
                latest_base = rollout
                base_path = path
        else:
            if latest_opt is None:
                latest_opt = rollout
                opt_path = path
        if latest_base is not None and latest_opt is not None:
            break

    return latest_base, latest_opt, base_path, opt_path


def plot_trajectory_timeline(
    rollout: Dict,
    game_name: str,
    level_name: str,
    policy_label: str,
    save_path: Optional[str] = None,
    show: bool = True,
    max_cols: int = 8,
) -> None:
    states = rollout["states"]
    actions = rollout["actions"]
    frame_indices = _timeline_frame_indices(len(states))
    num_frames = len(frame_indices)
    ncols = min(max_cols, num_frames)
    nrows = math.ceil(num_frames / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.8 * nrows))
    axes_list = [axes] if nrows == 1 and ncols == 1 else list(axes.flat)

    for i, ax in enumerate(axes_list):
        if i < num_frames:
            idx = frame_indices[i]
            prev_action = None if idx == 0 else actions[idx - 1]
            action_label = "" if prev_action is None else _format_action(game_name, prev_action)
            draw_state(ax, states[idx], game_name=game_name, step_idx=idx, action_label=action_label)
        else:
            ax.axis("off")

    solved = rollout["solved"]
    steps = rollout["steps"]
    reason = rollout["terminal_reason"]
    status = "SOLVED" if solved else f"UNSOLVED ({reason})"
    fig.suptitle(
        f"{game_name} trajectory — {policy_label} — {level_name} | {status} | steps={steps}",
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


def plot_trajectory_comparison(
    baseline_rollout: Dict,
    optimized_rollout: Dict,
    game_name: str,
    level_name: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    b_states = baseline_rollout["states"]
    b_actions = baseline_rollout["actions"]
    o_states = optimized_rollout["states"]
    o_actions = optimized_rollout["actions"]
    b_idx = _timeline_frame_indices(len(b_states))
    o_idx = _timeline_frame_indices(len(o_states))
    ncols = max(len(b_idx), len(o_idx))

    fig, axes = plt.subplots(2, ncols, figsize=(2.5 * ncols, 5.6))
    if ncols == 1:
        axes = [[axes[0]], [axes[1]]]

    for col in range(ncols):
        ax_b = axes[0][col]
        ax_o = axes[1][col]

        if col < len(b_idx):
            i = b_idx[col]
            prev_action = None if i == 0 else b_actions[i - 1]
            action_label = "" if prev_action is None else _format_action(game_name, prev_action)
            draw_state(ax_b, b_states[i], game_name=game_name, step_idx=i, action_label=action_label)
        else:
            ax_b.axis("off")

        if col < len(o_idx):
            i = o_idx[col]
            prev_action = None if i == 0 else o_actions[i - 1]
            action_label = "" if prev_action is None else _format_action(game_name, prev_action)
            draw_state(ax_o, o_states[i], game_name=game_name, step_idx=i, action_label=action_label)
        else:
            ax_o.axis("off")

    base_status = "SOLVED" if baseline_rollout["solved"] else f"UNSOLVED ({baseline_rollout['terminal_reason']})"
    opt_status = "SOLVED" if optimized_rollout["solved"] else f"UNSOLVED ({optimized_rollout['terminal_reason']})"
    axes[0][0].text(
        -0.02,
        1.08,
        f"Baseline: {base_status}, steps={baseline_rollout['steps']}",
        transform=axes[0][0].transAxes,
        fontsize=9,
        ha="left",
    )
    axes[1][0].text(
        -0.02,
        1.08,
        f"Optimized: {opt_status}, steps={optimized_rollout['steps']}",
        transform=axes[1][0].transAxes,
        fontsize=9,
        ha="left",
    )

    fig.suptitle(
        f"{game_name} trajectory comparison (sampled) — {level_name}",
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


def _gif_indices(total_states: int, max_frames: int = 40) -> List[int]:
    if total_states <= 1:
        return [0]
    if total_states <= max_frames:
        return list(range(total_states))
    step = max(1, (total_states - 1) // (max_frames - 1))
    out = list(range(0, total_states, step))
    if out[-1] != total_states - 1:
        out.append(total_states - 1)
    return out


def export_trajectory_gif(
    rollout: Dict,
    game_name: str,
    level_name: str,
    policy_label: str,
    save_path: str,
    fps: int = 2,
) -> None:
    states = rollout["states"]
    actions = rollout["actions"]
    # Step size 1: every timestep
    idxs = list(range(len(states)))
    fig, ax = plt.subplots(figsize=(5.2, 5.6))

    def _update(frame_idx: int):
        idx = idxs[frame_idx]
        ax.clear()
        prev_action = None if idx == 0 else actions[idx - 1]
        action_label = "" if prev_action is None else _format_action(game_name, prev_action)
        draw_state(ax, states[idx], game_name=game_name, step_idx=idx, action_label=action_label)
        ax.set_title(f"{policy_label} | t={idx}", fontsize=10)
        return []

    anim = FuncAnimation(fig, _update, frames=len(idxs), interval=int(1000 / max(1, fps)), blit=False)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)


def export_comparison_gif(
    baseline_rollout: Dict,
    optimized_rollout: Dict,
    game_name: str,
    level_name: str,
    save_path: str,
    fps: int = 2,
) -> None:
    b_states = baseline_rollout["states"]
    b_actions = baseline_rollout["actions"]
    o_states = optimized_rollout["states"]
    o_actions = optimized_rollout["actions"]
    max_t = max(len(b_states), len(o_states)) - 1
    if max_t < 0:
        max_t = 0
    # Step size 1: every timestep; same t on both sides until one stops, then that side shows (done)
    t_indices = list(range(max_t + 1))
    n = len(t_indices)

    fig, (ax_b, ax_o) = plt.subplots(1, 2, figsize=(10.0, 5.6))

    def _update(frame: int):
        t = t_indices[frame]
        ax_b.clear()
        ax_o.clear()

        i_b = min(t, len(b_states) - 1)
        i_o = min(t, len(o_states) - 1)
        b_action = None if i_b == 0 else b_actions[i_b - 1]
        o_action = None if i_o == 0 else o_actions[i_o - 1]
        draw_state(
            ax_b,
            b_states[i_b],
            game_name=game_name,
            step_idx=i_b,
            action_label="" if b_action is None else _format_action(game_name, b_action),
        )
        draw_state(
            ax_o,
            o_states[i_o],
            game_name=game_name,
            step_idx=i_o,
            action_label="" if o_action is None else _format_action(game_name, o_action),
        )
        # When a side has finished, its t stops and we show "(done)"
        t_b_label = f"Baseline | t={i_b}" + (" (done)" if t > len(b_states) - 1 else "")
        t_o_label = f"Optimized | t={i_o}" + (" (done)" if t > len(o_states) - 1 else "")
        ax_b.set_title(t_b_label, fontsize=10)
        ax_o.set_title(t_o_label, fontsize=10)
        fig.suptitle(f"{game_name} trajectory GIF comparison — {level_name}", fontsize=11)
        return []

    anim = FuncAnimation(fig, _update, frames=n, interval=int(1000 / max(1, fps)), blit=False)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)


def run_trajectory_demo(
    level_name: Optional[str] = None,
    iterations: Optional[int] = None,
    max_rollout_depth: Optional[int] = None,
    max_steps: int = MAX_STEPS,
    output_dir: Optional[str] = None,
    compare_baseline: bool = True,
    make_gif: bool = True,
    show: bool = True,
    game: Optional[str] = None,
    from_records: bool = False,
    records_dir: Optional[str] = None,
) -> None:
    import sys

    root_path = Path(__file__).resolve().parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from mcts import MCTSEngine

    cfg = get_game_config_for(game) if game else get_game_config()
    game_name = cfg["game_name"]
    game_class = cfg["game_class"]
    constructor_kwargs = cfg["constructor_kwargs"]
    levels = cfg["levels"]
    hp = cfg.get("hyperparams", {})
    tool_path = cfg.get("optimized_tool_path")
    iterations = int(iterations if iterations is not None else hp.get("iterations", DEFAULT_ITERATIONS))
    max_rollout_depth = int(max_rollout_depth if max_rollout_depth is not None else hp.get("max_rollout_depth", DEFAULT_MAX_ROLLOUT_DEPTH))
    if level_name is None:
        level_name = levels[0] if levels else ("level1" if "sokoban" in (game or "").lower() else "easy1")

    def _make_game(level: str):
        kwargs = dict(constructor_kwargs)
        if "max_steps" in kwargs:
            kwargs["max_steps"] = max_steps
        try:
            return game_class(level, **kwargs)
        except TypeError:
            return game_class(**kwargs)

    out_root = Path(output_dir) if output_dir else root_path / "visualization" / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / "trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = game_name.lower().replace(" ", "_")
    base_rollout: Optional[Dict] = None
    opt_rollout: Optional[Dict] = None
    base_json_path: Optional[Path] = None
    opt_json_path: Optional[Path] = None

    if from_records:
        rec_dir = Path(records_dir) if records_dir else (root_path / "mcts" / "records")
        base_rollout, opt_rollout, base_json_path, opt_json_path = _load_rollouts_from_latest_records(
            records_dir=rec_dir,
            game_name=game_name,
            level_name=level_name,
            make_game=_make_game,
        )
        if opt_rollout is None:
            print(
                f"[trajectory] No optimized trace found in {rec_dir} for "
                f"{game_name} ({level_name}); falling back to live MCTS run."
            )
        # Print which JSON files are used when loading from records
        if base_rollout is not None and base_json_path is not None:
            print(f"[trajectory] Baseline from records: {base_json_path.name}")
        if opt_rollout is not None and opt_json_path is not None:
            print(f"[trajectory] Optimized from records: {opt_json_path.name}")

    if opt_rollout is None:
        game_opt = _make_game(level_name)
        engine_opt = MCTSEngine(game_opt, iterations=iterations, max_rollout_depth=max_rollout_depth)
        if tool_path is not None:
            engine_opt.load_tool("simulation", tool_path)
        opt_rollout = rollout_with_mcts_policy(game_opt, engine_opt)
        # Print runtime tools for optimized
        print("[trajectory] Optimized (runtime): ", end="")
        if tool_path is not None:
            print(f"simulation from {Path(tool_path).name}, other phases default.")
        else:
            print("default tools for all phases.")

    opt_png = out_dir / f"{key}_trajectory_optimized_{level_name}.png"
    plot_trajectory_timeline(
        rollout=opt_rollout,
        game_name=game_name,
        level_name=level_name,
        policy_label="optimized MCTS",
        save_path=str(opt_png),
        show=show,
    )
    print(f"Saved: {opt_png}")

    if make_gif:
        opt_gif = out_dir / f"{key}_trajectory_optimized_{level_name}.gif"
        export_trajectory_gif(
            rollout=opt_rollout,
            game_name=game_name,
            level_name=level_name,
            policy_label="optimized",
            save_path=str(opt_gif),
        )
        print(f"Saved: {opt_gif}")

    if compare_baseline:
        if base_rollout is None:
            if from_records:
                print(
                    f"[trajectory] No baseline trace found for {game_name} ({level_name}); "
                    "falling back to live baseline MCTS run."
                )
            game_base = _make_game(level_name)
            engine_base = MCTSEngine(game_base, iterations=iterations, max_rollout_depth=max_rollout_depth)
            base_rollout = rollout_with_mcts_policy(game_base, engine_base)
            print("[trajectory] Baseline (runtime): default tools for all phases (selection, expansion, simulation, backpropagation).")

        cmp_png = out_dir / f"{key}_trajectory_compare_{level_name}.png"
        plot_trajectory_comparison(
            baseline_rollout=base_rollout,
            optimized_rollout=opt_rollout,
            game_name=game_name,
            level_name=level_name,
            save_path=str(cmp_png),
            show=show,
        )
        print(f"Saved: {cmp_png}")

        if make_gif:
            cmp_gif = out_dir / f"{key}_trajectory_compare_{level_name}.gif"
            export_comparison_gif(
                baseline_rollout=base_rollout,
                optimized_rollout=opt_rollout,
                game_name=game_name,
                level_name=level_name,
                save_path=str(cmp_gif),
            )
            print(f"Saved: {cmp_gif}")


def run_sokoban_trajectory_demo(**kwargs) -> None:
    """
    Backward-compatible alias for existing callers.
    """
    run_trajectory_demo(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trajectory visualization from MCTS step-by-step play.",
    )
    parser.add_argument("--level", default=None, help="Level name (uses active game config).")
    parser.add_argument("--game", default=None, choices=["sokoban", "rush_hour"], help="Force game (sokoban or rush_hour) for trajectory and GIFs.")
    parser.add_argument("--iterations", type=int, default=None, help="MCTS iterations per move.")
    parser.add_argument("--max-rollout-depth", type=int, default=None, help="MCTS max rollout depth.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum game steps/moves.")
    parser.add_argument("--no-compare", action="store_true", help="Disable baseline-vs-optimized comparison output.")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF export.")
    parser.add_argument(
        "--from-records",
        action="store_true",
        help="Load latest baseline/optimized trace JSONs from mcts/records instead of rerunning MCTS.",
    )
    parser.add_argument(
        "--records-dir",
        default=None,
        help="Override records directory for --from-records (default: Tool_Creation/mcts/records).",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    args = parser.parse_args()

    run_trajectory_demo(
        level_name=args.level,
        game=args.game,
        iterations=args.iterations,
        max_rollout_depth=args.max_rollout_depth,
        max_steps=args.max_steps,
        compare_baseline=not args.no_compare,
        make_gif=not args.no_gif,
        show=args.show,
        from_records=args.from_records,
        records_dir=args.records_dir,
    )

