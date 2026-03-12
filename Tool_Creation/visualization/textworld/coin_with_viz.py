#!/usr/bin/env python3
"""
TextWorld Coin Self-Evolving with Trajectory Visualization.

This wraps `textworld/tools/run_textworld_coin_self_evolving.py` and adds
trajectory visualization GIFs. All outputs are written under:
  Tool_Creation/visualization/output/textworld/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Global variable to store API keys
API_KEYS_LIST = None


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Tool_Creation" / "visualization" / "output" / "textworld"


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_paths = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        Path(".env"),
        Path("..") / ".env",
    ]

    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            env_loaded = True
            break

    if not env_loaded:
        load_dotenv()

    import os

    api_key = os.getenv("API_KEYS", "")
    if api_key:
        os.environ["API_KEYS"] = api_key
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://tritonai-api.ucsd.edu")
        os.environ["MODEL_NAME"] = os.getenv("MODEL_NAME", "api-gpt-oss-120b")
        API_KEYS_LIST = [k.strip() for k in api_key.split(",") if k.strip()]

except ImportError:
    pass


sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Tool_Creation"))

from visualization.textworld import TrajectoryVisualizer

# Import the original script functions
from textworld.tools.run_textworld_coin_self_evolving import *  # noqa: F403


def _build_branching_world_graph(num_locations: int) -> Dict[int, List[int]]:  # type: ignore[name-defined]
    """
    Build a small branching corridor graph.

    For the first 6 rooms, this matches the mock runner shape:
        0: 1,4
        1: 0,2
        2: 1,3
        3: 2
        4: 0,5
        5: 4
    Any additional rooms are attached as a tail after room 5.
    """
    base = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2],
        4: [0, 5],
        5: [4],
    }

    if num_locations <= 6:
        return {i: list(base.get(i, [])) for i in range(num_locations)}

    g: Dict[int, List[int]] = {i: list(neigh) for i, neigh in base.items()}
    # Extend a simple chain from room 5 onward.
    last = 5
    for room in range(6, num_locations):
        g.setdefault(last, []).append(room)
        g[room] = [last]
        last = room
    return g


def create_textworld_game_state(game_config):
    """Create a branching game_state graph for visualization from TextWorld coin config."""
    num_locations = game_config.num_locations

    world_graph = _build_branching_world_graph(num_locations)
    room_descriptions = {}
    room_names = {}

    for i in range(num_locations):
        if i == 0:
            desc = "Starting room"
        elif i == num_locations - 1:
            desc = "Final room containing the coin"
        else:
            desc = f"Room {i}"
        room_descriptions[i] = desc
        room_names[i] = f"Room {i}"

    return {"world_graph": world_graph, "room_descriptions": room_descriptions, "room_names": room_names}


def play_game_with_trajectory(engine, game_config):
    """Play a game and collect trajectory data for visualization."""
    state = engine.game.new_initial_state()
    moves = []
    trajectory = []

    initial_obs = {
        "room": state.room,
        "description": state.observation_text(),
        "inventory": list(state.inventory_items),
        "quest_progress": {"coin_taken": state.coin_taken},
        "available_actions": [str(a) for a in state.legal_actions()],
    }
    trajectory.append({"observation": initial_obs, "action": None, "reward": 0.0, "done": False})

    while not state.is_terminal():
        _root, action = engine._search_internal(state)
        prev_returns = state.returns()[0]

        state.apply_action(action)
        moves.append(action)

        reward = state.returns()[0] - prev_returns

        obs = {
            "room": state.room,
            "description": state.observation_text(),
            "inventory": list(state.inventory_items),
            "quest_progress": {"coin_taken": state.coin_taken},
            "available_actions": [str(a) for a in state.legal_actions()],
        }

        trajectory.append({"observation": obs, "action": str(action), "reward": reward, "done": state.is_terminal()})

    solved = state.returns()[0] >= 1.0
    result = {"solved": solved, "steps": len(moves), "returns": state.returns(), "moves": moves}
    return result, trajectory


def _viz_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def multi_eval_with_viz(
    phase: str,
    game_params: str,
    fn: Callable | None,  # noqa: F405
    n: int,
    iterations: int,
    max_depth: int,
    max_steps: int,
    create_viz: bool = False,
    viz_prefix: str = "eval",
) -> tuple[float, float, float, list[dict[str, Any]], float, list]:  # noqa: F405
    """Enhanced multi_eval that optionally creates trajectory visualizations."""
    t0 = time.time()  # noqa: F405
    results = []
    trajectories = []

    for i in range(n):
        engine = make_engine(game_params, iterations, max_depth, max_steps, logging=False)  # noqa: F405
        if fn is not None:
            engine.set_tool(phase, fn)

        result, trajectory = play_game_with_trajectory(engine, engine.game.config)
        results.append(result)
        trajectories.append(trajectory)

        if create_viz and (i == 0 or result["solved"]):
            game_state = create_textworld_game_state(engine.game.config)
            viz = TrajectoryVisualizer(trajectory, game_state=game_state)
            viz.visualize(output_path=_viz_path(f"{viz_prefix}_ep_{i+1}.gif"), fps=1)

    elapsed = time.time() - t0  # noqa: F405
    avg_ret = sum(r["returns"][0] for r in results) / n
    solve_rate = sum(1 for r in results if r["solved"]) / n
    avg_steps = sum(r["steps"] for r in results) / n
    return avg_ret, solve_rate, avg_steps, results, elapsed, trajectories


def main_with_viz() -> None:
    args = parse_args()  # noqa: F405
    random.seed(3)  # noqa: F405

    optimizer = Optimizer(  # noqa: F405
        game="textworld_coin",
        target_phase=args.phase,
        three_step=True,
        verbose=True,
        api_keys=API_KEYS_LIST,
    )

    best_fn = None
    current_fn = None
    all_results: list[dict[str, Any]] = []  # noqa: F405
    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}

    def get_baseline(game_params: str) -> dict[str, float]:
        if game_params not in baselines:
            avg, sr, steps, _, elapsed, _ = multi_eval_with_viz(
                args.phase,
                game_params,
                None,
                args.eval_runs,
                args.iterations,
                args.max_depth,
                args.max_steps,
                create_viz=False,
            )
            comp = composite_score(sr, avg)  # noqa: F405
            baselines[game_params] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "composite": comp,
            }
            best_scores[game_params] = comp
        return baselines[game_params]

    cur_params = args.game_params
    get_baseline(cur_params)

    for iteration in range(1, args.num_iters + 1):
        baseline = get_baseline(cur_params)
        reject_floor = baseline["composite"] * args.reject_threshold

        engine = make_engine(cur_params, args.iterations, args.max_depth, args.max_steps, logging=True)  # noqa: F405
        if current_fn is not None:
            engine.set_tool(args.phase, current_fn)

        play_result, trajectory = play_game_with_trajectory(engine, engine.game.config)

        game_state = create_textworld_game_state(engine.game.config)
        viz = TrajectoryVisualizer(trajectory, game_state=game_state)
        viz.visualize(output_path=_viz_path(f"optimization_iter_{iteration}.gif"), fps=1)

        tool_sources = engine.get_tool_source()
        play_trace = play_result.get("log_file", "")
        history = build_history(all_results[-3:], cur_params, baselines, best_scores) if all_results else None  # noqa: F405

        result = optimizer.run(
            record_files=[play_trace] if play_trace else [],
            tool_list=tool_sources,
            state_factory=lambda _p=cur_params: TextWorldCoin(_p, max_steps=args.max_steps).new_initial_state(),  # noqa: F405
            additional_context=history,
            session_tag=f"textworld_coin_{args.phase}_iter{iteration}",
        )

        rec = {
            "iteration": iteration,
            "params": cur_params,
            "solve_rate": 0.0,
            "avg_returns": baseline["avg_returns"],
            "avg_steps": args.max_steps,
            "composite": 0.0,
            "description": (result.get("parsed") or {}).get("description", ""),
            "adopted": False,
            "is_best": False,
        }

        fn = result.get("function")
        if fn is not None:
            avg_ret, solve_rate, avg_steps, _, eval_time, trajectories = multi_eval_with_viz(
                args.phase,
                cur_params,
                fn,
                args.eval_runs,
                args.iterations,
                args.max_depth,
                args.max_steps,
                create_viz=True,
                viz_prefix=f"eval_iter_{iteration}",
            )
            comp = composite_score(solve_rate, avg_ret)  # noqa: F405
            rec.update({"solve_rate": solve_rate, "avg_returns": avg_ret, "avg_steps": avg_steps, "composite": comp})

            prev_best = best_scores[cur_params]
            if comp > prev_best:
                best_scores[cur_params] = comp
                best_fn = fn
                current_fn = fn
                rec["adopted"] = True
                rec["is_best"] = True
            elif comp >= reject_floor:
                current_fn = fn
                rec["adopted"] = True
            else:
                current_fn = best_fn

        all_results.append(rec)

    compare_suite(args.phase, best_fn, DEFAULT_PARAMS, args.eval_runs, args.iterations, args.max_depth, args.max_steps)  # noqa: F405

    for params in DEFAULT_PARAMS:
        engine = make_engine(params, args.iterations, args.max_depth, args.max_steps, logging=False)  # noqa: F405
        if best_fn is not None:
            engine.set_tool(args.phase, best_fn)
        result, trajectory = play_game_with_trajectory(engine, engine.game.config)
        game_state = create_textworld_game_state(engine.game.config)
        viz = TrajectoryVisualizer(trajectory, game_state=game_state)
        safe_name = params.replace("=", "_").replace(",", "_")
        viz.visualize(output_path=_viz_path(f"final_{safe_name}.gif"), fps=1)


if __name__ == "__main__":
    main_with_viz()

