"""
Evaluate baseline, single-phase, pairwise, triple, and four-phase
combinations of LLM-optimized TextWorld heuristics on the original train
set and an intentionally harder test set.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcts import MCTSEngine
from mcts.games import TextWorldBenchmark


GAME_NAME = "textworld"
PHASES = ["selection", "expansion", "simulation", "backpropagation"]
SEEDS = [0, 1, 2]
TRAIN_SEEDS = [3, 4, 5]
ENV_VARIANTS = ["deterministic", "stochastic", "punishment"]
TRAIN_GAMES = {
    "coin": [
        "numLocations=2,includeDoors=1,numDistractorItems=0",
        "numLocations=4,includeDoors=1,numDistractorItems=0",
        "numLocations=6,includeDoors=1,numDistractorItems=0",
        "numLocations=8,includeDoors=1,numDistractorItems=0",
    ],
    "mapreader": [
        "numLocations=2,maxDistanceApart=1,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=4,maxDistanceApart=2,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=6,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=8,maxDistanceApart=4,includeDoors=0,maxDistractorItemsPerLocation=0",
    ],
}
TEST_GAMES = {
    "coin": [
        "numLocations=5,includeDoors=1,numDistractorItems=0",
        "numLocations=7,includeDoors=1,numDistractorItems=0",
        "numLocations=9,includeDoors=1,numDistractorItems=0",
        "numLocations=11,includeDoors=1,numDistractorItems=0",
        "numLocations=13,includeDoors=1,numDistractorItems=0",
        "numLocations=15,includeDoors=1,numDistractorItems=0",
        "numLocations=17,includeDoors=1,numDistractorItems=0",
    ],
    "mapreader": [
        "numLocations=5,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=7,maxDistanceApart=4,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=9,maxDistanceApart=5,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=11,maxDistanceApart=6,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=13,maxDistanceApart=7,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=15,maxDistanceApart=8,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=17,maxDistanceApart=9,includeDoors=0,maxDistractorItemsPerLocation=0",
    ],
}


def default_tool_path(phase: str) -> Path:
    return _ROOT / "MCTS_tools" / phase / f"{GAME_NAME}_opt_{phase}_heuristic.py"


def default_func_name(phase: str) -> str:
    return f"default_{phase}"


def load_function(tool_path: Path, func_name: str) -> Callable:
    spec = importlib.util.spec_from_file_location(f"tool_{tool_path.stem}", tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"{tool_path} does not define callable {func_name!r}")
    return fn


def build_case_list(games: dict[str, list[str]], seeds: list[int]) -> list[dict[str, Any]]:
    cases = []
    for variant in ENV_VARIANTS:
        for game_type, params_list in games.items():
            for params in params_list:
                for seed in seeds:
                    cases.append(
                        {
                            "variant": variant,
                            "game_type": game_type,
                            "game_params": params,
                            "seed": seed,
                        }
                    )
    return cases


def is_correct_plan(result: dict[str, Any]) -> bool:
    returns = result.get("returns", [0.0])
    ret = returns[0] if isinstance(returns, list) else float(returns)
    return bool(result.get("solved")) and ret >= 1.0


def combo_name(phases: tuple[str, ...]) -> str:
    if not phases:
        return "baseline"
    return "+".join(phases)


def iter_phase_combinations(phases: list[str]) -> list[tuple[str, ...]]:
    combos: list[tuple[str, ...]] = [()]
    for r in range(1, len(phases) + 1):
        combos.extend(itertools.combinations(phases, r))
    return combos


def run_case(
    case: dict[str, Any],
    phase_fns: dict[str, Callable],
    iterations: int,
    max_depth: int,
    max_steps: int,
    runs: int,
) -> tuple[int, float, float, int]:
    correct = 0
    total_ret = 0.0
    correct_steps_sum = 0.0
    correct_count = 0
    for _ in range(runs):
        game = TextWorldBenchmark(
            game_type=case["game_type"],
            game_params=case["game_params"],
            seed=case["seed"],
            variant=case["variant"],
            max_steps=max_steps,
        )
        engine = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=False)
        for phase, fn in phase_fns.items():
            engine.set_tool(phase, fn)
        result = engine.play_game(verbose=False)
        is_correct = is_correct_plan(result)
        correct += int(is_correct)
        if is_correct:
            correct_steps_sum += result["steps"]
            correct_count += 1
        total_ret += result["returns"][0]
    return correct, total_ret / runs, correct_steps_sum, correct_count


def evaluate_suite(
    cases: list[dict[str, Any]],
    phase_fns: dict[str, Callable],
    iterations: int,
    max_depth: int,
    max_steps: int,
    runs: int,
) -> tuple[int, float, float | None]:
    total_correct = 0
    ret_sum = 0.0
    total_correct_steps = 0.0
    total_correct_count = 0
    for case in cases:
        correct, avg_ret, correct_steps_sum, correct_count = run_case(
            case, phase_fns, iterations, max_depth, max_steps, runs
        )
        total_correct += correct
        ret_sum += avg_ret
        total_correct_steps += correct_steps_sum
        total_correct_count += correct_count
    avg_correct_steps = (
        total_correct_steps / total_correct_count if total_correct_count > 0 else None
    )
    return total_correct, ret_sum / len(cases), avg_correct_steps


def format_steps(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="all")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--selection-path", default=None)
    parser.add_argument("--expansion-path", default=None)
    parser.add_argument("--simulation-path", default=None)
    parser.add_argument("--backpropagation-path", default=None)
    return parser.parse_args()


def resolve_requested_phases(arg: str) -> list[str]:
    if arg.strip() == "all":
        return PHASES[:]
    phases = [p.strip() for p in arg.split(",") if p.strip()]
    for phase in phases:
        if phase not in PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Valid phases: {PHASES}")
    return phases


def resolve_tool_paths(args: argparse.Namespace, requested_phases: list[str]) -> dict[str, Path]:
    custom = {
        "selection": args.selection_path,
        "expansion": args.expansion_path,
        "simulation": args.simulation_path,
        "backpropagation": args.backpropagation_path,
    }
    paths: dict[str, Path] = {}
    for phase in requested_phases:
        path = Path(custom[phase]) if custom[phase] else default_tool_path(phase)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing heuristic for phase '{phase}': {path}. "
                "Run optimization first or pass a custom --<phase>-path."
            )
        paths[phase] = path
    return paths


def main() -> None:
    args = parse_args()
    requested_phases = resolve_requested_phases(args.phases)
    tool_paths = resolve_tool_paths(args, requested_phases)
    loaded_fns = {
        phase: load_function(path, default_func_name(phase))
        for phase, path in tool_paths.items()
    }

    print("Loaded heuristics:")
    for phase in requested_phases:
        print(f"  {phase}: {tool_paths[phase]}")
    print()

    train_cases = build_case_list(TRAIN_GAMES, TRAIN_SEEDS)
    test_cases = build_case_list(TEST_GAMES, SEEDS)
    train_slots = len(train_cases) * args.runs
    test_slots = len(test_cases) * args.runs

    baseline_train_total, baseline_train_ret, baseline_train_steps = evaluate_suite(
        train_cases, {}, args.iterations, args.max_depth, args.max_steps, args.runs
    )
    baseline_test_total, baseline_test_ret, baseline_test_steps = evaluate_suite(
        test_cases, {}, args.iterations, args.max_depth, args.max_steps, args.runs
    )

    print(
        f"{'Configuration':<30} "
        f"{'Train Correct':>16} {'Train Delta':>11} {'Train Ret':>11} {'Train Steps':>12} "
        f"{'Test Correct':>16} {'Test Delta':>10} {'Test Ret':>10} {'Test Steps':>11}"
    )
    print("-" * 136)
    print(
        f"{'baseline':<30} "
        f"{baseline_train_total:>7}/{train_slots:<8} {0:>9} {baseline_train_ret:>11.3f} {format_steps(baseline_train_steps):>12} "
        f"{baseline_test_total:>7}/{test_slots:<8} {0:>9} {baseline_test_ret:>10.3f} {format_steps(baseline_test_steps):>11}"
    )

    for combo in iter_phase_combinations(requested_phases):
        if not combo:
            continue
        phase_fns = {phase: loaded_fns[phase] for phase in combo}
        train_total, train_ret, train_steps = evaluate_suite(
            train_cases, phase_fns, args.iterations, args.max_depth, args.max_steps, args.runs
        )
        test_total, test_ret, test_steps = evaluate_suite(
            test_cases, phase_fns, args.iterations, args.max_depth, args.max_steps, args.runs
        )
        print(
            f"{combo_name(combo):<30} "
            f"{train_total:>7}/{train_slots:<8} {(train_total - baseline_train_total):>9} {train_ret:>11.3f} {format_steps(train_steps):>12} "
            f"{test_total:>7}/{test_slots:<8} {(test_total - baseline_test_total):>9} {test_ret:>10.3f} {format_steps(test_steps):>11}"
        )


if __name__ == "__main__":
    main()



