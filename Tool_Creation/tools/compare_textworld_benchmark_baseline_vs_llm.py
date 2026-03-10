"""
Compare baseline MCTS against a directly loaded LLM-optimized heuristic on
the hw2-style 63-case symbolic TextWorld benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcts import MCTSEngine
from mcts.games import TextWorldBenchmark


SEEDS = [0, 1, 2]
ENV_VARIANTS = ["deterministic", "stochastic", "punishment"]
TEST_GAMES = {
    "coin": [
        "numLocations=5,includeDoors=1,numDistractorItems=0",
        "numLocations=7,includeDoors=1,numDistractorItems=0",
        "numLocations=9,includeDoors=1,numDistractorItems=0",
        "numLocations=11,includeDoors=1,numDistractorItems=0",
        "numLocations=13,includeDoors=1,numDistractorItems=0",
    ],
    "mapreader": [
        "numLocations=5,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=7,maxDistanceApart=4,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=9,maxDistanceApart=5,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=11,maxDistanceApart=6,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=13,maxDistanceApart=7,includeDoors=0,maxDistractorItemsPerLocation=0",
    ],
}


def make_case_label(case: dict[str, Any]) -> str:
    return f"{case['variant']}|{case['game_type']}|{case['game_params']}|seed={case['seed']}"


def build_case_list() -> list[dict[str, Any]]:
    cases = []
    for variant in ENV_VARIANTS:
        for game_type, params_list in TEST_GAMES.items():
            for params in params_list:
                for seed in SEEDS:
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


def load_llm_function(tool_path: Path, func_name: str) -> Callable:
    spec = importlib.util.spec_from_file_location("llm_tool_module", tool_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"{tool_path} does not define callable {func_name!r}")
    return fn


def eval_case(
    case: dict[str, Any],
    phase: str,
    fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    runs: int,
) -> tuple[float, int]:
    correct = 0
    total_ret = 0.0
    for _ in range(runs):
        game = TextWorldBenchmark(
            game_type=case["game_type"],
            game_params=case["game_params"],
            seed=case["seed"],
            variant=case["variant"],
            max_steps=max_steps,
        )
        engine = MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=False)
        if fn is not None:
            engine.set_tool(phase, fn)
        result = engine.play_game(verbose=False)
        total_ret += result["returns"][0]
        correct += int(is_correct_plan(result))
    return total_ret / runs, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="selection")
    parser.add_argument("--tool-path", required=True)
    parser.add_argument("--func-name", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_path = Path(args.tool_path)
    if not tool_path.exists():
        raise FileNotFoundError(tool_path)
    func_name = args.func_name or f"default_{args.phase}"
    llm_fn = load_llm_function(tool_path, func_name)

    total_base = 0
    total_opt = 0
    total_cases = 0

    print(
        f"{'Variant':<14} {'Game':<10} {'Seed':>4} {'Base Correct':>12} "
        f"{'Opt Correct':>12} {'Base Ret':>9} {'Opt Ret':>8}"
    )
    print("-" * 86)

    for case in build_case_list():
        base_ret, base_correct = eval_case(
            case, args.phase, None, args.iterations, args.max_depth, args.max_steps, args.runs
        )
        opt_ret, opt_correct = eval_case(
            case, args.phase, llm_fn, args.iterations, args.max_depth, args.max_steps, args.runs
        )
        total_base += base_correct
        total_opt += opt_correct
        total_cases += args.runs
        print(
            f"{case['variant']:<14} {case['game_type']:<10} {case['seed']:>4} "
            f"{base_correct:>7}/{args.runs:<4} {opt_correct:>7}/{args.runs:<4} "
            f"{base_ret:>9.3f} {opt_ret:>8.3f}"
        )

    print()
    print(
        f"Correct plans summary: baseline={total_base}/{total_cases}, "
        f"optimized={total_opt}/{total_cases}"
    )


if __name__ == "__main__":
    main()
