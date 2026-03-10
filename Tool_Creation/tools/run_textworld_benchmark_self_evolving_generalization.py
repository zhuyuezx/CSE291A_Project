"""
Self-evolving LLM runner for generalization experiments on the symbolic
hw2-style TextWorld benchmark.

Training traces and validation cases come from a separate pool of
"non-hw2" settings. Final evaluation still runs on the original 63
hw2-style benchmark cases.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from LLM.optimizer import Optimizer
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
TRAIN_SEEDS = [3, 4, 5]
GAME_NAME = "textworld"
VALID_PHASES = ["selection", "expansion", "simulation", "backpropagation"]


def make_case_label(case: dict[str, Any]) -> str:
    return f"{case['variant']}|{case['game_type']}|{case['game_params']}|seed={case['seed']}"


def best_heuristic_path(phase: str) -> Path:
    return _ROOT / "MCTS_tools" / phase / f"{GAME_NAME}_opt_{phase}_heuristic.py"


def persist_best_heuristic(installed_path: str | Path, phase: str) -> Path:
    src = Path(installed_path)
    dst = best_heuristic_path(phase)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def make_engine(case: dict[str, Any], iterations: int, max_depth: int, max_steps: int, logging: bool) -> MCTSEngine:
    game = TextWorldBenchmark(
        game_type=case["game_type"],
        game_params=case["game_params"],
        seed=case["seed"],
        variant=case["variant"],
        max_steps=max_steps,
    )
    return MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=logging)


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


def correct_plan_rate(results: list[dict[str, Any]]) -> float:
    return sum(1 for r in results if is_correct_plan(r)) / max(1, len(results))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def eval_case(
    case: dict[str, Any],
    phase: str,
    fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    runs: int,
) -> tuple[float, float, float, list[dict[str, Any]], float]:
    t0 = time.time()
    results = []
    for _ in range(runs):
        engine = make_engine(case, iterations, max_depth, max_steps, logging=False)
        if fn is not None:
            engine.set_tool(phase, fn)
        results.append(engine.play_game(verbose=False))
    elapsed = time.time() - t0
    avg_ret = sum(r["returns"][0] for r in results) / runs
    solve_rate = sum(1 for r in results if r["solved"]) / runs
    avg_steps = sum(r["steps"] for r in results) / runs
    return avg_ret, solve_rate, avg_steps, results, elapsed


def eval_case_set(
    eval_cases: list[dict[str, Any]],
    phase: str,
    fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    runs: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for case in eval_cases:
        avg_ret, solve_rate, avg_steps, results, elapsed = eval_case(
            case, phase, fn, iterations, max_depth, max_steps, runs
        )
        per_case.append(
            {
                "label": make_case_label(case),
                "avg_returns": avg_ret,
                "solve_rate": solve_rate,
                "avg_steps": avg_steps,
                "correct_plan_rate": correct_plan_rate(results),
                "elapsed": elapsed,
            }
        )
        all_results.extend(results)
    return {
        "correct_plan_rate": correct_plan_rate(all_results),
        "solve_rate": mean([item["solve_rate"] for item in per_case]),
        "avg_returns": mean([item["avg_returns"] for item in per_case]),
        "avg_steps": mean([item["avg_steps"] for item in per_case]),
        "elapsed": sum(item["elapsed"] for item in per_case),
        "per_case": per_case,
    }


def rank_cases(
    cases: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: (
            baselines[make_case_label(case)]["correct_plan_rate"],
            baselines[make_case_label(case)]["avg_returns"],
            -baselines[make_case_label(case)]["avg_steps"],
            make_case_label(case),
        ),
    )


def select_balanced_top_cases(
    cases: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    count: int,
) -> list[dict[str, Any]]:
    count = max(1, count)
    ranked = rank_cases(cases, baselines)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_first_matching(predicate) -> None:
        for case in ranked:
            label = make_case_label(case)
            if label in seen:
                continue
            if predicate(case):
                selected.append(case)
                seen.add(label)
                return

    add_first_matching(lambda case: case["game_type"] == "coin")
    add_first_matching(lambda case: case["game_type"] == "mapreader")
    add_first_matching(lambda case: case["variant"] in {"stochastic", "punishment"})

    for case in ranked:
        if len(selected) >= count:
            break
        label = make_case_label(case)
        if label in seen:
            continue
        selected.append(case)
        seen.add(label)

    return selected[:count]


def collect_training_traces(
    train_cases: list[dict[str, Any]],
    phase: str,
    current_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
) -> tuple[list[str], list[str], dict[str, str]]:
    record_files: list[str] = []
    tool_sources: list[str] = []
    summary: dict[str, str] = {}
    for case in train_cases:
        label = make_case_label(case)
        t_play = time.time()
        engine = make_engine(case, iterations, max_depth, max_steps, logging=True)
        if current_fn is not None:
            engine.set_tool(phase, current_fn)
        play_result = engine.play_game(verbose=False)
        play_trace = play_result.get("log_file", "")
        if not play_trace:
            raise RuntimeError(
                f"Trace logging did not produce a log_file for {label}. "
                "The LLM would run without gameplay traces."
            )
        trace_path = Path(play_trace)
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file was reported but does not exist: {trace_path}")
        if not tool_sources:
            tool_sources = engine.get_tool_source()
        record_files.append(play_trace)
        summary[label] = (
            f"{'SOLVED' if play_result['solved'] else 'UNSOLVED'} in {play_result['steps']} steps "
            f"returns={play_result['returns'][0]:.4f} ({time.time() - t_play:.1f}s)"
        )
    return record_files, tool_sources, summary


def final_hw2_style_eval(
    phase: str,
    best_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
) -> None:
    cases = build_case_list(TEST_GAMES, SEEDS)
    total_base = 0
    total_opt = 0
    total = 0
    print(
        f"{'Variant':<14} {'Game':<10} {'Seed':>4} {'Base Correct':>12} "
        f"{'Opt Correct':>12} {'Base Ret':>9} {'Opt Ret':>8}"
    )
    print("-" * 86)
    for case in cases:
        rb = eval_case(case, phase, None, iterations, max_depth, max_steps, runs=1)
        ro = eval_case(case, phase, best_fn, iterations, max_depth, max_steps, runs=1)
        base_result = rb[3][0]
        opt_result = ro[3][0]
        base_correct = int(is_correct_plan(base_result))
        opt_correct = int(is_correct_plan(opt_result))
        total_base += base_correct
        total_opt += opt_correct
        total += 1
        print(
            f"{case['variant']:<14} {case['game_type']:<10} {case['seed']:>4} "
            f"{base_correct:>7}/1      {opt_correct:>7}/1      {rb[0]:>9.3f} {ro[0]:>8.3f}"
        )
    print()
    print(f"Correct plans summary: baseline={total_base}/{total}, optimized={total_opt}/{total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="simulation")
    p.add_argument("--phases", default=None)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--eval-runs", type=int, default=3)
    p.add_argument("--reject-threshold", type=float, default=0.5)
    p.add_argument("--trace-cases", type=int, default=3)
    p.add_argument("--validation-cases", type=int, default=5)
    p.add_argument("--anchor-game-type", default="coin")
    p.add_argument("--anchor-game-params", default=TRAIN_GAMES["coin"][2])
    p.add_argument("--anchor-seed", type=int, default=3)
    p.add_argument("--anchor-variant", default="deterministic")
    return p.parse_args()


def run_phase(args: argparse.Namespace, phase: str) -> None:
    optimizer = Optimizer(
        game="textworld_benchmark",
        target_phase=phase,
        three_step=True,
        verbose=True,
    )

    best_fn = None
    current_fn = None
    all_results: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}

    train_cases = build_case_list(TRAIN_GAMES, TRAIN_SEEDS)
    anchor = {
        "variant": args.anchor_variant,
        "game_type": args.anchor_game_type,
        "game_params": args.anchor_game_params,
        "seed": args.anchor_seed,
    }

    def get_baseline(case: dict[str, Any]) -> dict[str, float]:
        label = make_case_label(case)
        if label not in baselines:
            print(f"Computing training baseline for {label}...")
            avg, sr, steps, _, elapsed = eval_case(
                case, phase, None, args.iterations, args.max_depth, args.max_steps, args.eval_runs
            )
            baselines[label] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "correct_plan_rate": sr,
                "composite": sr,
            }
            best_scores[label] = sr
            print(
                f"  baseline: correct_plan_rate={sr:.4f}, solve_rate={sr:.0%}, "
                f"avg_returns={avg:.4f}, avg_steps={steps:.1f} ({elapsed:.1f}s)"
            )
        return baselines[label]

    for case in train_cases:
        get_baseline(case)

    cur_case = anchor
    if make_case_label(cur_case) not in baselines:
        get_baseline(cur_case)

    print("\nTraining pool uses non-hw2 settings:")
    print(f"  phase: {phase}")
    print(f"  coin params: {TRAIN_GAMES['coin']}")
    print(f"  mapreader params: {TRAIN_GAMES['mapreader']}")
    print(f"  seeds: {TRAIN_SEEDS}")

    for iteration in range(1, args.num_iters + 1):
        current_label = make_case_label(cur_case)
        baseline = baselines[current_label]
        iter_train_cases = select_balanced_top_cases(train_cases, baselines, args.trace_cases)
        validation_cases = select_balanced_top_cases(train_cases, baselines, args.validation_cases)
        validation_baseline = eval_case_set(
            validation_cases,
            phase,
            None,
            args.iterations,
            args.max_depth,
            args.max_steps,
            args.eval_runs,
        )
        reject_floor = validation_baseline["correct_plan_rate"] * args.reject_threshold

        print("\n" + "#" * 60)
        print(f"ITERATION {iteration}/{args.num_iters}, TRAIN-ANCHOR={current_label}")
        print(
            f"Anchor correct_plan_rate={baseline['correct_plan_rate']:.4f}, "
            f"validation_baseline={validation_baseline['correct_plan_rate']:.4f}, "
            f"reject_floor={reject_floor:.4f}"
        )
        print("Training trace cases:")
        for case in iter_train_cases:
            print(f"  - {make_case_label(case)}")
        print("Validation cases:")
        for case in validation_cases:
            print(f"  - {make_case_label(case)}")
        print("#" * 60)

        record_files, tool_sources, trace_summary = collect_training_traces(
            iter_train_cases, phase, current_fn, args.iterations, args.max_depth, args.max_steps
        )
        for label, status in trace_summary.items():
            print(f"Trace source [{label}]: {status}")
        for record_file in record_files:
            print(f"Trace: {record_file}")

        history = None
        if all_results:
            history_lines = []
            for r in all_results[-3:]:
                history_lines.append(
                    f"Iter {r['iteration']} anchor={r['label']} correct_plan_rate={r['composite']:.4f} "
                    f"solve_rate={r['solve_rate']:.0%} desc={r['description']}"
                )
            history = "\n".join(history_lines)

        t_opt = time.time()
        result = optimizer.run(
            record_files=record_files,
            tool_list=tool_sources,
            state_factory=lambda _c=cur_case: TextWorldBenchmark(
                _c["game_type"], _c["game_params"], seed=_c["seed"], variant=_c["variant"], max_steps=args.max_steps
            ).new_initial_state(),
            additional_context=history,
            session_tag=f"textworld_benchmark_generalization_{phase}_iter{iteration}",
        )
        print(f"Optimize: {time.time() - t_opt:.1f}s")

        rec = {
            "iteration": iteration,
            "label": current_label,
            "solve_rate": 0.0,
            "correct_plan_rate": 0.0,
            "avg_returns": baseline["avg_returns"],
            "avg_steps": args.max_steps,
            "composite": 0.0,
            "description": (result.get("parsed") or {}).get("description", ""),
            "adopted": False,
            "is_best": False,
        }

        fn = result.get("function")
        if fn is not None:
            eval_summary = eval_case_set(
                validation_cases,
                phase,
                fn,
                args.iterations,
                args.max_depth,
                args.max_steps,
                args.eval_runs,
            )
            comp = eval_summary["correct_plan_rate"]
            rec.update(
                {
                    "solve_rate": eval_summary["solve_rate"],
                    "correct_plan_rate": comp,
                    "avg_returns": eval_summary["avg_returns"],
                    "avg_steps": eval_summary["avg_steps"],
                    "composite": comp,
                }
            )
            print(
                f"Validation ({len(validation_cases)} cases x {args.eval_runs} runs): "
                f"correct_plan_rate={comp:.4f}, "
                f"solve_rate={eval_summary['solve_rate']:.0%}, "
                f"avg_returns={eval_summary['avg_returns']:.4f}, "
                f"avg_steps={eval_summary['avg_steps']:.1f} ({eval_summary['elapsed']:.1f}s)"
            )
            prev_best = max(best_scores[make_case_label(case)] for case in validation_cases)
            if comp > prev_best:
                print(f"NEW BEST on validation set (prev_correct_plan_rate={prev_best:.4f}) - adopting")
                for case in validation_cases:
                    label = make_case_label(case)
                    best_scores[label] = max(best_scores[label], comp)
                best_fn = fn
                current_fn = fn
                rec["adopted"] = True
                rec["is_best"] = True
                installed_path = result.get("installed_path")
                if installed_path:
                    saved_path = persist_best_heuristic(installed_path, phase)
                    print(f"Saved best heuristic to: {saved_path}")
            elif comp >= reject_floor:
                print(f"Accepted ({comp:.4f} >= {reject_floor:.4f})")
                current_fn = fn
                rec["adopted"] = True
            else:
                print(f"Rejected ({comp:.4f} < {reject_floor:.4f})")
                current_fn = best_fn
        else:
            print("Eval skipped due to smoke-test failure.")
            if result.get("error"):
                print(result["error"][:200])

        all_results.append(rec)
        if iteration < args.num_iters:
            hard_pool = select_balanced_top_cases(train_cases, baselines, max(args.validation_cases, 3))
            cur_case = random.choice(hard_pool)

    print("\nFinal hw2-style evaluation on held-out benchmark cases")
    final_hw2_style_eval(phase, best_fn, args.iterations, args.max_depth, args.max_steps)


def resolve_phases(args: argparse.Namespace) -> list[str]:
    if args.phases:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    else:
        phases = [args.phase]
    normalized: list[str] = []
    for phase in phases:
        if phase == "all":
            normalized.extend(VALID_PHASES)
            continue
        normalized.append(phase)
    deduped: list[str] = []
    for phase in normalized:
        if phase not in VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Valid phases: {VALID_PHASES}")
        if phase not in deduped:
            deduped.append(phase)
    return deduped


def main() -> None:
    args = parse_args()
    phases = resolve_phases(args)
    for idx, phase in enumerate(phases, start=1):
        random.seed(3)
        print("\n" + "=" * 72)
        print(f"PHASE RUN {idx}/{len(phases)}: {phase}")
        print("=" * 72)
        run_phase(args, phase)


if __name__ == "__main__":
    main()
