"""
Self-evolving LLM runner for the symbolic hw2-style TextWorld benchmark.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from LLM.optimizer import Optimizer
from mcts import MCTSEngine
from mcts.games import TextWorldBenchmark


SEEDS = [0, 1, 2]
ENV_VARIANTS = ["deterministic", "stochastic", "punishment"]
GAMES = {
    "coin": [
        "numLocations=5,includeDoors=1,numDistractorItems=0",
        "numLocations=6,includeDoors=1,numDistractorItems=0",
        "numLocations=7,includeDoors=1,numDistractorItems=0",
        "numLocations=10,includeDoors=1,numDistractorItems=0",
    ],
    "mapreader": [
        "numLocations=5,maxDistanceApart=3,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=8,maxDistanceApart=4,includeDoors=0,maxDistractorItemsPerLocation=0",
        "numLocations=11,maxDistanceApart=5,includeDoors=0,maxDistractorItemsPerLocation=0",
    ],
}


def composite_score(solve_rate: float, avg_returns: float) -> float:
    return 0.7 * solve_rate + 0.3 * avg_returns


def make_case_label(case: dict[str, Any]) -> str:
    return f"{case['variant']}|{case['game_type']}|{case['game_params']}|seed={case['seed']}"


def make_engine(case: dict[str, Any], iterations: int, max_depth: int, max_steps: int, logging: bool) -> MCTSEngine:
    game = TextWorldBenchmark(
        game_type=case["game_type"],
        game_params=case["game_params"],
        seed=case["seed"],
        variant=case["variant"],
        max_steps=max_steps,
    )
    return MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=logging)


def build_case_list() -> list[dict[str, Any]]:
    cases = []
    for variant in ENV_VARIANTS:
        for game_type, params_list in GAMES.items():
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


def correct_plan_rate(results: list[dict[str, Any]]) -> float:
    return sum(1 for r in results if is_correct_plan(r)) / max(1, len(results))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def select_hard_cases(
    cases: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    preferred: dict[str, Any] | None,
    count: int,
) -> list[dict[str, Any]]:
    count = max(1, count)
    ranked = sorted(
        cases,
        key=lambda case: (
            baselines[make_case_label(case)]["correct_plan_rate"],
            baselines[make_case_label(case)]["avg_returns"],
            -baselines[make_case_label(case)]["avg_steps"],
            make_case_label(case),
        ),
    )
    selected: list[dict[str, Any]] = []
    if preferred is not None:
        selected.append(preferred)
    for case in ranked:
        if len(selected) >= count:
            break
        if any(make_case_label(case) == make_case_label(existing) for existing in selected):
            continue
        selected.append(case)
    return selected[:count]


def select_single_hard_case(
    cases: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    preferred: dict[str, Any] | None,
) -> dict[str, Any]:
    failed = [
        case
        for case in cases
        if baselines[make_case_label(case)]["correct_plan_rate"] < 1.0
    ]
    pool = failed if failed else cases
    ranked = sorted(
        pool,
        key=lambda case: (
            baselines[make_case_label(case)]["correct_plan_rate"],
            baselines[make_case_label(case)]["avg_returns"],
            -baselines[make_case_label(case)]["avg_steps"],
            make_case_label(case),
        ),
    )
    if preferred is not None:
        preferred_label = make_case_label(preferred)
        for case in ranked:
            if make_case_label(case) == preferred_label:
                return case
    return ranked[0]


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

    # Force coverage across the two game families first.
    add_first_matching(lambda case: case["game_type"] == "coin")
    add_first_matching(lambda case: case["game_type"] == "mapreader")

    # Then prefer the hardest non-deterministic case to expose noisy / punitive failures.
    add_first_matching(lambda case: case["variant"] in {"stochastic", "punishment"})

    # Fill remaining slots by global hardness.
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


def final_hw2_style_eval(
    phase: str,
    best_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
) -> None:
    cases = build_case_list()
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
    p.add_argument("--num-iters", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--eval-runs", type=int, default=3)
    p.add_argument("--reject-threshold", type=float, default=0.5)
    p.add_argument("--anchor-game-type", default="coin")
    p.add_argument("--anchor-game-params", default=GAMES["coin"][0])
    p.add_argument("--anchor-seed", type=int, default=0)
    p.add_argument("--anchor-variant", default="deterministic")
    p.add_argument("--single-hard-trace", action="store_true")
    p.add_argument("--balanced-top-traces", action="store_true")
    p.add_argument("--trace-cases", type=int, default=3)
    p.add_argument("--validation-cases", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(3)

    optimizer = Optimizer(
        game="textworld_benchmark",
        target_phase=args.phase,
        three_step=True,
        verbose=True,
    )

    best_fn = None
    current_fn = None
    all_results: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}

    cases = build_case_list()
    anchor = {
        "variant": args.anchor_variant,
        "game_type": args.anchor_game_type,
        "game_params": args.anchor_game_params,
        "seed": args.anchor_seed,
    }

    def get_baseline(case: dict[str, Any]) -> dict[str, float]:
        label = make_case_label(case)
        if label not in baselines:
            print(f"Computing baseline for {label}...")
            avg, sr, steps, _, elapsed = eval_case(
                case, args.phase, None, args.iterations, args.max_depth, args.max_steps, args.eval_runs
            )
            # In the hw2-style benchmark, use correct-plan rate as the
            # primary acceptance score so the training objective matches
            # the final evaluation objective.
            comp = sr
            baselines[label] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "correct_plan_rate": sr,
                "composite": comp,
            }
            best_scores[label] = comp
            print(
                f"  baseline: correct_plan_rate={comp:.4f}, solve_rate={sr:.0%}, "
                f"avg_returns={avg:.4f}, avg_steps={steps:.1f} ({elapsed:.1f}s)"
            )
        return baselines[label]

    cur_case = anchor
    for case in cases:
        get_baseline(case)

    for iteration in range(1, args.num_iters + 1):
        if args.single_hard_trace:
            train_cases = [select_single_hard_case(cases, baselines, cur_case)]
        elif args.balanced_top_traces:
            train_cases = select_balanced_top_cases(cases, baselines, args.trace_cases)
        else:
            train_cases = select_hard_cases(cases, baselines, cur_case, args.trace_cases)
        validation_cases = select_hard_cases(cases, baselines, cur_case, args.validation_cases)
        cur_label = make_case_label(cur_case)
        baseline = baselines[cur_label]
        validation_baseline = eval_case_set(
            validation_cases,
            args.phase,
            None,
            args.iterations,
            args.max_depth,
            args.max_steps,
            args.eval_runs,
        )
        reject_floor = validation_baseline["correct_plan_rate"] * args.reject_threshold

        print("\n" + "#" * 60)
        print(f"ITERATION {iteration}/{args.num_iters}, CASE={cur_label}")
        print(
            f"Anchor correct_plan_rate={baseline['composite']:.4f}, "
            f"validation_baseline={validation_baseline['correct_plan_rate']:.4f}, "
            f"reject_floor={reject_floor:.4f}"
        )
        if args.single_hard_trace:
            train_label = make_case_label(train_cases[0])
            train_base = baselines[train_label]
            failed_count = sum(
                1 for case in cases if baselines[make_case_label(case)]["correct_plan_rate"] < 1.0
            )
            if failed_count:
                print(
                    f"Single-trace mode: using hardest failed case "
                    f"({train_label}, correct_plan_rate={train_base['correct_plan_rate']:.4f}, "
                    f"avg_returns={train_base['avg_returns']:.4f}, avg_steps={train_base['avg_steps']:.1f})"
                )
            else:
                print(
                    f"Single-trace mode: no failed cases available; falling back to hardest overall case "
                    f"({train_label}, correct_plan_rate={train_base['correct_plan_rate']:.4f}, "
                    f"avg_returns={train_base['avg_returns']:.4f}, avg_steps={train_base['avg_steps']:.1f})"
                )
        elif args.balanced_top_traces:
            print(
                f"Balanced-trace mode: selecting {len(train_cases)} hard traces with game-family and "
                f"variant coverage"
            )
        print("Trace cases:")
        for case in train_cases:
            print(f"  - {make_case_label(case)}")
        print("Validation cases:")
        for case in validation_cases:
            print(f"  - {make_case_label(case)}")
        print("#" * 60)

        record_files, tool_sources, trace_summary = collect_training_traces(
            train_cases, args.phase, current_fn, args.iterations, args.max_depth, args.max_steps
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
                    f"Iter {r['iteration']} case={r['label']} correct_plan_rate={r['composite']:.4f} "
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
            session_tag=f"textworld_benchmark_{args.phase}_iter{iteration}",
        )
        print(f"Optimize: {time.time() - t_opt:.1f}s")

        rec = {
            "iteration": iteration,
            "label": cur_label,
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
                args.phase,
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
            for item in eval_summary["per_case"]:
                print(
                    f"  {item['label']}: correct_plan_rate={item['correct_plan_rate']:.4f}, "
                    f"solve_rate={item['solve_rate']:.0%}, avg_returns={item['avg_returns']:.4f}, "
                    f"avg_steps={item['avg_steps']:.1f}"
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
            hard_pool = select_hard_cases(cases, baselines, None, max(args.validation_cases, args.trace_cases, 1))
            cur_case = random.choice(hard_pool)

    print("\nFinal hw2-style evaluation")
    final_hw2_style_eval(args.phase, best_fn, args.iterations, args.max_depth, args.max_steps)


if __name__ == "__main__":
    main()



