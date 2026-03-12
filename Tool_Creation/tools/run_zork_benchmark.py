"""
Self-evolving LLM runner for Zork benchmark.

Uses MCTS with LLM-generated heuristics to learn to play a symbolic
pure-Python Zork I game.  The self-evolving loop collects gameplay
traces, asks the LLM to improve the MCTS heuristic, and validates
improvements on held-out configurations.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from LLM.optimizer import Optimizer
from mcts import MCTSEngine
from mcts.games.zork import Zork

SEEDS = [0, 1, 2]
ENV_VARIANTS = ["deterministic", "stochastic", "punishment"]

# num_rooms is the primary difficulty axis (like TextWorld's numLocations).
# ALL_ROOMS is ordered surface→underground, so more rooms = harder game.
# Total available: 22 rooms.

# TEST: held-out evaluation — larger maps the heuristic must generalize to
TEST_GAMES = {
    "num_rooms": [12, 16, 18, 20, 22],
}

# TRAIN: smaller maps the heuristic learns from
TRAIN_GAMES = {
    "num_rooms": [8, 10, 12, 14],
}

TRAIN_SEEDS = [3, 4, 5]
GAME_NAME = "zork"
VALID_PHASES = ["selection", "expansion", "simulation", "backpropagation"]
CANDIDATE_TIMEOUT = 120  # seconds per candidate evaluation


def make_case_label(case: dict[str, Any]) -> str:
    return f"{case['variant']}|num_rooms={case['num_rooms']}|seed={case['seed']}"


def best_heuristic_path(phase: str) -> Path:
    return _ROOT / "MCTS_tools" / phase / f"{GAME_NAME}_opt_{phase}_heuristic.py"


def persist_best_heuristic(source_code: str, phase: str) -> Path:
    """Write the best heuristic source code to a persistent file."""
    dst = best_heuristic_path(phase)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(source_code)
    return dst


def make_engine(
    case: dict[str, Any], iterations: int, max_depth: int, max_steps: int, logging: bool
) -> MCTSEngine:
    game = Zork(
        num_rooms=case["num_rooms"],
        seed=case["seed"],
        variant=case["variant"],
        max_steps=max_steps,
    )
    return MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=logging)


def build_case_list(
    games: dict[str, list[int]], seeds: list[int]
) -> list[dict[str, Any]]:
    cases = []
    for variant in ENV_VARIANTS:
        for num_rooms in games["num_rooms"]:
            for seed in seeds:
                cases.append(
                    {
                        "variant": variant,
                        "num_rooms": num_rooms,
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
    parallel: int = 1,
) -> dict[str, Any]:
    """Evaluate a set of cases. When parallel > 1, uses a thread pool."""

    def _eval_one(case):
        avg_ret, solve_rate, avg_steps, results, elapsed = eval_case(
            case, phase, fn, iterations, max_depth, max_steps, runs
        )
        return {
            "label": make_case_label(case),
            "avg_returns": avg_ret,
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
            "correct_plan_rate": correct_plan_rate(results),
            "elapsed": elapsed,
            "_results": results,
        }

    if parallel > 1 and len(eval_cases) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            items = list(pool.map(_eval_one, eval_cases))
    else:
        items = [_eval_one(case) for case in eval_cases]

    per_case: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for item in items:
        all_results.extend(item.pop("_results"))
        per_case.append(item)

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

    # Ensure diversity across room counts and variants
    room_sizes = sorted(set(c["num_rooms"] for c in cases))
    if len(room_sizes) >= 2:
        add_first_matching(lambda case: case["num_rooms"] == room_sizes[0])  # smallest
        add_first_matching(lambda case: case["num_rooms"] == room_sizes[-1])  # largest
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


def final_eval(
    phase: str,
    best_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    games: dict[str, list[int]] | None = None,
    seeds: list[int] | None = None,
) -> None:
    cases = build_case_list(games or TEST_GAMES, seeds or SEEDS)
    total_base = 0
    total_opt = 0
    total = 0
    print(
        f"{'Variant':<14} {'Rooms':>5} {'Seed':>4} {'Base Correct':>12} "
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
            f"{case['variant']:<14} {case['num_rooms']:>5} {case['seed']:>4} "
            f"{base_correct:>7}/1      {opt_correct:>7}/1      {rb[0]:>9.3f} {ro[0]:>8.3f}"
        )
    print()
    print(f"Correct plans summary: baseline={total_base}/{total}, optimized={total_opt}/{total}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-evolving MCTS runner for Zork")
    p.add_argument("--phase", default="simulation")
    p.add_argument("--phases", default=None)
    p.add_argument("--num-iters", type=int, default=3)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=50)
    p.add_argument("--eval-runs", type=int, default=3)
    p.add_argument("--reject-threshold", type=float, default=0.5)
    p.add_argument("--trace-cases", type=int, default=3)
    p.add_argument("--validation-cases", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--anchor-num-rooms", type=int, default=TRAIN_GAMES["num_rooms"][1])
    p.add_argument("--anchor-seed", type=int, default=3)
    p.add_argument("--anchor-variant", default="deterministic")
    # Parallelization options
    p.add_argument("--num-candidates", type=int, default=1,
                   help="Number of parallel LLM candidates per iteration (use >1 with multiple API keys)")
    p.add_argument("--parallel-eval", type=int, default=1,
                   help="Thread pool size for parallel case evaluation")
    p.add_argument("--no-generalization", action="store_true",
                   help="Skip held-out test games; final eval uses training games only")
    return p.parse_args()


def run_phase(args: argparse.Namespace, phase: str) -> None:
    optimizer = Optimizer(
        game="zork",
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
        "num_rooms": args.anchor_num_rooms,
        "seed": args.anchor_seed,
    }

    def get_baseline(case: dict[str, Any]) -> dict[str, float]:
        label = make_case_label(case)
        if label not in baselines:
            avg, sr, steps, results, elapsed = eval_case(
                case, phase, None, args.iterations, args.max_depth, args.max_steps, args.eval_runs
            )
            cpr = correct_plan_rate(results)
            baselines[label] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "correct_plan_rate": cpr,
            }
            best_scores[label] = avg
            print(
                f"  baseline [{label}]: avg_returns={avg:.4f}, solve_rate={sr:.0%}, "
                f"correct_plan_rate={cpr:.4f}, avg_steps={steps:.1f} ({elapsed:.1f}s)"
            )
        return baselines[label]

    # Compute all baselines (parallel if requested)
    print(f"Computing baselines for {len(train_cases)} training cases (parallel={args.parallel_eval})...")
    if args.parallel_eval > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_eval) as pool:
            list(pool.map(get_baseline, train_cases))
    else:
        for case in train_cases:
            get_baseline(case)

    cur_case = anchor
    if make_case_label(cur_case) not in baselines:
        get_baseline(cur_case)

    print("\nTraining pool configuration:")
    print(f"  phase: {phase}")
    print(f"  num_rooms: {TRAIN_GAMES['num_rooms']}")
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
            parallel=args.parallel_eval,
        )
        reject_floor = validation_baseline["avg_returns"] * args.reject_threshold

        print("\n" + "#" * 60)
        print(f"ITERATION {iteration}/{args.num_iters}, TRAIN-ANCHOR={current_label}")
        print(
            f"Anchor avg_returns={baseline['avg_returns']:.4f}, "
            f"validation_baseline_avg_returns={validation_baseline['avg_returns']:.4f}, "
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
                line = (
                    f"Iter {r['iteration']} anchor={r['label']} correct_plan_rate={r['correct_plan_rate']:.4f} "
                    f"solve_rate={r['solve_rate']:.0%} avg_returns={r['avg_returns']:.4f} "
                    f"desc={r['description']}"
                )
                if r.get("timed_out"):
                    line += (
                        f" WARNING: {r['timeout_count']}/{r['total_candidates']} "
                        f"candidates TIMED OUT (took >{CANDIDATE_TIMEOUT}s each). "
                        f"The generated function was too slow — avoid complex loops, "
                        f"deep recursion, or expensive computations in the heuristic."
                    )
                history_lines.append(line)
            history = "\n".join(history_lines)

        t_opt = time.time()
        state_factory = lambda _c=cur_case: Zork(
            num_rooms=_c["num_rooms"],
            seed=_c["seed"],
            variant=_c["variant"],
            max_steps=args.max_steps,
        ).new_initial_state()
        session_tag_base = f"zork_benchmark_{phase}_iter{iteration}"

        # Generate candidate heuristics sequentially (single API key)
        candidate_results = []
        for ci in range(args.num_candidates):
            result = optimizer.run(
                record_files=record_files,
                tool_list=tool_sources,
                state_factory=state_factory,
                additional_context=history,
                session_tag=f"{session_tag_base}_c{ci}",
            )
            # Snapshot the installed file content before next candidate overwrites it
            ip = result.get("installed_path")
            if ip and Path(ip).exists():
                result["_source_snapshot"] = Path(ip).read_text()
            candidate_results.append(result)
        print(f"Optimize: {time.time() - t_opt:.1f}s ({len(candidate_results)} candidates)")

        # Evaluate all valid candidates and pick the best
        best_candidate = None
        best_comp = -1.0
        best_eval_summary = None
        timeout_count = 0

        for ci, result in enumerate(candidate_results):
            fn = result.get("function")
            if fn is None:
                err = result.get("error", "unknown")
                print(f"  Candidate {ci}: smoke-test failed ({str(err)[:100]})")
                continue
            # Run eval with a timeout to avoid hanging on bad heuristics
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as eval_pool:
                    future = eval_pool.submit(
                        eval_case_set,
                        validation_cases,
                        phase,
                        fn,
                        args.iterations,
                        args.max_depth,
                        args.max_steps,
                        args.eval_runs,
                        args.parallel_eval,
                    )
                    eval_summary = future.result(timeout=CANDIDATE_TIMEOUT)
            except concurrent.futures.TimeoutError:
                timeout_count += 1
                print(f"  Candidate {ci}: TIMED OUT after {CANDIDATE_TIMEOUT}s (likely infinite loop) - skipped")
                continue
            except Exception as exc:
                print(f"  Candidate {ci}: evaluation error ({str(exc)[:100]}) - skipped")
                continue
            comp = eval_summary["avg_returns"]
            desc = (result.get("parsed") or {}).get("description", "")[:60]
            print(
                f"  Candidate {ci}: avg_returns={comp:.4f}, "
                f"solve_rate={eval_summary['solve_rate']:.0%}, "
                f"correct_plan_rate={eval_summary['correct_plan_rate']:.4f} [{desc}]"
            )
            if comp > best_comp:
                best_comp = comp
                best_candidate = result
                best_eval_summary = eval_summary

        rec = {
            "iteration": iteration,
            "label": current_label,
            "solve_rate": 0.0,
            "correct_plan_rate": 0.0,
            "avg_returns": baseline["avg_returns"],
            "avg_steps": args.max_steps,
            "description": "",
            "adopted": False,
            "is_best": False,
            "timed_out": timeout_count > 0,
            "timeout_count": timeout_count,
            "total_candidates": args.num_candidates,
        }

        if best_candidate is not None and best_eval_summary is not None:
            fn = best_candidate["function"]
            comp = best_comp
            rec.update(
                {
                    "solve_rate": best_eval_summary["solve_rate"],
                    "correct_plan_rate": best_eval_summary["correct_plan_rate"],
                    "avg_returns": comp,
                    "avg_steps": best_eval_summary["avg_steps"],
                    "description": (best_candidate.get("parsed") or {}).get("description", ""),
                }
            )
            print(
                f"Best candidate: avg_returns={comp:.4f}, "
                f"solve_rate={best_eval_summary['solve_rate']:.0%}, "
                f"correct_plan_rate={best_eval_summary['correct_plan_rate']:.4f}, "
                f"avg_steps={best_eval_summary['avg_steps']:.1f} ({best_eval_summary['elapsed']:.1f}s)"
            )
            prev_best = max(best_scores[make_case_label(case)] for case in validation_cases)
            if comp > prev_best:
                print(f"NEW BEST on validation set (prev_avg_returns={prev_best:.4f}) - adopting")
                for case in validation_cases:
                    label = make_case_label(case)
                    best_scores[label] = max(best_scores[label], comp)
                best_fn = fn
                current_fn = fn
                rec["adopted"] = True
                rec["is_best"] = True
                snapshot = best_candidate.get("_source_snapshot")
                if snapshot:
                    saved_path = persist_best_heuristic(snapshot, phase)
                    print(f"Saved best heuristic to: {saved_path}")
            elif comp > reject_floor:
                print(f"Accepted ({comp:.4f} > {reject_floor:.4f})")
                current_fn = fn
                rec["adopted"] = True
            else:
                print(f"Rejected ({comp:.4f} < {reject_floor:.4f})")
                current_fn = best_fn
        else:
            print("No valid candidates produced this iteration.")

        all_results.append(rec)
        if iteration < args.num_iters:
            hard_pool = select_balanced_top_cases(train_cases, baselines, max(args.validation_cases, 3))
            cur_case = random.choice(hard_pool)

    # Reload best_fn from the persisted file to avoid stale in-memory refs
    # (subsequent LLM candidates overwrite simulation.py, invalidating the
    # module globals captured by the old function object).
    saved = best_heuristic_path(phase)
    if saved.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(saved.stem, str(saved))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func_name = f"default_{phase}"
        best_fn = getattr(mod, func_name, best_fn)
        print(f"Reloaded best heuristic from {saved}")

    if args.no_generalization:
        print("\nFinal evaluation on TRAINING configurations (--no-generalization)")
        final_eval(phase, best_fn, args.iterations, args.max_depth, args.max_steps,
                   games=TRAIN_GAMES, seeds=TRAIN_SEEDS)
    else:
        print("\nFinal evaluation on held-out test configurations")
        final_eval(phase, best_fn, args.iterations, args.max_depth, args.max_steps)


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
