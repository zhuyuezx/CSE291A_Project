"""
Self-evolving LLM runner for ScienceWorld benchmark.

Tests how the MCTS agent performs across ScienceWorld tasks, using the
LLM optimizer to iteratively improve MCTS heuristics (simulation,
expansion, etc.).

Task structure comes from the ScienceWorld curriculum:
  30 task types across 10 topics (Matter, Measurement, Electricity,
  Classification, Biology, Chemistry, Lifespan, Life Stages, Forces,
  Genetics).  Each task type has multiple variations.

Training traces and validation use a subset of easier tasks.
Final evaluation runs on a held-out set of harder tasks.

--------------------------------------------------------------------------------
HOW THE TEST PROCESSES THE GAME (MCTS TRAINING FLOW)
--------------------------------------------------------------------------------
The script does NOT run "hundreds of games" first. It runs games in stages:

1. BASELINE (once per training case)
   For each training case (e.g. task 1-1 var 0, 1-1 var 1, ...), we run
   *eval_runs* full games (default 1). Each "game" is one ScienceWorld
   episode: from reset until done or max_steps. For each in-game step we
   run MCTS (iterations per move, e.g. 30) to pick one action. So total
   games in this stage = (# train cases) × eval_runs (e.g. 8 × 1 = 8 games).

2. PER-ITERATION (num_iters times, e.g. 3)
   a) TRACE COLLECTION: We run *trace_cases* games (e.g. 3) with logging
      on. Same as above: each game is one episode; MCTS runs every step.
      So 3 games per iteration, each producing one JSON trace file.
   b) LLM OPTIMIZE: The optimizer reads those traces and proposes a new
      heuristic (e.g. simulation policy). No games run here.
   c) VALIDATION: We run *validation_cases* games (e.g. 4) using the new
      heuristic. Again 4 episodes, MCTS every step.
   Total games per iteration = trace_cases + validation_cases (e.g. 7).
   Over 3 iterations → 21 extra games.

3. FINAL EVALUATION
   For each test case we run eval_runs games with baseline (no custom
   heuristic) and eval_runs games with the best heuristic. So 2 ×
   (# test cases) × eval_runs games (e.g. 2 × 8 × 1 = 16 games).

TOTAL GAMES (defaults): 8 + 3×7 + 16 = 45 full episodes. Each episode
can be up to max_steps (e.g. 50) in-game steps, and each step runs
MCTS (e.g. 30 iterations × rollout depth 15). So the bottleneck is
MCTS work per step, not "hundreds of games" up front.
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
from mcts.games.scienceworld import (
    ScienceWorldGame,
    ScienceWorldState,
    TASK_ID_TO_NAME,
    get_simplifications,
    resolve_task_name,
    _SharedEnvPool,
)


GAME_NAME = "scienceworld"
VALID_PHASES = ["selection", "expansion", "simulation", "backpropagation"]

SIMPLIFICATION_PRESET = "easy"

TRAIN_TASK_IDS = ["1-1", "1-2", "4-2", "7-1"]
TRAIN_VARIATIONS = [0, 1]

TEST_TASK_IDS = [
    "1-3", "2-1", "4-1", "4-3",
    "5-1", "6-1", "6-2", "7-2",
]
TEST_VARIATIONS = [0]

ALL_TASK_IDS = list(TASK_ID_TO_NAME.keys())


# ── Helpers ───────────────────────────────────────────────────────────

def make_case_label(case: dict[str, Any]) -> str:
    return f"{case['task_id']}|{case['task_name']}|var={case['variation']}"


def best_heuristic_path(phase: str) -> Path:
    return _ROOT / "MCTS_tools" / phase / f"{GAME_NAME}_opt_{phase}_heuristic.py"


def persist_best_heuristic(installed_path: str | Path, phase: str) -> Path:
    src = Path(installed_path)
    dst = best_heuristic_path(phase)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def make_engine(
    case: dict[str, Any],
    iterations: int,
    max_depth: int,
    max_steps: int,
    max_actions: int,
    logging: bool,
) -> MCTSEngine:
    simplifications = get_simplifications(case["task_id"], SIMPLIFICATION_PRESET)
    game = ScienceWorldGame(
        task_name=case["task_name"],
        variation=case["variation"],
        simplifications=simplifications,
        max_steps=max_steps,
        max_actions=max_actions,
    )
    return MCTSEngine(
        game,
        iterations=iterations,
        max_rollout_depth=max_depth,
        logging=logging,
    )


def build_case_list(
    task_ids: list[str], variations: list[int],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_name = resolve_task_name(task_id)
        for var in variations:
            cases.append({
                "task_id": task_id,
                "task_name": task_name,
                "variation": var,
            })
    return cases


def is_correct_plan(result: dict[str, Any]) -> bool:
    returns = result.get("returns", [0.0])
    ret = returns[0] if isinstance(returns, list) else float(returns)
    return bool(result.get("solved")) and ret >= 1.0


def correct_plan_rate(results: list[dict[str, Any]]) -> float:
    return sum(1 for r in results if is_correct_plan(r)) / max(1, len(results))


def avg_normalized_score(results: list[dict[str, Any]]) -> float:
    scores = []
    for r in results:
        returns = r.get("returns", [0.0])
        ret = returns[0] if isinstance(returns, list) else float(returns)
        scores.append(ret)
    return sum(scores) / max(1, len(scores))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


# ── Evaluation ────────────────────────────────────────────────────────

def eval_case(
    case: dict[str, Any],
    phase: str,
    fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    max_actions: int,
    runs: int,
) -> tuple[float, float, float, list[dict[str, Any]], float]:
    label = make_case_label(case)
    t0 = time.time()
    results: list[dict[str, Any]] = []
    for run_idx in range(runs):
        if runs > 1:
            print(f"      Episode {run_idx + 1}/{runs} ({label})...", flush=True)
        engine = make_engine(
            case, iterations, max_depth, max_steps, max_actions, logging=False
        )
        if fn is not None:
            engine.set_tool(phase, fn)
        t_ep = time.time()
        results.append(engine.play_game(verbose=False))
        ep_elapsed = time.time() - t_ep
        print(f"      Episode {run_idx + 1}/{runs} done in {ep_elapsed:.1f}s", flush=True)
    elapsed = time.time() - t0
    avg_ret = mean([r["returns"][0] for r in results])
    solve_rate = sum(1 for r in results if r["solved"]) / runs
    avg_steps = mean([float(r["steps"]) for r in results])
    return avg_ret, solve_rate, avg_steps, results, elapsed


def eval_case_set(
    eval_cases: list[dict[str, Any]],
    phase: str,
    fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    max_actions: int,
    runs: int,
    progress_prefix: str = "",
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    n_cases = len(eval_cases)
    for case_idx, case in enumerate(eval_cases):
        if progress_prefix and n_cases > 1:
            print(f"  {progress_prefix} Case {case_idx + 1}/{n_cases}: {make_case_label(case)}", flush=True)
        avg_ret, solve_rate, avg_steps, results, elapsed = eval_case(
            case, phase, fn, iterations, max_depth, max_steps, max_actions, runs,
        )
        per_case.append({
            "label": make_case_label(case),
            "avg_returns": avg_ret,
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
            "correct_plan_rate": correct_plan_rate(results),
            "avg_normalized_score": avg_normalized_score(results),
            "elapsed": elapsed,
        })
        all_results.extend(results)
    return {
        "correct_plan_rate": correct_plan_rate(all_results),
        "avg_normalized_score": avg_normalized_score(all_results),
        "solve_rate": mean([item["solve_rate"] for item in per_case]),
        "avg_returns": mean([item["avg_returns"] for item in per_case]),
        "avg_steps": mean([item["avg_steps"] for item in per_case]),
        "elapsed": sum(item["elapsed"] for item in per_case),
        "per_case": per_case,
    }


# ── Case ranking / selection ─────────────────────────────────────────

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

    topic_ids = sorted({tid.split("-")[0] for tid in TRAIN_TASK_IDS})
    for topic in topic_ids:
        add_first_matching(lambda c, t=topic: c["task_id"].startswith(t + "-"))

    for case in ranked:
        if len(selected) >= count:
            break
        label = make_case_label(case)
        if label in seen:
            continue
        selected.append(case)
        seen.add(label)

    return selected[:count]


# ── Training traces ──────────────────────────────────────────────────

def collect_training_traces(
    train_cases: list[dict[str, Any]],
    phase: str,
    current_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    max_actions: int,
    epoch: int = 0,
) -> tuple[list[str], list[str], dict[str, str]]:
    record_files: list[str] = []
    tool_sources: list[str] = []
    summary: dict[str, str] = {}
    n_cases = len(train_cases)
    epoch_tag = f"[Epoch {epoch}] " if epoch else ""
    for trace_idx, case in enumerate(train_cases):
        label = make_case_label(case)
        print(f"  {epoch_tag}Trace game {trace_idx + 1}/{n_cases}: {label} ...", flush=True)
        t_play = time.time()
        engine = make_engine(
            case, iterations, max_depth, max_steps, max_actions, logging=True
        )
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
            raise FileNotFoundError(
                f"Trace file was reported but does not exist: {trace_path}"
            )
        if not tool_sources:
            tool_sources = engine.get_tool_source()
        record_files.append(play_trace)
        ret = play_result["returns"][0]
        elapsed_play = time.time() - t_play
        summary[label] = (
            f"{'SOLVED' if play_result['solved'] else 'UNSOLVED'} "
            f"in {play_result['steps']} steps "
            f"returns={ret:.4f} ({elapsed_play:.1f}s)"
        )
        print(f"  {epoch_tag}  -> done in {elapsed_play:.1f}s", flush=True)
    return record_files, tool_sources, summary


# ── Final evaluation ─────────────────────────────────────────────────

def final_eval(
    phase: str,
    best_fn: Callable | None,
    iterations: int,
    max_depth: int,
    max_steps: int,
    max_actions: int,
    test_task_ids: list[str] | None = None,
    test_variations: list[int] | None = None,
    report_average: bool = False,
) -> None:
    task_ids = test_task_ids if test_task_ids is not None else TEST_TASK_IDS
    variations = test_variations if test_variations is not None else TEST_VARIATIONS
    cases = build_case_list(task_ids, variations)
    total_base = 0
    total_opt = 0
    total = 0
    base_scores: list[float] = []
    opt_scores: list[float] = []
    n_test = len(cases)
    print(f"\n[Final Eval] {n_test} test case(s)", flush=True)
    print(
        f"  {'Task ID':<8} {'Task Name':<40} {'Var':>3} "
        f"{'Base Score':>10} {'Opt Score':>10} "
        f"{'Base Solved':>11} {'Opt Solved':>11}"
    )
    print("  " + "-" * 98)
    for eval_idx, case in enumerate(cases):
        if n_test > 1:
            print(f"  [Final Eval] Test case {eval_idx + 1}/{n_test}: {case['task_id']}|{case['task_name']}|var={case['variation']}", flush=True)
        rb = eval_case(
            case, phase, None, iterations, max_depth, max_steps, max_actions, runs=1
        )
        ro = eval_case(
            case, phase, best_fn, iterations, max_depth, max_steps, max_actions, runs=1
        )
        base_result = rb[3][0]
        opt_result = ro[3][0]
        base_correct = int(is_correct_plan(base_result))
        opt_correct = int(is_correct_plan(opt_result))
        base_score = base_result["returns"][0]
        opt_score = opt_result["returns"][0]
        base_scores.append(base_score)
        opt_scores.append(opt_score)
        total_base += base_correct
        total_opt += opt_correct
        total += 1
        print(
            f"  {case['task_id']:<8} {case['task_name']:<40} {case['variation']:>3} "
            f"{base_score:>10.4f} {opt_score:>10.4f} "
            f"{base_correct:>7}/1     {opt_correct:>7}/1"
        )
    print()
    print(
        f"Correct plans summary: baseline={total_base}/{total}, "
        f"optimized={total_opt}/{total}"
    )
    if report_average and base_scores and opt_scores:
        avg_base = sum(base_scores) / len(base_scores)
        avg_opt = sum(opt_scores) / len(opt_scores)
        print(
            f"Average baseline score (normalized): {avg_base:.4f}\n"
            f"Average optimized score (normalized): {avg_opt:.4f}"
        )


# ── CLI / orchestration ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-evolving MCTS benchmark for ScienceWorld",
    )
    p.add_argument("--phase", default="simulation")
    p.add_argument("--phases", default=None,
                   help="Comma-separated phases, or 'all'")
    p.add_argument("--num-iters", type=int, default=3,
                   help="LLM optimization iterations per phase")
    p.add_argument("--iterations", type=int, default=30,
                   help="MCTS iterations per move")
    p.add_argument("--max-depth", type=int, default=15,
                   help="Max MCTS rollout depth")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Max env steps per episode")
    p.add_argument("--max-actions", type=int, default=25,
                   help="Cap valid actions per step (0=no cap). Reduces MCTS branching; ScienceWorld can return 50-200+ actions.")
    p.add_argument("--eval-runs", type=int, default=1,
                   help="Episodes per case for evaluation")
    p.add_argument("--reject-threshold", type=float, default=0.5)
    p.add_argument("--trace-cases", type=int, default=3)
    p.add_argument("--validation-cases", type=int, default=4)
    p.add_argument("--simplification-preset", default="easy",
                   choices=["easy", "none"],
                   help="ScienceWorld simplification preset")
    p.add_argument("--train-task-ids", default=None,
                   help="Comma-separated task IDs for training (overrides default)")
    p.add_argument("--test-task-ids", default=None,
                   help="Comma-separated task IDs for testing (overrides default)")
    p.add_argument("--quick", action="store_true",
                   help="Minimal hyperparameters to verify MCTS+ScienceWorld in ~1–2 min (1 train case, 1 test case, few MCTS iters).")
    p.add_argument("--formal", action="store_true",
                   help="Formal evaluation: max_steps=100; final eval runs task 1-1 with 5 variations (seeds 0..4) and reports average score.")
    return p.parse_args()


def apply_quick_hyperparameters(args: argparse.Namespace) -> None:
    """Override args with small values for fast smoke-test / verification."""
    args.train_task_ids = "1-1"
    args.test_task_ids = "1-3"
    args.num_iters = 1
    args.iterations = 5
    args.max_depth = 3
    args.max_steps = 10
    args.max_actions = 10
    args.eval_runs = 1
    args.trace_cases = 1
    args.validation_cases = 1
    # Single variation so we get 1 train case and 1 test case
    args._train_variations_override = [0]
    args._test_variations_override = [0]


# Formal eval: task 1-1, 5 variations as "seeds", max_steps=100
FORMAL_EVAL_TASK_ID = "1-1"
FORMAL_EVAL_SEEDS = 5  # variation indices 0..4


def apply_formal_hyperparameters(args: argparse.Namespace) -> None:
    """Set max_steps=100 and final eval to task 1-1 with 5 seeds (variations), average score."""
    args.max_steps = 100
    args._test_task_ids_override = [FORMAL_EVAL_TASK_ID]
    args._test_variations_override = list(range(FORMAL_EVAL_SEEDS))
    args._formal_eval = True


def run_phase(args: argparse.Namespace, phase: str) -> None:
    global SIMPLIFICATION_PRESET
    SIMPLIFICATION_PRESET = args.simplification_preset

    train_task_ids = (
        [t.strip() for t in args.train_task_ids.split(",")]
        if args.train_task_ids else TRAIN_TASK_IDS
    )
    train_variations = getattr(args, "_train_variations_override", None) or TRAIN_VARIATIONS
    test_task_ids = (
        getattr(args, "_test_task_ids_override", None)
        or ([t.strip() for t in args.test_task_ids.split(",")] if args.test_task_ids else TEST_TASK_IDS)
    )
    test_variations = getattr(args, "_test_variations_override", None) or TEST_VARIATIONS
    formal_eval = getattr(args, "_formal_eval", False)

    optimizer = Optimizer(
        game=GAME_NAME,
        target_phase=phase,
        three_step=True,
        verbose=True,
    )

    best_fn: Callable | None = None
    current_fn: Callable | None = None
    all_results: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}

    train_cases = build_case_list(train_task_ids, train_variations)

    anchor = train_cases[0] if train_cases else {
        "task_id": "1-1",
        "task_name": "boil",
        "variation": 0,
    }

    def get_baseline(case: dict[str, Any], case_idx: int | None = None, total_cases: int | None = None) -> dict[str, float]:
        label = make_case_label(case)
        if label not in baselines:
            if case_idx is not None and total_cases is not None:
                print(f"  [Baseline] ({case_idx}/{total_cases}) {label} ...", flush=True)
            else:
                print(f"Computing training baseline for {label}...", flush=True)
            n_ep = args.eval_runs
            print(
                f"      Running {n_ep} episode(s) (each ~1–5 min with MCTS; please wait)...",
                flush=True,
            )
            avg, sr, steps, results, elapsed = eval_case(
                case, phase, None,
                args.iterations, args.max_depth, args.max_steps, args.max_actions,
                args.eval_runs,
            )
            cpr = correct_plan_rate(results)
            baselines[label] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "correct_plan_rate": cpr,
                "composite": cpr,
            }
            best_scores[label] = cpr
            print(
                f"  baseline: correct_plan_rate={cpr:.4f}, "
                f"solve_rate={sr:.0%}, avg_returns={avg:.4f}, "
                f"avg_steps={steps:.1f} ({elapsed:.1f}s)",
                flush=True,
            )
        return baselines[label]

    n_train = len(train_cases)
    print(
        f"[Baseline] Computing baselines for {n_train} training case(s) "
        f"(each episode can take 1–5 min with MCTS)...",
        flush=True,
    )
    for idx, case in enumerate(train_cases, start=1):
        get_baseline(case, case_idx=idx, total_cases=n_train)

    cur_case = anchor
    if make_case_label(cur_case) not in baselines:
        print(f"  [Baseline] (anchor) {make_case_label(cur_case)} ...", flush=True)
        get_baseline(cur_case)

    print("\nTraining pool configuration:", flush=True)
    print(f"  phase: {phase}", flush=True)
    print(f"  train tasks: {train_task_ids}")
    print(f"  train variations: {train_variations}")
    print(f"  simplifications: {SIMPLIFICATION_PRESET}")

    for iteration in range(1, args.num_iters + 1):
        current_label = make_case_label(cur_case)
        baseline = baselines[current_label]
        iter_train_cases = select_balanced_top_cases(
            train_cases, baselines, args.trace_cases,
        )
        validation_cases = select_balanced_top_cases(
            train_cases, baselines, args.validation_cases,
        )
        validation_baseline = eval_case_set(
            validation_cases, phase, None,
            args.iterations, args.max_depth, args.max_steps, args.max_actions,
            args.eval_runs,
        )
        reject_floor = (
            validation_baseline["correct_plan_rate"] * args.reject_threshold
        )

        print("\n" + "#" * 60)
        print(f"  EPOCH {iteration}/{args.num_iters}  |  ANCHOR={current_label}")
        print(
            f"  Anchor correct_plan_rate={baseline['correct_plan_rate']:.4f}, "
            f"validation_baseline={validation_baseline['correct_plan_rate']:.4f}, "
            f"reject_floor={reject_floor:.4f}"
        )
        print("  Training trace cases:")
        for case in iter_train_cases:
            print(f"    - {make_case_label(case)}")
        print("  Validation cases:")
        for case in validation_cases:
            print(f"    - {make_case_label(case)}")
        print("#" * 60)

        print(f"\n  [Epoch {iteration}] Step 1/3: Collecting training traces ({len(iter_train_cases)} games)...", flush=True)
        record_files, tool_sources, trace_summary = collect_training_traces(
            iter_train_cases, phase, current_fn,
            args.iterations, args.max_depth, args.max_steps, args.max_actions,
            epoch=iteration,
        )
        for label, status in trace_summary.items():
            print(f"  Trace source [{label}]: {status}")
        for record_file in record_files:
            print(f"  Trace file: {record_file}")

        print(f"  [Epoch {iteration}] Step 2/3: LLM optimizer (analysis → draft → critique)...", flush=True)
        history = None
        if all_results:
            history_lines = []
            for r in all_results[-3:]:
                history_lines.append(
                    f"Iter {r['iteration']} anchor={r['label']} "
                    f"correct_plan_rate={r['composite']:.4f} "
                    f"solve_rate={r['solve_rate']:.0%} "
                    f"desc={r['description']}"
                )
            history = "\n".join(history_lines)

        simplifications = get_simplifications(
            cur_case["task_id"], SIMPLIFICATION_PRESET,
        )
        t_opt = time.time()
        result = optimizer.run(
            record_files=record_files,
            tool_list=tool_sources,
            state_factory=lambda _c=cur_case: ScienceWorldGame(
                task_name=_c["task_name"],
                variation=_c["variation"],
                simplifications=simplifications,
                max_steps=args.max_steps,
                max_actions=args.max_actions,
            ).new_initial_state(),
            additional_context=history,
            session_tag=f"scienceworld_{phase}_iter{iteration}",
        )
        print(f"  Optimizer finished in {time.time() - t_opt:.1f}s", flush=True)

        print(f"  [Epoch {iteration}] Step 3/3: Validation ({len(validation_cases)} cases)...", flush=True)

        rec: dict[str, Any] = {
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
                validation_cases, phase, fn,
                args.iterations, args.max_depth, args.max_steps, args.max_actions,
                args.eval_runs,
                progress_prefix=f"[Epoch {iteration}] Validation",
            )
            comp = eval_summary["correct_plan_rate"]
            rec.update({
                "solve_rate": eval_summary["solve_rate"],
                "correct_plan_rate": comp,
                "avg_returns": eval_summary["avg_returns"],
                "avg_steps": eval_summary["avg_steps"],
                "composite": comp,
            })
            print(
                f"Validation ({len(validation_cases)} cases x "
                f"{args.eval_runs} runs): "
                f"correct_plan_rate={comp:.4f}, "
                f"solve_rate={eval_summary['solve_rate']:.0%}, "
                f"avg_returns={eval_summary['avg_returns']:.4f}, "
                f"avg_normalized_score={eval_summary['avg_normalized_score']:.4f}, "
                f"avg_steps={eval_summary['avg_steps']:.1f} "
                f"({eval_summary['elapsed']:.1f}s)"
            )
            prev_best = max(
                best_scores[make_case_label(case)] for case in validation_cases
            )
            if comp > prev_best:
                print(
                    f"NEW BEST on validation set "
                    f"(prev_correct_plan_rate={prev_best:.4f}) - adopting"
                )
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
            hard_pool = select_balanced_top_cases(
                train_cases, baselines,
                max(args.validation_cases, 3),
            )
            cur_case = random.choice(hard_pool)

    print("\nFinal evaluation on held-out ScienceWorld tasks", flush=True)
    if formal_eval:
        print(
            f"Formal eval: task {FORMAL_EVAL_TASK_ID}, {FORMAL_EVAL_SEEDS} variations (seeds), max_steps={args.max_steps}",
            flush=True,
        )
    final_eval(
        phase, best_fn, args.iterations, args.max_depth, args.max_steps,
        args.max_actions,
        test_task_ids=test_task_ids,
        test_variations=test_variations,
        report_average=formal_eval,
    )

    _SharedEnvPool.close()


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
            raise ValueError(
                f"Invalid phase '{phase}'. Valid phases: {VALID_PHASES}"
            )
        if phase not in deduped:
            deduped.append(phase)
    return deduped


def main() -> None:
    args = parse_args()
    if args.quick:
        apply_quick_hyperparameters(args)
        print("Quick mode: using minimal hyperparameters for verification.", flush=True)
    if args.formal:
        apply_formal_hyperparameters(args)
        print(
            f"Formal mode: max_steps=100, final eval = task {FORMAL_EVAL_TASK_ID} "
            f"with {FORMAL_EVAL_SEEDS} variations (seeds), average score reported.",
            flush=True,
        )
    phases = resolve_phases(args)
    for idx, phase in enumerate(phases, start=1):
        random.seed(42)
        print("\n" + "=" * 72, flush=True)
        print(f"  PHASE {idx}/{len(phases)}: {phase}", flush=True)
        print("=" * 72, flush=True)
        run_phase(args, phase)


if __name__ == "__main__":
    main()
