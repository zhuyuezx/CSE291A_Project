"""
Self-evolving LLM optimization loop for the pure-Python TextWorld coin game.
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
from mcts.games import TextWorldCoin


DEFAULT_PARAMS = [
    "numLocations=5,includeDoors=1,numDistractorItems=0",
    "numLocations=6,includeDoors=1,numDistractorItems=0",
    "numLocations=7,includeDoors=1,numDistractorItems=0",
    "numLocations=10,includeDoors=1,numDistractorItems=0",
]


def composite_score(solve_rate: float, avg_returns: float) -> float:
    return 0.7 * solve_rate + 0.3 * avg_returns


def is_correct_plan(result: dict[str, Any]) -> bool:
    """
    Homework-style success proxy.

    In hw2_part2.ipynb the grading language is "correct plans". For this
    symbolic coin game, the closest analogue is a run that actually
    solves the task (coin taken), which is represented by solved=True and
    return 1.0.
    """
    returns = result.get("returns", [0.0])
    ret = returns[0] if isinstance(returns, list) else float(returns)
    return bool(result.get("solved")) and ret >= 1.0


def make_engine(game_params: str, iterations: int, max_depth: int, max_steps: int, logging: bool):
    game = TextWorldCoin(game_params=game_params, max_steps=max_steps)
    return MCTSEngine(game, iterations=iterations, max_rollout_depth=max_depth, logging=logging)


def multi_eval(
    phase: str,
    game_params: str,
    fn: Callable | None,
    n: int,
    iterations: int,
    max_depth: int,
    max_steps: int,
) -> tuple[float, float, float, list[dict[str, Any]], float]:
    t0 = time.time()
    results = []
    for _ in range(n):
        engine = make_engine(game_params, iterations, max_depth, max_steps, logging=False)
        if fn is not None:
            engine.set_tool(phase, fn)
        results.append(engine.play_game(verbose=False))
    elapsed = time.time() - t0
    avg_ret = sum(r["returns"][0] for r in results) / n
    solve_rate = sum(1 for r in results if r["solved"]) / n
    avg_steps = sum(r["steps"] for r in results) / n
    return avg_ret, solve_rate, avg_steps, results, elapsed


def compare_suite(
    phase: str,
    best_fn: Callable | None,
    params_list: list[str],
    eval_runs: int,
    iterations: int,
    max_depth: int,
    max_steps: int,
) -> None:
    print(
        f"{'Params':<42} {'Base Correct':>12} {'Opt Correct':>12} "
        f"{'Base Solve%':>12} {'Opt Solve%':>12} {'Base AvgRet':>12} {'Opt AvgRet':>12}"
    )
    print("-" * 122)
    total_base_correct = 0
    total_opt_correct = 0
    total_cases = 0
    for params in params_list:
        rb = multi_eval(phase, params, None, eval_runs, iterations, max_depth, max_steps)
        ro = multi_eval(phase, params, best_fn, eval_runs, iterations, max_depth, max_steps)
        base_ret, base_solve, _base_steps, base_results = rb[0], rb[1], rb[2], rb[3]
        opt_ret, opt_solve, _opt_steps, opt_results = ro[0], ro[1], ro[2], ro[3]
        base_correct = sum(1 for r in base_results if is_correct_plan(r))
        opt_correct = sum(1 for r in opt_results if is_correct_plan(r))
        total_base_correct += base_correct
        total_opt_correct += opt_correct
        total_cases += eval_runs
        print(
            f"{params:<42} {base_correct:>7}/{eval_runs:<4} {opt_correct:>7}/{eval_runs:<4} "
            f"{base_solve*100:>11.0f}% {opt_solve*100:>11.0f}% "
            f"{base_ret:>12.3f} {opt_ret:>12.3f}"
        )
    print()
    print(f"Correct plans summary: baseline={total_base_correct}/{total_cases}, optimized={total_opt_correct}/{total_cases}")


def build_history(
    all_results: list[dict[str, Any]],
    current_params: str,
    baselines: dict[str, dict[str, float]],
    best_scores: dict[str, float],
) -> str:
    bl = baselines[current_params]
    lines = [
        f"Current task: {current_params}",
        (
            f"Baseline for current task: composite={bl['composite']:.4f}, "
            f"solve_rate={bl['solve_rate']:.0%}, avg_returns={bl['avg_returns']:.4f}"
        ),
        "",
        "Per-task best composites so far:",
    ]
    for params in sorted(best_scores):
        lines.append(
            f"  {params}: best={best_scores[params]:.4f} "
            f"(baseline={baselines[params]['composite']:.4f})"
        )
    lines += ["", "Recent iterations:"]
    for r in all_results:
        status = "best" if r.get("is_best") else ("accepted" if r.get("adopted") else "rejected")
        lines.append(
            f"  Iter {r['iteration']} [{r['params']}]: composite={r['composite']:.4f}, "
            f"solve_rate={r['solve_rate']:.0%}, avg_returns={r['avg_returns']:.4f}, {status}, "
            f"desc={r.get('description', 'n/a')}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="simulation")
    p.add_argument("--num-iters", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--eval-runs", type=int, default=3)
    p.add_argument("--reject-threshold", type=float, default=0.5)
    p.add_argument("--game-params", default=DEFAULT_PARAMS[0])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(3)

    optimizer = Optimizer(
        game="textworld_coin",
        target_phase=args.phase,
        three_step=True,
        verbose=True,
    )

    best_fn = None
    current_fn = None
    all_results: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}

    def get_baseline(game_params: str) -> dict[str, float]:
        if game_params not in baselines:
            print(f"Computing baseline for {game_params}...")
            avg, sr, steps, _, elapsed = multi_eval(
                args.phase, game_params, None, args.eval_runs, args.iterations, args.max_depth, args.max_steps
            )
            comp = composite_score(sr, avg)
            baselines[game_params] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": elapsed,
                "composite": comp,
            }
            best_scores[game_params] = comp
            print(
                f"  baseline: composite={comp:.4f}, solve_rate={sr:.0%}, "
                f"avg_returns={avg:.4f}, avg_steps={steps:.1f} ({elapsed:.1f}s)"
            )
        return baselines[game_params]

    cur_params = args.game_params
    get_baseline(cur_params)

    for iteration in range(1, args.num_iters + 1):
        baseline = get_baseline(cur_params)
        reject_floor = baseline["composite"] * args.reject_threshold

        print("\n" + "#" * 60)
        print(f"ITERATION {iteration}/{args.num_iters}, PARAMS={cur_params}")
        print(f"Baseline composite={baseline['composite']:.4f}, reject_floor={reject_floor:.4f}")
        print("#" * 60)

        t_play = time.time()
        engine = make_engine(cur_params, args.iterations, args.max_depth, args.max_steps, logging=True)
        if current_fn is not None:
            engine.set_tool(args.phase, current_fn)
        play_result = engine.play_game(verbose=False)
        tool_sources = engine.get_tool_source()
        play_trace = play_result.get("log_file", "")
        print(
            f"Play: {'SOLVED' if play_result['solved'] else 'UNSOLVED'} in {play_result['steps']} steps "
            f"returns={play_result['returns'][0]:.4f} ({time.time() - t_play:.1f}s)"
        )

        history = build_history(all_results[-3:], cur_params, baselines, best_scores) if all_results else None
        t_opt = time.time()
        result = optimizer.run(
            record_files=[play_trace] if play_trace else [],
            tool_list=tool_sources,
            state_factory=lambda _p=cur_params: TextWorldCoin(_p, max_steps=args.max_steps).new_initial_state(),
            additional_context=history,
            session_tag=f"textworld_coin_{args.phase}_iter{iteration}",
        )
        print(f"Optimize: {time.time() - t_opt:.1f}s")

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
            avg_ret, solve_rate, avg_steps, _, eval_time = multi_eval(
                args.phase, cur_params, fn, args.eval_runs, args.iterations, args.max_depth, args.max_steps
            )
            comp = composite_score(solve_rate, avg_ret)
            rec.update(
                {
                    "solve_rate": solve_rate,
                    "avg_returns": avg_ret,
                    "avg_steps": avg_steps,
                    "composite": comp,
                }
            )
            print(
                f"Eval ({args.eval_runs} runs): avg_returns={avg_ret:.4f}, solve_rate={solve_rate:.0%}, "
                f"composite={comp:.4f}, avg_steps={avg_steps:.1f} ({eval_time:.1f}s)"
            )

            prev_best = best_scores[cur_params]
            if comp > prev_best:
                print(f"NEW BEST for {cur_params} (prev={prev_best:.4f}) - adopting")
                best_scores[cur_params] = comp
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
            cur_params = random.choice(DEFAULT_PARAMS)
            get_baseline(cur_params)

    print("\nFinal suite evaluation")
    compare_suite(args.phase, best_fn, DEFAULT_PARAMS, args.eval_runs, args.iterations, args.max_depth, args.max_steps)


if __name__ == "__main__":
    main()



