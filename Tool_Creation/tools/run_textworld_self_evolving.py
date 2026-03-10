"""
Self-evolving LLM optimization loop for TextWorld-Express.

This is the TextWorld counterpart to the iterative logic in
test_llm_pipeline copy.ipynb:
    - keep per-task baselines
    - accept/reject newly generated tools
    - maintain best_fn/current_fn
    - run a final baseline vs optimized suite comparison

Default NUM_ITERS is intentionally small because TextWorld state cloning
replays the action history and is therefore slower than Sokoban.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any, Callable

from LLM.optimizer import Optimizer
from tools.textworld_llm_pipeline import (
    DEFAULT_CONFIGS,
    TextWorldEvalConfig,
    compare_suite,
    composite_score,
    format_rows,
    make_engine,
    make_game,
    multi_eval,
)


def config_label(cfg: TextWorldEvalConfig) -> str:
    return f"{cfg.game_type}:{cfg.game_params}:seed={cfg.seed}"


def default_task_pool(
    seed: int,
    iterations: int,
    max_rollout_depth: int,
    max_steps: int,
    env_step_limit: int,
) -> list[TextWorldEvalConfig]:
    out: list[TextWorldEvalConfig] = []
    for game_type, params_list in DEFAULT_CONFIGS.items():
        for params in params_list:
            out.append(
                TextWorldEvalConfig(
                    game_type=game_type,
                    game_params=params,
                    seed=seed,
                    iterations=iterations,
                    max_rollout_depth=max_rollout_depth,
                    max_steps=max_steps,
                    env_step_limit=env_step_limit,
                )
            )
    return out


def collect_trace_and_sources_for_fn(
    cfg: TextWorldEvalConfig,
    phase: str,
    current_fn: Callable | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    engine = make_engine(cfg, logging=True)
    if current_fn is not None:
        engine.set_tool(phase, current_fn)
    result = engine.play_game(verbose=False)
    return result, engine.get_tool_source()


def build_history(
    results: list[dict[str, Any]],
    current_label: str,
    baselines: dict[str, dict[str, float]],
    best_scores: dict[str, float],
    active_labels: list[str],
    mastered_labels: set[str],
) -> str:
    bl = baselines[current_label]
    agg_best = sum(best_scores.values()) / len(best_scores) if best_scores else 0.0
    lines = [
        f"Current task: {current_label}",
        (
            f"Baseline for current task: composite={bl['composite']:.4f}, "
            f"solve_rate={bl['solve_rate']:.0%}, avg_returns={bl['avg_returns']:.4f}"
        ),
        f"Aggregate best (avg across {len(best_scores)} tasks): {agg_best:.4f}",
        "",
        "Per-task best composites so far:",
    ]
    for label in sorted(best_scores):
        tag = " [MASTERED]" if label in mastered_labels else ""
        lines.append(
            f"  {label}: best={best_scores[label]:.4f} "
            f"(baseline={baselines[label]['composite']:.4f}){tag}"
        )
    lines += [
        "",
        f"Active tasks: {active_labels}",
        f"Mastered tasks: {sorted(mastered_labels)}" if mastered_labels else "",
        "",
        "SCORING: composite = 0.7 * solve_rate + 0.3 * avg_returns",
        "SOLVING the task matters more than raw return shaping.",
        "",
        "Prefer incremental improvements to the current tool. Restructure only",
        "when the current approach is fundamentally limited.",
        "",
        "Recent iterations:",
    ]
    for r in results:
        status = "accepted" if r.get("adopted") else "rejected"
        if r.get("is_best"):
            status = "best"
        lines.append(
            f"  Iter {r['iteration']} [{r['task']}]: composite={r['composite']:.4f}, "
            f"solve_rate={r['solve_rate']:.0%}, avg_returns={r['avg_returns']:.4f}, "
            f"avg_steps={r['avg_steps']:.1f}, {status}, "
            f"desc={r.get('description', 'n/a')}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="simulation")
    parser.add_argument("--num-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--env-step-limit", type=int, default=100)
    parser.add_argument("--eval-runs", type=int, default=3)
    parser.add_argument("--reject-threshold", type=float, default=0.5)
    parser.add_argument("--mastery-solve-rate", type=float, default=1.0)
    parser.add_argument("--mastery-confirm-runs", type=int, default=5)
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--task-game-type", default="", help="Optional fixed task type, e.g. coin")
    parser.add_argument("--task-game-params", default="", help="Optional fixed task params")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    task_pool = default_task_pool(
        seed=args.seed,
        iterations=args.iterations,
        max_rollout_depth=args.max_depth,
        max_steps=args.max_steps,
        env_step_limit=args.env_step_limit,
    )
    if args.task_game_type and args.task_game_params:
        anchor = TextWorldEvalConfig(
            game_type=args.task_game_type,
            game_params=args.task_game_params,
            seed=args.seed,
            iterations=args.iterations,
            max_rollout_depth=args.max_depth,
            max_steps=args.max_steps,
            env_step_limit=args.env_step_limit,
        )
    else:
        anchor = task_pool[0]

    opt = Optimizer(
        game="textworld_express",
        target_phase=args.phase,
        three_step=True,
        verbose=True,
    )

    best_fn = None
    current_fn = None
    all_results: list[dict[str, Any]] = []

    baselines: dict[str, dict[str, float]] = {}
    best_scores: dict[str, float] = {}
    mastered_labels: set[str] = set()
    cfg_by_label = {config_label(cfg): cfg for cfg in task_pool}
    active_labels = list(cfg_by_label.keys())

    def get_baseline(cfg: TextWorldEvalConfig) -> dict[str, float]:
        label = config_label(cfg)
        if label not in baselines:
            print(f"Computing baseline for {label}...")
            avg_ret, solve_rate, avg_steps, _, elapsed = multi_eval(
                cfg, args.phase, fn=None, n=args.eval_runs
            )
            comp = composite_score(solve_rate, avg_ret)
            baselines[label] = {
                "avg_returns": avg_ret,
                "solve_rate": solve_rate,
                "avg_steps": avg_steps,
                "eval_time": elapsed,
                "composite": comp,
            }
            best_scores[label] = comp
            print(
                f"  baseline: composite={comp:.4f}, solve_rate={solve_rate:.0%}, "
                f"avg_returns={avg_ret:.4f}, avg_steps={avg_steps:.1f} ({elapsed:.1f}s)"
            )
            if solve_rate >= args.mastery_solve_rate and label in active_labels:
                mastered_labels.add(label)
                active_labels.remove(label)
        return baselines[label]

    first_bl = get_baseline(anchor)
    print(f"Starting task: {config_label(anchor)}")
    print(f"Reject floor: {first_bl['composite'] * args.reject_threshold:.4f}")
    print(f"Active tasks: {len(active_labels)}")

    for iteration in range(1, args.num_iters + 1):
        if not active_labels:
            print(f"All tasks mastered after {iteration - 1} iterations. Stopping early.")
            break

        cur_label = config_label(anchor) if config_label(anchor) in active_labels else random.choice(active_labels)
        cur_cfg = cfg_by_label[cur_label]
        bl = get_baseline(cur_cfg)
        reject_floor = bl["composite"] * args.reject_threshold

        print("\n" + "#" * 72)
        print(f"ITERATION {iteration}/{args.num_iters} TASK={cur_label}")
        print(
            f"Baseline composite={bl['composite']:.4f}, reject_floor={reject_floor:.4f}, "
            f"active={len(active_labels)}, mastered={len(mastered_labels)}"
        )
        print("#" * 72)

        state_factory = lambda _cfg=cur_cfg: make_game(_cfg).new_initial_state()

        t_play_start = time.time()
        play_result, tool_sources = collect_trace_and_sources_for_fn(cur_cfg, args.phase, current_fn)
        t_play = time.time() - t_play_start
        play_trace = play_result.get("log_file", "")
        ptag = "SOLVED" if play_result.get("solved") else "UNSOLVED"
        print(
            f"Play: {ptag} in {play_result.get('steps', '?')} steps "
            f"returns={play_result['returns'][0]:.4f} ({t_play:.1f}s)"
        )

        history = build_history(
            all_results[-args.history_window:],
            cur_label,
            baselines,
            best_scores,
            active_labels,
            mastered_labels,
        ) if all_results else None

        t_opt_start = time.time()
        result = opt.run(
            record_files=[play_trace] if play_trace else [],
            tool_list=tool_sources,
            state_factory=state_factory,
            additional_context=history,
            session_tag=f"textworld_{args.phase}_iter{iteration}",
        )
        t_opt = time.time() - t_opt_start
        print(f"Optimize: {t_opt:.1f}s")

        iter_record = {
            "iteration": iteration,
            "task": cur_label,
            "smoke_test": result["smoke_test"],
            "avg_returns": bl["avg_returns"],
            "solve_rate": 0.0,
            "composite": 0.0,
            "avg_steps": args.max_steps,
            "description": (result.get("parsed") or {}).get("description", ""),
            "error": result.get("error"),
            "adopted": False,
            "is_best": False,
            "play_time": t_play,
            "opt_time": t_opt,
            "eval_time": None,
        }

        fn = result.get("function")
        if fn is not None:
            avg_ret, solve_rate, avg_steps, _, eval_time = multi_eval(
                cur_cfg, args.phase, fn=fn, n=args.eval_runs
            )
            comp = composite_score(solve_rate, avg_ret)
            iter_record["avg_returns"] = avg_ret
            iter_record["solve_rate"] = solve_rate
            iter_record["composite"] = comp
            iter_record["avg_steps"] = avg_steps
            iter_record["eval_time"] = eval_time

            print(
                f"Eval ({args.eval_runs} runs): avg_returns={avg_ret:.4f}, "
                f"solve_rate={solve_rate:.0%}, composite={comp:.4f}, "
                f"avg_steps={avg_steps:.1f} ({eval_time:.1f}s)"
            )

            if solve_rate >= args.mastery_solve_rate:
                print(
                    f"Confirming mastery for {cur_label} with "
                    f"{args.mastery_confirm_runs} more runs..."
                )
                _, confirm_sr, confirm_steps, _, confirm_t = multi_eval(
                    cur_cfg, args.phase, fn=fn, n=args.mastery_confirm_runs
                )
                if confirm_sr >= args.mastery_solve_rate:
                    print(
                        f"MASTERED {cur_label}: confirm solve_rate={confirm_sr:.0%}, "
                        f"avg_steps={confirm_steps:.1f} ({confirm_t:.1f}s)"
                    )
                    mastered_labels.add(cur_label)
                    if cur_label in active_labels:
                        active_labels.remove(cur_label)
                else:
                    print(
                        f"Mastery confirmation failed for {cur_label}: "
                        f"{confirm_sr:.0%} ({confirm_t:.1f}s)"
                    )

            prev_best = best_scores.get(cur_label, bl["composite"])
            if comp > prev_best:
                print(f"NEW BEST for {cur_label} (prev={prev_best:.4f}) - adopting")
                best_scores[cur_label] = comp
                best_fn = fn
                current_fn = fn
                iter_record["adopted"] = True
                iter_record["is_best"] = True
            elif comp >= reject_floor:
                print(
                    f"Accepted on {cur_label} (comp={comp:.4f} >= floor={reject_floor:.4f}, "
                    f"level_best={prev_best:.4f})"
                )
                current_fn = fn
                iter_record["adopted"] = True
            else:
                print(
                    f"Rejected on {cur_label} (comp={comp:.4f} < floor={reject_floor:.4f}) "
                    f"- reverting to best"
                )
                current_fn = best_fn
        else:
            print("Eval: SKIPPED (smoke test failed or error)")
            if result.get("error"):
                print(f"       {result['error'][:200]}")

        total_iter_time = t_play + t_opt + (iter_record["eval_time"] or 0.0)
        print(f"Iteration total: {total_iter_time:.1f}s")
        all_results.append(iter_record)

        if active_labels:
            anchor = cfg_by_label[random.choice(active_labels)]

    print("\nFinal suite evaluation")
    rows = compare_suite(args.phase, best_fn, task_pool, games_per_config=args.eval_runs)
    print(format_rows(rows))


if __name__ == "__main__":
    main()
