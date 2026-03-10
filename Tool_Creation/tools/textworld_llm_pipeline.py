"""
Helpers for running the LLM self-evolving pipeline on TextWorld-Express.

This mirrors the high-level workflow of test_llm_pipeline copy.ipynb:
    1. Play a baseline MCTS game and log a trace
    2. Send trace + tool sources to the Optimizer
    3. Hot-swap the optimized function
    4. Evaluate baseline vs optimized on the same task or a small suite
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable

from LLM.optimizer import Optimizer
from mcts import MCTSEngine
from mcts.games import TextWorldExpressGame


DEFAULT_CONFIGS = {
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


@dataclass
class TextWorldEvalConfig:
    game_type: str = "coin"
    game_params: str = DEFAULT_CONFIGS["coin"][0]
    seed: int = 3
    iterations: int = 100
    max_rollout_depth: int = 50
    max_steps: int = 50
    env_step_limit: int = 100


def make_game(cfg: TextWorldEvalConfig) -> TextWorldExpressGame:
    return TextWorldExpressGame(
        game_type=cfg.game_type,
        game_params=cfg.game_params,
        seed=cfg.seed,
        env_step_limit=cfg.env_step_limit,
        max_steps=cfg.max_steps,
    )


def make_engine(cfg: TextWorldEvalConfig, logging: bool = False) -> MCTSEngine:
    return MCTSEngine(
        make_game(cfg),
        iterations=cfg.iterations,
        max_rollout_depth=cfg.max_rollout_depth,
        logging=logging,
    )


def collect_trace_and_sources(cfg: TextWorldEvalConfig) -> tuple[dict[str, Any], dict[str, str]]:
    engine = make_engine(cfg, logging=True)
    result = engine.play_game(verbose=False)
    return result, engine.get_tool_source()


def multi_eval(
    cfg: TextWorldEvalConfig,
    phase: str,
    fn: Callable | None = None,
    n: int = 3,
) -> tuple[float, float, float, list[dict[str, Any]], float]:
    t0 = time()
    engine = make_engine(cfg, logging=False)
    if fn is not None:
        engine.set_tool(phase, fn)
    results = []
    for _ in range(n):
        results.append(engine.play_game(verbose=False))
    elapsed = time() - t0
    avg_ret = sum(r["returns"][0] for r in results) / len(results)
    solve_rate = sum(1 for r in results if r["solved"]) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    return avg_ret, solve_rate, avg_steps, results, elapsed


def compare_suite(
    phase: str,
    optimized_fn: Callable | None,
    configs: list[TextWorldEvalConfig],
    games_per_config: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        base_ret, base_solve, base_steps, _, base_time = multi_eval(
            cfg, phase, fn=None, n=games_per_config
        )
        opt_ret, opt_solve, opt_steps, _, opt_time = multi_eval(
            cfg, phase, fn=optimized_fn, n=games_per_config
        )
        rows.append(
            {
                "game_type": cfg.game_type,
                "game_params": cfg.game_params,
                "base_solve": base_solve,
                "opt_solve": opt_solve,
                "base_ret": base_ret,
                "opt_ret": opt_ret,
                "base_steps": base_steps,
                "opt_steps": opt_steps,
                "base_time": base_time,
                "opt_time": opt_time,
            }
        )
    return rows


def composite_score(solve_rate: float, avg_ret: float) -> float:
    return 0.7 * solve_rate + 0.3 * avg_ret


def run_single_iteration(
    cfg: TextWorldEvalConfig,
    phase: str = "simulation",
    eval_runs: int = 3,
    max_moves_per_trace: int = 30,
    additional_context: str | None = None,
    session_tag: str | None = None,
) -> dict[str, Any]:
    trace_result, tool_sources = collect_trace_and_sources(cfg)
    log_file = trace_result.get("log_file")
    if not log_file:
        raise RuntimeError("Trace logging failed; no log_file was returned.")

    optimizer = Optimizer(
        game="textworld_express",
        target_phase=phase,
        max_moves_per_trace=max_moves_per_trace,
        two_step=True,
        three_step=True,
        verbose=True,
    )
    result = optimizer.run(
        record_files=[log_file],
        tool_list=tool_sources,
        state_factory=lambda: make_game(cfg).new_initial_state(),
        additional_context=additional_context,
        session_tag=session_tag,
    )

    fn = result.get("function")
    base_ret, base_solve, base_steps, _, base_time = multi_eval(cfg, phase, None, eval_runs)
    if fn is not None:
        opt_ret, opt_solve, opt_steps, _, opt_time = multi_eval(cfg, phase, fn, eval_runs)
    else:
        opt_ret, opt_solve, opt_steps, opt_time = 0.0, 0.0, float(cfg.max_steps), 0.0

    result["baseline_eval"] = {
        "avg_ret": base_ret,
        "solve_rate": base_solve,
        "avg_steps": base_steps,
        "elapsed": base_time,
    }
    result["optimized_eval"] = {
        "avg_ret": opt_ret,
        "solve_rate": opt_solve,
        "avg_steps": opt_steps,
        "elapsed": opt_time,
    }
    result["trace_result"] = trace_result
    return result


def format_rows(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'Game':<10} {'Params':<52} {'Base%':>6} {'Opt%':>6} "
        f"{'Base Ret':>9} {'Opt Ret':>8} {'Base Steps':>11} {'Opt Steps':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        params = row["game_params"]
        if len(params) > 52:
            params = params[:49] + "..."
        lines.append(
            f"{row['game_type']:<10} {params:<52} "
            f"{row['base_solve']*100:>5.0f}% {row['opt_solve']*100:>5.0f}% "
            f"{row['base_ret']:>9.3f} {row['opt_ret']:>8.3f} "
            f"{row['base_steps']:>11.1f} {row['opt_steps']:>10.1f}"
        )
    return "\n".join(lines)


def default_suite(seed: int = 3, iterations: int = 100, max_rollout_depth: int = 50, max_steps: int = 50) -> list[TextWorldEvalConfig]:
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
                )
            )
    return out
