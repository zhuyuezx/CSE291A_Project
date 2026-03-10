"""
Compare three methods on Sokoban:
  1) UCT baseline
  2) PUCT + heuristic prior
  3) LLM-optimized tool (single phase hot-swap)

Example:
  python tools/compare_three_methods.py ^
    --llm-tool-path MCTS_tools/simulation/my_sim.py ^
    --llm-phase simulation --levels level1,level2,level3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts import MCTSEngine, make_dqn_prior_fn, make_puct_expansion, make_puct_selection
from mcts.games import Sokoban
from mcts.games.sokoban import LEVELS
from dqn_sokoban_module import q_model, encode_state_fn, action_to_index_fn


def _parse_levels(raw: str) -> list[str]:
    levels = [x.strip() for x in raw.split(",") if x.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"Unknown level '{lv}'")
    return levels


def _avg_return(stats: dict[str, Any]) -> float:
    vals = [float(r["returns"][0]) for r in stats["results"]]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _run_uct(
    level: str, iterations: int, games: int, max_steps: int, max_depth: int
) -> tuple[dict, float]:
    t0 = time.perf_counter()
    engine = MCTSEngine(
        Sokoban(level_name=level, max_steps=max_steps),
        iterations=iterations,
        max_rollout_depth=max_depth,
        logging=False,
    )
    stats = engine.play_many(num_games=games, verbose=False)
    return stats, time.perf_counter() - t0


def _run_puct_prior(
    level: str,
    iterations: int,
    games: int,
    max_steps: int,
    max_depth: int,
    cpuct: float,
    temperature: float,
    expansion_strategy: str,
    epsilon: float,
) -> tuple[dict, float]:
    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=action_to_index_fn,
        temperature=temperature,
    )
    t0 = time.perf_counter()
    engine = MCTSEngine(
        Sokoban(level_name=level, max_steps=max_steps),
        iterations=iterations,
        max_rollout_depth=max_depth,
        logging=False,
    )
    engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=cpuct))
    engine.set_tool(
        "expansion",
        make_puct_expansion(prior_fn, strategy=expansion_strategy, epsilon=epsilon),
    )
    stats = engine.play_many(num_games=games, verbose=False)
    return stats, time.perf_counter() - t0


def _run_llm_tool(
    level: str,
    iterations: int,
    games: int,
    max_steps: int,
    max_depth: int,
    phase: str,
    llm_tool_path: str,
) -> tuple[dict, float]:
    t0 = time.perf_counter()
    engine = MCTSEngine(
        Sokoban(level_name=level, max_steps=max_steps),
        iterations=iterations,
        max_rollout_depth=max_depth,
        logging=False,
    )
    engine.load_tool(phase, llm_tool_path)
    stats = engine.play_many(num_games=games, verbose=False)
    return stats, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        default="level1,level2,level3,level4,level5,level6,level7,level8,level9,level10",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=500)
    parser.add_argument("--cpuct", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--expansion-strategy", choices=["greedy", "sample", "epsilon_greedy"], default="sample")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--llm-phase", default="simulation")
    parser.add_argument("--llm-tool-path", required=True)
    args = parser.parse_args()

    levels = _parse_levels(args.levels)
    llm_path = Path(args.llm_tool_path)
    if not llm_path.is_absolute():
        llm_path = (ROOT / llm_path).resolve()
    if not llm_path.exists():
        raise FileNotFoundError(f"LLM tool not found: {llm_path}")

    rows = []
    for lv in levels:
        base, t_base = _run_uct(
            lv, args.iterations, args.games, args.max_steps, args.max_depth
        )
        puct, t_puct = _run_puct_prior(
            lv,
            args.iterations,
            args.games,
            args.max_steps,
            args.max_depth,
            cpuct=args.cpuct,
            temperature=args.temperature,
            expansion_strategy=args.expansion_strategy,
            epsilon=args.epsilon,
        )
        llm, t_llm = _run_llm_tool(
            lv,
            args.iterations,
            args.games,
            args.max_steps,
            args.max_depth,
            phase=args.llm_phase,
            llm_tool_path=str(llm_path),
        )

        print(
            f"Evaluating {lv}... "
            f"UCT={base['solve_rate']:.3f} ({base['solve_rate']*100:.0f}%)  "
            f"PUCT={puct['solve_rate']:.3f} ({puct['solve_rate']*100:.0f}%)  "
            f"LLM={llm['solve_rate']:.3f} ({llm['solve_rate']*100:.0f}%)  "
            f"[{t_base:.1f}s + {t_puct:.1f}s + {t_llm:.1f}s]"
        )

        rows.append(
            {
                "level": lv,
                "uct_solve": float(base["solve_rate"]),
                "puct_solve": float(puct["solve_rate"]),
                "llm_solve": float(llm["solve_rate"]),
                "uct_ret": _avg_return(base),
                "puct_ret": _avg_return(puct),
                "llm_ret": _avg_return(llm),
                "uct_steps": float(base["avg_steps"]),
                "puct_steps": float(puct["avg_steps"]),
                "llm_steps": float(llm["avg_steps"]),
            }
        )

    print()
    print(
        f"{'Level':<8} {'UCT%':>6} {'PUCT%':>6} {'LLM%':>6} "
        f"{'UCT Ret':>8} {'PUCT Ret':>9} {'LLM Ret':>8} "
        f"{'UCT Steps':>10} {'PUCT Steps':>11} {'LLM Steps':>10}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['level']:<8} "
            f"{r['uct_solve']*100:>5.0f}% "
            f"{r['puct_solve']*100:>5.0f}% "
            f"{r['llm_solve']*100:>5.0f}% "
            f"{r['uct_ret']:>8.3f} "
            f"{r['puct_ret']:>9.3f} "
            f"{r['llm_ret']:>8.3f} "
            f"{r['uct_steps']:>10.1f} "
            f"{r['puct_steps']:>11.1f} "
            f"{r['llm_steps']:>10.1f}"
        )


if __name__ == "__main__":
    main()
