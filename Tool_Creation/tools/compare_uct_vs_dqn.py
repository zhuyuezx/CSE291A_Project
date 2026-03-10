"""
Compare UCT baseline vs PUCT+DQN across Sokoban levels.

Example:
  python tools/compare_uct_vs_dqn.py \
    --checkpoint checkpoints/sokoban_dqn_l1_l10.pt \
    --iterations 500 --games 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_sokoban_torch import (  # noqa: E402
    action_to_index_fn,
    encode_state_fn,
    load_checkpoint,
    q_model,
)
from mcts import (  # noqa: E402
    MCTSEngine,
    make_dqn_prior_fn,
    make_puct_expansion,
    make_puct_selection,
)
from mcts.games import Sokoban  # noqa: E402
from mcts.games.sokoban import LEVELS  # noqa: E402


def _parse_levels(raw: str) -> list[str]:
    levels = [x.strip() for x in raw.split(",") if x.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"Unknown level '{lv}'. Available: {list(LEVELS.keys())}")
    return levels


def _avg_return(stats: dict) -> float:
    vals = [float(r["returns"][0]) for r in stats["results"]]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _run_uct(level: str, iterations: int, games: int, verbose: bool) -> tuple[dict, float]:
    start = time.perf_counter()
    engine = MCTSEngine(Sokoban(level), iterations=iterations, logging=False)
    stats = engine.play_many(num_games=games, verbose=verbose)
    elapsed = time.perf_counter() - start
    return stats, elapsed


def _run_puct_dqn(
    level: str,
    iterations: int,
    games: int,
    cpuct: float,
    temperature: float,
    expansion_strategy: str,
    epsilon: float,
    verbose: bool,
) -> tuple[dict, float]:
    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=action_to_index_fn,
        temperature=temperature,
    )

    start = time.perf_counter()
    engine = MCTSEngine(Sokoban(level), iterations=iterations, logging=False)
    engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=cpuct))
    engine.set_tool(
        "expansion",
        make_puct_expansion(
            prior_fn,
            strategy=expansion_strategy,
            epsilon=epsilon,
        ),
    )
    stats = engine.play_many(num_games=games, verbose=verbose)
    elapsed = time.perf_counter() - start
    return stats, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--levels",
        default="level1,level2,level3,level4,level5,level6,level7,level8,level9,level10",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--cpuct", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument(
        "--expansion-strategy",
        choices=["greedy", "sample", "epsilon_greedy"],
        default="sample",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    levels = _parse_levels(args.levels)
    load_checkpoint(args.checkpoint, device=args.device)

    rows = []

    for lv in levels:
        base, t_base = _run_uct(lv, args.iterations, args.games, args.verbose)
        opt, t_opt = _run_puct_dqn(
            lv,
            args.iterations,
            args.games,
            cpuct=args.cpuct,
            temperature=args.temperature,
            expansion_strategy=args.expansion_strategy,
            epsilon=args.epsilon,
            verbose=args.verbose,
        )

        base_rate = float(base["solve_rate"])
        opt_rate = float(opt["solve_rate"])

        print(
            f"Evaluating {lv}... "
            f"baseline={base_rate:.3f} ({base_rate*100:.0f}%)  "
            f"optimized={opt_rate:.3f} ({opt_rate*100:.0f}%)  "
            f"[{t_base:.1f}s + {t_opt:.1f}s]"
        )

        rows.append(
            {
                "level": lv,
                "base_solve": base_rate,
                "opt_solve": opt_rate,
                "base_ret": _avg_return(base),
                "opt_ret": _avg_return(opt),
                "base_steps": float(base["avg_steps"]),
                "opt_steps": float(opt["avg_steps"]),
            }
        )

    print()
    print(
        f"{'Level':<8} {'Base Solve%':>10} {'Opt Solve%':>10} "
        f"{'Base AvgRet':>11} {'Opt AvgRet':>10} {'Base Steps':>10} {'Opt Steps':>9}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['level']:<8} "
            f"{r['base_solve']*100:>9.0f}% "
            f"{r['opt_solve']*100:>9.0f}% "
            f"{r['base_ret']:>11.3f} "
            f"{r['opt_ret']:>10.3f} "
            f"{r['base_steps']:>10.1f} "
            f"{r['opt_steps']:>9.1f}"
        )


if __name__ == "__main__":
    main()

