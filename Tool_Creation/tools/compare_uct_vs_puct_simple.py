"""
Compare UCT vs PUCT using a simple heuristic prior (no trained DQN checkpoint).

This script is intentionally separate from checkpoint-based DQN comparison
to make experiment variants explicit.

Example:
  python tools/compare_uct_vs_puct_simple.py --iterations 200 --games 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dqn_sokoban_module import action_to_index_fn, encode_state_fn, q_model
from mcts import MCTSEngine, make_dqn_prior_fn, make_puct_expansion, make_puct_selection
from mcts.games import Sokoban
from mcts.games.sokoban import LEVELS


def _parse_levels(raw: str) -> list[str]:
    levels = [x.strip() for x in raw.split(",") if x.strip()]
    for lv in levels:
        if lv not in LEVELS:
            raise ValueError(f"Unknown level '{lv}'. Available: {list(LEVELS.keys())}")
    return levels


def _avg_return(stats: dict) -> float:
    vals = [float(r["returns"][0]) for r in stats["results"]]
    return (sum(vals) / len(vals)) if vals else 0.0


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
    parser.add_argument(
        "--expansion-strategy",
        choices=["greedy", "sample", "epsilon_greedy"],
        default="sample",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    levels = _parse_levels(args.levels)
    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=action_to_index_fn,
        temperature=args.temperature,
    )

    rows = []
    for lv in levels:
        uct_engine = MCTSEngine(
            Sokoban(level_name=lv, max_steps=args.max_steps),
            iterations=args.iterations,
            max_rollout_depth=args.max_depth,
            logging=False,
        )
        uct = uct_engine.play_many(num_games=args.games, verbose=False)

        puct_engine = MCTSEngine(
            Sokoban(level_name=lv, max_steps=args.max_steps),
            iterations=args.iterations,
            max_rollout_depth=args.max_depth,
            logging=False,
        )
        puct_engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=args.cpuct))
        puct_engine.set_tool(
            "expansion",
            make_puct_expansion(
                prior_fn,
                strategy=args.expansion_strategy,
                epsilon=args.epsilon,
            ),
        )
        puct = puct_engine.play_many(num_games=args.games, verbose=False)

        rows.append(
            {
                "level": lv,
                "uct_solve": float(uct["solve_rate"]),
                "puct_solve": float(puct["solve_rate"]),
                "uct_ret": _avg_return(uct),
                "puct_ret": _avg_return(puct),
                "uct_steps": float(uct["avg_steps"]),
                "puct_steps": float(puct["avg_steps"]),
            }
        )

    print(
        f"{'Level':<8} {'UCT%':>6} {'PUCT%':>6} "
        f"{'UCT Ret':>8} {'PUCT Ret':>9} {'UCT Steps':>10} {'PUCT Steps':>11}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"{r['level']:<8} "
            f"{r['uct_solve']*100:>5.0f}% "
            f"{r['puct_solve']*100:>5.0f}% "
            f"{r['uct_ret']:>8.3f} "
            f"{r['puct_ret']:>9.3f} "
            f"{r['uct_steps']:>10.1f} "
            f"{r['puct_steps']:>11.1f}"
        )


if __name__ == "__main__":
    main()

