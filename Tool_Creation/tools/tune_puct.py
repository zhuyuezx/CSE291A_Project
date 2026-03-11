"""
Simple hyperparameter sweep for Sokoban PUCT.

Example:
  python tools/tune_puct.py --level level4 --iterations 500 --games 20
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from statistics import mean

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts import (
    MCTSEngine,
    make_dqn_prior_fn,
    make_puct_expansion,
    make_puct_selection,
)
from mcts.games import Sokoban
from dqn_sokoban_module import action_to_index_fn, encode_state_fn, q_model


def _parse_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _run_uct(level: str, iterations: int, games: int, verbose: bool) -> dict:
    engine = MCTSEngine(Sokoban(level), iterations=iterations, logging=False)
    return engine.play_many(num_games=games, verbose=verbose)


def _run_puct(
    level: str,
    iterations: int,
    games: int,
    cpuct: float,
    temperature: float,
    expansion_strategy: str,
    epsilon: float,
    seed: int,
    verbose: bool,
) -> dict:
    prior_fn = make_dqn_prior_fn(
        q_model=q_model,
        encode_state_fn=encode_state_fn,
        action_to_index_fn=action_to_index_fn,
        temperature=temperature,
    )
    rng = random.Random(seed)
    engine = MCTSEngine(Sokoban(level), iterations=iterations, logging=False)
    engine.set_tool("selection", make_puct_selection(prior_fn, c_puct=cpuct))
    engine.set_tool(
        "expansion",
        make_puct_expansion(
            prior_fn,
            strategy=expansion_strategy,
            epsilon=epsilon,
            rng=rng,
        ),
    )
    return engine.play_many(num_games=games, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="level4")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--cpuct", default="0.3,0.6,1.0,1.5")
    parser.add_argument("--temperature", default="0.7,1.0,1.5,2.0")
    parser.add_argument(
        "--expansion-strategy",
        choices=["greedy", "sample", "epsilon_greedy"],
        default="sample",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-uct", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(
        f"Sweep level={args.level} iterations={args.iterations} games={args.games} "
        f"strategy={args.expansion_strategy}"
    )

    if not args.skip_uct:
        uct = _run_uct(args.level, args.iterations, args.games, args.verbose)
        print(
            f"BASELINE UCT: solved={uct['solved']} solve_rate={uct['solve_rate']} "
            f"avg_steps={uct['avg_steps']}"
        )

    cpucts = _parse_list(args.cpuct)
    temps = _parse_list(args.temperature)
    results = []

    for c, t in itertools.product(cpucts, temps):
        stats = _run_puct(
            level=args.level,
            iterations=args.iterations,
            games=args.games,
            cpuct=c,
            temperature=t,
            expansion_strategy=args.expansion_strategy,
            epsilon=args.epsilon,
            seed=args.seed,
            verbose=args.verbose,
        )
        row = {
            "cpuct": c,
            "temperature": t,
            "solved": stats["solved"],
            "solve_rate": stats["solve_rate"],
            "avg_steps": stats["avg_steps"],
        }
        results.append(row)
        print(
            f"PUCT cpuct={c:.3f} temp={t:.3f}: solved={row['solved']} "
            f"solve_rate={row['solve_rate']} avg_steps={row['avg_steps']}"
        )

    # Rank by solve_rate desc, then avg_steps asc.
    best = sorted(results, key=lambda r: (-r["solve_rate"], r["avg_steps"]))[0]
    print(
        f"BEST: cpuct={best['cpuct']} temperature={best['temperature']} "
        f"solve_rate={best['solve_rate']} avg_steps={best['avg_steps']}"
    )
    print(
        f"MEAN over configs: solve_rate={round(mean(r['solve_rate'] for r in results), 4)} "
        f"avg_steps={round(mean(r['avg_steps'] for r in results), 2)}"
    )


if __name__ == "__main__":
    main()
