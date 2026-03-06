#!/usr/bin/env python3
"""
Run a full Sokoban example: MCTS play → trace → prompt build.

Usage:
    python run_sokoban.py                         # defaults: level1, 200 iters
    python run_sokoban.py --level level3
    python run_sokoban.py --level level5 --iterations 500 --games 3
    python run_sokoban.py --level level1 --phase backpropagation
    python run_sokoban.py --no-prompt              # skip prompt building
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the Tool_Creation directory is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from mcts import MCTSEngine
from mcts.games import Sokoban
from LLM.prompt_builder import PromptBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MCTS on Sokoban with tracing and prompt generation."
    )
    p.add_argument(
        "--level", default="level1",
        help="Sokoban level name (level1–level10). Default: level1",
    )
    p.add_argument(
        "--iterations", type=int, default=200,
        help="MCTS iterations per move. Default: 200",
    )
    p.add_argument(
        "--games", type=int, default=1,
        help="Number of games to play. Default: 1",
    )
    p.add_argument(
        "--max-rollout-depth", type=int, default=50,
        help="Max rollout depth for simulation. Default: 50",
    )
    p.add_argument(
        "--phase", default="simulation",
        choices=["selection", "expansion", "simulation", "backpropagation"],
        help="Target MCTS phase for prompt building. Default: simulation",
    )
    p.add_argument(
        "--max-moves", type=int, default=None,
        help="Max moves per trace shown in the prompt. Default: all",
    )
    p.add_argument(
        "--no-prompt", action="store_true",
        help="Skip prompt building (only play and log).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print each move during play.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Play game(s) with logging ─────────────────────────────────
    print(f"=== Sokoban ({args.level}) | {args.iterations} iters | {args.games} game(s) ===\n")

    game = Sokoban(args.level)
    engine = MCTSEngine(
        game,
        iterations=args.iterations,
        max_rollout_depth=args.max_rollout_depth,
        logging=True,
    )

    if args.games == 1:
        result = engine.play_game(verbose=args.verbose)
        results = [result]
        tag = "SOLVED" if result["solved"] else "UNSOLVED"
        print(f"Result: {tag} in {result['steps']} steps")
        print(f"Trace:  {result.get('log_file', 'N/A')}")
    else:
        stats = engine.play_many(num_games=args.games, verbose=args.verbose)
        results = stats["results"]
        print(f"\nAggregate: {stats['solved']}/{stats['total']} solved "
              f"({stats['solve_rate']*100:.1f}%), avg {stats['avg_steps']} steps")

    # ── 2. Build prompt from traces ──────────────────────────────────
    if args.no_prompt:
        print("\n(Prompt building skipped)")
        return

    log_files = [r["log_file"] for r in results if r.get("log_file")]
    if not log_files:
        print("\nNo trace files generated — skipping prompt.")
        return

    print(f"\n=== Building prompt (phase: {args.phase}) ===")

    pb = PromptBuilder(game="sokoban", target_phase=args.phase)
    sources = engine.get_tool_source()
    prompt = pb.build(
        record_files=log_files,
        tool_source=sources.get(args.phase),
        max_moves_per_trace=args.max_moves,
    )
    path = pb.save(prompt)
    print(f"Prompt saved to: {path}")
    print(f"Prompt length:   {len(prompt)} chars")


if __name__ == "__main__":
    main()
