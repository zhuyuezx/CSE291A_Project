# main.py
"""
Self-Evolving Game Agent - Main Entry Point

Usage:
  python main.py --game connect_four --mode train --sims 100 --games 50
  python main.py --game connect_four --mode eval --sims 100 --games 50
  python main.py --game connect_four --mode evolve --sims 100 --games 100
"""
import argparse
import json
import os

from src.config import load_config
from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.tools.manager import ToolPoolManager
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer, PlateauDetector


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Game Agent")
    parser.add_argument("--game", default="connect_four", help="OpenSpiel game name")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "evolve"],
        default="eval",
        help="Run mode",
    )
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--uct-c", type=float, default=1.41, help="UCT exploration constant")
    parser.add_argument("--tool-pool", default="tool_pool", help="Tool pool directory")
    parser.add_argument("--no-tools", action="store_true", help="Run vanilla MCTS (no tools)")
    args = parser.parse_args()

    adapter = GameAdapter(args.game)
    print(f"Game: {adapter.game_description()}")

    # Load tools
    registry = ToolRegistry()
    if not args.no_tools:
        pool_manager = ToolPoolManager(args.tool_pool)
        tool_paths = pool_manager.get_all_tools_for_game(args.game)
        for path in tool_paths:
            try:
                from src.tools.base import load_tool_from_file
                meta, run_fn = load_tool_from_file(path)
                registry.register(meta.name, meta.type, run_fn)
                print(f"  Loaded tool: {meta.name} ({meta.type.value})")
            except Exception as e:
                print(f"  Failed to load {path}: {e}")

    tools_loaded = registry.list_all()
    print(f"Tools loaded: {len(tools_loaded)} ({', '.join(tools_loaded) if tools_loaded else 'none'})")

    engine = MCTSEngine(adapter, registry, simulations=args.sims, uct_c=args.uct_c)
    evaluator = Evaluator(adapter)

    if args.mode == "eval":
        print(f"\nEvaluating vs random ({args.games} games, {args.sims} sims/move)...")
        result = evaluator.evaluate_vs_random(engine, num_games=args.games)
        print(f"Results: {result['wins']}W / {result['losses']}L / {result['draws']}D")
        print(f"Win rate: {result['win_rate']:.1%}")

    elif args.mode == "train":
        detector = PlateauDetector(window_size=max(10, args.games // 5))
        trainer = Trainer(
            adapter, registry, simulations=args.sims, uct_c=args.uct_c,
            plateau_detector=detector,
        )
        print(f"\nTraining vs random ({args.games} games, {args.sims} sims/move)...")
        result = trainer.train(num_games=args.games)
        print(f"Results: {result['wins']}W / {result['games'] - result['wins']}L")
        print(f"Win rate: {result['win_rate']:.1%}")

        wr = detector.current_win_rate()
        if wr is not None:
            print(f"Current rolling win rate: {wr:.1%}")
        if detector.is_plateau():
            print("PLATEAU DETECTED - tool evolution recommended")

    elif args.mode == "evolve":
        print("\nEvolution mode requires LLM API configuration in conf.yaml")
        print("Not yet fully integrated - use Phase 2 tasks")


if __name__ == "__main__":
    main()
