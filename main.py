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
        from src.llm.client import LLMClient
        from src.training.trace_recorder import TraceRecorder
        from src.tools.generator import ToolGenerator

        config = load_config("conf.yaml")
        trace_client = LLMClient.from_config(config, "TRACE_ANALYZER")
        code_client = LLMClient.from_config(config, "CODE_GENERATOR")
        val_client = LLMClient.from_config(config, "TOOL_VALIDATOR")

        generator = ToolGenerator(
            trace_analyzer_client=trace_client,
            code_generator_client=code_client,
            validator_client=val_client,
            game_name=args.game,
        )
        pool_manager = ToolPoolManager(args.tool_pool)

        # Step 1: Play games and collect traces
        print(f"\n[1/4] Playing {args.games} games to collect traces...")
        trainer = Trainer(adapter, registry, simulations=args.sims, uct_c=args.uct_c)
        result = trainer.train(num_games=args.games)
        print(f"  Results: {result['wins']}W / {result['games'] - result['wins']}L")
        print(f"  Win rate: {result['win_rate']:.1%}")

        # Step 2: Select informative traces (losses / close games)
        traces = trainer.recorder.select_informative_traces(player=0, n=5)
        if not traces:
            print("  No informative traces found (all wins). Try more games or fewer sims.")
            return

        traces_text = "\n---\n".join(t.to_string() for t in traces)
        current_tools_desc = ", ".join(registry.list_all()) if registry.list_all() else "None"
        print(f"  Selected {len(traces)} informative traces for analysis")

        # Step 3: Generate a new tool via LLM
        print(f"\n[2/4] Analyzing traces with LLM...")
        gen_result = generator.generate_tool(
            traces_text=traces_text,
            game_description=adapter.game_description(),
            current_tools_desc=current_tools_desc,
        )

        if not gen_result.valid:
            print(f"  Tool generation failed: {gen_result.error}")
            if gen_result.code:
                print(f"  Last code attempt:\n{gen_result.code[:500]}")
            return

        print(f"  Generated tool: {gen_result.spec['name']} ({gen_result.spec['type']})")
        print(f"  Description: {gen_result.spec['description']}")

        # Step 4: Save the tool
        print(f"\n[3/4] Saving tool to pool...")
        path = pool_manager.save_tool(args.game, gen_result.spec["name"], gen_result.code)
        pool_manager.update_metadata(gen_result.spec["name"], {
            "type": gen_result.spec["type"],
            "description": gen_result.spec["description"],
            "origin_game": args.game,
        })
        print(f"  Saved to: {path}")

        # Step 5: Quick A/B test
        print(f"\n[4/4] Quick A/B test (10 games)...")
        from src.tools.base import load_tool_from_file
        new_registry = ToolRegistry()
        new_registry.load_from_directory(f"{args.tool_pool}/{args.game}")
        new_engine = MCTSEngine(adapter, new_registry, simulations=args.sims, uct_c=args.uct_c)
        new_evaluator = Evaluator(adapter)
        ab_result = new_evaluator.evaluate_vs_random(new_engine, num_games=10)
        print(f"  With new tool: {ab_result['wins']}W / {ab_result['losses']}L / {ab_result['draws']}D ({ab_result['win_rate']:.0%})")
        print(f"\n  Tool code:\n{'='*60}")
        print(gen_result.code)
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
