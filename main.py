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
        print(f"\nEvaluating ({args.games} games, {args.sims} sims/move)...")
        result = evaluator.measure(engine, n_games=args.games)
        print(f"Results: {result.metric_name}={result.raw_value:.2f} (norm={result.normalized_value:.3f})")

    elif args.mode == "train":
        detector = PlateauDetector(window_size=max(10, args.games // 5))
        trainer = Trainer(
            adapter, registry, simulations=args.sims, uct_c=args.uct_c,
            plateau_detector=detector,
        )
        print(f"\nTraining ({args.games} games, {args.sims} sims/move)...")
        result = trainer.train(num_games=args.games)
        metric = adapter.meta.metric_name
        print(f"Results: {metric}={result[metric]:.2f}")

        wr = detector.current_win_rate()
        if wr is not None:
            print(f"Current rolling win rate: {wr:.1%}")
        if detector.is_plateau():
            print("PLATEAU DETECTED - tool evolution recommended")

    elif args.mode == "evolve":
        import time
        from datetime import datetime
        from src.llm.client import LLMClient
        from src.training.trace_recorder import TraceRecorder
        from src.tools.generator import ToolGenerator

        # Create timestamped log directory
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/evolve_{args.game}_{run_ts}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  EVOLVE MODE — {args.game}")
        print(f"  Logs → {log_dir}")
        print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        config = load_config("conf.yaml")
        trace_client = LLMClient.from_config(config, "TRACE_ANALYZER", log_dir=log_dir)
        code_client = LLMClient.from_config(config, "CODE_GENERATOR", log_dir=log_dir)
        val_client = LLMClient.from_config(config, "TOOL_VALIDATOR", log_dir=log_dir)

        generator = ToolGenerator(
            trace_analyzer_client=trace_client,
            code_generator_client=code_client,
            validator_client=val_client,
            game_name=args.game,
        )
        pool_manager = ToolPoolManager(args.tool_pool)
        is_sp = adapter.meta.is_single_player
        metric = adapter.meta.metric_name

        # ─── Step 1: Play games and collect traces ──────────────────
        print(f"\n[1/4] Playing {args.games} games to collect traces...")
        print(f"  Config: sims={args.sims}, uct_c={args.uct_c}, "
              f"{'single-player' if is_sp else 'vs random'}")
        t0 = time.time()
        trainer = Trainer(adapter, registry, simulations=args.sims, uct_c=args.uct_c)

        # Play games with per-game logging
        scores = []
        for i in range(args.games):
            gt0 = time.time()
            if is_sp:
                score = trainer.play_episode()
            else:
                score = trainer.play_game_vs_random(0)
            ge = time.time() - gt0
            scores.append(score)
            print(f"  Game {i+1}/{args.games}: score={score:.2f} ({ge:.1f}s)")

        avg_score = sum(scores) / len(scores) if scores else 0
        elapsed = time.time() - t0
        print(f"  ── Summary: {metric}={avg_score:.2f} "
              f"(min={min(scores):.2f}, max={max(scores):.2f}) in {elapsed:.1f}s")

        # Save game scores to log
        with open(os.path.join(log_dir, "game_scores.json"), "w") as f:
            json.dump({"game": args.game, "sims": args.sims, "scores": scores,
                       "avg": avg_score, "metric": metric}, f, indent=2)

        # ─── Step 2: Select informative traces ──────────────────────
        print(f"\n[2/4] Selecting informative traces...")
        if is_sp:
            # For single-player: pick lowest-scoring games
            all_traces = trainer.recorder.get_traces()
            scored = [(t, t.outcome[0] if t.outcome else 0) for t in all_traces]
            scored.sort(key=lambda x: x[1])
            traces = [t for t, _ in scored[:5]]
            print(f"  Selected {len(traces)} lowest-scoring traces from {len(all_traces)} total")
            for i, (t, s) in enumerate(scored[:5]):
                print(f"    Trace {i+1}: score={s:.2f}, steps={len(t.actions)}")
        else:
            traces = trainer.recorder.select_informative_traces(player=0, n=5)
            print(f"  Selected {len(traces)} informative traces (losses/draws prioritized)")

        if not traces:
            print("  ✗ No traces found. Try more games or fewer sims.")
            return

        traces_text = "\n---\n".join(t.to_string() for t in traces)
        current_tools_desc = ", ".join(registry.list_all()) if registry.list_all() else "None"
        print(f"  Traces text: {len(traces_text)} chars")
        print(f"  Current tools: {current_tools_desc}")

        # Save traces to log
        with open(os.path.join(log_dir, "traces.txt"), "w") as f:
            f.write(traces_text)

        # ─── Step 3: Generate a new tool via LLM ───────────────────
        print(f"\n[3/4] LLM Tool Generation Pipeline...")
        gen_t0 = time.time()
        gen_result = generator.generate_tool(
            traces_text=traces_text,
            game_description=adapter.game_description(),
            current_tools_desc=current_tools_desc,
        )
        gen_elapsed = time.time() - gen_t0

        if not gen_result.valid:
            print(f"\n  ✗ Tool generation failed ({gen_elapsed:.1f}s): {gen_result.error}")
            if gen_result.code:
                code_path = os.path.join(log_dir, "failed_code.py")
                with open(code_path, "w") as f:
                    f.write(gen_result.code)
                print(f"  Failed code saved → {code_path}")
            return

        print(f"\n  ✓ Generated tool in {gen_elapsed:.1f}s:")
        print(f"    Name: {gen_result.spec['name']}")
        print(f"    Type: {gen_result.spec['type']}")
        print(f"    Description: {gen_result.spec['description']}")

        # Save generated code to log
        with open(os.path.join(log_dir, "generated_tool.py"), "w") as f:
            f.write(gen_result.code)
        with open(os.path.join(log_dir, "tool_spec.json"), "w") as f:
            json.dump(gen_result.spec, f, indent=2)

        # ─── Step 4: Save tool + A/B test ───────────────────────────
        print(f"\n[4/4] Save & A/B test...")
        path = pool_manager.save_tool(args.game, gen_result.spec["name"], gen_result.code)
        pool_manager.update_metadata(gen_result.spec["name"], {
            "type": gen_result.spec["type"],
            "description": gen_result.spec["description"],
            "origin_game": args.game,
        })
        print(f"  Saved to: {path}")

        # A/B test
        print(f"  Running 10-game A/B test...")
        ab_t0 = time.time()
        from src.tools.base import load_tool_from_file
        new_registry = ToolRegistry()
        new_registry.load_from_directory(f"{args.tool_pool}/{args.game}")
        new_engine = MCTSEngine(adapter, new_registry, simulations=args.sims, uct_c=args.uct_c)
        new_evaluator = Evaluator(adapter)
        ab_result = new_evaluator.measure(new_engine, n_games=10)
        ab_elapsed = time.time() - ab_t0
        print(f"  A/B result: {ab_result.metric_name}={ab_result.raw_value:.2f} "
              f"(norm={ab_result.normalized_value:.3f}) in {ab_elapsed:.1f}s")

        # Save A/B results
        with open(os.path.join(log_dir, "ab_test.json"), "w") as f:
            json.dump({
                "metric": ab_result.metric_name,
                "raw_value": ab_result.raw_value,
                "normalized_value": ab_result.normalized_value,
                "n_games": ab_result.n_games,
                "baseline_avg": avg_score,
            }, f, indent=2)

        # ─── Summary ────────────────────────────────────────────────
        total = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  EVOLVE COMPLETE — {args.game}")
        print(f"  Total time: {total:.1f}s")
        print(f"  Baseline {metric}: {avg_score:.2f}")
        print(f"  With tool {metric}: {ab_result.raw_value:.2f}")
        delta = ab_result.raw_value - avg_score
        pct = (delta / avg_score * 100) if avg_score != 0 else 0
        print(f"  Delta: {delta:+.2f} ({pct:+.1f}%)")
        print(f"  All logs → {log_dir}/")
        print(f"{'='*60}")

        # Print the tool code
        print(f"\n  Generated tool code:")
        print(f"{'─'*60}")
        print(gen_result.code)
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()

