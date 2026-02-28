# experiments/run_evaluation.py
"""
Full evaluation suite:
1. Win rate tables at various sim budgets
2. Sample efficiency curves
3. Cross-game transfer analysis
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.tools.manager import ToolPoolManager
from src.tools.base import load_tool_from_file
from src.training.evaluator import Evaluator


GAMES = ["connect_four", "quoridor"]
SIM_BUDGETS = [50, 100, 500, 1000]
GAMES_PER_EVAL = 50
TOOL_POOL_DIR = "tool_pool"


def load_registry(game_name: str, pool_dir: str) -> ToolRegistry:
    registry = ToolRegistry()
    manager = ToolPoolManager(pool_dir)
    for path in manager.get_all_tools_for_game(game_name):
        try:
            meta, run_fn = load_tool_from_file(path)
            registry.register(meta.name, meta.type, run_fn)
        except Exception:
            continue
    return registry


def run_win_rate_table(game_name: str):
    print(f"\n{'='*60}")
    print(f"Win Rate Table: {game_name}")
    print(f"{'='*60}")

    adapter = GameAdapter(game_name)
    evaluator = Evaluator(adapter)

    results = {}
    for sims in SIM_BUDGETS:
        # Vanilla
        vanilla_engine = MCTSEngine(adapter, ToolRegistry(), simulations=sims)
        vanilla_result = evaluator.evaluate_vs_random(vanilla_engine, GAMES_PER_EVAL)

        # With tools
        tool_registry = load_registry(game_name, TOOL_POOL_DIR)
        tool_engine = MCTSEngine(adapter, tool_registry, simulations=sims)
        tool_result = evaluator.evaluate_vs_random(tool_engine, GAMES_PER_EVAL)

        results[sims] = {
            "vanilla": vanilla_result["win_rate"],
            "with_tools": tool_result["win_rate"],
        }

        print(f"  {sims} sims: vanilla={vanilla_result['win_rate']:.1%}, tools={tool_result['win_rate']:.1%}")

    return results


def run_sample_efficiency(game_name: str):
    print(f"\n{'='*60}")
    print(f"Sample Efficiency: {game_name}")
    print(f"{'='*60}")

    adapter = GameAdapter(game_name)
    evaluator = Evaluator(adapter)
    tool_registry = load_registry(game_name, TOOL_POOL_DIR)

    vanilla_curve = evaluator.sample_efficiency_curve(
        lambda sims: MCTSEngine(adapter, ToolRegistry(), simulations=sims),
        SIM_BUDGETS, GAMES_PER_EVAL,
    )
    tool_curve = evaluator.sample_efficiency_curve(
        lambda sims: MCTSEngine(adapter, tool_registry, simulations=sims),
        SIM_BUDGETS, GAMES_PER_EVAL,
    )

    for sims in SIM_BUDGETS:
        print(f"  {sims} sims: vanilla={vanilla_curve[sims]:.1%}, tools={tool_curve[sims]:.1%}")

    return {"vanilla": vanilla_curve, "with_tools": tool_curve}


if __name__ == "__main__":
    all_results = {}
    for game in GAMES:
        all_results[game] = {
            "win_rate_table": run_win_rate_table(game),
            "sample_efficiency": run_sample_efficiency(game),
        }

    output_path = "experiments/results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
