#!/usr/bin/env python3
"""
Simple TextWorld Coin Visualization Demo.

Runs two MCTS configurations (baseline vs high-iteration) and exports:
- trajectory_baseline.gif
- trajectory_high_iter.gif
- trajectory_compare.gif (side-by-side)

All outputs are written under: Tool_Creation/visualization/output/textworld/
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUTPUT_DIR = PROJECT_ROOT / "Tool_Creation" / "visualization" / "output" / "textworld"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Tool_Creation"))

from textworld.tools.run_textworld_coin_self_evolving import DEFAULT_PARAMS, make_engine
from visualization.textworld import TrajectoryVisualizer

# Local helper (kept under visualization/textworld/ as requested)
from visualization.textworld.coin_with_viz import create_textworld_game_state, play_game_with_trajectory


def run_visualization_demo() -> None:
    print("TextWorld Coin Trajectory Visualization Demo")
    print("=" * 60)

    game_params = DEFAULT_PARAMS[0]  # "numLocations=5,includeDoors=1,numDistractorItems=0"
    print(f"Game parameters: {game_params}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = make_engine(game_params, iterations=100, max_depth=50, max_steps=50, logging=False)

    print("Running baseline MCTS game...")
    result, trajectory = play_game_with_trajectory(engine, engine.game.config)
    print(f"  solved={result['solved']} steps={result['steps']} return={result['returns'][0]:.3f}")

    game_state = create_textworld_game_state(engine.game.config)
    viz = TrajectoryVisualizer(trajectory, game_state=game_state)
    viz.visualize(output_path=OUTPUT_DIR / "trajectory_baseline.gif", fps=1)

    print("Running high-iteration MCTS game (500)...")
    engine_high_iter = make_engine(game_params, iterations=500, max_depth=50, max_steps=50, logging=False)
    result2, trajectory2 = play_game_with_trajectory(engine_high_iter, engine_high_iter.game.config)
    print(f"  solved={result2['solved']} steps={result2['steps']} return={result2['returns'][0]:.3f}")

    viz2 = TrajectoryVisualizer(trajectory2, game_state=game_state)
    viz2.visualize(output_path=OUTPUT_DIR / "trajectory_high_iter.gif", fps=1)

    viz.compare_side_by_side(
        trajectory2,
        output_path=OUTPUT_DIR / "trajectory_compare.gif",
        other_game_state=game_state,
        agent_names=("Baseline", "High-Iter"),
        fps=1,
    )

    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_visualization_demo()

