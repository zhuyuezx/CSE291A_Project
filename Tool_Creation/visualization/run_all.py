"""
CLI entry point to generate all core visualization figures.

Usage (from Tool_Creation/):

    python -m visualization.run_all
    python -m visualization.run_all --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_game_config
from .plot_solve_rate import plot_solve_rate, plot_cross_game_summary
from .plot_steps import plot_steps_comparison
from .plot_optimization_progress import plot_optimization_progress
from .demo_visit_distribution import main as demo_visit_main
from .plot_mcts_tree import run_mcts_tree_demo
from .plot_game_trajectory import run_trajectory_demo
from .plot_principal_variation import run_principal_variation_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving PNGs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_game_config()
    game_name = cfg["game_name"]
    levels = cfg["levels"]
    level_name = levels[0] if levels else "level1"
    summary_path = str(cfg["summary_path"])
    records_dir = str(cfg["records_dir"])
    game_label = game_name.lower().replace(" ", "_")

    # 1. Baseline vs optimized solve rate (active game)
    plot_solve_rate(
        save_path=str(out_dir / f"{game_label}_solve_rate.png"),
        show=args.show,
        levels=levels,
        game_name=game_name,
        summary_path=summary_path,
        records_dir=records_dir,
    )

    # 2. Steps-to-solve comparison (active game)
    plot_steps_comparison(
        save_path=str(out_dir / f"{game_label}_steps.png"),
        show=args.show,
        levels=levels,
        game_name=game_name,
        summary_path=summary_path,
        records_dir=records_dir,
    )

    # 3. Optimization progress over iterations (from summary JSON if present)
    plot_optimization_progress(
        save_path=str(out_dir / "optimization_progress.png"),
        show=args.show,
        summary_path=summary_path,
    )

    # 4. Cross-game summary
    plot_cross_game_summary(
        save_path=str(out_dir / "cross_game_summary.png"),
        show=args.show,
    )

    # 5. MCTS root search behavior — per-level PNGs for all configured levels
    #    (also launches interactive viewer when --show is used)
    demo_visit_main(show=args.show)

    # 6. MCTS tree snapshot — baseline vs optimized on same state
    run_mcts_tree_demo(
        level_name=level_name,
        max_depth=3,
        save_path=str(out_dir / "mcts_tree_snapshot.png"),
        show=args.show,
    )

    # 7. Game trajectory (PNG + GIF), optional baseline comparison (active game)
    run_trajectory_demo(
        level_name=level_name,
        output_dir=str(out_dir),
        compare_baseline=True,
        make_gif=True,
        show=args.show,
    )

    # 7b. Sokoban trajectory (PNG + GIF) so Sokoban always has GIFs like Rush Hour
    run_trajectory_demo(
        game="sokoban",
        level_name="level3",
        output_dir=str(out_dir),
        compare_baseline=True,
        make_gif=True,
        show=args.show,
    )

    # 8. Principal Variation (best path by visits), top-10
    run_principal_variation_demo(
        level_name=level_name,
        max_depth=10,
        output_dir=str(out_dir),
        show=args.show,
    )


if __name__ == "__main__":
    main()

