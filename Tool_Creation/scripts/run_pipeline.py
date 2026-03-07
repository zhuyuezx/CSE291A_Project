#!/usr/bin/env python3
"""
Run the full MCTS + LLM optimization pipeline.

This is the one-step command-line entry point that mirrors the notebook
``scripts/test_llm_pipeline.ipynb``.

Configuration comes from two files in MCTS_tools/:
  - hyperparams/default_hyperparams.py  — engine params, game identity,
    optimisation settings (phases, num_iters, …)
  - training_logic/<game>_training.py   — levels, mastery, pick_next_level()

Usage:
    python scripts/run_pipeline.py              # default config
    python scripts/run_pipeline.py --iters 20   # override iteration count
    python scripts/run_pipeline.py --quiet       # minimal output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure Tool_Creation (parent of scripts/) is on sys.path
_TOOL_CREATION_DIR = Path(__file__).resolve().parent.parent
if str(_TOOL_CREATION_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_CREATION_DIR))

from orchestrator import OptimizationRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full MCTS + LLM optimisation pipeline.",
    )
    p.add_argument(
        "--iters", type=int, default=None,
        help="Override NUM_ITERS from hyperparams file.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    # ── Build runner from config ─────────────────────────────────────
    runner = OptimizationRunner.from_config(verbose=verbose)

    if args.iters is not None:
        runner.num_iters = args.iters

    if verbose:
        hp = runner.current_hyperparams
        print(f"Game         : {runner.game_name}")
        print(f"Phases       : {runner.phases}")
        print(f"Iterations   : {runner.num_iters}")
        print(f"Hyperparams  : {hp}")
        print(f"Levels       : {runner.levels}")
        print(f"Start level  : {runner.start_level}")
        print()

    # ── Run the iterative optimisation loop ──────────────────────────
    summary = runner.run()

    # ── Print final summary ──────────────────────────────────────────
    best_fns = summary["best_fns"]
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best fns     : { {p: ('set' if f else 'None') for p, f in best_fns.items()} }")
    print(f"Hyperparams  : {summary['current_hyperparams']}")
    print(f"Mastered     : {sorted(summary.get('mastered_levels', set()))}")
    print(f"Active       : {summary['active_levels']}")
    if summary.get("level_best_scores"):
        print("Per-level best composites:")
        for lv in sorted(summary["level_best_scores"]):
            bl = summary["level_baselines"].get(lv, {})
            delta = summary["level_best_scores"][lv] - bl.get("composite", 0)
            tag = " [MASTERED]" if lv in summary.get("mastered_levels", set()) else ""
            print(f"  {lv}: {summary['level_best_scores'][lv]:.4f} "
                  f"(baseline={bl.get('composite', 0):.4f}, Δ={delta:+.4f}){tag}")
    print("=" * 60)


if __name__ == "__main__":
    main()
