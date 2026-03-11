"""
Training strategy for Sokoban.

Defines levels, mastery criteria, evaluation parameters, and level
selection strategy specific to the Sokoban game.

This module is loaded by the orchestrator's OptimizationRunner.
Different games should have their own training logic file (e.g.
tictactoe_training.py, sliding_puzzle_training.py) with the same
interface.

Required interface
------------------
Constants : LEVELS, START_LEVEL, EVAL_RUNS, SOLVE_WEIGHT, RETURN_WEIGHT,
            REJECT_THRESHOLD, MASTERY_SOLVE_RATE, MASTERY_CONFIRM_RUNS,
            MASTERY_MAX_STEPS
Function  : pick_next_level(active_levels, all_levels, history) -> str
"""

import random


# ── Level configuration ──────────────────────────────────────────────
LEVELS = [ "level1", "level2", "level3", "level4", "level5", "level6", "level7", "level8", "level9", "level10"]
START_LEVEL = "level5"

# ── Evaluation parameters ────────────────────────────────────────────
EVAL_RUNS = 3
SOLVE_WEIGHT = 0.6
RETURN_WEIGHT = 0.4
REJECT_THRESHOLD = 0.5

# ── Mastery criteria ─────────────────────────────────────────────────
MASTERY_SOLVE_RATE = 1.0
MASTERY_CONFIRM_RUNS = 7
MASTERY_MAX_STEPS = None


def pick_next_level(active_levels, all_levels, history):
    """
    Select the next level to train on.

    Parameters
    ----------
    active_levels : list[str]
        Levels not yet mastered.
    all_levels : list[str]
        All available levels.
    history : list[dict]
        Past iteration records (can inform curriculum strategies).

    Returns
    -------
    str : the chosen level name.
    """
    if active_levels:
        return random.choice(active_levels)
    return random.choice(all_levels)
