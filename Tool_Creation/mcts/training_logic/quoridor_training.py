"""
Training strategy for Quoridor.

Quoridor is a two-player adversarial game, so there are no puzzle
"levels" to progress through.  Instead, training repeatedly plays
self-play games and evaluates whether the MCTS agent is improving.

Required interface
------------------
Constants : LEVELS, START_LEVEL, EVAL_RUNS, SOLVE_WEIGHT, RETURN_WEIGHT,
            REJECT_THRESHOLD, MASTERY_SOLVE_RATE, MASTERY_CONFIRM_RUNS,
            MASTERY_MAX_STEPS
Function  : pick_next_level(active_levels, all_levels, history) -> str
"""

import random

# ── Level configuration ──────────────────────────────────────────────
# For a two-player adversarial game there are no distinct levels.
# We use a single "self_play" level and keep the interface compatible
# with the orchestrator.
LEVELS = ["self_play"]
START_LEVEL = "self_play"

# ── Evaluation parameters ────────────────────────────────────────────
EVAL_RUNS = 5          # self-play games per evaluation round
SOLVE_WEIGHT = 0.6     # weight for P1 win-rate (proxy for "solved")
RETURN_WEIGHT = 0.4    # weight for mean return
REJECT_THRESHOLD = 0.3

# ── Mastery criteria ─────────────────────────────────────────────────
MASTERY_SOLVE_RATE = 0.8     # P1 win-rate target (or combined metric)
MASTERY_CONFIRM_RUNS = 10
MASTERY_MAX_STEPS = 200      # max moves per game before draw


def pick_next_level(active_levels, all_levels, history):
    """Always returns 'self_play' — there is only one mode."""
    return "self_play"
