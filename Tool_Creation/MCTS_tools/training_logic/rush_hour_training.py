"""
Training strategy for Rush Hour.

Defines puzzles, mastery criteria, evaluation parameters, and puzzle
selection strategy specific to the Rush Hour game.

Uses curriculum learning: only unlocks harder tiers after easier ones
are solved at least once.  This avoids wasting time on 50+ move puzzles
when the agent can't yet solve 10-move ones.

Required interface
------------------
Constants : LEVELS, START_LEVEL, EVAL_RUNS, SOLVE_WEIGHT, RETURN_WEIGHT,
            REJECT_THRESHOLD, MASTERY_SOLVE_RATE, MASTERY_CONFIRM_RUNS,
            MASTERY_MAX_STEPS
Function  : pick_next_level(active_levels, all_levels, history) -> str
"""

import random


# ── Level configuration ──────────────────────────────────────────────
# Puzzles ordered by difficulty (optimal move count in parentheses)
LEVELS = [
    "easy1",    # 2 moves optimal
    "easy2",    # 8 moves
    "easy3",    # 10 moves
    "medium1",  # 50 moves
    "medium2",  # 50 moves
    "hard1",    # 51 moves
    "hard2",    # 60 moves
    "hard3",    # 58 moves
]
START_LEVEL = "easy2"

# Tiers for curriculum progression
_TIERS = {
    "easy":   ["easy1", "easy2", "easy3"],
    "medium": ["medium1", "medium2"],
    "hard":   ["hard1", "hard2", "hard3"],
}
_TIER_ORDER = ["easy", "medium", "hard"]

# ── Evaluation parameters ────────────────────────────────────────────
EVAL_RUNS = 3
SOLVE_WEIGHT = 0.6
RETURN_WEIGHT = 0.4
REJECT_THRESHOLD = 0.5

# ── Mastery criteria ─────────────────────────────────────────────────
MASTERY_SOLVE_RATE = 1.0
MASTERY_CONFIRM_RUNS = 7
MASTERY_MAX_STEPS = None


def _solved_levels(history):
    """Return set of levels that were solved at least once in history."""
    solved = set()
    for record in history:
        results = record.get("results") or record.get("eval_results") or []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and r.get("solved"):
                    lvl = record.get("level")
                    if lvl:
                        solved.add(lvl)
        if record.get("solved"):
            lvl = record.get("level")
            if lvl:
                solved.add(lvl)
    return solved


def _unlocked_levels(history):
    """Return levels available given curriculum progress.

    A tier is unlocked when at least one level from the previous tier
    has been solved.  The easy tier is always unlocked.
    """
    solved = _solved_levels(history)
    unlocked = list(_TIERS["easy"])         # easy always available

    # Unlock medium if any easy solved
    if solved & set(_TIERS["easy"]):
        unlocked.extend(_TIERS["medium"])

    # Unlock hard if any medium solved
    if solved & set(_TIERS["medium"]):
        unlocked.extend(_TIERS["hard"])

    return unlocked


def pick_next_level(active_levels, all_levels, history):
    """
    Select the next puzzle to train on using curriculum learning.

    Only offers puzzles from unlocked difficulty tiers.  Within
    unlocked tiers, picks from active (non-mastered) levels.

    Parameters
    ----------
    active_levels : list[str]
        Levels not yet mastered.
    all_levels : list[str]
        All available levels.
    history : list[dict]
        Past iteration records.

    Returns
    -------
    str : the chosen level name.
    """
    unlocked = _unlocked_levels(history)
    candidates = [l for l in active_levels if l in unlocked]
    if candidates:
        return random.choice(candidates)
    # All unlocked levels mastered — try remaining active
    if active_levels:
        return random.choice(active_levels)
    return random.choice(all_levels)
