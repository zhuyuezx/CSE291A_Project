"""
LLM-generated MCTS tool: simulation
Description: No changes required; the function is correct and efficient.
Generated:   2026-03-11T00:20:45.112314
"""

import math
from typing import Any

def default_simulation(
    state: Any,
    perspective_player: int,
    max_depth: int = 0,
) -> float:
    """
    Refined leaf evaluation for MCTS.

    • Returns 1.0 for solved states.
    • Returns the terminal payoff for any terminal (dead‑locked or max‑step) state.
    • Detects simple corner dead‑locks early and returns 0.0 (hard penalty).
    • Otherwise computes a blended cost based on:
        – h : sum of Manhattan distances from each box to its nearest target
        – g : total steps taken so far (walk + pushes)
      and combines it with a progress bonus proportional to the fraction of
      boxes already placed on targets.
    The final reward is clamped to [0.0, 1.0] to give the back‑propagation
    a rich, discriminative signal.
    """
    # --------------------------------------------------------------------- #
    # Helper: simple corner dead‑lock detection
    # --------------------------------------------------------------------- #
    def _simple_corner_deadlock(s) -> bool:
        """
        Detect classic corner dead‑locks: a non‑target box trapped by two
        orthogonal obstacles (walls or other boxes). This cheap check catches
        the most common unsolvable patterns.
        """
        walls = s.walls
        boxes = s.boxes
        targets = s.targets

        corner_dirs = [((-1, 0), (0, -1)),  # up & left
                       ((-1, 0), (0, 1)),   # up & right
                       ((1, 0), (0, -1)),   # down & left
                       ((1, 0), (0, 1))]    # down & right

        for bx, by in boxes:
            if (bx, by) in targets:
                continue
            for (dx1, dy1), (dx2, dy2) in corner_dirs:
                n1 = (bx + dx1, by + dy1)
                n2 = (bx + dx2, by + dy2)
                if (n1 in walls or n1 in boxes) and (n2 in walls or n2 in boxes):
                    return True
        return False

    # --------------------------------------------------------------------- #
    # 1) Terminal / solved handling
    # --------------------------------------------------------------------- #
    if state.is_terminal():
        # Terminal includes dead‑lock or max‑step limit; rely on GameState's
        # return vector.
        return float(state.returns()[perspective_player])

    if state.boxes_on_targets() == state.num_targets:
        # All boxes are on goals – highest possible reward.
        return 1.0

    # --------------------------------------------------------------------- #
    # 2) Early dead‑lock detection (hard penalty)
    # --------------------------------------------------------------------- #
    if _simple_corner_deadlock(state):
        # Immediate dead‑lock – treat as the worst possible leaf.
        return 0.0

    # --------------------------------------------------------------------- #
    # 3) Core heuristic components
    # --------------------------------------------------------------------- #
    # h – box‑to‑target Manhattan distance (lower is better)
    try:
        h = state.total_box_distance()
    except Exception:
        # Fallback to the shared box‑only heuristic if the method is absent.
        from astar_globals import h_sokoban_box_only as _h_box_only
        h = _h_box_only(state)

    # g – total steps already spent (walk + pushes)
    g = getattr(state, "steps", 0)

    # progress – fraction of boxes already placed on targets
    progress = (
        state.boxes_on_targets() / state.num_targets
        if state.num_targets > 0
        else 0.0
    )

    # --------------------------------------------------------------------- #
    # 4) Parameterised blend
    # --------------------------------------------------------------------- #
    # Weight for step cost – increased to give walks comparable influence.
    LAMBDA = 1.0      # step‑cost multiplier
    # Exponential decay factor – smaller γ yields a wider value range.
    GAMMA = 0.15
    # Bonus weight for each box already on a target.
    BONUS_WEIGHT = 0.4

    blended_cost = h + LAMBDA * g
    exp_term = math.exp(-GAMMA * blended_cost)

    # Add a linear progress bonus (0 … BONUS_WEIGHT)
    reward = exp_term + BONUS_WEIGHT * progress

    # Clamp to the valid range.
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    return float(reward)
