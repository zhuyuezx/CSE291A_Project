"""
LLM-generated MCTS tool: simulation
Description: Fixed BenchmarkConfig handling and ensured robust heuristic scoring.
Generated:   2026-03-11T00:40:13.209703
"""

import random
from typing import List, Any


def _action_score(state: Any, action: str) -> float:
    """
    Compute a cheap heuristic score for an action.

    Higher scores correspond to actions that are more likely to make progress
    toward the goal while avoiding useless “look/inventory/task” actions and
    discouraging door oscillations.

    The function only relies on the public attributes / methods of the game
    state.
    """
    a = action.lower()

    # ``state.config`` is a ``BenchmarkConfig`` object, not a dict.
    cfg = getattr(state, "config", None)
    variant = getattr(cfg, "variant", "deterministic") if cfg is not None else "deterministic"

    # ----- movement actions -------------------------------------------------
    if a.startswith("go ") or a.startswith("move ") or a.startswith("walk "):
        weight = 10.0
        # In stochastic mode, favour moves that actually reduce distance.
        if variant == "stochastic" and hasattr(state, "distance_to_goal"):
            try:
                before = state.distance_to_goal()
                tmp = state.clone()
                tmp.apply_action(action)
                after = tmp.distance_to_goal()
                if after < before:          # safe move
                    weight *= 1.2
            except Exception:
                # Any failure (e.g., action not applicable) falls back to the base weight.
                pass
        return weight

    # ----- mapreader specific actions ---------------------------------------
    if "take map" in a:
        # Taking the map is essential before reading it.
        return 9.0 if not getattr(state, "map_read", False) else 1.0
    if "read map" in a:
        # Reading the map is the second crucial step.
        return 8.0 if not getattr(state, "map_read", False) else 1.0

    # ----- generic take actions (coin, other items) -------------------------
    if a.startswith("take "):
        return 7.0

    # ----- door actions -------------------------------------------------------
    if a.startswith("open ") or a.startswith("close "):
        weight = 2.0
        if hasattr(state, "distance_to_goal"):
            try:
                before = state.distance_to_goal()
                tmp = state.clone()
                tmp.apply_action(action)
                after = tmp.distance_to_goal()
                if after < before:
                    weight *= 1.5
            except Exception:
                pass
        return weight

    # ----- cheap/no‑progress actions -----------------------------------------
    if a.startswith("look") or a.startswith("inventory") or a.startswith("task"):
        return 0.5

    # ----- fallback -----------------------------------------------------------
    return 1.0


def _select_weighted_action(state: Any) -> str:
    """
    Choose an action from the legal set using the scores from
    ``_action_score``. Falls back to uniform random if all scores sum to 0.
    """
    legal: List[str] = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal actions available for simulation.")

    scores = [_action_score(state, act) for act in legal]
    total = sum(scores)

    if total == 0:
        # All scores are zero – revert to uniform random choice.
        return random.choice(legal)

    # ``random.choices`` returns a list; we need the single selected element.
    return random.choices(legal, weights=scores, k=1)[0]


def _realistic_max_depth(state: Any, supplied_max: int) -> int:
    """
    Clamp the depth to a sensible bound.

    Episodes in this benchmark never exceed ~50 steps; we allow a few
    extra steps for stochastic hiccups.
    """
    base_steps = getattr(state, "steps", 0)
    heuristic = base_steps + 30            # planned steps + safety margin
    # Clamp between 10 and 70, then respect the caller's supplied max.
    return min(supplied_max, max(10, min(70, heuristic)))


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Progress‑biased simulation rollout.

    * Samples actions proportionally to a cheap progress heuristic.
    * Terminates as soon as the task is completed (or terminal).
    * Caps depth to a realistic bound (default ≈ 50‑70 steps).
    * In stochastic variants, slightly favours moves whose accidental
      substitutes are also beneficial.

    Returns:
        The reward for ``perspective_player`` extracted from the final state.
    """
    # Work on a fresh copy so the original MCTS tree node is unchanged.
    sim_state = state.clone()

    # Make sure we never run an excessively long rollout.
    max_depth = _realistic_max_depth(sim_state, max_depth)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        # Choose an action guided by the heuristic.
        action = _select_weighted_action(sim_state)

        # Apply the chosen action and advance the simulation.
        sim_state.apply_action(action)
        depth += 1

        # Early exit if the task has been solved.
        if getattr(sim_state, "task_completed", False):
            break

    # ``returns()`` gives a sequence (one entry per player).  We index the
    # requested player's reward.
    return sim_state.returns()[perspective_player]
