"""
LLM-generated MCTS tool: simulation
Description: 
Generated:   2026-03-11T15:10:02.590579
"""

import random
from typing import List, Tuple

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _apply_and_eval(state, action) -> Tuple[object, float]:
    """
    Clone `state`, apply `action`, and return the new state together with
    the change in distance to the goal (negative = closer, positive = farther).
    """
    cloned = state.clone()
    old_dist = cloned.distance_to_goal()
    cloned.apply_action(action)
    new_dist = cloned.distance_to_goal()
    if old_dist is None or new_dist is None:
        delta = 0.0
    else:
        delta = new_dist - old_dist  # >0 = farther, <0 = closer
    return cloned, delta


def _state_changed(state, action) -> bool:
    """
    Return True if applying `action` would change any progress‑relevant
    attribute (distance to goal, map_read flag, or task_completed flag).
    """
    before_dist = state.distance_to_goal()
    before_map = getattr(state, "map_read", False)
    before_task = getattr(state, "task_completed", False)

    cloned = state.clone()
    cloned.apply_action(action)

    after_dist = cloned.distance_to_goal()
    after_map = getattr(cloned, "map_read", False)
    after_task = getattr(cloned, "task_completed", False)

    return (
        after_dist != before_dist
        or after_map != before_map
        or after_task != before_task
    )


def _weighted_random(actions: List[str], weights: List[float]) -> str:
    """
    Choose an action proportionally to the supplied positive weights.
    Falls back to uniform random choice if all weights are non‑positive.
    """
    total = sum(weights)
    if total <= 0.0:
        return random.choice(actions)
    r = random.random() * total
    cum = 0.0
    for a, w in zip(actions, weights):
        cum += w
        if r <= cum:
            return a
    return actions[-1]  # safety fallback


def _progress_score(state) -> float:
    """
    Heuristic score indicating rollout progress (higher = better).
    """
    dist = state.distance_to_goal()
    if dist is None:
        dist = 0
    score = -0.1 * dist
    if getattr(state, "map_read", False):
        score += 0.5
    if getattr(state, "task_completed", False):
        score += 0.9
    return score


# ----------------------------------------------------------------------
# Main simulation function (incrementally improved)
# ----------------------------------------------------------------------
def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Progress‑biased rollout with stronger discouragement of loops,
    no‑op informational actions, and distance‑scaled move weighting.
    Returns a shaped reward mixing the terminal game reward with an
    intermediate progress estimate.
    """
    sim_state = state.clone()
    depth = 0

    # History for stagnation detection and door‑cycle avoidance
    recent_deltas: List[float] = []          # last 3 distance deltas
    recent_door_actions: List[str] = []      # last 2 door actions (open/close)

    # Normaliser for distance‑based weighting (avoid division by zero)
    max_dist = sim_state.distance_to_goal()
    if max_dist is None:
        max_dist = 0
    max_dist = max(max_dist, 1)

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        weights: List[float] = []
        for act in legal:
            w = 1.0  # baseline weight

            # --------------------------------------------------------------
            # Informational actions: heavy penalty if they do not change state
            # --------------------------------------------------------------
            if act in ("look", "inventory", "task"):
                if not _state_changed(sim_state, act):
                    w = 0.0          # pure no‑op, avoid completely
                else:
                    w -= 0.5         # minor penalty if they happen to change something

            # --------------------------------------------------------------
            # Map handling
            # --------------------------------------------------------------
            if not getattr(sim_state, "map_read", False):
                if act == "take map":
                    w += 2.0
                if act == "read map":
                    w += 2.0          # can read after taking; gives progress
            else:
                # Map already read → further reads are useless
                if act == "read map":
                    w = 0.0

            # --------------------------------------------------------------
            # Main task item (coin, etc.)
            # --------------------------------------------------------------
            if not getattr(sim_state, "task_completed", False) and act.startswith("take"):
                w += 1.0

            # --------------------------------------------------------------
            # Door actions with short‑term cycle suppression
            # --------------------------------------------------------------
            if act.startswith("open "):
                w += 0.8
            if act.startswith("close "):
                # Discourage closing a door that was opened within the last 2 actions
                opened_dirs = [
                    a.split()[1] for a in recent_door_actions if a.startswith("open ")
                ]
                closed_dir = act.split()[1]
                if closed_dir in opened_dirs:
                    w = 0.0
                else:
                    w -= 0.4

            # --------------------------------------------------------------
            # Move actions – distance‑scaled weighting
            # --------------------------------------------------------------
            if act.startswith("move "):
                _, delta = _apply_and_eval(sim_state, act)
                if delta < 0:          # moved closer
                    progress_frac = (-delta) / max_dist
                    w += 2.0 + 3.0 * progress_frac
                else:
                    # No progress or moved away – penalise
                    w -= 0.8

            # Ensure non‑negative weight
            if w < 0.0:
                w = 0.0
            weights.append(w)

        # Choose action according to the computed weights
        action = _weighted_random(legal, weights)

        # Apply selected action and record distance change
        before_dist = sim_state.distance_to_goal()
        sim_state.apply_action(action)
        after_dist = sim_state.distance_to_goal()
        delta_dist = (
            (after_dist - before_dist)
            if (before_dist is not None and after_dist is not None)
            else 0.0
        )
        recent_deltas.append(delta_dist)
        if len(recent_deltas) > 3:
            recent_deltas.pop(0)

        # Update door‑action history *after* the action is taken
        if action.startswith(("open ", "close ")):
            recent_door_actions.append(action)
            if len(recent_door_actions) > 2:
                recent_door_actions.pop(0)

        # Early‑exit if we have been stuck (no negative delta) for 3 steps
        if len(recent_deltas) == 3 and all(d >= 0 for d in recent_deltas):
            terminal_reward = sim_state.returns()[perspective_player]
            return terminal_reward + _progress_score(sim_state)

        depth += 1

        # Early successful termination based on high progress score
        prog = _progress_score(sim_state)
        if prog >= 0.8:
            terminal_reward = sim_state.returns()[perspective_player]
            return terminal_reward + prog

    # Rollout finished (terminal or depth limit reached)
    terminal_reward = sim_state.returns()[perspective_player]
    final_progress = _progress_score(sim_state)
    return terminal_reward + final_progress
