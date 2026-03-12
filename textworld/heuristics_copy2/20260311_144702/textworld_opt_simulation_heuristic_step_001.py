"""
LLM-generated MCTS tool: simulation
Description: Remove duplicate look‑ahead calls and streamline weight computation for move actions.
Generated:   2026-03-11T15:07:36.563396
"""

import random
from typing import List, Tuple

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _apply_and_eval(state, action) -> Tuple[object, float]:
    """
    Apply `action` to a cloned copy of `state` and return the new state
    together with the change in distance to the goal (negative if distance
    decreased, positive otherwise).  This is an inexpensive look‑ahead used
    for weighting actions during the rollout.
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


def _weighted_random(actions: List[str], weights: List[float]) -> str:
    """
    Choose an action proportionally to the supplied positive weights.
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
    Heuristic that reflects how close the rollout is to success.
    Components:
        - distance to goal (negative weight)
        - map already read (bonus)
        - main task completed (bonus)
    Returns a float where higher is better.
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
# Main simulation function (improved)
# ----------------------------------------------------------------------
def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Progress‑biased rollout simulation.

    The rollout prefers actions that move the agent closer to the goal,
    avoid immediate reversals, and encourage taking/reading the map or
    picking up the coin.  It may terminate early when a high progress
    score is achieved, returning a shaped reward that combines the game's
    terminal reward with the intermediate progress estimate.
    """
    sim_state = state.clone()
    depth = 0
    recent_actions: List[str] = []

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        weights: List[float] = []
        for act in legal:
            w = 1.0  # baseline

            # Down‑weight pure informational actions unless nothing else exists.
            if act in ("look", "inventory", "task"):
                w -= 0.5

            # Encourage map handling.
            if not getattr(sim_state, "map_read", False):
                if act == "take map":
                    w += 1.0
                if act == "read map":
                    w -= 0.5  # can't read before taking
            else:
                if act == "read map":
                    w += 1.0

            # Encourage taking the main task item (coin or similar).
            if not getattr(sim_state, "task_completed", False) and act.startswith("take"):
                w += 0.8

            # Prefer moves that reduce distance to goal.
            if act.startswith("move "):
                _, delta = _apply_and_eval(sim_state, act)
                if delta < 0:          # got closer
                    w += 2.0
                else:
                    w -= 0.3

            # Door handling.
            if act.startswith("open "):
                w += 0.6
            if act.startswith("close "):
                # Discourage closing a door just opened.
                if recent_actions and recent_actions[-1].startswith("open "):
                    opened_dir = recent_actions[-1].split()[1]
                    closed_dir = act.split()[1]
                    if opened_dir == closed_dir:
                        w = 0.0
                w -= 0.4

            if w < 0.0:
                w = 0.0
            weights.append(w)

        action = _weighted_random(legal, weights)
        sim_state.apply_action(action)

        recent_actions.append(action)
        if len(recent_actions) > 2:
            recent_actions.pop(0)

        depth += 1

        prog = _progress_score(sim_state)
        if prog >= 0.7:
            terminal_reward = sim_state.returns()[perspective_player]
            return terminal_reward + prog

    terminal_reward = sim_state.returns()[perspective_player]
    final_progress = _progress_score(sim_state)
    return terminal_reward + final_progress
