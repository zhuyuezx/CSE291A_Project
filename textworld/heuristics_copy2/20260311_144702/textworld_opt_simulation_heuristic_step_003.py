"""
LLM-generated MCTS tool: simulation
Description: No changes required; the draft implementation is correct and efficient.
Generated:   2026-03-11T15:12:05.496504
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


def _door_is_open(state, direction: str) -> bool:
    """
    Heuristic check whether a door in `direction` is already open.
    The benchmark stores doors as a dict ``state.doors`` mapping direction
    strings to a boolean ``True`` when open.  If the structure is missing,
    conservatively return False (treat as unknown).
    """
    doors = getattr(state, "doors", None)
    if isinstance(doors, dict):
        return doors.get(direction, False)
    return False


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
    Scaled up (×2) later when added to the terminal reward.
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
    Progress‑biased rollout with:
      * door‑state awareness (avoid reopening/closing already‑correct doors)
      * dynamic distance‑based move weighting
      * stronger penalty for non‑progress informational actions
      * longer stagnation detection (5 steps) with average‑delta check
      * amplified shaping term (×2) to give intermediate progress more impact.
    Returns a shaped reward mixing the terminal game reward with an
    intermediate progress estimate.
    """
    sim_state = state.clone()
    depth = 0

    # History for stagnation detection and recent door actions
    recent_deltas: List[float] = []          # last 5 distance deltas
    recent_door_actions: List[str] = []      # last 2 door actions (open/close)

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        weights: List[float] = []
        before_dist = sim_state.distance_to_goal() or 0  # for dynamic scaling

        for act in legal:
            w = 1.0  # baseline weight

            # --------------------------------------------------------------
            # Informational actions: keep only if they produce *progress*
            # --------------------------------------------------------------
            if act in ("look", "inventory", "task"):
                if not _state_changed(sim_state, act):
                    w = 0.0
                else:
                    # evaluate distance change; also give credit if map_read becomes True
                    _, delta = _apply_and_eval(sim_state, act)
                    map_now = getattr(sim_state, "map_read", False)
                    after_state, _ = _apply_and_eval(sim_state, act)
                    map_after = getattr(after_state, "map_read", False)
                    if delta < 0 or (not map_now and map_after):
                        w += 0.5   # modest reward for genuine progress
                    else:
                        w = 0.0    # otherwise discard

            # --------------------------------------------------------------
            # Map handling
            # --------------------------------------------------------------
            if not getattr(sim_state, "map_read", False):
                if act == "take map":
                    w += 2.0
                if act == "read map":
                    w += 2.0
            else:
                if act == "read map":
                    w = 0.0

            # --------------------------------------------------------------
            # Main task item (coin, etc.)
            # --------------------------------------------------------------
            if not getattr(sim_state, "task_completed", False) and act.startswith("take"):
                w += 1.0

            # --------------------------------------------------------------
            # Door actions with state awareness and short‑term cycle suppression
            # --------------------------------------------------------------
            if act.startswith("open "):
                direction = act.split()[1]
                if _door_is_open(sim_state, direction):
                    w = 0.0          # already open → useless
                else:
                    w += 0.8
            if act.startswith("close "):
                direction = act.split()[1]
                if not _door_is_open(sim_state, direction):
                    w = 0.0          # already closed
                else:
                    # Discourage closing a door that was just opened
                    opened_dirs = [
                        a.split()[1] for a in recent_door_actions if a.startswith("open ")
                    ]
                    if direction in opened_dirs:
                        w = 0.0
                    else:
                        w -= 2.0       # generally penalise closing doors (often backtrack)

            # --------------------------------------------------------------
            # Move actions – dynamic distance‑scaled weighting
            # --------------------------------------------------------------
            if act.startswith("move "):
                _, delta = _apply_and_eval(sim_state, act)
                if delta < 0:  # moved closer
                    progress_frac = (-delta) / max(1, before_dist)
                    w += 2.0 + 3.0 * progress_frac
                else:
                    w -= 0.8   # moved away or no change

            # Clamp to non‑negative
            if w < 0.0:
                w = 0.0
            weights.append(w)

        # Choose action according to computed non‑negative weights
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
        if len(recent_deltas) > 5:
            recent_deltas.pop(0)

        # Update door‑action history after the action is taken
        if action.startswith(("open ", "close ")):
            recent_door_actions.append(action)
            if len(recent_door_actions) > 2:
                recent_door_actions.pop(0)

        # Extended stagnation detection:
        # 5 consecutive non‑negative deltas AND average delta >= 0.
        if len(recent_deltas) == 5 and all(d >= 0 for d in recent_deltas):
            avg_delta = sum(recent_deltas) / 5.0
            if avg_delta >= 0.0:
                terminal_reward = sim_state.returns()[perspective_player]
                return terminal_reward + 2.0 * _progress_score(sim_state)

        depth += 1

        # Early successful termination when progress is already high
        prog = _progress_score(sim_state)
        if prog >= 0.9:
            terminal_reward = sim_state.returns()[perspective_player]
            return terminal_reward + 2.0 * prog

    # Rollout finished (terminal or depth limit reached)
    terminal_reward = sim_state.returns()[perspective_player]
    final_progress = _progress_score(sim_state)
    return terminal_reward + 2.0 * final_progress
