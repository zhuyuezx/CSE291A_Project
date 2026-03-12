"""
LLM-generated MCTS tool: simulation
Description: Provide the finalized simulation heuristic implementation.
Generated:   2026-03-11T00:44:53.582264
"""

import random
from typing import List, Any


def _variant(state: Any) -> str:
    """Return the benchmark variant (deterministic / stochastic / punishment)."""
    cfg = getattr(state, "config", None)
    return getattr(cfg, "variant", "deterministic") if cfg is not None else "deterministic"


def _simulate_action(state: Any, action: str):
    """
    Apply *action* to a clone of *state* and return a tuple describing the effect.

    Returns:
        success (bool): whether ``apply_action`` succeeded.
        room_changed (bool): whether the player changed rooms.
        dist_before (float|None), dist_after (float|None): distance to goal before/after
            (both None if ``distance_to_goal`` is unavailable).
    """
    try:
        tmp = state.clone()
        tmp.apply_action(action)
    except Exception:
        return False, False, (None, None)

    room_before = getattr(state, "room", None)
    room_after = getattr(tmp, "room", None)
    room_changed = room_before != room_after

    if hasattr(state, "distance_to_goal"):
        try:
            before = state.distance_to_goal()
            after = tmp.distance_to_goal()
        except Exception:
            before, after = None, None
    else:
        before, after = None, None

    return True, room_changed, (before, after)


def _action_score(state: Any, action: str) -> float:
    """
    Heuristic score for *action* in *state*.

    The score favours actions that make genuine progress (opening doors,
    moving closer to the goal, taking new items) and heavily penalises
    no‑op or already‑satisfied actions, especially in the punishment variant.
    """
    a = action.lower()
    var = _variant(state)

    # ------------------------------------------------------------------ #
    # 1. Movement actions
    # ------------------------------------------------------------------ #
    if a.startswith(("go ", "move ", "walk ")):
        base = 10.0
        success, room_changed, (before, after) = _simulate_action(state, action)
        if not success:
            return 0.1  # illegal move
        if room_changed:
            base *= 1.5
        if before is not None and after is not None:
            if after < before:
                base *= 1.5
            elif after == before:
                base *= 0.5
        return base

    # ------------------------------------------------------------------ #
    # 2. Mapreader specific actions
    # ------------------------------------------------------------------ #
    if "take map" in a:
        inv = getattr(state, "inventory_items", [])
        has_map = any("map" in str(item).lower() for item in inv)
        return 0.2 if has_map else 9.0

    if "read map" in a:
        return 0.2 if getattr(state, "map_read", False) else 8.0

    # ------------------------------------------------------------------ #
    # 3. Generic take actions (coins, other items)
    # ------------------------------------------------------------------ #
    if a.startswith("take "):
        obj = a[5:].strip()
        inv = getattr(state, "inventory_items", [])
        already = any(obj in str(item).lower() for item in inv)
        return 0.2 if already else 7.0

    # ------------------------------------------------------------------ #
    # 4. Door actions
    # ------------------------------------------------------------------ #
    if a.startswith(("open ", "close ")):
        base = 2.0
        success, room_changed, (before, after) = _simulate_action(state, action)
        if not success:
            return 0.1  # illegal door command
        if not (room_changed or (before is not None and after is not None and after != before)):
            return 0.1  # no effect
        if before is not None and after is not None and after < before:
            base *= 1.5
        return base

    # ------------------------------------------------------------------ #
    # 5. Cheap / no‑progress actions
    # ------------------------------------------------------------------ #
    if a.startswith(("look", "inventory", "task")):
        return 0.05 if var == "punishment" else 0.1

    # ------------------------------------------------------------------ #
    # 6. Fallback
    # ------------------------------------------------------------------ #
    return 1.0


def _select_weighted_action(state: Any) -> str:
    """Choose a legal action proportionally to its heuristic score."""
    legal: List[str] = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal actions available for simulation.")

    scores = [_action_score(state, act) for act in legal]
    total = sum(scores)

    if total == 0:
        return random.choice(legal)

    return random.choices(legal, weights=scores, k=1)[0]


def _realistic_max_depth(state: Any, supplied_max: int) -> int:
    """
    Clamp rollout depth to a sensible bound.
    Episodes never exceed ~50 steps; we give a small safety margin.
    """
    base_steps = getattr(state, "steps", 0)
    heuristic = base_steps + 30            # planned steps + safety margin
    return min(supplied_max, max(10, min(70, heuristic)))


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Progress‑biased simulation rollout.

    * Samples actions according to an inexpensive progress heuristic.
    * Stops early when the task is completed or the rollout exceeds a
      realistic depth bound.
    * Returns the reward for *perspective_player* from the final state.
    """
    sim_state = state.clone()
    max_depth = _realistic_max_depth(sim_state, max_depth)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        action = _select_weighted_action(sim_state)
        sim_state.apply_action(action)
        depth += 1

        if getattr(sim_state, "task_completed", False):
            break

    return sim_state.returns()[perspective_player]
