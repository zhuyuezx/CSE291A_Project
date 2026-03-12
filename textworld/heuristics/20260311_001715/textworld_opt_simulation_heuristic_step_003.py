"""
LLM-generated MCTS tool: simulation
Description: Fixed `_is_opposite` to safely handle `None` and added defensive checks; updated imports and retained original heuristic logic.
Generated:   2026-03-11T00:51:09.366017
"""

import random
import collections
from typing import List, Any, Tuple, Optional

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _variant(state: Any) -> str:
    """Return the benchmark variant (deterministic / stochastic / punishment)."""
    cfg = getattr(state, "config", None)
    return getattr(cfg, "variant", "deterministic") if cfg is not None else "deterministic"


def _estimated_distance(state: Any) -> Optional[float]:
    """
    Cheap fallback distance estimator.

    Uses ``state.distance_to_goal`` when available; otherwise, if a
    ``known_goal_room`` attribute exists, performs BFS on ``state.graph``
    (assumed to be a dict mapping room -> {dir: neighbor_room}) to compute
    shortest path length. Returns None if no sensible estimate can be made.
    """
    if hasattr(state, "distance_to_goal"):
        try:
            d = state.distance_to_goal()
            return None if d is None else float(d)
        except Exception:
            pass

    goal = getattr(state, "known_goal_room", None)
    if goal is None:
        return None

    graph = getattr(state, "graph", {})
    start = getattr(state, "room", None)
    if start is None or start not in graph:
        return None

    # Simple BFS
    frontier = collections.deque([(start, 0)])
    visited = {start}
    while frontier:
        node, dist = frontier.popleft()
        if node == goal:
            return float(dist)
        for neigh in graph.get(node, {}).values():
            if neigh not in visited:
                visited.add(neigh)
                frontier.append((neigh, dist + 1))
    return None  # unreachable


def _simulate_action(state: Any, action: str) -> Tuple[bool, bool, Tuple[Optional[float], Optional[float]]]:
    """
    Apply *action* to a clone of *state* and return a tuple describing the effect.

    Returns:
        success (bool): whether ``apply_action`` succeeded.
        room_changed (bool): whether the player changed rooms.
        (dist_before, dist_after): distance to goal before/after (None if unavailable).
    """
    try:
        tmp = state.clone()
        tmp.apply_action(action)
    except Exception:
        return False, False, (None, None)

    room_before = getattr(state, "room", None)
    room_after = getattr(tmp, "room", None)
    room_changed = room_before != room_after

    # Prefer the official distance_to_goal if it exists and returns a number.
    if hasattr(state, "distance_to_goal"):
        try:
            before = state.distance_to_goal()
            after = tmp.distance_to_goal()
            before = None if before is None else float(before)
            after = None if after is None else float(after)
            return True, room_changed, (before, after)
        except Exception:
            pass

    # Fallback: try to estimate distance using graph+known goal (if any)
    before = _estimated_distance(state)
    after = _estimated_distance(tmp)
    return True, room_changed, (before, after)


def _is_opposite(action_a: Optional[str], action_b: Optional[str]) -> bool:
    """
    Detect immediate opposite door actions (open X ↔ close X).

    Returns ``False`` if either argument is ``None``.
    """
    if not action_a or not action_b:
        return False

    a = action_a.lower().split()
    b = action_b.lower().split()
    if len(a) >= 2 and len(b) >= 2 and a[0] in ("open", "close") and b[0] in ("open", "close"):
        if a[1] == b[1] and a[0] != b[0]:
            return True
    return False


def _action_score(state: Any, action: str) -> float:
    """
    Heuristic score for *action* in *state*.

    The score favours genuine progress and heavily penalises
    no‑op or regressive actions.
    """
    a = action.lower()
    var = _variant(state)

    # --------------------------------------------------------------
    # 1. Movement actions
    # --------------------------------------------------------------
    if a.startswith(("go ", "move ", "walk ")):
        base = 10.0
        success, room_changed, (before, after) = _simulate_action(state, action)
        if not success:
            return 0.1  # illegal move
        if room_changed:
            base *= 1.5
        if before is not None and after is not None:
            if after < before:
                base *= 1.5           # getting closer
            elif after > before:
                base *= 0.5           # moving away
        return base

    # --------------------------------------------------------------
    # 2. Mapreader specific actions
    # --------------------------------------------------------------
    if "take map" in a:
        inv = getattr(state, "inventory_items", [])
        has_map = any("map" in str(item).lower() for item in inv)
        return 0.2 if has_map else 9.0

    if "read map" in a:
        return 0.2 if getattr(state, "map_read", False) else 8.0

    # --------------------------------------------------------------
    # 3. Generic take actions (coins, other items)
    # --------------------------------------------------------------
    if a.startswith("take "):
        obj = a[5:].strip()
        inv = getattr(state, "inventory_items", [])
        already = any(obj in str(item).lower() for item in inv)

        # If the object is visible in the current room (look_text contains it),
        # dramatically increase the incentive to take it.
        look = getattr(state, "look_text", lambda: "")()
        if not already and obj in look.lower():
            return 20.0
        return 0.2 if already else 7.0

    # --------------------------------------------------------------
    # 4. Door actions (open / close)
    # --------------------------------------------------------------
    if a.startswith(("open ", "close ")):
        # Base is deliberately low to avoid frivolous door fiddling.
        base = 0.5
        success, room_changed, (before, after) = _simulate_action(state, action)
        if not success:
            return 0.1  # illegal door command

        # If action has no observable effect, heavily penalise.
        if not (room_changed or (before is not None and after is not None and after != before)):
            return 0.1

        # Reward actions that reduce distance to goal.
        if before is not None and after is not None and after < before:
            base *= 2.0

        # In punishment mode, penalise actions that increase distance.
        if var == "punishment" and before is not None and after is not None and after > before:
            return 0.05

        return base

    # --------------------------------------------------------------
    # 5. Cheap / no‑progress actions
    # --------------------------------------------------------------
    if a.startswith(("look", "inventory", "task")):
        return 0.05 if var == "punishment" else 0.1

    # --------------------------------------------------------------
    # 6. Fallback
    # --------------------------------------------------------------
    return 1.0


def _select_weighted_action(state: Any, last_action: Optional[str] = None) -> str:
    """
    Choose a legal action proportionally to its heuristic score.
    Avoids picking the exact opposite of ``last_action`` (e.g. close after open).
    """
    legal: List[str] = state.legal_actions()
    if not legal:
        raise RuntimeError("No legal actions available for simulation.")

    # Compute scores once
    scores = [_action_score(state, act) for act in legal]

    # If we have a last_action, down‑weight the opposite action heavily.
    if last_action:
        for idx, act in enumerate(legal):
            if _is_opposite(act, last_action):
                scores[idx] *= 0.01  # near‑zero probability

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
    last_action: Optional[str] = None

    while not sim_state.is_terminal() and depth < max_depth:
        # Try a few times to avoid picking the immediate opposite action.
        for _ in range(3):
            action = _select_weighted_action(sim_state, last_action)
            if not _is_opposite(action, last_action):
                break
        sim_state.apply_action(action)
        last_action = action
        depth += 1

        if getattr(sim_state, "task_completed", False):
            break

    return sim_state.returns()[perspective_player]
