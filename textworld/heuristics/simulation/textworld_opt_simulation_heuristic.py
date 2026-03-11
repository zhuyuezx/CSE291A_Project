"""
LLM-generated MCTS tool: simulation
Description: Finalized heuristic‑guided rollout with improved dead‑lock detection and door handling.
Generated:   2026-03-10T01:11:48.392746
"""

import random
from typing import List, Tuple

# ----------------------------------------------------------------------
# Helper predicates
# ----------------------------------------------------------------------
def _is_movement(action: str) -> bool:
    """Return True if the action looks like a movement command."""
    move_keywords = ["go ", "move ", "north", "south", "east", "west", "up", "down"]
    a = action.lower()
    return any(k in a for k in move_keywords)


def _is_noop(action: str) -> bool:
    """Detect actions that usually do not change the environment."""
    noop_keywords = ["look", "inventory", "task", "examine"]
    a = action.lower()
    return any(k in a for k in noop_keywords)


def _item_in_inventory(state, item: str) -> bool:
    """Check whether *item* (lower‑cased) is already held."""
    inv = getattr(state, "inventory_items", [])
    return any(item == i.lower() for i in inv)


def _item_present_in_room(state, item: str) -> bool:
    """
    Rough heuristic: assume the item is present if it is not already in
    the inventory and its name appears in the current room description.
    """
    if _item_in_inventory(state, item):
        return False
    try:
        room_desc = state.look_text()
    except Exception:
        try:
            room_desc = state.observation_text()
        except Exception:
            room_desc = ""
    return item.lower() in room_desc.lower()


def _door_status(state, door_name: str) -> bool:
    """
    Return True if *door_name* is currently open.
    Supports common representations:
      - dict mapping name -> bool (True=open)
      - list/tuple where index corresponds to a door id (True=open)
    If unknown, returns False.
    """
    doors = getattr(state, "doors", {})
    if isinstance(doors, dict):
        return bool(doors.get(door_name, False))
    # Fallback: treat doors as an iterable of booleans; we cannot map name,
    # so assume closed.
    return False


# ----------------------------------------------------------------------
# Scoring function
# ----------------------------------------------------------------------
def _heuristic_score(state, action: str) -> float:
    """
    State‑aware scoring for rollout actions.
    Higher scores → more promising for reaching the goal.
    """
    score = 0.0
    act = action.lower()

    # ------------------------------------------------------------------
    # 1. Take actions – only reward if the item is actually reachable.
    # ------------------------------------------------------------------
    if act.startswith("take "):
        target = act[5:].strip()
        if target == "map":
            if not _item_in_inventory(state, "map"):
                score += 12.0
        elif target == "coin":
            if not _item_in_inventory(state, "coin"):
                score += 9.0
        else:
            if _item_present_in_room(state, target):
                score += 5.0

    # ------------------------------------------------------------------
    # 2. Read map – only useful when map is in inventory and not yet read.
    # ------------------------------------------------------------------
    if act.startswith("read"):
        if not getattr(state, "map_read", False) and _item_in_inventory(state, "map"):
            score += 10.0

    # ------------------------------------------------------------------
    # 3. Movement / door actions – reward distance reduction heavily.
    # ------------------------------------------------------------------
    if _is_movement(action) or "open " in act or "close " in act:
        # generic movement boost
        if _is_movement(action):
            score += 3.0

        # distance before & after
        try:
            before = float(state.distance_to_goal())
        except Exception:
            before = None

        try:
            after_state = state.clone()
            after_state.apply_action(action)
            after = float(after_state.distance_to_goal())
        except Exception:
            after = None

        if before is not None and after is not None:
            diff = before - after          # positive → we got closer
            # Stronger reward for genuine progress
            score += diff * 4.0            # larger multiplier than before
            if diff > 0:
                score += 2.0               # extra bonus for any reduction
        else:
            # fallback when distance info unavailable
            if _is_movement(action):
                score += 1.0

        # ------------------------------------------------------------------
        # Door specific tweaks
        # ------------------------------------------------------------------
        if "open " in act:
            # extract door name
            door_name = act.split("open ", 1)[1].strip()
            # reward only if the door is currently closed
            if not _door_status(state, door_name):
                score += 0.1                # tiny incentive to open needed doors
            else:
                score -= 5.0                # penalise opening an already‑open door
            score -= 0.5                    # small constant toggle penalty

        if "close " in act:
            door_name = act.split("close ", 1)[1].strip()
            if _door_status(state, door_name):
                # door is open; closing it is usually bad unless it improves distance
                if before is not None and after is not None and after > before:
                    score -= 1.0            # closing helped (rare) – small penalty
                else:
                    score -= 5.0            # heavy penalty for unnecessary close
            else:
                score -= 8.0                # trying to close an already‑closed door
            score -= 0.5                    # constant toggle penalty

    # ------------------------------------------------------------------
    # 4. No‑op actions – stronger penalty.
    # ------------------------------------------------------------------
    if _is_noop(action):
        score -= 8.0

    # ------------------------------------------------------------------
    # 5. Minor stochastic safety tweaks.
    # ------------------------------------------------------------------
    if "close" in act:
        score -= 2.0
    if "open" in act:
        score += 0.5

    # ------------------------------------------------------------------
    # 6. Small random jitter (kept tiny).
    # ------------------------------------------------------------------
    score += random.random() * 0.1

    return score


# ----------------------------------------------------------------------
# Main simulation routine
# ----------------------------------------------------------------------
def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided rollout with improved dead‑lock detection and
    stronger distance‑based incentives.
    """
    sim_state = state.clone()
    depth = 0
    non_progress_streak = 0          # counts consecutive low‑score steps

    def _snapshot(st):
        """Create a cheap observable snapshot for stagnation detection."""
        inv = tuple(sorted(getattr(st, "inventory_items", [])))
        room = getattr(st, "room", None)
        map_read = getattr(st, "map_read", False)

        # Encode door state – support dict or iterable.
        doors_raw = getattr(st, "doors", {})
        if isinstance(doors_raw, dict):
            # Sort items to obtain deterministic hashable representation
            doors = tuple(sorted((k, bool(v)) for k, v in doors_raw.items()))
        else:
            # Assume iterable of booleans; convert to tuple
            try:
                doors = tuple(bool(d) for d in doors_raw)
            except Exception:
                doors = ()
        return (inv, room, map_read, doors)

    prev_snapshot = _snapshot(sim_state)

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        scored: List[Tuple[float, str]] = [
            (_heuristic_score(sim_state, act), act) for act in legal
        ]

        max_score = max(scored, key=lambda x: x[0])[0]
        top_actions = [act for sc, act in scored if sc == max_score]
        chosen = random.choice(top_actions)

        # Update streak of non‑progressing steps
        if max_score <= 0:
            non_progress_streak += 1
        else:
            non_progress_streak = 0

        if non_progress_streak >= 3:
            # Too many low‑score moves – abort as dead‑end.
            return 0.0

        sim_state.apply_action(chosen)
        depth += 1

        cur_snapshot = _snapshot(sim_state)
        if cur_snapshot == prev_snapshot:
            # No observable progress (including door changes) – dead‑end.
            return 0.0
        prev_snapshot = cur_snapshot

    try:
        return sim_state.returns()[perspective_player]
    except Exception:
        return 0.0
