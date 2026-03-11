"""
LLM-generated MCTS tool: simulation
Description: 
Generated:   2026-03-10T01:09:21.412828
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
    room_desc = ""
    try:
        room_desc = state.look_text()
    except Exception:
        try:
            room_desc = state.observation_text()
        except Exception:
            room_desc = ""
    return item.lower() in room_desc.lower()


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

    # 1. Take actions – only reward if the item is actually reachable.
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

    # 2. Read map – only useful when map is in inventory and not yet read.
    if act.startswith("read"):
        if not getattr(state, "map_read", False) and _item_in_inventory(state, "map"):
            score += 10.0

    # 3. Movement / door actions – look one step ahead and reward distance reduction.
    if _is_movement(action) or "open " in act or "close " in act:
        if _is_movement(action):
            score += 3.0  # generic movement boost

        # Attempt to evaluate distance change.
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
            score += diff * 2.0            # encourage distance reduction

            if "open " in act:
                score += 1.0
            if "close " in act:
                if diff < 0:
                    score -= 4.0
                else:
                    score -= 1.0
        else:
            if _is_movement(action):
                score += 1.0  # fallback small boost

    # 4. No‑op actions – stronger penalty.
    if _is_noop(action):
        score -= 8.0

    # 5. Minor stochastic safety tweaks.
    if "close" in act:
        score -= 2.0
    if "open" in act:
        score += 0.5

    # 6. Small random jitter (reduced magnitude to keep deterministic bias dominant).
    score += random.random() * 0.1

    return score


# ----------------------------------------------------------------------
# Main simulation routine
# ----------------------------------------------------------------------
def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided rollout with state‑aware scoring.
    Stops early if an action leaves the observable state unchanged.
    """
    sim_state = state.clone()
    depth = 0

    def _snapshot(st):
        """Create a cheap observable snapshot for stagnation detection."""
        inv = tuple(sorted(getattr(st, "inventory_items", [])))
        room = getattr(st, "room", None)
        map_read = getattr(st, "map_read", False)
        return (inv, room, map_read)

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

        sim_state.apply_action(chosen)
        depth += 1

        cur_snapshot = _snapshot(sim_state)
        if cur_snapshot == prev_snapshot:
            # No observable progress – treat as dead‑end.
            return 0.0
        prev_snapshot = cur_snapshot

    try:
        return sim_state.returns()[perspective_player]
    except Exception:
        return 0.0
