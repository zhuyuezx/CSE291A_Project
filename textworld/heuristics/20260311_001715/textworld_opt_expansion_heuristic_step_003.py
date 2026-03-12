"""
LLM-generated MCTS tool: expansion
Description: Fix undefined `sim_state` in movement scoring and clean up imports.
Generated:   2026-03-11T00:34:00.778122
"""

"""
Improved expansion tool for the TextWorld Benchmark.

Key enhancements:
  * Stronger task‑specific bonuses (e.g., taking the coin, reading the map).
  * Higher weight on distance‑delta for movement actions.
  * Penalty for reverting/opening‑then‑closing doors.
  * Heavier penalty for pure observation/no‑op actions while movement
    actions are still untried.
  * Larger repeat‑action penalty to suppress loops.
"""

import random
from typing import Any, List

# -------------------------------------------------------------------------
# Helper utilities – self‑contained so the function can run independently
# -------------------------------------------------------------------------

def _is_movement(action: str) -> bool:
    """Return True if the action looks like a navigation or door‑opening command."""
    lowered = action.lower()
    move_tokens = {"north", "south", "east", "west", "up", "down"}
    verb_tokens = {"go ", "move ", "walk ", "run ", "open door", "door to", "enter "}
    if any(tok in lowered for tok in move_tokens):
        return True
    if any(tok in lowered for tok in verb_tokens):
        return True
    if lowered.startswith("open door") and any(dir_ in lowered for dir_ in move_tokens):
        return True
    return False


def _action_changes_state(state: Any, action: str) -> bool:
    """Cheap test whether applying *action* would change the observable state."""
    try:
        cloned = state.clone()
        cloned.apply_action(action)
        return cloned.state_key() != state.state_key()
    except Exception:
        # Assume the action changes something if we cannot simulate safely.
        return True


def _pop_action(untried: Any, chosen: str) -> None:
    """Remove *chosen* from the *untried* container (set, list or similar)."""
    if isinstance(untried, set):
        untried.remove(chosen)
    elif isinstance(untried, list):
        untried.remove(chosen)
    else:
        # generic fallback for other collection types
        try:
            untried.discard(chosen)  # type: ignore
        except Exception:
            try:
                untried.remove(chosen)  # type: ignore
            except Exception:
                pass


def _door_reversal_penalty(before_doors: Any, after_doors: Any) -> float:
    """
    Return a negative penalty if an action closes a door that was open before.
    The benchmark’s door dict is expected to map identifiers → bool (True=open).
    """
    if not isinstance(before_doors, dict) or not isinstance(after_doors, dict):
        return 0.0
    penalty = 0.0
    for d_id, before_state in before_doors.items():
        after_state = after_doors.get(d_id, before_state)
        # Closing a door that was open → penalty
        if before_state is True and after_state is False:
            penalty -= 3.0
    return penalty


def _score_action(state: Any, action: str, node: Any) -> float:
    """
    Heuristic score for *action* used during expansion.
    Higher values are preferred.
    """
    score = 0.0
    act_low = action.lower()

    # -----------------------------------------------------------------
    # 1️⃣ Movement actions – distance delta, state change, door penalty
    # -----------------------------------------------------------------
    if _is_movement(action):
        # Distance delta (weight 2.0 for stronger guidance)
        try:
            cur_dist = state.distance_to_goal()
        except Exception:
            cur_dist = 0.0

        # Simulate once; always have a `sim_state` object.
        sim_state = state.clone()
        try:
            sim_state.apply_action(action)
            new_dist = sim_state.distance_to_goal()
            changed = sim_state.state_key() != state.state_key()
        except Exception:
            # If simulation fails, treat as no progress.
            new_dist = cur_dist
            changed = False

        # Positive when we get closer, negative otherwise.
        score += 2.0 * (cur_dist - new_dist)

        # Bonus / penalty for genuine state change.
        if changed:
            score += 2.0
        else:
            score -= 2.0  # harsher penalty for a movement that does nothing.

        # Door reversal penalty (closing a previously open door).
        before_doors = getattr(state, "doors", {})
        after_doors = getattr(sim_state, "doors", {})
        score += _door_reversal_penalty(before_doors, after_doors)

    else:
        # -----------------------------------------------------------------
        # 2️⃣ Non‑movement actions – task‑specific bonuses & observation penalty
        # -----------------------------------------------------------------
        inv = getattr(state, "inventory_items", [])

        # Map‑reader specific bonuses (kept from previous version).
        if "take map" in act_low and "map" not in inv:
            score += 5.0
        if "read map" in act_low and "map" in inv and not getattr(state, "map_read", False):
            score += 5.0

        # New coin‑specific bonus.
        if "take coin" in act_low and "coin" not in inv:
            score += 10.0

        # Observation / no‑op penalty – heavier when movement still possible.
        movement_untried = any(_is_movement(a) for a in getattr(node, "_untried_actions", []))
        obs_tokens = {"look", "look around", "inventory", "task", "examine"}
        if any(tok in act_low for tok in obs_tokens):
            score -= 2.0 if movement_untried else 0.5

        # Reward if the action truly changes the world.
        if _action_changes_state(state, action):
            score += 2.0
        else:
            score -= 2.0

    # -----------------------------------------------------------------
    # 3️⃣ Repeat‑action penalty – discourage loops
    # -----------------------------------------------------------------
    repeat_cnt = sum(1 for a in getattr(node, "children", {}) if a == action)
    score -= 1.5 * repeat_cnt

    return score


def default_expansion(node):
    """
    Expand one untried action from the given node, preferring actions that
    are more likely to make progress according to an enhanced heuristic.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    untried = node._untried_actions
    if not untried:
        raise ValueError("default_expansion called on a node with no untried actions")

    # -------------------------------------------------------------
    # 0️⃣ Determine candidate actions:
    #    Prefer movement actions; if none left, fall back to all actions.
    # -------------------------------------------------------------
    untried_list = list(untried)          # snapshot (set, list, etc.)
    movement_actions = [a for a in untried_list if _is_movement(a)]
    candidate_actions = movement_actions if movement_actions else untried_list

    # -------------------------------------------------------------
    # 1️⃣ Choose the best‑scoring action from the candidate set
    # -------------------------------------------------------------
    best_score = -float("inf")
    best_actions: List[str] = []

    for action in candidate_actions:
        sc = _score_action(node.state, action, node)
        if sc > best_score:
            best_score = sc
            best_actions = [action]
        elif sc == best_score:
            best_actions.append(action)

    # Break ties randomly to preserve exploration.
    chosen_action = random.choice(best_actions)

    # Remove the chosen action from the untried container.
    _pop_action(untried, chosen_action)

    # -------------------------------------------------------------
    # 2️⃣ Create the child node with the resulting state
    # -------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    # Local import to avoid circular dependencies.
    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # Store for future look‑ups.
    node.children[chosen_action] = child

    return child
