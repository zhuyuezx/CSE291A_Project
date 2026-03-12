"""
LLM-generated MCTS tool: expansion
Description: Refine expansion with lightweight heuristic ordering and small performance tweaks.
Generated:   2026-03-11T00:30:15.911324
"""

import random
from typing import Any, List

# -------------------------------------------------------------------------
# Helper utilities – all self‑contained so the function can run independently
# -------------------------------------------------------------------------

def _is_movement(action: str) -> bool:
    """Return True if the action resembles a move command."""
    move_tokens = {
        "north", "south", "east", "west", "up", "down",
        "go ", "move ", "walk ", "run "
    }
    lowered = action.lower()
    return any(tok in lowered for tok in move_tokens)


def _action_changes_state(state: Any, action: str) -> bool:
    """
    Determine whether applying *action* would change the observable state.
    A cheap clone‑apply‑compare is used only when needed.
    """
    try:
        cloned = state.clone()
        cloned.apply_action(action)
        return cloned.state_key() != state.state_key()
    except Exception:
        # If anything goes wrong (e.g., illegal action), assume it changes state.
        return True


def _score_action(state: Any, action: str, node: Any) -> float:
    """
    Produce a cheap heuristic score for *action* given the current *state*.
    Higher scores are preferred for expansion.

    Heuristic components:
      * Progress towards goal for movement actions.
      * Bonuses for map‑related actions in mapreader tasks.
      * Penalties for likely no‑ops (look/inventory/task).
      * Penalty if the action is known not to change the state.
      * Small penalty for actions already expanded from this node.
    """
    score = 0.0
    act_low = action.lower()

    # 1️⃣ Movement – favour actions that (potentially) reduce distance to goal.
    if _is_movement(action):
        try:
            # Smaller distance → higher (less negative) score.
            score += -state.distance_to_goal()
        except Exception:
            pass

    # 2️⃣ Map‑reader specific bonuses.
    inv = getattr(state, "inventory_items", [])
    if "take map" in act_low and "map" not in inv:
        score += 5.0
    if "read map" in act_low and "map" in inv and not getattr(state, "map_read", False):
        score += 5.0

    # 3️⃣ Discourage pure observation actions when they are unlikely to help.
    no_op_set = {"look", "look around", "inventory", "task", "examine"}
    if any(tok in act_low for tok in no_op_set):
        score -= 2.0

    # 4️⃣ Penalise actions that do not change the state.
    # Skip the expensive clone‑apply check for actions we already know affect the world.
    if not (_is_movement(action) or "take map" in act_low or "read map" in act_low):
        if not _action_changes_state(state, action):
            score -= 3.0

    # 5️⃣ Small repeat penalty – actions already expanded from this node.
    # In typical MCTS each action appears only once, but we keep the term for safety.
    repeat_cnt = sum(1 for a in getattr(node, "children", {}) if a == action)
    score -= 0.5 * repeat_cnt

    return score


def _pop_action(untried: Any, chosen: str) -> None:
    """Remove *chosen* from the *untried* container (set or list)."""
    if isinstance(untried, set):
        untried.remove(chosen)
    elif isinstance(untried, list):
        untried.remove(chosen)
    else:
        # Fallback for any container supporting discard/remove.
        try:
            untried.discard(chosen)  # type: ignore
        except Exception:
            try:
                untried.remove(chosen)  # type: ignore
            except Exception:
                pass


def default_expansion(node):
    """
    Expand one untried action from the given node, preferring actions that are
    more likely to make progress according to a lightweight heuristic.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    untried = node._untried_actions
    if not untried:
        raise ValueError("default_expansion called on a node with no untried actions")

    # -------------------------------------------------------------
    # 1. Choose the best‑scoring action from the untried set
    # -------------------------------------------------------------
    best_score = -float("inf")
    best_actions: List[str] = []

    for action in list(untried):  # snapshot because we will modify the container
        sc = _score_action(node.state, action, node)
        if sc > best_score:
            best_score = sc
            best_actions = [action]
        elif sc == best_score:
            best_actions.append(action)

    # Break ties randomly to keep exploration alive.
    chosen_action = random.choice(best_actions)

    # Remove the chosen action from the untried container.
    _pop_action(untried, chosen_action)

    # -------------------------------------------------------------
    # 2. Create the child node with the resulting state
    # -------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    from mcts.node import MCTSNode  # Local import to avoid circular dependencies.
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # Store the child for future reference.
    node.children[chosen_action] = child

    return child
