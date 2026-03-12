"""
LLM-generated MCTS tool: expansion
Description: Improved expansion with progress‑oriented scoring and no‑op filtering
Generated:   2026-03-11T02:22:32.171097
"""

import math
import random
from typing import Any

# ----------------------------------------------------------------------
# Helper functions – all self‑contained
# ----------------------------------------------------------------------
def _clone_and_apply(state: Any, action: Any) -> Any:
    """Clone a GameState and apply an action to the clone."""
    child = state.clone()
    child.apply_action(action)
    return child


def _is_noop(parent_state: Any, child_state: Any) -> bool:
    """
    Detect actions that do not change the observable description.
    In punishment mode these are heavily penalised, so we filter them.
    """
    try:
        return parent_state.observation_text() == child_state.observation_text()
    except Exception:
        # If the method is missing or fails, fall back to a safe false.
        return False


def _distance_improvement(parent_state: Any, child_state: Any) -> float:
    """
    Return positive value proportional to reduction in distance to the goal.
    If distance information is unavailable, return 0.
    """
    try:
        old_dist = parent_state.distance_to_goal()
        new_dist = child_state.distance_to_goal()
        if old_dist is None or new_dist is None:
            return 0.0
        return max(0.0, old_dist - new_dist)  # only reward improvement
    except Exception:
        return 0.0


def _inventory_gain(parent_state: Any, child_state: Any) -> int:
    """Count newly acquired items (e.g., map, coin)."""
    try:
        old_items = set(parent_state.inventory_items)
        new_items = set(child_state.inventory_items)
        return len(new_items - old_items)
    except Exception:
        return 0


def _map_read_gain(parent_state: Any, child_state: Any) -> int:
    """Detect transition from unread map to read map."""
    try:
        return int(child_state.map_read and not parent_state.map_read)
    except Exception:
        return 0


def _door_state_change(parent_state: Any, child_state: Any, action: Any) -> int:
    """
    Reward opening/closing a door only when the state actually changes.
    Assumes `doors` is a dict mapping door identifiers to a bool (open).
    """
    try:
        if not isinstance(action, str):
            return 0
        lowered = action.lower()
        if not ("open" in lowered or "close" in lowered):
            return 0
        return int(parent_state.doors != child_state.doors)
    except Exception:
        return 0


def _action_score(node: Any, action: Any) -> float:
    """
    Compute a cheap heuristic score for an action.
    Higher score = more promising for expansion.
    """
    parent = node.state
    child = _clone_and_apply(parent, action)

    # Discard obvious no‑ops early (they will get a huge negative score)
    if _is_noop(parent, child):
        return -math.inf

    score = 0.0

    # Terminal states are the best
    if child.is_terminal():
        score += 1e6

    # Distance reduction (relevant for mapreader)
    score += 5.0 * _distance_improvement(parent, child)

    # Acquiring new items (map, coin, etc.)
    score += 3.0 * _inventory_gain(parent, child)

    # Reading the map (important for mapreader)
    score += 3.0 * _map_read_gain(parent, child)

    # Door state changes (opening/closing a closed door)
    score += 1.0 * _door_state_change(parent, child, action)

    # Small random tie‑breaker to keep exploration alive
    score += random.random() * 0.01

    return score


# ----------------------------------------------------------------------
# Modified expansion function
# ----------------------------------------------------------------------
def default_expansion(node):
    """
    Expand one untried action from the given node, preferring actions that
    make progress (move towards goal, acquire items, read map, change door
    state) and discarding actions that have no observable effect.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    if not hasattr(node, "_untried_actions") or not node._untried_actions:
        # Do not crash search if expansion is called on a fully expanded node.
        return node

    # Work on a snapshot to avoid mutating during iteration
    untried = list(node._untried_actions)

    best_action = None
    best_score = -math.inf

    for action in untried:
        score = _action_score(node, action)

        if score == -math.inf:
            # Remove permanent no‑ops to avoid future waste
            try:
                node._untried_actions.remove(action)  # set or list
            except Exception:
                try:
                    node._untried_actions.discard(action)  # set fallback
                except Exception:
                    pass
            continue

        if score > best_score:
            best_score = score
            best_action = action

    # If every action was filtered as no-op, avoid popping from an empty list.
    if best_action is None:
        if not node._untried_actions:
            return node
        best_action = node._untried_actions.pop()
    else:
        # Remove the selected best action from the collection
        try:
            node._untried_actions.remove(best_action)
        except Exception:
            node._untried_actions.discard(best_action)

    # Create the child node with the chosen action
    child_state = node.state.clone()
    child_state.apply_action(best_action)

    # Import here to avoid circular imports in the original project layout
    from mcts.node import MCTSNode

    child = MCTSNode(child_state, parent=node, parent_action=best_action)
    node.children[best_action] = child
    return child
