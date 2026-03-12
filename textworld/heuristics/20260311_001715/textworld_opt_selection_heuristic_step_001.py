"""
LLM-generated MCTS tool: selection
Description: Fix attribute name, prevent division by zero, and cache repeated state lookups.
Generated:   2026-03-11T00:20:17.051520
"""

import math
from typing import Any

def default_selection(
    node,
    exploration_weight: float = 1.41,
    progress_weight: float = 0.5,
    noop_penalty: float = -0.3,
    negative_value_threshold: float = -0.8,
    min_visits_for_negative_penalty: int = 5,
    negative_exploration_factor: float = 0.5,
) -> Any:
    """
    Enhanced UCB1 tree policy with domain‑specific heuristics.

    Adds a progress‑based bonus (distance reduction, action category),
    penalises actions that leave the observation unchanged (no‑ops),
    and reduces exploration for children that consistently yield strong
    negative rewards (e.g., door‑toggle loops).

    Args:
        node:                Root MCTSNode to start selection from.
        exploration_weight: Standard UCB1 exploration constant C.
        progress_weight:    Scaling factor for the progress bonus.
        noop_penalty:        Fixed penalty added when an action does not
                            change the room description.
        negative_value_threshold: Average value below which a child is
                            considered a dead‑end.
        min_visits_for_negative_penalty: Minimum visits before applying
                            the negative‑exploration reduction.
        negative_exploration_factor: Factor (<1) that shrinks the
                            exploration term for dead‑end children.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    while not node.is_terminal:
        # If there are still actions to try, hand off to expansion.
        if not node.is_fully_expanded:
            return node

        # Cache parent data that are reused for every child.
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)
        parent_look = None
        try:
            parent_look = node.state.look_text()
        except Exception:
            pass
        parent_dist = None
        try:
            parent_dist = node.state.distance_to_goal()
        except Exception:
            pass

        best_child = None
        best_score = -math.inf

        for child in node.children.values():
            # ---- Exploit / Explore ----
            child_visits = max(child.visits, 1)
            exploit = child.value / child_visits
            explore = exploration_weight * math.sqrt(log_parent / child_visits)

            # ---- Progress bonus (distance reduction) ----
            progress_bonus = 0.0
            if parent_dist is not None:
                try:
                    child_dist = child.state.distance_to_goal()
                    if child_dist is not None:
                        delta = parent_dist - child_dist  # positive if we got closer
                        if delta > 0:
                            cat_weight = _action_category_weight(
                                getattr(child, "action", None)
                            )
                            progress_bonus = progress_weight * delta * cat_weight
                except Exception:
                    pass

            # ---- No‑op penalty (observation unchanged) ----
            if parent_look is not None:
                try:
                    child_look = child.state.look_text()
                    if parent_look == child_look:
                        progress_bonus += noop_penalty
                except Exception:
                    pass

            # ---- Reduce exploration for consistently bad children ----
            avg_value = child.value / child_visits if child.visits > 0 else 0.0
            if (
                avg_value < negative_value_threshold
                and child.visits >= min_visits_for_negative_penalty
            ):
                explore *= negative_exploration_factor

            # ---- Combined UCB score ----
            score = exploit + explore + progress_bonus

            if score > best_score:
                best_child = child
                best_score = score

        # Safety fallback
        if best_child is None:
            return node
        node = best_child

    return node


def _action_category_weight(action: Any) -> float:
    """
    Assign a static weight to an action based on its textual category.
    Higher weights boost progress bonus for useful actions.

    Args:
        action: The action object or string stored in the node.

    Returns:
        A float weight (>=0).
    """
    if not isinstance(action, str):
        return 1.0

    act = action.lower().strip()
    if act.startswith(("go ", "move ", "walk ")):
        return 1.2
    if act.startswith("take "):
        return 1.3
    if act.startswith("read "):
        return 1.4
    if act in ("look", "look around", "inventory", "task"):
        return 0.5
    return 1.0
