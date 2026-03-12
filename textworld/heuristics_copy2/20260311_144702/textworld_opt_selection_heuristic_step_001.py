"""
LLM-generated MCTS tool: selection
Description: Updated selection function with domain‑aware heuristics
Generated:   2026-03-11T14:49:45.501294
"""

"""
Improved selection: UCB1 tree policy with lightweight domain heuristics.

The original pure UCB1 often cannot distinguish between
progress actions (e.g., moving toward the goal) and
no‑op or harmful actions (e.g., repeatedly opening/closing a door)
because their raw average rewards are identical in punishment
mode.  This version augments the UCB1 score with three cheap
biases:

1. State‑change penalty – if applying the action leaves the
   observable state unchanged (state_key identical), a small
   negative bias is applied.
2. Goal‑distance incentive – actions that decrease the estimated
   distance to the goal receive a positive boost proportional to
   the distance reduction.
3. Action‑type weighting – movement‑type actions get a modest
   bonus, while pure observation actions (look, inventory, task)
   are slightly penalised.

These heuristics are inexpensive, keep the original exploration/
exploitation balance, and dramatically improve tie‑breaking in
environments where raw rewards are flat.
"""

import math
from typing import Any


def default_selection(node, exploration_weight: float = 1.41) -> Any:
    """
    UCB1 tree walk with additional domain‑specific bias terms.

    Walks down the tree choosing the child with the highest
    (UCB1 + heuristic) score. Stops when reaching a node that
    is terminal or has untried actions (needs expansion).

    Args:
        node:               Root MCTSNode to start selection from.
        exploration_weight:  UCB1 exploration constant C.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    # Helper: small bonus/penalty based on the textual action.
    def _action_type_bonus(action_str: str) -> float:
        action = action_str.lower()
        # movement actions
        if any(action.startswith(p) for p in ("move ", "go ", "walk ")):
            return 0.2
        # door manipulation, taking items, reading map – useful but less than movement
        if any(action.startswith(p) for p in ("open door", "close door", "take ", "read ")):
            return 0.1
        # pure observation actions
        if action in ("look", "inventory", "task"):
            return -0.1
        return 0.0

    while not node.is_terminal:
        if not node.is_fully_expanded:
            # Node still has actions that have never been tried – expand it.
            return node

        # Protect against log(0); a node with 0 visits shouldn't be selected here.
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        best_child = None
        best_score = -math.inf

        for action_str, child in node.children.items():
            child_visits = child.visits
            if child_visits == 0:
                exploit = 0.0
                explore = float('inf')   # unseen child gets maximal exploration
            else:
                exploit = child.value / child_visits
                explore = exploration_weight * math.sqrt(log_parent / child_visits)

            # ---- Heuristic bias -------------------------------------------------
            heuristic = 0.0

            # 1) State‑change penalty
            try:
                if child.state.state_key() == node.state.state_key():
                    heuristic -= 0.1
            except Exception:
                pass

            # 2) Distance‑to‑goal incentive (if the method exists)
            try:
                dist = child.state.distance_to_goal()
                heuristic -= 0.05 * dist
            except Exception:
                pass

            # 3) Action‑type weighting
            heuristic += _action_type_bonus(action_str)

            # ---- Combined score -------------------------------------------------
            score = exploit + explore + heuristic

            if score > best_score:
                best_score = score
                best_child = child

        # Defensive fallback – should not happen but avoids infinite loops.
        if best_child is None:
            return node

        node = best_child

    return node
