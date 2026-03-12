"""
LLM-generated MCTS tool: selection
Description: No changes required; the function is correct and efficient.
Generated:   2026-03-11T14:51:11.009164
"""

"""
Improved selection with finer‑grained domain heuristics.

Changes compared to the previous version:
- Uses *delta* distance to goal instead of raw distance penalty.
- Adds a directional move bonus/penalty based on whether the move
  reduces the distance to the known goal.
- Penalises immediate door‑toggle cycles (open‑then‑close or vice‑versa).
- Gives a larger boost to “take …” actions, reduced if the item is
  already in the inventory.
All other behaviour (UCB1 core, exploration weight) is unchanged.
"""

import math
from typing import Any, Dict


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

    # ------------------------------------------------------------------
    # Helper functions – keep everything self‑contained.
    # ------------------------------------------------------------------
    def _action_type_bonus(action_str: str) -> float:
        """Base bonus/penalty based on generic action categories."""
        a = action_str.lower()
        if any(a.startswith(p) for p in ("move ", "go ", "walk ")):
            return 0.2
        if any(a.startswith(p) for p in ("open door", "close door", "take ", "read ")):
            return 0.1
        if a in ("look", "inventory", "task"):
            return -0.1
        return 0.0

    def _directional_move_bonus(parent_dist: float, child_dist: float) -> float:
        """
        Bonus (+) if the move brings us closer to the goal,
        penalty (‑) if it moves us farther away.
        """
        delta = parent_dist - child_dist  # >0 means improvement
        if delta > 0:
            return 0.1   # encouragement for progress moves
        if delta < 0:
            return -0.1  # discourage regress moves
        return 0.0

    def _door_toggle_penalty(parent_state, child_state) -> float:
        """
        Detect a door being toggled back to its previous state
        (e.g., open then close immediately) and penalise.
        Works if both states expose a ``doors`` mapping
        ``door_id -> bool(open)``.
        """
        try:
            p_doors: Dict = getattr(parent_state, "doors", {})
            c_doors: Dict = getattr(child_state, "doors", {})
            if not isinstance(p_doors, dict) or not isinstance(c_doors, dict):
                return 0.0
            # Identify doors whose status changed this step
            changed = [d for d in p_doors if p_doors.get(d) != c_doors.get(d)]
            if len(changed) == 1:
                # If the total number of open doors is unchanged, we likely
                # toggled the same door back.
                if sum(bool(v) for v in p_doors.values()) == sum(bool(v) for v in c_doors.values()):
                    return -0.05
        except Exception:
            pass
        return 0.0

    def _take_action_bonus(action_str: str, child_state) -> float:
        """
        Stronger incentive for taking an item, reduced if already owned.
        """
        a = action_str.lower()
        if a.startswith("take "):
            item = a[5:].strip()
            inv = getattr(child_state, "inventory_items", [])
            if isinstance(inv, (list, set, tuple)) and item in inv:
                return 0.0  # already have it – no extra boost
            return 0.3
        return 0.0

    # ------------------------------------------------------------------
    # Main selection loop.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            # There are still actions that have never been tried – expand here.
            return node

        # Parent statistics for the UCB term.
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        best_child = None
        best_score = -math.inf

        # Cache parent distance once (used for delta calculations).
        try:
            parent_dist = float(node.state.distance_to_goal())
        except Exception:
            parent_dist = 0.0

        for action_str, child in node.children.items():
            child_visits = child.visits

            # ---- Base UCB1 components ------------------------------------
            if child_visits == 0:
                exploit = 0.0
                explore = float('inf')   # unseen child gets maximal exploration
            else:
                exploit = child.value / child_visits
                explore = exploration_weight * math.sqrt(log_parent / child_visits)

            # ---- Heuristic bias -----------------------------------------
            heuristic = 0.0

            # 1) State‑change penalty (unchanged observable state)
            try:
                if child.state.state_key() == node.state.state_key():
                    heuristic -= 0.1
            except Exception:
                pass

            # 2) Delta distance to goal (positive if we get closer)
            try:
                child_dist = float(child.state.distance_to_goal())
                delta = parent_dist - child_dist
                heuristic += 0.3 * delta                     # main delta bonus
                heuristic += _directional_move_bonus(parent_dist, child_dist)
            except Exception:
                pass

            # 3) Action‑type weighting (generic)
            heuristic += _action_type_bonus(action_str)

            # 4) Door‑toggle penalty
            heuristic += _door_toggle_penalty(node.state, child.state)

            # 5) Take‑action boost
            heuristic += _take_action_bonus(action_str, child.state)

            # ---- Combined score -----------------------------------------
            score = exploit + explore + heuristic

            if score > best_score:
                best_score = score
                best_child = child

        # Defensive fallback – should not happen but avoids infinite loops.
        if best_child is None:
            return node

        node = best_child

    return node
