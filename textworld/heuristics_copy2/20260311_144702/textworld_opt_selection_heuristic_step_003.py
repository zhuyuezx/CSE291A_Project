"""
LLM-generated MCTS tool: selection
Description: 
Generated:   2026-03-11T14:52:32.445765
"""

"""
Improved selection with task‑aware heuristics for the TextWorld benchmark.

Key changes:
- Strong positive bias for “take map” and “read map” actions when the map
  has not yet been read.
- Reduced generic move bonus so it no longer drowns out map‑related signals.
- Door‑toggle penalty increased to discourage open‑close cycles.
- Distance‑delta bonus applied only after the goal is known (map_read=True).
- Minor refactoring of helper functions for clarity.
"""

import math
from typing import Any, Dict, List


def default_selection(node, exploration_weight: float = 1.41) -> Any:
    """
    UCB1 tree walk with additional domain‑specific bias terms.

    Walks down the tree choosing the child with the highest
    (UCB1 + heuristic) score. Stops when reaching a node that
    is terminal or has untried actions (needs expansion).

    Args:
        node:               Root MCTSNode to start selection from.
        exploration_weight: UCB1 exploration constant C.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """

    # ------------------------------------------------------------------
    # Helper functions – all self‑contained.
    # ------------------------------------------------------------------
    def _action_type_bonus(action_str: str) -> float:
        """Base bonus/penalty based on generic action categories."""
        a = action_str.lower()
        # Move actions are still useful but receive a smaller static boost.
        if any(a.startswith(p) for p in ("move ", "go ", "walk ")):
            return 0.05
        if any(a.startswith(p) for p in ("open door", "close door", "take ", "read ")):
            return 0.1
        if a in ("look", "inventory", "task"):
            return -0.1
        return 0.0

    def _directional_move_bonus(parent_dist: float, child_dist: float) -> float:
        """Bonus for moves that reduce the distance to the known goal."""
        delta = parent_dist - child_dist
        if delta > 0:
            return 0.1
        if delta < 0:
            return -0.1
        return 0.0

    def _door_toggle_penalty(parent_state, child_state) -> float:
        """Detect immediate door‑state reversals and penalise strongly."""
        try:
            p_doors: Dict = getattr(parent_state, "doors", {})
            c_doors: Dict = getattr(child_state, "doors", {})
            if not isinstance(p_doors, dict) or not isinstance(c_doors, dict):
                return 0.0
            # doors whose boolean(open) status changed this step
            changed = [d for d in p_doors if p_doors.get(d) != c_doors.get(d)]
            if len(changed) == 1:
                # if total number of open doors is unchanged → toggled back
                if sum(bool(v) for v in p_doors.values()) == sum(bool(v) for v in c_doors.values()):
                    return -0.2
        except Exception:
            pass
        return 0.0

    def _take_action_bonus(action_str: str, child_state) -> float:
        """Boost for taking new items; no bonus if already owned."""
        a = action_str.lower()
        if a.startswith("take "):
            item = a[5:].strip()
            inv: List[str] = getattr(child_state, "inventory_items", [])
            if isinstance(inv, (list, set, tuple)) and item in inv:
                return 0.0
            return 0.3
        return 0.0

    def _map_read_bonus(parent_state, action_str: str) -> float:
        """
        Large bonuses for map‑acquisition actions when the map has not yet
        been read.  Rewards:
          - taking the map: +1.0
          - reading the map (when it is already in inventory): +0.8
        """
        try:
            map_read = bool(getattr(parent_state, "map_read", False))
        except Exception:
            map_read = False

        # No map‑related incentives once the map has been read.
        if map_read:
            return 0.0

        a = action_str.lower()
        # Take the map
        if "take map" in a:
            return 1.0

        # Read the map – only useful if we already possess it.
        if "read map" in a:
            inv = getattr(parent_state, "inventory_items", [])
            if isinstance(inv, (list, set, tuple)):
                if any(item.lower() == "map" for item in inv):
                    return 0.8
        return 0.0

    # ------------------------------------------------------------------
    # Main selection loop.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            # There are still actions that have never been tried – expand here.
            return node

        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        best_child = None
        best_score = -math.inf

        # Cache parent distance, but only meaningful when goal is known.
        try:
            parent_dist = float(node.state.distance_to_goal())
        except Exception:
            parent_dist = 0.0

        # Determine if the goal is currently known.
        goal_known = bool(getattr(node.state, "map_read", False))

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

            # 1) Penalty for state that did not change.
            try:
                if child.state.state_key() == node.state.state_key():
                    heuristic -= 0.1
            except Exception:
                pass

            # 2) Map‑related bonuses (take/read map) – high priority.
            heuristic += _map_read_bonus(node.state, action_str)

            # 3) Delta distance to goal – only when the goal is known.
            if goal_known:
                try:
                    child_dist = float(child.state.distance_to_goal())
                    delta = parent_dist - child_dist
                    heuristic += 0.3 * delta               # primary delta reward
                    heuristic += _directional_move_bonus(parent_dist, child_dist)
                except Exception:
                    pass

            # 4) Generic action‑type weighting.
            heuristic += _action_type_bonus(action_str)

            # 5) Door‑toggle penalty.
            heuristic += _door_toggle_penalty(node.state, child.state)

            # 6) Take‑action boost for new items.
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
