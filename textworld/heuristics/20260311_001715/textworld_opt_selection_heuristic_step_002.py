"""
LLM-generated MCTS tool: selection
Description: 
Generated:   2026-03-11T00:21:57.315916
"""

"""
Enhanced selection (UCB1) for TextWorld Benchmark.

Changes compared to the previous version:
- Gives a fixed information‑gain bonus when the action takes or reads the map
  (critical for the map‑reader variant).
- Penalises actions that increase the distance to the goal, preventing
  oscillatory back‑and‑forth moves.
- Reduces the threshold for applying the negative‑exploration penalty
  (visits ≥ 2) and makes the penalty stronger (factor 0.3).
- Adds a tiny stochastic‑safety factor that slightly lowers exploration
  in stochastic variants.
- Keeps caching of repeated state queries and division‑by‑zero guards.
"""

import math
from typing import Any


def default_selection(
    node,
    exploration_weight: float = 1.41,
    progress_weight: float = 0.5,
    noop_penalty: float = -0.3,
    info_gain_bonus: float = 0.8,
    negative_value_threshold: float = -0.8,
    min_visits_for_negative_penalty: int = 2,
    negative_exploration_factor: float = 0.3,
    stochastic_explore_factor: float = 0.9,
) -> Any:
    """
    Tree policy that mixes classic UCB1 with domain‑specific heuristics.

    Args:
        node: MCTSNode to start selection from.
        exploration_weight: Standard UCB1 constant C.
        progress_weight: Scaling for distance‑based progress (positive or negative).
        noop_penalty: Penalty added when the observation does not change.
        info_gain_bonus: Fixed bonus for actions that acquire or read the map.
        negative_value_threshold: Average value below which a child is treated as a dead‑end.
        min_visits_for_negative_penalty: Minimum visits before applying the dead‑end reduction.
        negative_exploration_factor: Multiplicative factor (<1) applied to the exploration term
                                    for dead‑end children.
        stochastic_explore_factor: Factor (<1) that reduces exploration in stochastic variants
                                    (applied per child).

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    while not node.is_terminal:
        # If there are actions left to try, stop here for expansion.
        if not node.is_fully_expanded:
            return node

        # ---- Cached parent information ----
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        # Observation text (for no‑op detection)
        parent_look = None
        try:
            parent_look = node.state.look_text()
        except Exception:
            pass

        # Distance to goal (may be None if map unread)
        parent_dist = None
        try:
            parent_dist = node.state.distance_to_goal()
        except Exception:
            pass

        # Detect whether the map has already been read (used for info‑gain)
        map_already_read = False
        try:
            map_already_read = bool(getattr(node.state, "map_read", False))
        except Exception:
            pass

        # Determine if we are in a stochastic variant (optional)
        is_stochastic = False
        try:
            cfg = getattr(node.state, "config", {})
            if isinstance(cfg, dict):
                is_stochastic = cfg.get("variant") == "stochastic"
            else:
                is_stochastic = getattr(cfg, "variant", None) == "stochastic"
        except Exception:
            pass

        best_child = None
        best_score = -math.inf

        for child in node.children.values():
            # ---- Exploit / Explore ----
            child_visits = max(child.visits, 1)
            exploit = child.value / child_visits
            explore = exploration_weight * math.sqrt(log_parent / child_visits)

            # Apply stochastic safety factor early
            if is_stochastic:
                explore *= stochastic_explore_factor

            # ---- Progress bonus / penalty (distance) ----
            progress_bonus = 0.0
            if parent_dist is not None:
                try:
                    child_dist = child.state.distance_to_goal()
                    if child_dist is not None:
                        delta = parent_dist - child_dist  # >0 means closer
                        cat_weight = _action_category_weight(
                            getattr(child, "action", None)
                        )
                        if delta > 0:
                            # Positive progress
                            progress_bonus = progress_weight * delta * cat_weight
                        elif delta < 0:
                            # Penalise moving away
                            progress_bonus = -progress_weight * (-delta) * cat_weight
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

            # ---- Information‑gain bonus (map handling) ----
            if not map_already_read:
                act_str = getattr(child, "action", "")
                if isinstance(act_str, str):
                    act_low = act_str.lower().strip()
                    if act_low == "take map" or act_low.startswith("read "):
                        progress_bonus += info_gain_bonus

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

        # Safety fallback – if for any reason no child was selected
        if best_child is None:
            return node

        node = best_child

    return node


def _action_category_weight(action: Any) -> float:
    """
    Static weight according to textual action category.
    Higher weights boost the progress contribution for generally useful actions.

    Args:
        action: Action object or string stored in the node.

    Returns:
        Weight ≥ 0.
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
