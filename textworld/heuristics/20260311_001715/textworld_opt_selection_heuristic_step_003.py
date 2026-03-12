"""
LLM-generated MCTS tool: selection
Description: Add robust handling for `is_fully_expanded`/`is_terminal` callables and ensure map‑acquisition bonus works even when inventory attribute is absent.
Generated:   2026-03-11T00:23:26.963555
"""

"""
Enhanced selection (UCB1) for TextWorld Benchmark with
additional domain‑specific signals:
  • Door‑state change penalty to discourage useless open/close loops.
  • Scaled map‑acquisition bonus when the map is not yet in inventory.
  • Dynamic exploration weight that shrinks as the parent node is
    visited many times, letting progress/bonus terms dominate early.
"""

import math
from typing import Any


def default_selection(
    node,
    exploration_weight: float = 1.41,
    progress_weight: float = 0.5,
    noop_penalty: float = -0.3,
    info_gain_bonus: float = 0.8,
    door_penalty: float = -0.4,
    negative_value_threshold: float = -0.8,
    min_visits_for_negative_penalty: int = 2,
    negative_exploration_factor: float = 0.3,
    stochastic_explore_factor: float = 0.9,
) -> Any:
    """
    Tree policy mixing classic UCB1 with domain‑specific heuristics.

    Args:
        node: MCTSNode to start selection from.
        exploration_weight: Base UCB1 constant C.
        progress_weight: Scaling for distance‑based progress (positive or negative).
        noop_penalty: Penalty when observation (look_text) does not change.
        info_gain_bonus: Base bonus for actions that acquire or read the map.
        door_penalty: Fixed penalty applied when a child changes the door configuration.
        negative_value_threshold: Average value below which a child is considered a dead‑end.
        min_visits_for_negative_penalty: Minimum visits before applying the dead‑end reduction.
        negative_exploration_factor: Multiplicative factor (<1) applied to the exploration term
                                    for dead‑end children.
        stochastic_explore_factor: Factor (<1) that reduces exploration in stochastic variants.
    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    while True:
        # Termination check – support both attribute and callable forms.
        is_terminal = (
            node.is_terminal()
            if callable(getattr(node, "is_terminal", None))
            else getattr(node, "is_terminal", False)
        )
        if is_terminal:
            return node

        # Expansion check – support both attribute and callable forms.
        fully_expanded = (
            node.is_fully_expanded()
            if callable(getattr(node, "is_fully_expanded", None))
            else getattr(node, "is_fully_expanded", False)
        )
        if not fully_expanded:
            return node

        # ---- Cached parent information ----
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        # Dynamic exploration weight: shrink as we gather more visits.
        dyn_exploration = exploration_weight / (1.0 + math.log(parent_visits))

        # Look text (for noop detection)
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

        # Door configuration (cached for door‑penalty)
        parent_doors = None
        try:
            parent_doors = node.state.doors
        except Exception:
            pass

        # Inventory items – used to know whether the map is already owned.
        # Fallback to empty set if attribute missing.
        try:
            inventory = set(node.state.inventory_items)
        except Exception:
            inventory = set()

        # Map‑read flag (for info‑gain bonus)
        map_already_read = False
        try:
            map_already_read = bool(getattr(node.state, "map_read", False))
        except Exception:
            pass

        # Stochastic variant detection
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
            explore = dyn_exploration * math.sqrt(log_parent / child_visits)

            # Stochastic safety factor
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
                            progress_bonus = progress_weight * delta * cat_weight
                        elif delta < 0:
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

            # ---- Door‑state change penalty ----
            if parent_doors is not None:
                try:
                    child_doors = child.state.doors
                    if child_doors != parent_doors:
                        progress_bonus += door_penalty
                except Exception:
                    pass

            # ---- Information‑gain bonus (map handling) ----
            map_missing = "map" not in inventory
            act_str = getattr(child, "action", "")
            if isinstance(act_str, str):
                act_low = act_str.lower().strip()
                if map_missing and (act_low == "take map" or act_low.startswith("read ")):
                    # Stronger bonus, scaled by 2× the base info_gain_bonus
                    progress_bonus += 2.0 * info_gain_bonus

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
