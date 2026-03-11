"""
LLM-generated MCTS tool: selection
Description: No changes needed; the draft implementation is correct and efficient.
Generated:   2026-03-09T22:37:15.019990
"""

"""
Improved selection policy for TextWorld Benchmark.

Key enhancements:
* Incentivises reading the map when it has not been read yet.
* Rewards actual room changes (useful after opening doors).
* Applies a stronger and broader no‑op penalty (look, inventory, task,
  opening/closing already‑open doors, etc.).
* Uses a dynamic exploration floor that slowly rises with the number of
  visits, keeping enough exploration pressure deep in the tree.
* Raises the default progress weight so that a single step of progress
  outweighs the (now smaller) exploration term.
"""

import math
from typing import Any

def default_selection(
    node: Any,
    exploration_weight: float = 1.41,
    progress_weight: float = 0.7,      # stronger influence of progress
    nop_penalty: float = 0.4,          # harsher penalty for useless actions
    map_read_bonus: float = 2.0,       # bonus for the first “read map” action
    room_change_bonus: float = 0.5,    # bonus when the agent actually moves rooms
    decay_visits_threshold: int = 30, # start decaying later
    exploration_floor: float = 0.3,    # minimal exploration constant
) -> Any:
    """
    Enhanced UCB1 tree policy for TextWorld Benchmark.

    Returns a node that is either terminal or has at least one
    untried action (so the expansion phase can act).

    Parameters
    ----------
    node : Any
        Root MCTSNode to start selection from.
    exploration_weight : float, optional
        Base UCB1 exploration constant C (default 1.41).
    progress_weight : float, optional
        Scaling factor for the progress heuristic (default 0.7).
    nop_penalty : float, optional
        Penalty subtracted when an action appears to do nothing
        (default 0.4).
    map_read_bonus : float, optional
        Extra progress reward for the first successful ``read map``.
    room_change_bonus : float, optional
        Extra reward when the child state is in a different room.
    decay_visits_threshold : int, optional
        Number of visits after which C starts to decay (default 30).
    exploration_floor : float, optional
        Minimal exploration constant; will grow slowly with visits.

    Returns
    -------
    Any
        Selected MCTSNode.
    """
    while not node.is_terminal:
        # If there are still actions we haven't tried from this node,
        # hand over control to the expansion phase.
        if not node.is_fully_expanded:
            return node

        # ------------------------------------------------------------
        # Exploration constant (C) – decay gently, with a slowly rising floor.
        # ------------------------------------------------------------
        if node.visits > decay_visits_threshold:
            # Decay inversely with sqrt(visits) but never below a floor that
            # rises a little with log(visits) to retain some exploration.
            floor = exploration_floor * (1.0 + 0.01 * math.log(node.visits + 1))
            C = max(exploration_weight / math.sqrt(node.visits), floor)
        else:
            C = exploration_weight

        # Guard against log(0) for the very first visit.
        log_parent = math.log(node.visits) if node.visits > 0 else 0.0

        # ------------------------------------------------------------
        # Cache parent information once per iteration.
        # ------------------------------------------------------------
        # Distance to goal (if available)
        parent_dist = None
        if hasattr(node.state, "distance_to_goal"):
            try:
                parent_dist = node.state.distance_to_goal()
            except Exception:
                parent_dist = None

        # Observation text (used for no‑op detection)
        parent_obs = None
        if hasattr(node.state, "look_text"):
            try:
                parent_obs = node.state.look_text()
            except Exception:
                parent_obs = None

        # Current room identifier (if available)
        parent_room = getattr(node.state, "room", None)

        # Whether the map has already been read
        parent_map_read = getattr(node.state, "map_read", False)

        best_child = None
        best_score = -math.inf

        for child in node.children.values():
            # --------------------------------------------------------
            # Immediate selection of never‑visited children.
            # --------------------------------------------------------
            if child.visits == 0:
                best_child = child
                break

            # Classic UCB1 components
            exploit = child.value / child.visits
            explore = C * math.sqrt(log_parent / child.visits)

            # --------------------------------------------------------
            # Progress heuristic (aggregate several domain signals).
            # --------------------------------------------------------
            progress = 0.0

            # ----- distance delta -------------------------------------------------
            child_dist = None
            if hasattr(child.state, "distance_to_goal"):
                try:
                    child_dist = child.state.distance_to_goal()
                except Exception:
                    child_dist = None

            if parent_dist is not None and child_dist is not None:
                progress += (parent_dist - child_dist)  # positive if we get closer

            # ----- room change bonus -----------------------------------------------
            child_room = getattr(child.state, "room", None)
            if parent_room is not None and child_room is not None and parent_room != child_room:
                progress += room_change_bonus

            # ----- action‑based heuristics -----------------------------------------
            act = getattr(child, "parent_action", "")
            if isinstance(act, str):
                act_low = act.lower()

                # Incentive for taking goal items (coin or map)
                if ("take" in act_low or "pick" in act_low) and (
                    "coin" in act_low or "map" in act_low
                ):
                    progress += 1.0

                # Small bias towards movement actions
                if act_low.startswith(("go ", "move ", "walk ")):
                    progress += 0.1

                # ----- map‑read incentive -----------------------------------------
                if not parent_map_read and ("read" in act_low or "inspect" in act_low):
                    if "map" in act_low:
                        progress += map_read_bonus

                # ----- no‑op detection & penalty -----------------------------------
                # Look / inventory / generic task that do not change observation.
                if act_low.startswith(("look", "inventory", "task")):
                    if (
                        parent_obs is not None
                        and hasattr(child.state, "look_text")
                    ):
                        try:
                            if child.state.look_text() == parent_obs:
                                progress -= nop_penalty
                        except Exception:
                            pass

                # Moving into a blocked direction (observation unchanged)
                if act_low.startswith(("go ", "move ")):
                    if (
                        parent_obs is not None
                        and hasattr(child.state, "look_text")
                    ):
                        try:
                            if child.state.look_text() == parent_obs:
                                progress -= nop_penalty
                        except Exception:
                            pass

                # Opening / closing a door that yields no observation change
                if act_low.startswith(("open", "close")):
                    if (
                        parent_obs is not None
                        and hasattr(child.state, "look_text")
                    ):
                        try:
                            if child.state.look_text() == parent_obs:
                                progress -= nop_penalty
                        except Exception:
                            pass

            # --------------------------------------------------------
            # Combine UCB components
            # --------------------------------------------------------
            score = exploit + explore + progress_weight * progress

            if score > best_score:
                best_child = child
                best_score = score

        # Safety fallback – keep current node if something went wrong.
        node = best_child if best_child is not None else node

    return node
