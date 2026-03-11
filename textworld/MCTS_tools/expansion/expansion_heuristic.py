"""
LLM-generated MCTS tool: expansion
Description: 
Generated:   2026-03-10T00:59:25.438817
"""

"""
Improved expansion phase for TextWorld Benchmark.

Key changes over the original heuristic:
  * Motion actions receive a positive base bonus and a scaled reward
    proportional to how much they reduce the distance to the goal.
  * Opening/closing doors is only rewarded when it leads to a
    strictly shorter distance; otherwise it is penalised.
  * Meta‑actions (look, inventory, task) and any action that leaves the
    state unchanged receive an additional penalty, discouraging wasteful
    expansions.
  * Weight constants are tuned to keep useful actions clearly ahead of
    trivial ones while preserving stochasticity via the margin‑based
    top‑action selection.
"""

import random
from typing import Any, List, Tuple

# ----------------------------------------------------------------------
# Heuristic parameters – can be tuned further if needed.
# ----------------------------------------------------------------------
MOVE_BASE_BONUS = 0.5          # small positive boost for any movement attempt
DISTANCE_BONUS = 2.0           # multiplier for distance reduction (higher = more aggressive)
DOOR_BONUS = 1.0               # baseline reward for a useful door action
DOOR_PENALTY = -3.0            # penalty for opening/closing an already‑in‑desired state
META_PENALTY = -1.5            # base penalty for look/inventory/task when they do nothing
NOOP_PENALTY = -2.0            # extra penalty if the action does not change the state
TAKE_BONUS = 5.0               # reward for taking coin or map
READ_MAP_BONUS = 4.0           # reward for reading the map when applicable


def _state_signature(state: Any) -> Any:
    """
    Helper to obtain a cheap immutable representation of a state.
    We use the state's ``state_key`` if available; otherwise fall back
    to ``repr(state)`` which is still deterministic for the benchmark.
    """
    try:
        return state.state_key()
    except Exception:
        return repr(state)


def _action_score(state: Any, action: str) -> float:
    """
    Compute a lightweight, scaled heuristic score for a single action.
    Positive scores indicate actions that are expected to move the
    game toward completion.
    """
    score = 0.0

    # ------------------------------------------------------------------
    # 1. Meta‑actions that usually do not change the state.
    # ------------------------------------------------------------------
    if action in ("look", "look around", "inventory", "task"):
        # Apply a baseline penalty; if the action truly changes the state
        # (e.g., reading a new map) we will add additional bonuses later.
        score += META_PENALTY

    # ------------------------------------------------------------------
    # 2. Taking valuable objects.
    # ------------------------------------------------------------------
    if action.startswith("take "):
        obj = action.split(" ", 1)[1]
        if obj in ("coin", "map"):
            score += TAKE_BONUS

    # ------------------------------------------------------------------
    # 3. Reading the map.
    # ------------------------------------------------------------------
    if action == "read map":
        has_map = "map" in getattr(state, "inventory_items", [])
        map_read = getattr(state, "map_read", False)
        if has_map and not map_read:
            score += READ_MAP_BONUS

    # ------------------------------------------------------------------
    # 4. Movement actions (go / move).
    # ------------------------------------------------------------------
    if action.startswith(("go ", "move ")):
        # Base bonus so movement is never the worst option.
        score += MOVE_BASE_BONUS

        # Simulate the move to evaluate distance impact.
        try:
            next_state = state.clone()
            next_state.apply_action(action)
            if hasattr(state, "distance_to_goal") and hasattr(next_state, "distance_to_goal"):
                cur_dist = float(state.distance_to_goal())
                new_dist = float(next_state.distance_to_goal())
                # Positive contribution proportional to distance reduction.
                dist_reduction = cur_dist - new_dist
                if dist_reduction > 0:
                    score += DISTANCE_BONUS * dist_reduction
        except Exception:
            # If simulation fails, keep only the base bonus.
            pass

    # ------------------------------------------------------------------
    # 5. Door manipulation – reward only if it shortens the path.
    # ------------------------------------------------------------------
    doors = getattr(state, "doors", {})
    if action.startswith("open "):
        direction = action.split(" ", 1)[1]
        # Penalise redundant opening.
        if doors.get(direction) is True:
            score += DOOR_PENALTY
        else:
            # Simulate opening to see distance effect.
            try:
                next_state = state.clone()
                next_state.apply_action(action)
                if hasattr(state, "distance_to_goal") and hasattr(next_state, "distance_to_goal"):
                    cur_dist = float(state.distance_to_goal())
                    new_dist = float(next_state.distance_to_goal())
                    if new_dist < cur_dist:
                        # Useful door – give base bonus plus distance improvement.
                        score += DOOR_BONUS + DISTANCE_BONUS * (cur_dist - new_dist)
                    else:
                        # Door opens but does not help; small penalty.
                        score += DOOR_PENALTY
            except Exception:
                score += DOOR_PENALTY

    if action.startswith("close "):
        direction = action.split(" ", 1)[1]
        # Penalise redundant closing.
        if doors.get(direction) is False:
            score += DOOR_PENALTY
        else:
            # Generally closing is rarely useful for these tasks; keep a small
            # negative bias unless it demonstrably shortens the path.
            try:
                next_state = state.clone()
                next_state.apply_action(action)
                if hasattr(state, "distance_to_goal") and hasattr(next_state, "distance_to_goal"):
                    cur_dist = float(state.distance_to_goal())
                    new_dist = float(next_state.distance_to_goal())
                    if new_dist < cur_dist:
                        score += DOOR_BONUS + DISTANCE_BONUS * (cur_dist - new_dist)
                    else:
                        score += DOOR_PENALTY
            except Exception:
                score += DOOR_PENALTY

    # ------------------------------------------------------------------
    # 6. General no‑op detection for any action.
    # ------------------------------------------------------------------
    try:
        sim_state = state.clone()
        sim_state.apply_action(action)
        if _state_signature(sim_state) == _state_signature(state):
            # Action left the world unchanged → additional penalty.
            score += NOOP_PENALTY
    except Exception:
        # If simulation crashes we assume the action is non‑trivial enough
        # to keep its existing score.
        pass

    return score


def default_expansion(node):
    """
    Expand one untried action from the given node, preferring actions
    that are judged useful by a lightweight, distance‑aware heuristic.

    Parameters
    ----------
    node: MCTSNode
        Node that has at least one untried action.  The node must expose
        the mutable container `node._untried_actions`, the current
        `node.state`, and the dictionary `node.children`.

    Returns
    -------
    MCTSNode
        The newly created child node.
    """
    # --------------------------------------------------------------------
    # 1️⃣ Gather the list of currently untried actions.
    # --------------------------------------------------------------------
    try:
        actions = list(node._untried_actions)   # works for set, list, tuple
    except TypeError:
        actions = [node._untried_actions]

    if not actions:
        raise ValueError("default_expansion called on a node with no untried actions")

    # --------------------------------------------------------------------
    # 2️⃣ Score every candidate action using the improved heuristic.
    # --------------------------------------------------------------------
    scored_actions: List[Tuple[str, float]] = [
        (action, _action_score(node.state, action)) for action in actions
    ]

    # --------------------------------------------------------------------
    # 3️⃣ Determine the top‑scoring actions, keeping a small margin for
    #     stochasticity while strongly biasing toward the best.
    # --------------------------------------------------------------------
    best_score = max(score for _, score in scored_actions)
    margin = 0.5   # tighter margin because scores are now better separated
    top_actions = [action for action, score in scored_actions if score >= best_score - margin]

    # --------------------------------------------------------------------
    # 4️⃣ Pick one of the top actions uniformly at random.
    # --------------------------------------------------------------------
    chosen_action = random.choice(top_actions)

    # --------------------------------------------------------------------
    # 5️⃣ Remove the chosen action from the node's untried pool.
    # --------------------------------------------------------------------
    if hasattr(node._untried_actions, "discard"):
        node._untried_actions.discard(chosen_action)
    elif hasattr(node._untried_actions, "remove"):
        try:
            node._untried_actions.remove(chosen_action)
        except ValueError:
            pass
    else:
        # Fallback: rebuild the container without the chosen action.
        node._untried_actions = [a for a in actions if a != chosen_action]

    # --------------------------------------------------------------------
    # 6️⃣ Create the child node with the resulting state.
    # --------------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    # Import locally to avoid circular import problems.
    from mcts.node import MCTSNode

    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
