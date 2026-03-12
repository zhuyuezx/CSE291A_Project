"""
LLM-generated MCTS tool: expansion
Description: 
Generated:   2026-03-11T00:32:05.853119
"""

"""
Improved expansion tool for the TextWorld Benchmark.

Key enhancements over the original version:
  * Recognises a broader set of movement commands (e.g., "open door", "go ").
  * Computes a one‑step look‑ahead distance delta to favor actions that
    bring the agent closer to the goal.
  * Rewards actions that truly change the observable state (+2.0) while
    keeping a modest penalty for genuine no‑ops.
  * Softens the blanket penalty for observation actions and ignores them
    while any movement‑type actions remain untried, focusing expansion on
    progress‑making moves.
  * Reduces cloning overhead by re‑using a single simulated state per
    movement action.
"""

import random
from typing import Any, List

# -------------------------------------------------------------------------
# Helper utilities – self‑contained so the function can run independently
# -------------------------------------------------------------------------

def _is_movement(action: str) -> bool:
    """Return True if the action looks like a navigation or door‑opening command."""
    lowered = action.lower()
    # Basic compass directions
    move_tokens = {"north", "south", "east", "west", "up", "down"}
    # Verbs that trigger movement in TextWorld
    verb_tokens = {"go ", "move ", "walk ", "run ", "open door", "door to", "enter "}
    if any(tok in lowered for tok in move_tokens):
        return True
    if any(tok in lowered for tok in verb_tokens):
        return True
    # Catch patterns like "open door east" without the word "to"
    if lowered.startswith("open door") and any(dir_ in lowered for dir_ in move_tokens):
        return True
    return False


def _action_changes_state(state: Any, action: str) -> bool:
    """
    Cheap test whether applying *action* would change the observable state.
    Returns True if the state key differs after the action.
    """
    try:
        cloned = state.clone()
        cloned.apply_action(action)
        return cloned.state_key() != state.state_key()
    except Exception:
        # If the action is illegal or something goes wrong we assume it changes state.
        return True


def _pop_action(untried: Any, chosen: str) -> None:
    """Remove *chosen* from the *untried* container (set, list or similar)."""
    if isinstance(untried, set):
        untried.remove(chosen)
    elif isinstance(untried, list):
        untried.remove(chosen)
    else:
        # generic fallback
        try:
            untried.discard(chosen)  # type: ignore
        except Exception:
            try:
                untried.remove(chosen)  # type: ignore
            except Exception:
                pass


def _score_action(state: Any, action: str, node: Any) -> float:
    """
    Heuristic score for *action*.
    Higher scores are preferred during expansion.

    Components:
      * Distance delta for movement actions (reward moving closer).
      * Bonuses for map‑related actions in mapreader tasks.
      * Mild penalty for observation/no‑op actions.
      * Reward (+2) if the action actually changes the state.
      * Small repeat penalty for actions already expanded from this node.
    """
    score = 0.0
    act_low = action.lower()

    # --- 1️⃣ Distance delta & state‑change detection for movement actions ---
    if _is_movement(action):
        try:
            cur_dist = state.distance_to_goal()
        except Exception:
            cur_dist = 0.0

        # Simulate once to obtain both new distance and change information.
        try:
            sim_state = state.clone()
            sim_state.apply_action(action)
            new_dist = sim_state.distance_to_goal()
            changed = sim_state.state_key() != state.state_key()
        except Exception:
            new_dist = cur_dist
            changed = False

        # Positive when we get closer, negative when we go farther.
        score += (cur_dist - new_dist) * 1.0

        if changed:
            score += 2.0
        else:
            # Small extra negative for a movement that leaves the world unchanged.
            score -= 1.0
    else:
        # --- 2️⃣ Non‑movement actions ---------------------------------------
        # Map‑reader specific bonuses.
        inv = getattr(state, "inventory_items", [])
        if "take map" in act_low and "map" not in inv:
            score += 5.0
        if "read map" in act_low and "map" in inv and not getattr(state, "map_read", False):
            score += 5.0

        # Observation / no‑op penalty (softened).
        no_op_set = {"look", "look around", "inventory", "task", "examine"}
        if any(tok in act_low for tok in no_op_set):
            score -= 0.5

        # Reward if the action truly changes the world.
        if _action_changes_state(state, action):
            score += 2.0
        else:
            score -= 1.0

    # --- 3️⃣ Small repeat penalty -----------------------------------------
    repeat_cnt = sum(1 for a in getattr(node, "children", {}) if a == action)
    score -= 0.5 * repeat_cnt

    return score


def default_expansion(node):
    """
    Expand one untried action from the given node, preferring actions that are
    more likely to make progress according to an enhanced heuristic.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    untried = node._untried_actions
    if not untried:
        raise ValueError("default_expansion called on a node with no untried actions")

    # -------------------------------------------------------------
    # 0️⃣ Determine candidate actions:
    #    If any movement‑type actions are still present, ignore pure
    #    observation actions for this expansion step.
    # -------------------------------------------------------------
    untried_list = list(untried)          # snapshot (set, list, etc.)
    movement_actions = [a for a in untried_list if _is_movement(a)]
    candidate_actions = movement_actions if movement_actions else untried_list

    # -------------------------------------------------------------
    # 1️⃣ Choose the best‑scoring action from the candidate set
    # -------------------------------------------------------------
    best_score = -float("inf")
    best_actions: List[str] = []

    for action in candidate_actions:
        sc = _score_action(node.state, action, node)
        if sc > best_score:
            best_score = sc
            best_actions = [action]
        elif sc == best_score:
            best_actions.append(action)

    # Break ties randomly to preserve exploration.
    chosen_action = random.choice(best_actions)

    # Remove the chosen action from the untried container.
    _pop_action(untried, chosen_action)

    # -------------------------------------------------------------
    # 2️⃣ Create the child node with the resulting state
    # -------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    # Local import to avoid circular dependencies.
    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # Store for future look‑ups.
    node.children[chosen_action] = child

    return child
