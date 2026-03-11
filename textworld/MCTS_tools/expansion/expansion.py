"""
LLM-generated MCTS tool: expansion
Description: Fixed SyntaxError by moving/removing __future__ import, added missing imports, ensured safe handling of list‑ and set‑based untried‑action containers, and made the function self‑contained.
Generated:   2026-03-10T00:55:07.002901
"""

"""
Improved expansion for TextWorld Benchmark.

Ranks untried actions with a cheap heuristic and expands the highest‑scoring
action. Low‑utility actions (e.g., look/inventory when they produce no
information, door toggles that do not change rooms) are given low scores but
are **not** permanently discarded here – this avoids emptying the untried‑action
pool and guarantees a child is always created.
"""

from typing import List, Tuple, Any

# --------------------------------------------------------------------------- #
# Helper: lightweight heuristic scoring for a candidate action.
# --------------------------------------------------------------------------- #
def _action_score(state: Any, action: str, prev_distance: float | None) -> float:
    """
    Compute a lightweight score for ``action`` taken in ``state``.

    Factors:
      * reduction in distance_to_goal (positive if distance shrinks)
      * map‑reader bonuses (take / read map)
      * door relevance (moving through a door is slightly rewarded)
      * no‑op penalty for look/inventory/task that leave the description unchanged
    """
    # Simulate the action once – cheap compared to a full rollout.
    sim = state.clone()
    sim.apply_action(action)

    score = 0.0

    # ----- distance reduction -------------------------------------------------
    try:
        new_dist = sim.distance_to_goal()
    except Exception:
        new_dist = None

    if prev_distance is not None and new_dist is not None:
        # Positive if we get closer to the goal.
        score += (prev_distance - new_dist) * 1.0

    # ----- map‑reader specific bonuses ----------------------------------------
    cfg = getattr(state, "config", {})
    game_type = cfg.get("game_type", "")
    if game_type == "mapreader" and not getattr(state, "map_read", False):
        low = action.lower()
        if "take map" in low:
            score += 2.0          # must obtain the map early
        elif "read map" in low:
            score += 1.5          # reading reveals the goal

    # ----- door relevance ------------------------------------------------------
    low = action.lower()
    if "open" in low or "close" in low:
        # If the action changes the room, it is probably a useful door use.
        if getattr(sim, "room", None) != getattr(state, "room", None):
            score += 0.5
        else:
            score -= 0.5          # a pure toggle without moving is rarely helpful

    # ----- no‑op penalty (especially important in punishment mode) ------------
    try:
        before = state.look_text()
        after = sim.look_text()
        if before == after and any(tok in low for tok in ("look", "inventory", "task")):
            score -= 2.0          # produces no new information → penalise
    except Exception:
        pass

    return score


# --------------------------------------------------------------------------- #
# Main expansion routine used by the MCTS implementation.
# --------------------------------------------------------------------------- #
def default_expansion(node: Any) -> Any:
    """
    Expand one untried action from ``node`` using a heuristic ordering.

    Steps:
      1. Score every currently untried action.
      2. Select the highest‑scoring action.
      3. Remove that action from the node's untried pool.
      4. Create and return the corresponding child node.
    """
    # ------------------------------------------------------------------- #
    # Retrieve the container that stores untried actions.
    # The benchmark may expose it as a set, list, or any mutable Sequence.
    # ------------------------------------------------------------------- #
    untried = getattr(node, "_untried_actions", None)
    if untried is None:
        raise RuntimeError("Node does not expose a '_untried_actions' attribute.")

    # If the container is empty we cannot expand.
    if not untried:
        raise RuntimeError("Attempted to expand a node with no untried actions.")

    # Work with a concrete list for deterministic processing.
    actions: List[str] = list(untried)

    # ------------------------------------------------------------------- #
    # Pre‑compute the current distance to goal if the method exists.
    # ------------------------------------------------------------------- #
    try:
        prev_dist = node.state.distance_to_goal()
    except Exception:
        prev_dist = None

    # ------------------------------------------------------------------- #
    # Score every candidate action using the cheap heuristic.
    # ------------------------------------------------------------------- #
    scored: List[Tuple[str, float]] = [
        (act, _action_score(node.state, act, prev_dist)) for act in actions
    ]

    # ------------------------------------------------------------------- #
    # Choose the highest‑scoring action.
    # ------------------------------------------------------------------- #
    chosen_action, _ = max(scored, key=lambda x: x[1])

    # ------------------------------------------------------------------- #
    # Remove the chosen action from the original untried‑action container.
    # ------------------------------------------------------------------- #
    if isinstance(untried, set):
        # set.discard is safe even if the element is missing.
        untried.discard(chosen_action)
    else:
        # Assume list‑like (or any mutable sequence supporting .remove).
        try:
            untried.remove(chosen_action)
        except ValueError:
            # Should not happen, but we silently ignore to keep the algorithm robust.
            pass

    # ------------------------------------------------------------------- #
    # Apply the chosen action to obtain the child state and create the node.
    # ------------------------------------------------------------------- #
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    # Import here to avoid circular‑import problems if the module is used
    # in a context where MCTSNode is defined elsewhere.
    from mcts.node import MCTSNode  # type: ignore

    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # Register the child for future traversals.
    if not hasattr(node, "children"):
        node.children = {}
    node.children[chosen_action] = child

    return child
