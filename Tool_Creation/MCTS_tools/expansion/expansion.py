"""
LLM-generated MCTS tool: expansion
Description: Fixed attribute errors by handling both `MCTSNode` objects (with `_untried_actions`) and raw `SokobanState` objects (using `legal_actions`). Adjusted all state accesses accordingly and safely created the child node only when a true MCTS node is supplied.
Generated:   2026-03-06T23:48:18.958247
"""

import random
from typing import List, Tuple, Any

# Safe import of the dead‑lock detector used by rollouts.
# If the exact helper is named differently in the project, adjust accordingly.
try:
    from mcts.rollout import _is_deadlock  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: assume no dead‑lock detection is available.
    def _is_deadlock(state) -> bool:  # pylint: disable=unused-argument
        return False


def default_expansion(node: Any):
    """
    Expand one untried action from the given node using a cheap
    heuristic look‑ahead.

    The function now:
      • Scores each untried action by its effect on boxes‑on‑targets,
        total box distance and whether it is a safe push or walk.
      • Discards actions that immediately create a dead‑lock.
      • Chooses the highest‑scoring action, but with a small ε‑greedy
        chance to keep exploration.
      • Removes the chosen action from the node's untried pool (if
        present) and creates the corresponding child node (if possible).

    Works with both a full ``MCTSNode`` (which stores ``_untried_actions``)
    and a plain ``SokobanState`` (in which case the legal actions are
    derived from ``state.legal_actions()``).

    Args:
        node: Either an ``MCTSNode`` or a ``SokobanState``.

    Returns:
        The newly created child ``MCTSNode`` if ``node`` is an MCTS node,
        otherwise the child ``SokobanState``.
    """
    # ------------------------------------------------------------------
    # 1. Determine whether we received an MCTS node or a raw state.
    # ------------------------------------------------------------------
    is_mcts_node = hasattr(node, "_untried_actions") and hasattr(node, "children")

    if is_mcts_node:
        # MCTSNode – use its cached untried actions and its internal state.
        mcts_node = node
        cur_state = mcts_node.state
        untried_actions = list(mcts_node._untried_actions)
    else:
        # Plain SokobanState – treat every legal action as “untried”.
        cur_state = node
        untried_actions = list(cur_state.legal_actions())

    if not untried_actions:
        raise RuntimeError(
            "default_expansion called on a node/state with no untried actions"
        )

    # ------------------------------------------------------------------
    # 2. Compute cheap heuristics for the current state.
    # ------------------------------------------------------------------
    cur_targets = cur_state.boxes_on_targets()
    cur_dist = cur_state.total_box_distance()

    # ------------------------------------------------------------------
    # 3. Score each candidate action.
    # ------------------------------------------------------------------
    EPSILON = 0.05               # probability of random expansion
    TARGET_WEIGHT = 5.0          # importance of placing a box on a target
    DIST_WEIGHT = 1.0            # importance of reducing total box distance
    PUSH_PENALTY = -0.5           # penalty when a push makes distance worse
    WALK_BIAS = 0.2              # small bonus for pure moves (no push)
    DEADLOCK_PENALTY = -1e9      # huge negative score to prune dead‑locks

    action_scores: List[Tuple[Any, float]] = []
    action_states: dict[Any, Any] = {}

    for a in untried_actions:
        # Clone once, apply the action, and keep the resulting state.
        child_state = cur_state.clone()
        child_state.apply_action(a)

        # Immediate dead‑lock detection – assign huge negative score.
        if _is_deadlock(child_state):
            score = DEADLOCK_PENALTY
        else:
            # Metric deltas
            delta_targets = child_state.boxes_on_targets() - cur_targets
            delta_dist = cur_dist - child_state.total_box_distance()

            # Base score from progress
            score = TARGET_WEIGHT * delta_targets + DIST_WEIGHT * delta_dist

            # Determine whether the action was a push.
            # A push occurs if the player moves onto a square that previously held a box.
            # For both node types we can compare the player's new position with the
            # previous set of boxes.
            is_push = child_state.player in cur_state.boxes

            if is_push:
                # Penalise pushes that increase the distance to the nearest target.
                if delta_dist < 0:
                    score += PUSH_PENALTY * abs(delta_dist)
            else:
                # Small bias for pure walks.
                score += WALK_BIAS

        action_scores.append((a, score))
        action_states[a] = child_state

    # ------------------------------------------------------------------
    # 4. Choose an action (ε‑greedy).
    # ------------------------------------------------------------------
    # Filter out actions that are known dead‑locks for the random branch as well.
    non_deadlocked = [(a, s) for (a, s) in action_scores if s != DEADLOCK_PENALTY]

    if not non_deadlocked:
        # All remaining actions are dead‑locks; fall back to any action.
        chosen_action = random.choice(untried_actions)
    elif random.random() < EPSILON:
        chosen_action = random.choice([a for a, _ in non_deadlocked])
    else:
        # Greedy choice: highest scored action(s)
        max_score = max(s for _, s in non_deadlocked)
        best_actions = [a for a, s in non_deadlocked if s == max_score]
        chosen_action = random.choice(best_actions)

    # ------------------------------------------------------------------
    # 5. Update the untried‑action container (if we are dealing with an MCTS node).
    # ------------------------------------------------------------------
    if is_mcts_node:
        try:
            mcts_node._untried_actions.remove(chosen_action)   # list or set
        except (AttributeError, ValueError):
            try:
                mcts_node._untried_actions.discard(chosen_action)  # set‑like
            except Exception:
                # Re‑build the container without the chosen action as a last resort.
                mcts_node._untried_actions = [
                    a for a in mcts_node._untried_actions if a != chosen_action
                ]

    # ------------------------------------------------------------------
    # 6. Return the child (node or state).
    # ------------------------------------------------------------------
    child_state = action_states[chosen_action]

    if is_mcts_node:
        # Import locally to avoid circular imports.
        from mcts.node import MCTSNode  # pylint: disable=import-outside-toplevel
        child = MCTSNode(child_state, parent=mcts_node, parent_action=chosen_action)
        mcts_node.children[chosen_action] = child
        return child
    else:
        # No MCTS wrapper – just return the new SokobanState.
        return child_state
