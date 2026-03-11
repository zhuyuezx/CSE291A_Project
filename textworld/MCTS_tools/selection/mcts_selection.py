"""
LLM-generated MCTS tool: selection
Description: Improved UCB1 selection with progressive bias (no code changes needed)
Generated:   2026-03-09T23:44:23.545645
"""

"""
Enhanced selection: UCB1 with progressive bias.

Walks down the tree choosing the child with the highest combined score:
    exploit + explore + λ * h(state)

The heuristic h(state) favours actions that:
  * Reduce distance to the goal (negative distance → larger value).
  * Have read the map in map‑reader tasks (small bonus).
  * Actually change the world (state_key differs from parent) – penalises
    repeated “look”, “inventory”, “task” actions.

When scores are equal (within a tiny epsilon) the child whose state has the
smaller distance to the goal is preferred, providing a deterministic
tie‑breaker that avoids oscillating open/close loops.

All constants are chosen to be small so that the heuristic only guides
early exploration; real simulation outcomes still dominate the final
policy.
"""

import math
from typing import Any

# --------------------------------------------------------------------------- #
# Helper: compute a domain‑specific heuristic for a GameState.
# --------------------------------------------------------------------------- #
def _state_heuristic(state: Any, parent_state: Any) -> float:
    """
    Returns a scalar hint of how promising *state* is relative to *parent_state*.

    Parameters
    ----------
    state : GameState
        The child state whose promise we evaluate.
    parent_state : GameState
        The parent state, used to detect whether the action changed the world.

    Returns
    -------
    float
        Heuristic value (higher = better). The value is deliberately small
        compared with typical UCB scores.
    """
    # Base: negative distance → larger is better.
    # distance_to_goal() should be defined for both coin and mapreader tasks.
    try:
        dist = state.distance_to_goal()
    except Exception:
        # Fallback if not implemented – treat unknown distance as 0.
        dist = 0.0
    h = -dist  # closer to goal -> higher value

    # Bonus for having read the map (only relevant for mapreader tasks).
    # Guard against missing attribute.
    if getattr(state, "map_read", False):
        h += 0.5  # modest boost when the map is available

    # Bonus if the world actually changed (different description / state key).
    # In punishment mode, actions that leave the description unchanged are
    # penalised, so we give a small positive reward when the key differs.
    try:
        if state.state_key() != parent_state.state_key():
            h += 0.2
    except Exception:
        # If state_key is unavailable we ignore this component.
        pass

    return h


# --------------------------------------------------------------------------- #
# Modified selection function.
# --------------------------------------------------------------------------- #
def default_selection(node, exploration_weight: float = 1.41, heuristic_weight: float = 0.1):
    """
    UCB1 tree walk with a progressive bias based on game‑specific heuristics.

    Descends the tree choosing the child with the highest combined score.
    Stops when reaching a node that is either terminal or has untried actions
    (so the expansion phase can create a new child).

    Parameters
    ----------
    node : MCTSNode
        Root node to start selection from.
    exploration_weight : float, optional
        Classic UCB exploration constant C (default 1.41).
    heuristic_weight : float, optional
        Weight λ for the heuristic term h(state) (default 0.1).

    Returns
    -------
    MCTSNode
        A node that is either terminal or has at least one untried action.
    """
    EPS = 1e-9  # tolerance for floating‑point tie handling

    while not node.is_terminal:
        # If we still have actions to try, stop here for expansion.
        if not node.is_fully_expanded:
            return node

        # All children are fully expanded; pick the best according to the
        # augmented UCB formula.
        parent_visits = node.visits
        # Guard against log(0); if parent hasn't been visited yet, treat it
        # as 1 (the exploration term will be large, which is fine).
        log_parent = math.log(parent_visits) if parent_visits > 0 else 0.0

        best_child = None
        best_score = -math.inf
        best_dist = math.inf  # for deterministic tie‑break

        for child in node.children.values():
            # Exploitation term (average value). Visits should be >0 for
            # fully‑expanded children; guard against accidental zero.
            if child.visits == 0:
                exploit = 0.0
                explore = float('inf')  # force exploration of never‑visited node
            else:
                exploit = child.value / child.visits
                explore = exploration_weight * math.sqrt(log_parent / child.visits)

            # Heuristic term based on child state vs parent state.
            h = _state_heuristic(child.state, node.state)

            score = exploit + explore + heuristic_weight * h

            # Tie‑breaker: prefer child that is closer to the goal.
            try:
                child_dist = child.state.distance_to_goal()
            except Exception:
                child_dist = math.inf

            if (score > best_score + EPS) or (
                abs(score - best_score) <= EPS and child_dist < best_dist
            ):
                best_child = child
                best_score = score
                best_dist = child_dist

        # Defensive programming: if something went wrong and we couldn't pick
        # a child, break out to avoid an infinite loop.
        if best_child is None:
            return node

        node = best_child

    # Reached a terminal node.
    return node
