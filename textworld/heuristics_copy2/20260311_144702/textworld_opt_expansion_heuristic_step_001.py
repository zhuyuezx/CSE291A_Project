"""
LLM-generated MCTS tool: expansion
Description: No changes needed; the function is correct and efficient.
Generated:   2026-03-11T14:58:36.368625
"""

import random
from typing import List, Tuple

# ----------------------------------------------------------------------
# Helper utilities – all self‑contained, using only the public GameState API
# ----------------------------------------------------------------------
def _is_noop(action: str) -> bool:
    """Return True for actions that only provide information and never change position."""
    low = action.lower()
    return any(noop in low for noop in ("look", "inventory", "task"))


def _action_category(action: str) -> str:
    """
    Very simple categorical split based on substrings.
    Returns one of: move, open, close, take, read, noop, other
    """
    low = action.lower()
    if any(dir_ in low for dir_ in ("north", "south", "east", "west", "go ", "move ")):
        return "move"
    if "open" in low:
        return "open"
    if "close" in low:
        return "close"
    if "take" in low:
        return "take"
    if "read" in low:
        return "read"
    if _is_noop(action):
        return "noop"
    return "other"


def _simulate(state, action: str):
    """Clone the state, apply the action and return the new state."""
    new_state = state.clone()
    new_state.apply_action(action)
    return new_state


def _score_action(action: str, state) -> float:
    """
    Heuristic score for an action in the *current* state.
    Higher scores indicate more promising expansions.
    """
    cat = _action_category(action)

    # Base scores per category
    base_scores = {
        "move": 0.0,
        "open": 0.0,
        "close": -2.0,
        "take": 1.5,
        "read": 1.0,
        "noop": -5.0,
        "other": 0.0,
    }
    score = base_scores.get(cat, 0.0)

    # Refine with a one‑step lookahead where cheap
    try:
        after = _simulate(state, action)
    except Exception:  # safety net – if action crashes, give it a low score
        return -float("inf")

    # Movement: prefer actions that reduce distance to goal
    if cat == "move":
        try:
            dist = after.distance_to_goal()
            # smaller distance -> higher score
            score += -dist
            if dist == 0:  # reached goal directly
                score += 10.0
        except Exception:
            pass

    # Opening doors: reward if distance shrinks after opening
    if cat == "open":
        try:
            before = state.distance_to_goal()
            after_dist = after.distance_to_goal()
            if after_dist < before:
                score += 5.0 - after_dist  # positive boost
        except Exception:
            pass

    # Reading the map: only useful once the map is in inventory and not yet read
    if cat == "read":
        if getattr(state, "map_read", False):
            score -= 3.0  # already read, treat as low value
        else:
            # ensure map is in inventory (heuristic – we cannot check directly)
            # give a moderate bump
            score += 3.0

    # Taking objects (coin, map) – high value if not already in inventory
    if cat == "take":
        # later states may have the object already; we cannot query inventory,
        # but a small penalty for repeat takes helps.
        score += 2.0

    # Small random tie‑breaker to avoid deterministic loops
    score += random.uniform(-0.1, 0.1)
    return score


def _opposite_door_action(action: str) -> str:
    """
    Given an 'open door …' action, return the matching 'close door …' action.
    If the action is not an open‑door command, return empty string.
    """
    low = action.lower()
    if low.startswith("open"):
        return "close" + action[4:]  # replace 'open' with 'close'
    return ""


# ----------------------------------------------------------------------
# Modified expansion routine
# ----------------------------------------------------------------------
def default_expansion(node):
    """
    Expand one untried action from the given node using a
    lightweight heuristic ranking.

    The routine:
      1. Scores all currently untried actions.
      2. Prefers progress‑making actions (movement, opening needed doors,
         taking/reading objects) and deprioritises no‑ops.
      3. After expanding an 'open door' action, removes the complementary
         'close door' from the child's untried actions to avoid immediate toggle.
      4. Falls back to a random choice if scoring fails.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    # ------------------------------------------------------------------
    # Gather current untried actions – they may be a set, list or other iterable
    # ------------------------------------------------------------------
    try:
        actions = list(node._untried_actions)
    except Exception:
        actions = []

    if not actions:
        # No untried actions – cannot expand; return node itself
        return node

    # ------------------------------------------------------------------
    # Filter out pure informational actions if any progress actions exist
    # ------------------------------------------------------------------
    progress_actions = [a for a in actions if not _is_noop(a)]
    candidate_actions = progress_actions if progress_actions else actions

    # ------------------------------------------------------------------
    # Score each candidate action
    # ------------------------------------------------------------------
    scored: List[Tuple[float, str]] = []
    for act in candidate_actions:
        try:
            sc = _score_action(act, node.state)
        except Exception:
            sc = -float("inf")
        scored.append((sc, act))

    # Choose the highest‑scoring action (break ties randomly)
    if scored:
        max_score = max(scored, key=lambda x: x[0])[0]
        best_actions = [act for (sc, act) in scored if sc == max_score]
        chosen_action = random.choice(best_actions)
    else:
        # Fallback – random pop
        chosen_action = node._untried_actions.pop()
        child_state = node.state.clone()
        child_state.apply_action(chosen_action)
        from mcts.node import MCTSNode
        child = MCTSNode(child_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        return child

    # ------------------------------------------------------------------
    # Remove the selected action from the parent's untried set
    # ------------------------------------------------------------------
    try:
        node._untried_actions.discard(chosen_action)  # set‑like
    except AttributeError:
        try:
            node._untried_actions.remove(chosen_action)  # list‑like
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Create the child node with the chosen action
    # ------------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # ------------------------------------------------------------------
    # Door‑toggle guard: if we just opened a door, prune the opposite close action
    # ------------------------------------------------------------------
    if _action_category(chosen_action) == "open":
        opposite = _opposite_door_action(chosen_action)
        if opposite:
            try:
                child._untried_actions.discard(opposite)
            except AttributeError:
                try:
                    child._untried_actions.remove(opposite)
                except Exception:
                    pass

    # Register child and return
    node.children[chosen_action] = child
    return child
