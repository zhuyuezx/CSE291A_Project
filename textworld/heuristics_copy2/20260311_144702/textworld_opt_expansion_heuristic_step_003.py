"""
LLM-generated MCTS tool: expansion
Description: 
Generated:   2026-03-11T15:01:39.360684
"""

"""
Improved MCTS expansion heuristic for TextWorld Benchmark.

Key enhancements:
  * Movement now receives *negative* score when it does not reduce the
    distance to the goal (reduces oscillations).
  * When the map has been read, movement baseline and distance‑gain are
    amplified to push directly toward the revealed goal.
  * Opening doors carries a modest positive baseline and is penalised
    when it does not bring the agent closer.
  * “task” actions are treated as no‑ops and heavily penalised.
  * Repeated “look”/“inventory” that leave the observation unchanged receive
    an extra small penalty.
  * Door‑toggle pruning remains, but the opposite action is also removed
    from the parent’s untried‑action container.
"""

import random
from typing import List, Tuple, Iterable


# ----------------------------------------------------------------------
# Helper utilities
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


def _opposite_door_action(action: str) -> str:
    """
    Given an 'open door …' action, return the matching 'close door …' action.
    If the action is not an open‑door command, return empty string.
    """
    low = action.lower()
    if low.startswith("open"):
        return "close" + action[4:]  # replace 'open' with 'close'
    return ""


def _remove_action_untried(container: Iterable[str], action: str):
    """
    Safely remove an action from a container that may be a set or a list.
    Silently ignores missing entries.
    """
    try:
        container.discard(action)          # set‑like
    except AttributeError:
        try:
            container.remove(action)       # list‑like
        except Exception:
            pass


# ----------------------------------------------------------------------
# Scoring function
# ----------------------------------------------------------------------
def _score_action(action: str, state) -> float:
    """
    Heuristic score for an action in the *current* state.
    Higher scores indicate more promising expansions.
    """
    cat = _action_category(action)

    # ------------------------------------------------------------------
    # Base scores (tuned)
    # ------------------------------------------------------------------
    base_scores = {
        "move": 0.3,    # low positive baseline, can become negative after penalty
        "open": 0.5,    # modest encouragement to open useful doors
        "close": -2.0,
        "take": 1.5,
        "read": 1.0,
        "noop": -5.0,
        "other": 0.0,
    }
    score = base_scores.get(cat, 0.0)

    # ------------------------------------------------------------------
    # One‑step lookahead (cheap but informative)
    # ------------------------------------------------------------------
    try:
        after = _simulate(state, action)
    except Exception:
        # Invalid action – deprioritise heavily
        return -float("inf")

    # ------------------------------------------------------------------
    # Distance information (may be None)
    # ------------------------------------------------------------------
    before_dist = None
    after_dist = None
    try:
        before_dist = state.distance_to_goal()
        after_dist = after.distance_to_goal()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Movement: reward/penalty based on distance change
    # ------------------------------------------------------------------
    if cat == "move":
        # If the map has already been read we trust distance more
        map_known = getattr(state, "map_read", False)
        if map_known:
            # ensure a slightly higher baseline for informed movement
            score = max(score, 0.6)

        if before_dist is not None and after_dist is not None:
            delta = before_dist - after_dist  # positive -> closer
            mult = 1.2 if map_known else 1.0
            score += mult * delta

            # Penalty for staying at the same distance
            if delta == 0:
                score -= 0.4

            # Large bonus for reaching the goal instantly
            if after_dist == 0:
                score += 10.0
        else:
            # When we cannot compute a distance, discourage random wandering
            score -= 0.2

    # ------------------------------------------------------------------
    # Opening doors: encourage only when it helps
    # ------------------------------------------------------------------
    if cat == "open":
        if before_dist is not None and after_dist is not None:
            if after_dist < before_dist:
                # positive reinforcement proportional to improvement
                score += 5.0 + (before_dist - after_dist)
            else:
                # opening a door that does not improve distance is discouraged
                score -= 0.3

    # ------------------------------------------------------------------
    # Reading the map
    # ------------------------------------------------------------------
    if cat == "read":
        if getattr(state, "map_read", False):
            score -= 3.0
        else:
            score += 3.0

    # ------------------------------------------------------------------
    # Taking objects: reward only if inventory actually grows
    # ------------------------------------------------------------------
    if cat == "take":
        try:
            inv_before = len(state.inventory_items)
            inv_after = len(after.inventory_items)
            if inv_after > inv_before:
                score += 2.0
            else:
                score -= 2.0
        except Exception:
            pass

    # ------------------------------------------------------------------
    # No‑op actions (including “task”): small extra penalty if they
    # do not change the visible observation.
    # ------------------------------------------------------------------
    if cat == "noop":
        try:
            before_obs = state.look_text()
            after_obs = after.look_text()
            if before_obs == after_obs:
                score -= 0.5
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tiny random tie‑breaker (reduced magnitude)
    # ------------------------------------------------------------------
    score += random.uniform(-0.01, 0.01)

    return score


# ----------------------------------------------------------------------
# Main expansion routine
# ----------------------------------------------------------------------
def default_expansion(node):
    """
    Expand one untried action from the given node using an improved
    heuristic ranking.

    The routine:
      1. Scores all currently untried actions.
      2. Prefers genuine progress (movement, opening useful doors,
         taking new items, reading the map) and de‑prioritises no‑ops.
      3. After expanding an 'open' action, prunes the opposite 'close'
         from both child **and parent** untried‑action containers.
      4. Falls back to a random choice if scoring fails.
    """
    # --------------------------------------------------------------
    # Gather current untried actions – they may be a set, list or other iterable
    # --------------------------------------------------------------
    try:
        actions = list(node._untried_actions)
    except Exception:
        actions = []

    if not actions:
        # No untried actions – cannot expand; return node itself
        return node

    # --------------------------------------------------------------
    # Prefer progress actions over pure informational ones
    # --------------------------------------------------------------
    progress_actions = [a for a in actions if not _is_noop(a)]
    candidate_actions = progress_actions if progress_actions else actions

    # --------------------------------------------------------------
    # Score each candidate action
    # --------------------------------------------------------------
    scored: List[Tuple[float, str]] = []
    for act in candidate_actions:
        try:
            sc = _score_action(act, node.state)
        except Exception:
            sc = -float("inf")
        scored.append((sc, act))

    # --------------------------------------------------------------
    # Choose the highest‑scoring action (break ties randomly)
    # --------------------------------------------------------------
    if scored:
        max_score = max(scored, key=lambda x: x[0])[0]
        best_actions = [act for (sc, act) in scored if sc == max_score]
        chosen_action = random.choice(best_actions)
    else:
        # Fallback – pick any untried action at random
        chosen_action = node._untried_actions.pop()
        child_state = node.state.clone()
        child_state.apply_action(chosen_action)
        from mcts.node import MCTSNode
        child = MCTSNode(child_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        return child

    # --------------------------------------------------------------
    # Remove the selected action from the parent's untried set
    # --------------------------------------------------------------
    _remove_action_untried(node._untried_actions, chosen_action)

    # --------------------------------------------------------------
    # Create the child node with the chosen action
    # --------------------------------------------------------------
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)

    # --------------------------------------------------------------
    # Door‑toggle guard: prune opposite action from both child and parent
    # --------------------------------------------------------------
    if _action_category(chosen_action) == "open":
        opposite = _opposite_door_action(chosen_action)
        if opposite:
            _remove_action_untried(child._untried_actions, opposite)
            _remove_action_untried(node._untried_actions, opposite)

    # Register child and return
    node.children[chosen_action] = child
    return child
