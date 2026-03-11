"""
LLM-generated MCTS tool: selection
Description: No changes needed; existing implementation is correct.
Generated:   2026-03-08T16:19:07.262482
"""

import math
from typing import Optional

def _sokoban_heuristic(state) -> float:
    """
    Light‑weight estimate of state quality for Sokoban.

    Returns a value in roughly [-inf, 1].  A large negative value is
    returned for obvious dead‑locks (box stuck in a non‑target corner).
    Otherwise the score combines:
        * fraction of boxes already on targets   (weight 0.7)
        * normalized distance of remaining boxes to any target (weight 0.3)
    """
    # --- dead‑lock detection (simple corner check) ---
    for (r, c) in state.boxes:
        if (r, c) in state.targets:
            continue          # box already safe
        # orthogonal neighbours
        up = (r - 1, c) in state.walls
        down = (r + 1, c) in state.walls
        left = (r, c - 1) in state.walls
        right = (r, c + 1) in state.walls
        # corner if blocked in two orthogonal directions
        if (up or down) and (left or right):
            return -1e9       # treat as fatal dead‑lock

    # --- progress / distance components ---
    # fraction of boxes on targets
    frac_on_target = state.boxes_on_targets() / state.num_targets

    # total box distance to nearest target (already provided)
    total_dist = state.total_box_distance()

    # Upper bound for total distance: each box could be at farthest corner
    # of the board from any target. Use board diameter as a cheap bound.
    max_single = state.height + state.width
    max_total = max_single * state.num_targets

    dist_score = 1.0 - (total_dist / max_total) if max_total > 0 else 0.0

    # weighted combination (mirrors the reward shape)
    return 0.7 * frac_on_target + 0.3 * dist_score


def default_selection(node, exploration_weight: float = 1.41):
    """
    UCB1 tree walk enriched with a Sokoban‑specific heuristic prior.

    The exploitation term is blended with the heuristic so that
    unvisited or lightly visited children are guided by domain knowledge.
    Children that are already dead‑locked receive a huge negative estimate
    and are effectively pruned.

    Args:
        node:               Root MCTSNode to start selection from.
        exploration_weight:  UCB1 exploration constant C.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    # hyper‑parameter controlling how strongly the heuristic influences early picks
    heuristic_alpha = 2.0   # can be tuned; higher = more bias early on

    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion phase

        # Pre‑compute log(parent_visits) once (add 1 to avoid log(0))
        log_parent = math.log(node.visits + 1)

        best_child: Optional[object] = None
        best_score = -float('inf')

        for child in node.children.values():
            visits = child.visits
            # Compute heuristic for the child's state
            h = _sokoban_heuristic(child.state)

            # If the child is a dead‑lock, give it the worst possible score
            if h <= -1e8:
                score = -float('inf')
            else:
                if visits == 0:
                    # First time we see this child: use heuristic as exploitation,
                    # and a maximal exploration term.
                    exploit = h
                    explore = exploration_weight * math.sqrt(log_parent)
                else:
                    # Blend observed value with heuristic prior.
                    exploit = (child.value + heuristic_alpha * h) / (visits + heuristic_alpha)
                    explore = exploration_weight * math.sqrt(log_parent / visits)

                score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        # Defensive fallback (should never happen)
        if best_child is None:
            return node

        node = best_child

    return node
