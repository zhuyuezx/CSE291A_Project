"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function meets the intended improvements.
Generated:   2026-03-06T23:44:46.528466
"""

import random
from typing import List, Tuple, FrozenSet

def _is_deadlock(state) -> bool:
    """
    Simple dead‑lock detector for Sokoban.
    Returns True if any box (not on a target) is stuck in a corner
    formed by two orthogonal walls. This is a lightweight check
    that runs in O(number_of_boxes).
    """
    walls: FrozenSet[Tuple[int, int]] = state.walls
    targets: FrozenSet[Tuple[int, int]] = state.targets
    for (r, c) in state.boxes:
        if (r, c) in targets:
            continue  # box already on a target is safe
        # check four corner configurations
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided simulation for Sokoban.

    Improves over a pure random rollout by:
      * biasing action selection towards moves that increase boxes on targets
        or decrease total box distance,
      * penalising actions that create an obvious dead‑lock (corner box),
      * early‑terminating when a dead‑lock is detected,
      * returning the current shaped reward if the depth limit is reached
        without reaching a terminal state.

    Args:
        state:              GameState to simulate from (will be cloned).
        perspective_player: Player index whose reward we return.
        max_depth:          Maximum rollout steps (default large, but
                            may be cut short by early termination).

    Returns:
        Float reward from perspective_player's viewpoint.
    """
    # Parameters for the rollout policy
    EPSILON = 0.2               # chance to pick a random action (exploration)
    TARGET_WEIGHT = 5.0         # importance of placing a box on a target
    DIST_WEIGHT = 1.0           # importance of reducing total box distance
    DEADLOCK_PENALTY = -1e6      # huge negative score for a move that dead‑locks

    sim_state = state.clone()
    depth = 0

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break  # no moves possible

        # Score each legal action using a cheap look‑ahead
        scores: List[float] = []
        for a in legal:
            # cheap clone + apply
            child = sim_state.clone()
            child.apply_action(a)

            # dead‑lock check first – give it a massive negative score
            if _is_deadlock(child):
                scores.append(DEADLOCK_PENALTY)
                continue

            # progress measures
            delta_targets = child.boxes_on_targets() - sim_state.boxes_on_targets()
            delta_dist = sim_state.total_box_distance() - child.total_box_distance()

            score = TARGET_WEIGHT * delta_targets + DIST_WEIGHT * delta_dist
            scores.append(score)

        # Choose action: ε‑greedy over the scored distribution
        if random.random() < EPSILON:
            chosen_action = random.choice(legal)
        else:
            max_score = max(scores)
            best_actions = [a for a, s in zip(legal, scores) if s == max_score]
            chosen_action = random.choice(best_actions)

        # Apply the selected action to the real rollout state
        sim_state.apply_action(chosen_action)

        # Immediate dead‑lock abort – return 0 reward (worst possible)
        if _is_deadlock(sim_state):
            return 0.0

        depth += 1

    # If we stopped because of depth limit (non‑terminal), use the shaped reward.
    # This gives the tree a meaningful signal instead of a blind random value.
    return sim_state.returns()[perspective_player]
