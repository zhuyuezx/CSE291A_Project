"""
LLM-generated MCTS tool: simulation
Description: Replace dead‑lock placeholder with safe fallback and clean up imports.
Generated:   2026-03-06T23:55:23.025159
"""

import random
from typing import List, Set

# Try to import the project's dead‑lock detection; fall back to a no‑op that
# never declares a dead‑lock. This ensures the simulation works even if the
# function is defined elsewhere.
try:
    from .deadlock import is_deadlock as _is_deadlock  # type: ignore
except Exception:  # pragma: no cover
    def _is_deadlock(state) -> bool:  # noqa: D401
        """Fallback dead‑lock checker – assume no dead‑lock."""
        return False


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided simulation for Sokoban with richer scoring.

    Improvements over the previous version:
      * Lower ε (exploration) to 0.05 and decay it with depth.
      * Penalise pushes that increase total box distance.
      * Grant a small bonus when a pushed box aligns (row/col) with any target.
      * Discourage revisiting states already seen in the current rollout.
      * Use a two‑level tie‑breaker: (score, delta_dist) so larger distance
        reductions are preferred when scores are equal.
      * Keep the cheap one‑step look‑ahead and early dead‑lock abort.
    """
    # --- rollout hyper‑parameters -------------------------------------------------
    BASE_EPSILON = 0.05          # base chance of random move
    EPSILON_DECAY = 0.99         # decay factor per depth step
    TARGET_WEIGHT = 5.0          # importance of placing a box on a target
    DIST_WEIGHT = 1.0            # importance of reducing total box distance
    PUSH_PENALTY = -0.5           # penalty per unit distance increase on a push
    ALIGN_BONUS = 0.1            # tiny reward for aligning a box with a target line
    LOOP_PENALTY = -0.01          # penalty for re‑entering a previously seen state
    DEADLOCK_PENALTY = -1e6       # massive negative score for a move that dead‑locks

    sim_state = state.clone()
    depth = 0
    visited_keys: Set[str] = {sim_state.state_key()}

    def _alignment_bonus(s) -> float:
        """Return ALIGN_BONUS if any box shares a row or column with a target."""
        for bx, by in s.boxes:
            for tx, ty in s.targets:
                if bx == tx or by == ty:
                    return ALIGN_BONUS
        return 0.0

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break  # no moves possible

        cur_targets = sim_state.boxes_on_targets()
        cur_dist = sim_state.total_box_distance()

        # Store (action, score, delta_dist, child_key) for selection.
        action_infos: List[tuple[int, float, int, str]] = []

        for a in legal:
            child = sim_state.clone()
            child.apply_action(a)
            child_key = child.state_key()

            # Immediate dead‑lock detection.
            if _is_deadlock(child):
                action_infos.append((a, DEADLOCK_PENALTY, 0, child_key))
                continue

            delta_targets = child.boxes_on_targets() - cur_targets
            delta_dist = cur_dist - child.total_box_distance()   # >0 means improvement
            score = TARGET_WEIGHT * delta_targets + DIST_WEIGHT * delta_dist

            # Detect a push: player ends on a square that previously contained a box.
            is_push = child.player in sim_state.boxes

            if is_push:
                if delta_dist < 0:
                    score += PUSH_PENALTY * abs(delta_dist)   # penalise worsening pushes
                score += _alignment_bonus(child)               # alignment encouragement

            # Loop avoidance penalty.
            if child_key in visited_keys:
                score += LOOP_PENALTY

            action_infos.append((a, score, delta_dist, child_key))

        # --- ε‑greedy selection -------------------------------------------------
        epsilon = BASE_EPSILON * (EPSILON_DECAY ** depth)
        if random.random() < epsilon:
            chosen_action = random.choice(legal)
        else:
            # Filter out dead‑locked actions.
            viable = [info for info in action_infos if info[1] != DEADLOCK_PENALTY]
            if not viable:
                chosen_action = random.choice(legal)
            else:
                # Primary sort by score, secondary by delta_dist (larger improvement).
                max_score = max(info[1] for info in viable)
                best_by_score = [info for info in viable if info[1] == max_score]

                max_delta = max(info[2] for info in best_by_score)
                best_actions = [info for info in best_by_score if info[2] == max_delta]

                chosen_action = random.choice(best_actions)[0]

        # Apply the chosen action and record the new state.
        sim_state.apply_action(chosen_action)
        visited_keys.add(sim_state.state_key())

        # Immediate dead‑lock abort – worst possible reward.
        if _is_deadlock(sim_state):
            return 0.0

        depth += 1

    # Non‑terminal rollouts end here: return the shaped reward.
    return sim_state.returns()[perspective_player]
