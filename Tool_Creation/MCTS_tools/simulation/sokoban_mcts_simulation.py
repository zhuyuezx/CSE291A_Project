"""
LLM-generated MCTS tool: simulation
Description: Clean up unused imports and cache distance computation for efficiency.
Generated:   2026-03-06T20:57:39.506685
"""

import random
from typing import List

def default_simulation(state, perspective_player: int, max_depth: int = 200) -> float:
    """
    Heuristic‑guided rollout for Sokoban.

    Improves over the pure random rollout by:
      • Giving higher probability to actions that push a box.
      • Preferring pushes that reduce the total Manhattan distance of boxes
        to their nearest targets.
      • Rejecting moves that create an obvious dead‑lock (box in a non‑target
        corner or against a wall with no target on that wall).
      • Stopping early when a push places a box on a target or yields a clear
        distance improvement.
    """
    # ------------------------------------------------------------------ #
    # Helper utilities (kept local to avoid external dependencies)
    # ------------------------------------------------------------------ #
    DIRS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def _apply(state_obj, action: int):
        """Apply `action` on a clone of `state_obj` and return the new state."""
        new_state = state_obj.clone()
        new_state.apply_action(action)
        return new_state

    def _action_pushes_box(state_obj, action: int) -> bool:
        """True iff `action` moves the player onto a box (i.e. pushes it)."""
        dr, dc = DIRS[action]
        pr, pc = state_obj.player
        nr, nc = pr + dr, pc + dc
        return (nr, nc) in state_obj.boxes

    def _corner_deadlock(s) -> bool:
        """
        Detect simple corner dead‑locks:
          - Box not on a target that is adjacent to two orthogonal walls.
        """
        for b in s.boxes:
            if b in s.targets:
                continue
            r, c = b
            if ((r - 1, c) in s.walls and (r, c - 1) in s.walls) or \
               ((r - 1, c) in s.walls and (r, c + 1) in s.walls) or \
               ((r + 1, c) in s.walls and (r, c - 1) in s.walls) or \
               ((r + 1, c) in s.walls and (r, c + 1) in s.walls):
                return True
        return False

    def _wall_deadlock(s) -> bool:
        """
        Simple wall‑line dead‑lock:
        a box is pushed against a wall (or wall + boxes) where no target
        exists on that whole line in the push direction.
        """
        for br, bc in s.boxes:
            if (br, bc) in s.targets:
                continue
            # up
            if (br - 1, bc) in s.walls:
                if not any((r, bc) in s.targets for r in range(br)):
                    return True
            # down
            if (br + 1, bc) in s.walls:
                if not any((r, bc) in s.targets for r in range(br + 1, s.height)):
                    return True
            # left
            if (br, bc - 1) in s.walls:
                if not any((br, c) in s.targets for c in range(bc)):
                    return True
            # right
            if (br, bc + 1) in s.walls:
                if not any((br, c) in s.targets for c in range(bc + 1, s.width)):
                    return True
        return False

    def _is_deadlocked(s) -> bool:
        """Combined dead‑lock test used during rollouts."""
        return _corner_deadlock(s) or _wall_deadlock(s)

    # ------------------------------------------------------------------ #
    # Begin the guided rollout
    # ------------------------------------------------------------------ #
    sim_state = state.clone()
    depth = 0
    baseline_boxes_on = state.boxes_on_targets()
    baseline_dist = state.total_box_distance()

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break   # safety guard

        # -------------------------------------------------------------- #
        # Score each legal action.
        # -------------------------------------------------------------- #
        scores: List[float] = []
        base_dist = sim_state.total_box_distance()  # cached for this step

        for a in legal:
            trial = _apply(sim_state, a)

            # Discard actions that immediately cause a dead‑lock.
            if _is_deadlocked(trial):
                scores.append(0.0)
                continue

            pushes = _action_pushes_box(sim_state, a)

            after = trial.total_box_distance()
            dist_gain = max(0, base_dist - after)

            # Linear combination: push gives base 1, each unit distance gain adds 0.5
            score = (1.0 if pushes else 0.0) + 0.5 * dist_gain
            scores.append(score)

        # -------------------------------------------------------------- #
        # Choose an action – weighted random if any positive score,
        # otherwise fall back to uniform random.
        # -------------------------------------------------------------- #
        if any(s > 0 for s in scores):
            total = sum(scores)
            pick = random.random() * total
            cumulative = 0.0
            for a, sc in zip(legal, scores):
                cumulative += sc
                if pick <= cumulative:
                    chosen = a
                    break
        else:
            chosen = random.choice(legal)

        sim_state.apply_action(chosen)
        depth += 1

        # -------------------------------------------------------------- #
        # Early termination checks
        # -------------------------------------------------------------- #
        if sim_state.boxes_on_targets() > baseline_boxes_on:
            break

        if sim_state.total_box_distance() < baseline_dist:
            baseline_dist = sim_state.total_box_distance()

        if _is_deadlocked(sim_state):
            break

    # Return the shaped reward from the perspective of `perspective_player`.
    return sim_state.returns()[perspective_player]
