"""
LLM-generated MCTS tool: simulation
Description: Fixed undefined helpers `_push_bonus` and `_player_to_nearest_pushable` by implementing lightweight in‑function equivalents; adjusted their usage accordingly while preserving the original heuristic logic.
Generated:   2026-03-06T20:56:48.982337
"""

import math
import random

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided rollout simulation for Sokoban with stronger push bias,
    explicit target‑placement reward, tighter stagnation handling and an
    extended dead‑lock detector.

    Fixed:
        • Defined the missing helper functions `_is_push` and
          `_player_to_nearest_pushable` inside the simulation.
        • Replaced the now‑undefined `_push_bonus` calls with the new helpers.
        • Kept the original scoring and early‑termination strategy unchanged.
    """
    # ------------------------------------------------------------------ #
    #  Helper utilities (local to the simulation)                         #
    # ------------------------------------------------------------------ #
    def _is_push(s, a):
        """
        Returns True if action `a` moves a box (i.e., a push), False otherwise.
        Detects a push by checking whether the set of box positions changes
        after the action is applied.
        """
        tmp = s.clone()
        tmp.apply_action(a)
        return tmp.boxes != s.boxes

    def _player_to_nearest_pushable(s):
        """
        Approximate distance from the player to the closest *pushable* box.
        For speed we treat every box as pushable and return the Manhattan
        distance to the nearest one. If no boxes exist, a large constant is
        returned.
        """
        if not s.boxes:
            return 0  # no boxes → distance irrelevant
        pr, pc = s.player
        # Manhattan distance to each box
        distances = [abs(pr - br) + abs(pc - bc) for (br, bc) in s.boxes]
        return min(distances) if distances else 0

    # ------------------------------------------------------------------ #
    #  Simulation preparation                                             #
    # ------------------------------------------------------------------ #
    sim_state = state.clone()

    # ---------- adaptive depth ----------
    placed = sim_state.boxes_on_targets()
    remaining = sim_state.num_targets - placed
    adaptive_limit = 4 * placed + 6 * remaining + 20
    max_steps = min(max_depth, adaptive_limit)

    # ---------- dead‑lock detector ----------
    def _deadlock_extended(s):
        """
        Detect simple dead‑locks:
          * classic corner (wall/box on two orthogonal sides) not on a target
          * box against a wall where no target lies on that wall line
          * two boxes forming a frozen corner against a wall
        """
        for b in s.boxes:
            if b in s.targets:
                continue
            r, c = b

            # corner with walls/boxes
            up = (r - 1, c) in s.walls or (r - 1, c) in s.boxes
            down = (r + 1, c) in s.walls or (r + 1, c) in s.boxes
            left = (r, c - 1) in s.walls or (r, c - 1) in s.boxes
            right = (r, c + 1) in s.walls or (r, c + 1) in s.boxes

            if (up and left) or (up and right) or (down and left) or (down and right):
                return True

            # wall‑line dead‑lock (no target on that line)
            if left and not any(t[0] == r and t[1] < c for t in s.targets):
                return True
            if right and not any(t[0] == r and t[1] > c for t in s.targets):
                return True
            if up and not any(t[1] == c and t[0] < r for t in s.targets):
                return True
            if down and not any(t[1] == c and t[0] > r for t in s.targets):
                return True
        return False

    # ---------- rollout state ----------
    visited_keys = set()
    depth = 0
    stalled_counter = 0
    STALL_LIMIT = 8                     # tighter than before
    prev_distance = sim_state.total_box_distance()
    prev_boxes_on_targets = sim_state.boxes_on_targets()
    nonpush_streak = 0
    MAX_NONPUSH_STREAK = 3

    # ------------------------------------------------------------------ #
    #  Main rollout loop                                                 #
    # ------------------------------------------------------------------ #
    while not sim_state.is_terminal() and depth < max_steps:
        # loop detection – abort if we revisit the same rollout state
        key = sim_state.state_key()
        if key in visited_keys:
            stalled_counter = STALL_LIMIT   # force early termination
        else:
            visited_keys.add(key)

        legal = sim_state.legal_actions()
        if not legal:
            break

        best_actions = []
        best_score = math.inf   # lower score = more attractive

        for a in legal:
            # simulate the action
            tmp_state = sim_state.clone()
            tmp_state.apply_action(a)

            new_dist = tmp_state.total_box_distance()
            delta = new_dist - prev_distance                    # negative → improvement
            new_boxes = tmp_state.boxes_on_targets()
            target_gain = new_boxes - prev_boxes_on_targets      # >0 if a box reaches a target

            # push detection for this candidate action
            push = _is_push(sim_state, a)

            # push incentive (stronger)
            push_bonus = -0.30
            if target_gain > 0:
                push_bonus -= 0.5 * target_gain                # reward for hitting a target
            if delta < 0:
                push_bonus += -0.05 * (-delta)                 # extra for distance reduction
            # (optional) add a tiny bonus for actually pushing a box
            if push:
                push_bonus -= 0.02

            # explicit target‑placement term (makes score lower for progress)
            target_gain_term = -0.80 * target_gain

            # distance of player to nearest pushable box (cheap tie‑breaker)
            player_dist = _player_to_nearest_pushable(tmp_state)

            # composite score
            score = delta + push_bonus + target_gain_term + 0.01 * player_dist

            # keep best scoring actions
            if score < best_score - 1e-9:
                best_score = score
                best_actions = [a]
            elif abs(score - best_score) < 1e-9:
                best_actions.append(a)

        # ε‑greedy selection among the best actions
        if best_score < 0:                     # any improving (or push‑biased) move
            chosen = random.choice(best_actions)
        else:
            if random.random() < 0.2:
                chosen = random.choice(legal)
            else:
                chosen = random.choice(best_actions)

        # -------------------------------------------------
        # early cut‑off if we are stuck in a non‑push streak
        if not _is_push(sim_state, chosen):
            nonpush_streak += 1
        else:
            nonpush_streak = 0
        if nonpush_streak >= MAX_NONPUSH_STREAK:
            return sim_state.returns()[perspective_player]
        # -------------------------------------------------

        # Apply the chosen action
        sim_state.apply_action(chosen)
        depth += 1

        # early dead‑lock detection
        if _deadlock_extended(sim_state):
            return 0.0

        # progress tracking for early termination
        cur_distance = sim_state.total_box_distance()
        cur_boxes = sim_state.boxes_on_targets()

        if cur_distance < prev_distance or cur_boxes > prev_boxes_on_targets:
            stalled_counter = 0
        else:
            stalled_counter += 1

        prev_distance = cur_distance
        prev_boxes_on_targets = cur_boxes

        if stalled_counter >= STALL_LIMIT:
            return sim_state.returns()[perspective_player]

    # Return the shaped reward of the final (possibly non‑terminal) state
    return sim_state.returns()[perspective_player]
