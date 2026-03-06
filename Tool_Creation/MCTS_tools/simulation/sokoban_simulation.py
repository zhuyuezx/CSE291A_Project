"""
LLM-generated MCTS tool: simulation
Description: No changes required; the function is correct and efficient.
Generated:   2026-03-06T03:38:48.611924
"""

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Heuristic‑guided rollout for Sokoban with targeted improvements:
      • Refined wall‑stuck detection to avoid false negatives.
      • Small bonus for actions that position the player next to a pushable box.
      • Stronger weighting of global distance reduction (no negative clamp).
      • Reduced dead‑lock penalty and lower ε‑greedy randomness.
    """
    # ----------------------------------------------------------------------
    # Imports (local to keep the function self‑contained)
    # ----------------------------------------------------------------------
    import random
    from typing import List, Tuple

    # ----------------------------------------------------------------------
    # Tunable parameters (incrementally tuned)
    # ----------------------------------------------------------------------
    EPSILON = 0.10               # ε‑greedy random move probability (more exploitation)
    TOP_K = 4                    # actions kept for exploitation
    PROGRESS_WINDOW = 120       # steps without improvement before stop
    PUSH_BONUS = 2.0             # base bonus for any push
    NEUTRAL_PUSH = 0.3           # constant bonus for performing a push
    SETUP_BONUS = 1.0            # reward for ending up next to a pushable box
    BOX_IMP_WEIGHT = 3.0         # weight for raw per‑box distance change
    DIR_WEIGHT = 2.0             # extra weight for positive per‑box progress
    DIST_WEIGHT = 12.0           # weight for global distance reduction (stronger)
    LOOKAHEAD_WEIGHT = 1.5       # weight for depth‑2 look‑ahead improvement
    WALL_STUCK_PENALTY = 0.8     # milder penalty for creating a frozen box
    DEADLOCK_PENALTY = -1e6      # huge negative value to discard dead‑locked states

    # ----------------------------------------------------------------------
    # Helper functions – local for speed and self‑containment
    # ----------------------------------------------------------------------
    def _action_is_push(prev_state, nxt_state) -> bool:
        """True if the action moved at least one box."""
        return prev_state.boxes != nxt_state.boxes

    def _min_dist_to_target(pos: Tuple[int, int], targets) -> int:
        """Manhattan distance from pos to its nearest target."""
        r, c = pos
        return min(abs(r - tr) + abs(c - tc) for tr, tc in targets)

    def _box_frozen_against_wall(box: Tuple[int, int], st) -> bool:
        """
        Detect a box that is pressed against a wall (or another box) without any
        reachable target along that wall direction. This is a stricter version
        of the previous wall‑stuck test to avoid false penalties.
        """
        r, c = box
        # left side
        if (r, c - 1) in st.walls:
            if not any(t[0] == r and t[1] < c for t in st.targets):
                if (r, c + 1) in st.walls or (r, c + 1) in st.boxes:
                    return True
        # right side
        if (r, c + 1) in st.walls:
            if not any(t[0] == r and t[1] > c for t in st.targets):
                if (r, c - 1) in st.walls or (r, c - 1) in st.boxes:
                    return True
        # upper side
        if (r - 1, c) in st.walls:
            if not any(t[1] == c and t[0] < r for t in st.targets):
                if (r + 1, c) in st.walls or (r + 1, c) in st.boxes:
                    return True
        # lower side
        if (r + 1, c) in st.walls:
            if not any(t[1] == c and t[0] > r for t in st.targets):
                if (r - 1, c) in st.walls or (r - 1, c) in st.boxes:
                    return True
        return False

    def _is_deadlock(st) -> bool:
        """Extended dead‑lock detection (corner + refined freeze)."""
        for b in st.boxes:
            if b in st.targets:
                continue

            r, c = b
            up = (r - 1, c) in st.walls or (r - 1, c) in st.boxes
            down = (r + 1, c) in st.walls or (r + 1, c) in st.boxes
            left = (r, c - 1) in st.walls or (r, c - 1) in st.boxes
            right = (r, c + 1) in st.walls or (r, c + 1) in st.boxes

            # corner dead‑lock
            if (up and left) or (up and right) or (down and left) or (down and right):
                return True

            # refined freeze dead‑lock
            if _box_frozen_against_wall(b, st):
                return True
        return False

    def _depth_two_lookahead(push_state) -> float:
        """
        Cheap depth‑2 search: from push_state explore up to two actions
        (any actions, not just pushes) and return the maximal distance
        reduction achievable.
        """
        best_imp = 0.0
        base_dist = push_state.total_box_distance()
        for a1 in push_state.legal_actions():
            s1 = push_state.clone()
            s1.apply_action(a1)
            if _is_deadlock(s1):
                continue
            imp1 = base_dist - s1.total_box_distance()
            if imp1 > best_imp:
                best_imp = imp1
            for a2 in s1.legal_actions():
                s2 = s1.clone()
                s2.apply_action(a2)
                if _is_deadlock(s2):
                    continue
                imp2 = base_dist - s2.total_box_distance()
                if imp2 > best_imp:
                    best_imp = imp2
        return best_imp

    def _player_next_to_pushable_box(st) -> bool:
        """
        Returns True if the player stands adjacent to a box that can be pushed
        (i.e., the cell on the opposite side of the box is empty floor or a target).
        """
        pr, pc = st.player
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            box_pos = (pr + dr, pc + dc)
            if box_pos in st.boxes:
                beyond = (box_pos[0] + dr, box_pos[1] + dc)
                if (beyond not in st.walls) and (beyond not in st.boxes):
                    return True
        return False

    # ----------------------------------------------------------------------
    # Simulation loop
    # ----------------------------------------------------------------------
    sim_state = state.clone()
    depth = 0
    last_distance = sim_state.total_box_distance()
    no_progress = 0

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        scored_actions: List[Tuple[int, float]] = []   # (action, score)

        for a in legal:
            nxt = sim_state.clone()
            nxt.apply_action(a)

            # ------------------------------------------------------------------
            # Dead‑lock handling
            # ------------------------------------------------------------------
            if _is_deadlock(nxt):
                scored_actions.append((a, DEADLOCK_PENALTY))
                continue

            new_dist = nxt.total_box_distance()
            # Global distance improvement (no lower clamp)
            distance_score = (last_distance - new_dist) * DIST_WEIGHT

            # ------------------------------------------------------------------
            # Push detection and per‑box improvement
            # ------------------------------------------------------------------
            push = _action_is_push(sim_state, nxt)
            push_bonus = 0.0
            per_box_imp = 0.0          # signed improvement
            dir_imp = 0.0              # positive‑only improvement
            setup_bonus = 0.0

            if push:
                # Identify the moved box (exactly one box moves per push)
                old_pos_set = sim_state.boxes - nxt.boxes
                new_pos_set = nxt.boxes - sim_state.boxes
                if old_pos_set and new_pos_set:
                    old_pos = next(iter(old_pos_set))
                    new_pos = next(iter(new_pos_set))
                    old_dist = _min_dist_to_target(old_pos, sim_state.targets)
                    new_dist_box = _min_dist_to_target(new_pos, nxt.targets)
                    per_box_imp = old_dist - new_dist_box
                    dir_imp = max(0.0, per_box_imp)

                # Base + neutral push bonus (does not depend on per‑box gain)
                push_bonus = PUSH_BONUS + NEUTRAL_PUSH

                # Wall‑stuck penalty – milder and based on refined detection
                if any(b not in nxt.targets and _box_frozen_against_wall(b, nxt)
                       for b in nxt.boxes):
                    push_bonus -= WALL_STUCK_PENALTY

                # Depth‑2 look‑ahead for pushes
                lookahead_imp = _depth_two_lookahead(nxt)
                distance_score += LOOKAHEAD_WEIGHT * lookahead_imp

            # Reward for ending up next to a pushable box (even if this step wasn't a push)
            if _player_next_to_pushable_box(nxt):
                setup_bonus = SETUP_BONUS

            # ------------------------------------------------------------------
            # Combine scores
            # ------------------------------------------------------------------
            const = 0.01
            box_imp_score = per_box_imp * BOX_IMP_WEIGHT          # may be negative
            dir_score = dir_imp * DIR_WEIGHT                      # always non‑negative
            total_score = (distance_score + box_imp_score + dir_score +
                           push_bonus + setup_bonus + const)
            scored_actions.append((a, total_score))

        # ----------------------------------------------------------------------
        # ε‑greedy selection with enlarged top‑K exploitation
        # ----------------------------------------------------------------------
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        top_actions = [act for act, _ in scored_actions[:TOP_K]]

        if random.random() < (1.0 - EPSILON):
            chosen = random.choice(top_actions)
        else:
            chosen = random.choice(legal)

        # Apply chosen action
        sim_state.apply_action(chosen)
        depth += 1

        # ----------------------------------------------------------------------
        # Progress monitoring & early termination
        # ----------------------------------------------------------------------
        cur_dist = sim_state.total_box_distance()
        if cur_dist < last_distance:
            no_progress = 0
            last_distance = cur_dist
        else:
            no_progress += 1

        if _is_deadlock(sim_state):
            return 0.0

        if no_progress >= PROGRESS_WINDOW:
            break

    # Return the shaped reward from the final rollout state
    return sim_state.returns()[perspective_player]
