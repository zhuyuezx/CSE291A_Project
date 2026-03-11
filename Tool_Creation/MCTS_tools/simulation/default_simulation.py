"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function is correct and efficient.
Generated:   2026-03-11T08:45:59.377710
"""

def default_simulation(state, perspective_player: int, max_depth: int = 50) -> float:
    """
    Heuristic‑enhanced rollout simulation for Sokoban with additional
    loop‑avoidance and stronger push incentives.

    Improvements over the previous version:
      • Detects and penalises revisiting a board state within the same rollout
        (breaks idle ↑/↓ loops).
      • Gives a flat bonus for *any* box movement, not only when the total
        Manhattan distance to targets decreases.
      • Rewards decreasing the distance to the nearest push‑able box.
      • Increases the idle‑step penalty and re‑balances push‑related weights.
      • Keeps dead‑lock avoidance and other useful heuristics.
    """
    import random
    import math

    # ------------------------------------------------------------------
    # Helper: Detect corner dead‑locks (four orthogonal wall combos)
    # ------------------------------------------------------------------
    def _corner_deadlock(sim_state):
        walls = sim_state.walls
        targets = sim_state.targets
        for bx, by in sim_state.boxes:
            if (bx, by) in targets:
                continue
            if ((bx - 1, by) in walls and (bx, by - 1) in walls):
                return True
            if ((bx + 1, by) in walls and (bx, by - 1) in walls):
                return True
            if ((bx - 1, by) in walls and (bx, by + 1) in walls):
                return True
            if ((bx + 1, by) in walls and (bx, by + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: Detect simple wall‑deadlocks (box against a wall with no
    # target on that wall line). Returns True if any box is in such a
    # situation.
    # ------------------------------------------------------------------
    def _wall_deadlock(sim_state):
        walls = sim_state.walls
        targets = sim_state.targets
        for bx, by in sim_state.boxes:
            if (bx, by) in targets:
                continue

            # left wall
            if (bx, by - 1) in walls:
                if not any((bx, t) in targets for t in range(by - 1)):
                    return True
            # right wall
            if (bx, by + 1) in walls:
                if not any((bx, t) in targets for t in range(by + 2, sim_state.width)):
                    return True
            # upper wall
            if (bx - 1, by) in walls:
                if not any((t, by) in targets for t in range(bx - 1)):
                    return True
            # lower wall
            if (bx + 1, by) in walls:
                if not any((t, by) in targets for t in range(bx + 2, sim_state.height)):
                    return True
        return False

    # ------------------------------------------------------------------
    # Combined dead‑lock test used during rollouts
    # ------------------------------------------------------------------
    def _any_deadlock(sim_state):
        return _corner_deadlock(sim_state) or _wall_deadlock(sim_state)

    # ------------------------------------------------------------------
    # Helper: Manhattan distance from player to the nearest box
    # ------------------------------------------------------------------
    def _player_to_nearest_box(sim_state):
        if not sim_state.boxes:
            return 0
        px, py = sim_state.player
        return min(abs(px - bx) + abs(py - by) for bx, by in sim_state.boxes)

    # ------------------------------------------------------------------
    # Helper: Does the player stand adjacent to a box that can be pushed?
    # ------------------------------------------------------------------
    def _has_adjacent_pushable(sim_state):
        px, py = sim_state.player
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            bx, by = px + dx, py + dy
            if (bx, by) in sim_state.boxes:
                tx, ty = bx + dx, by + dy
                if (tx, ty) not in sim_state.walls and (tx, ty) not in sim_state.boxes:
                    return True
        return False

    # ------------------------------------------------------------------
    # Helper: List of boxes that can currently be pushed (has free cell
    # behind them in at least one direction)
    # ------------------------------------------------------------------
    def _pushable_boxes(sim_state):
        result = []
        for bx, by in sim_state.boxes:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = bx + dx, by + dy          # player would stand here
                tx, ty = bx + 2 * dx, by + 2 * dy  # cell beyond the box
                if (nx, ny) not in sim_state.walls and (nx, ny) not in sim_state.boxes \
                   and (tx, ty) not in sim_state.walls and (tx, ty) not in sim_state.boxes:
                    result.append((bx, by))
                    break
        return result

    # ------------------------------------------------------------------
    # Helper: Distance from player to the closest pushable box.
    # Returns a large number if none are pushable.
    # ------------------------------------------------------------------
    def _dist_to_nearest_pushable(sim_state):
        pushables = _pushable_boxes(sim_state)
        if not pushables:
            return 1_000_000_000
        px, py = sim_state.player
        return min(abs(px - bx) + abs(py - by) for bx, by in pushables)

    # ------------------------------------------------------------------
    # Helper: Count boxes that are orthogonally adjacent to any target.
    # This is a simple “push‑potential” metric.
    # ------------------------------------------------------------------
    def _push_potential(sim_state):
        cnt = 0
        for bx, by in sim_state.boxes:
            for tx, ty in sim_state.targets:
                if abs(bx - tx) + abs(by - ty) == 1:
                    cnt += 1
                    break
        return cnt

    # ------------------------------------------------------------------
    # Helper: Sample an action using a temperature‑scaled soft‑max over a
    # richer scoring function that includes loop‑avoidance.
    # ------------------------------------------------------------------
    def _sample_weighted_action(sim_state, legal_actions, visited_keys):
        # -------------------------- Tunable constants --------------------------
        PUSH_BASE = 15.0               # base reward for a genuine push
        PUSH_DIST_WEIGHT = 4.0         # extra per unit of distance reduction during a push
        PUSH_MOVED_BONUS = 5.0         # flat bonus for *any* box movement
        PRE_PUSH_BONUS = 8.0           # incentive for ending the step adjacent to a pushable box
        TARGET_ADJ_BONUS = 4.0         # per box now orthogonal to a target (push potential)
        DIST_REWARD_WEIGHT = 8.0       # reward per unit of distance reduction (any action)
        DIST_PENALTY_WEIGHT = -3.0     # penalty per unit of distance increase
        PLAYER_PROX_WEIGHT = 1.2       # reward for getting closer to the nearest box
        MOVE_AWAY_PENALTY = -1.0       # penalty for moving farther away
        PUSHABLE_DIST_REWARD = 2.0     # reward for reducing distance to nearest pushable box
        DEADLOCK_PENALTY = -5000.0      # huge negative to discourage dead‑locks
        IDLE_PENALTY = -12.0           # stronger flat penalty for idle steps
        LOOP_PENALTY = -30.0           # penalty for revisiting a state on the current rollout
        TEMP = 1.0                     # temperature for soft‑max (lower ⇒ more deterministic)

        base_dist = sim_state.total_box_distance()
        base_player_dist = _player_to_nearest_box(sim_state)
        base_pushable_dist = _dist_to_nearest_pushable(sim_state)

        scores = []
        for a in legal_actions:
            trial = sim_state.clone()
            trial.apply_action(a)

            # Immediate dead‑lock check – heavily penalised
            if _any_deadlock(trial):
                scores.append(DEADLOCK_PENALTY)
                continue

            # Loop detection (state repeated on current rollout path)
            trial_key = trial.state_key()
            if trial_key in visited_keys:
                scores.append(LOOP_PENALTY)
                continue

            new_dist = trial.total_box_distance()
            moved_box = trial.boxes != sim_state.boxes

            # ---- Push‑related bonus (distance reduction) ----
            dist_red = max(base_dist - new_dist, 0)
            push_bonus = PUSH_BASE + PUSH_DIST_WEIGHT * dist_red

            # ---- Bonus for *any* box movement (even without distance gain) ----
            move_bonus = PUSH_MOVED_BONUS if moved_box else 0.0

            # ---- Pre‑push positioning bonus ----
            pre_push_bonus = PRE_PUSH_BONUS if _has_adjacent_pushable(trial) else 0.0

            # ---- Push‑potential bonus (boxes adjacent to targets) ----
            potential_bonus = TARGET_ADJ_BONUS * _push_potential(trial)

            # ---- Distance change component (any action) ----
            if new_dist < base_dist:
                dist_score = DIST_REWARD_WEIGHT * (base_dist - new_dist)
            elif new_dist > base_dist:
                dist_score = DIST_PENALTY_WEIGHT * (new_dist - base_dist)
            else:
                dist_score = 0.0

            # ---- Player proximity component ----
            new_player_dist = _player_to_nearest_box(trial)
            player_delta = base_player_dist - new_player_dist
            if player_delta > 0:
                proximity_score = PLAYER_PROX_WEIGHT * player_delta
            else:
                proximity_score = MOVE_AWAY_PENALTY * (-player_delta)

            # ---- Push‑able distance reduction component ----
            new_pushable_dist = _dist_to_nearest_pushable(trial)
            pushable_score = PUSHABLE_DIST_REWARD * max(base_pushable_dist - new_pushable_dist, 0)

            # ---- Idle penalty (no box movement && no distance improvement) ----
            idle_pen = IDLE_PENALTY if not moved_box and new_dist >= base_dist else 0.0

            total_score = (
                push_bonus
                + move_bonus
                + pre_push_bonus
                + potential_bonus
                + dist_score
                + proximity_score
                + pushable_score
                + idle_pen
            )
            scores.append(total_score)

        # Soft‑max selection (numerically stable)
        max_score = max(scores)
        exps = [math.exp((s - max_score) / TEMP) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]

        return random.choices(legal_actions, weights=probs, k=1)[0]

    # ------------------------------------------------------------------
    # Heuristic rollout parameters (tuned)
    # ------------------------------------------------------------------
    BIG_TARGET_BONUS = 0.90          # reward per newly placed box on a target
    DIST_REDUCTION_BONUS = 0.20      # reward per unit of distance reduction
    NO_PROGRESS_LIMIT_BASE = 3      # tighter abort for wandering loops
    MAX_GAMMA = 0.10                 # maximum adaptive discount factor
    PROGRESS_CAP = 1.0               # upper bound for accumulated progress bonus

    sim_state = state.clone()
    progress_bonus = 0.0
    depth = 0
    no_progress_counter = 0
    visited_keys = {sim_state.state_key()}   # keep track of states on this rollout

    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break  # dead‑end

        # Choose an action with the richer weighted policy, passing visited set
        action = _sample_weighted_action(sim_state, legal, visited_keys)

        # Record pre‑action metrics
        before_targets = sim_state.boxes_on_targets()
        before_dist = sim_state.total_box_distance()
        before_player_dist = _player_to_nearest_box(sim_state)
        before_boxes = set(sim_state.boxes)

        # Apply the chosen action
        sim_state.apply_action(action)

        # Record post‑action metrics
        after_targets = sim_state.boxes_on_targets()
        after_dist = sim_state.total_box_distance()
        after_player_dist = _player_to_nearest_box(sim_state)
        after_boxes = set(sim_state.boxes)

        # ------------------------------------------------------------------
        # Accumulate progress‑related bonus
        # ------------------------------------------------------------------
        if after_targets > before_targets:
            progress_bonus += BIG_TARGET_BONUS * (after_targets - before_targets)

        if after_dist < before_dist:
            progress_bonus += DIST_REDUCTION_BONUS * (before_dist - after_dist)

        if after_player_dist < before_player_dist:
            progress_bonus += 0.01 * (before_player_dist - after_player_dist)

        # ------------------------------------------------------------------
        # Early‑abort progress tracking: count idle / non‑progress steps
        # ------------------------------------------------------------------
        moved_this_step = after_boxes != before_boxes
        if not moved_this_step and after_targets == before_targets and after_dist >= before_dist:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        dynamic_limit = NO_PROGRESS_LIMIT_BASE + int(0.02 * depth)
        if no_progress_counter >= dynamic_limit:
            break

        # Abort immediately if a dead‑lock is reached
        if _any_deadlock(sim_state):
            break

        # Record the new state key for loop detection in subsequent steps
        visited_keys.add(sim_state.state_key())

        depth += 1

    # ------------------------------------------------------------------
    # Normalise / cap progress bonus
    # ------------------------------------------------------------------
    if progress_bonus > PROGRESS_CAP:
        progress_bonus = PROGRESS_CAP
    progress_bonus *= (1.0 - depth / max_depth)

    # Terminal shaped reward (already in [0,1])
    final_shaped = sim_state.returns()[perspective_player]

    # Adaptive discount (more aggressive for deeper rollouts)
    adaptive_gamma = MAX_GAMMA * (1.0 - depth / max_depth) if depth > 0 else MAX_GAMMA

    return progress_bonus + (1.0 - adaptive_gamma) * final_shaped
