"""
LLM-generated MCTS tool: simulation
Description: Added stronger dead‑lock checks, re‑balanced heuristic, adaptive rollout depth, lower ε‑greedy, push‑bonus scaled by distance reduction and cumulative shaped rewards to create a more discriminative rollout signal.
Generated:   2026-03-06T00:51:05.993415
"""

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Refined rollout for Sokoban.

    Main improvements:
      • Extended dead‑lock detection (corner, wall‑line, 2×2 clusters,
        frozen corridor, simple dead‑square detection).
      • Adaptive rollout horizon (longer when more boxes are already placed).
      • Lower ε (0.15) with *improving* push bias – random pushes are only
        taken if they reduce Manhattan distance to a target.
      • Re‑balanced 6‑term heuristic:
            0.45 – boxes on targets
            0.20 – normalised total Manhattan distance
            0.15 – fraction of boxes that have at least one reachable improving push
            0.10 – alignment (clear row/col with a free target)
            0.05 – player mobility (reachable area)
            0.05 – proximity to the nearest improving push cell
      • Scaled push‑bonus proportional to the distance reduction achieved.
      • Cumulative shaped reward (state.returns()) is summed during the rollout;
        the final reward is the maximum of this average and the best heuristic
        observed, ensuring early progress is reflected.
    """

    from collections import deque
    import random
    import math

    # ------------------------------------------------------------------
    # Helper 1 – richer dead‑lock detection
    # ------------------------------------------------------------------
    def is_deadlocked(st) -> bool:
        """Detect common dead‑locks plus simple frozen‑square patterns."""
        # pre‑compute set for speed
        walls = st.walls
        boxes = st.boxes
        targets = st.targets

        # 1) classic corner dead‑lock (non‑target)
        for (bx, by) in boxes:
            if (bx, by) in targets:
                continue
            up = (bx - 1, by) in walls
            down = (bx + 1, by) in walls
            left = (bx, by - 1) in walls
            right = (bx, by + 1) in walls
            if (up and left) or (up and right) or (down and left) or (down and right):
                return True

            # 2) wall‑line dead‑lock (box stuck against wall with no target beyond)
            if up and not any(tx == bx and ty > by for (tx, ty) in targets):
                if ((bx + 1, by) in walls or (bx + 1, by) in boxes) and \
                   ((bx - 1, by) in walls or (bx - 1, by) in boxes):
                    return True
            if down and not any(tx == bx and ty < by for (tx, ty) in targets):
                if ((bx + 1, by) in walls or (bx + 1, by) in boxes) and \
                   ((bx - 1, by) in walls or (bx - 1, by) in boxes):
                    return True
            if left and not any(ty == by and tx > bx for (tx, ty) in targets):
                if ((bx, by + 1) in walls or (bx, by + 1) in boxes) and \
                   ((bx, by - 1) in walls or (bx, by - 1) in boxes):
                    return True
            if right and not any(ty == by and tx < bx for (tx, ty) in targets):
                if ((bx, by + 1) in walls or (bx, by + 1) in boxes) and \
                   ((bx, by - 1) in walls or (bx, by - 1) in boxes):
                    return True

            # 3) box‑against‑wall with another box blocking retreat side
            if up and (bx + 1, by) in boxes and not any(tx == bx and ty > by for (tx, ty) in targets):
                return True
            if down and (bx - 1, by) in boxes and not any(tx == bx and ty < by for (tx, ty) in targets):
                return True
            if left and (bx, by + 1) in boxes and not any(ty == by and tx > bx for (tx, ty) in targets):
                return True
            if right and (bx, by - 1) in boxes and not any(ty == by and tx < bx for (tx, ty) in targets):
                return True

            # 4) 2×2 box cluster without a target inside
            if ((bx + 1, by) in boxes and (bx, by + 1) in boxes and (bx + 1, by + 1) in boxes):
                if not any(p in targets for p in [(bx, by), (bx + 1, by),
                                                 (bx, by + 1), (bx + 1, by + 1)]):
                    return True

            # 5) simple dead‑square (non‑target floor surrounded on two orthogonal sides)
            # If a cell has walls/boxes on both north&south OR east&west and is not a target,
            # any box on it cannot move.
            if (up and down) or (left and right):
                return True

            # 6) frozen corridor: box on a straight corridor with walls/boxes at both ends
            # and no target lies further along that line.
            # Horizontal corridor
            if not up and not down:  # same row, free vertical movement
                # look left
                l = bx
                while (l - 1, by) not in walls and (l - 1, by) not in boxes:
                    l -= 1
                # look right
                r = bx
                while (r + 1, by) not in walls and (r + 1, by) not in boxes:
                    r += 1
                # if both ends are walls/boxes and no target in between, dead
                if ((l - 1, by) in walls or (l - 1, by) in boxes) and \
                   ((r + 1, by) in walls or (r + 1, by) in boxes):
                    if not any(tx == bx and l < tx < r for (tx, ty) in targets):
                        return True
            # Vertical corridor
            if not left and not right:
                u = by
                while (bx, u - 1) not in walls and (bx, u - 1) not in boxes:
                    u -= 1
                d = by
                while (bx, d + 1) not in walls and (bx, d + 1) not in boxes:
                    d += 1
                if ((bx, u - 1) in walls or (bx, u - 1) in boxes) and \
                   ((bx, d + 1) in walls or (bx, d + 1) in boxes):
                    if not any(ty == by and u < ty < d for (tx, ty) in targets):
                        return True

        return False

    # ------------------------------------------------------------------
    # Helper 2 – reachable cells (BFS)
    # ------------------------------------------------------------------
    def reachable_cells(st):
        """Set of floor cells the player can stand on (boxes act as walls)."""
        visited = set()
        q = deque([st.player])
        while q:
            r, c = q.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in st.walls or (nr, nc) in st.boxes:
                    continue
                if 0 <= nr < st.height and 0 <= nc < st.width:
                    q.append((nr, nc))
        return visited

    # ------------------------------------------------------------------
    # Helper 3 – pushability, proximity and distance‑reduction bonus
    # ------------------------------------------------------------------
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_FROM_DIR = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}

    def pushability_proximity(st):
        """
        Returns:
            pushable_frac – fraction of non‑target boxes that have at least one
                            reachable push that reduces Manhattan distance.
            proximity     – normalised closeness of the player to the nearest such push cell.
            max_reduction – maximal distance reduction offered by any improving push.
        """
        reachable = reachable_cells(st)
        total_boxes = max(1, len(st.boxes))
        pushable = 0
        best_dist = None
        max_reduction = 0

        for (bx, by) in st.boxes:
            if (bx, by) in st.targets:
                continue

            has_good = False
            cur_dist = min(abs(bx - tx) + abs(by - ty) for (tx, ty) in st.targets)

            for dx, dy in DIRS:
                nbx, nby = bx + dx, by + dy          # destination cell for the box
                px, py = bx - dx, by - dy            # player must stand here

                if (nbx, nby) in st.walls or (nbx, nby) in st.boxes:
                    continue
                if (px, py) in st.walls or (px, py) in st.boxes:
                    continue
                if (px, py) not in reachable:
                    continue

                new_dist = min(abs(nbx - tx) + abs(nby - ty) for (tx, ty) in st.targets)
                if new_dist >= cur_dist:
                    continue        # not an improving push

                # quick dead‑lock test on the resulting state (no full clone needed)
                fake = st.clone()
                fake.player = (px, py)
                fake.apply_action(ACTION_FROM_DIR[(dx, dy)])
                if is_deadlocked(fake):
                    continue

                has_good = True
                d = abs(st.player[0] - px) + abs(st.player[1] - py)
                if best_dist is None or d < best_dist:
                    best_dist = d
                reduction = cur_dist - new_dist
                if reduction > max_reduction:
                    max_reduction = reduction

            if has_good:
                pushable += 1

        pushable_frac = pushable / total_boxes

        if best_dist is None:
            proximity = 0.0
        else:
            max_d = st.height + st.width
            proximity = (max_d - best_dist) / max_d

        return pushable_frac, proximity, max_reduction

    # ------------------------------------------------------------------
    # Helper 4 – alignment fraction
    # ------------------------------------------------------------------
    def alignment_fraction(st):
        """Fraction of non‑target boxes that share a clear row/column with a free target."""
        count = 0
        for (bx, by) in st.boxes:
            if (bx, by) in st.targets:
                continue
            aligned = False
            for (tx, ty) in st.targets:
                if tx == bx:
                    step = 1 if ty > by else -1
                    clear = True
                    for c in range(by + step, ty, step):
                        if (bx, c) in st.walls or (bx, c) in st.boxes:
                            clear = False
                            break
                    if clear:
                        aligned = True
                        break
                if ty == by:
                    step = 1 if tx > bx else -1
                    clear = True
                    for r in range(bx + step, tx, step):
                        if (r, by) in st.walls or (r, by) in st.boxes:
                            clear = False
                            break
                    if clear:
                        aligned = True
                        break
            if aligned:
                count += 1
        return count / max(1, len(st.boxes))

    # ------------------------------------------------------------------
    # Helper 5 – mobility term
    # ------------------------------------------------------------------
    def mobility(st) -> float:
        reachable = reachable_cells(st)
        total_floor = st.height * st.width - len(st.walls)
        return len(reachable) / total_floor if total_floor else 0.0

    # ------------------------------------------------------------------
    # Helper 6 – heuristic evaluation (re‑balanced 6‑term sum)
    # ------------------------------------------------------------------
    def heuristic(st) -> float:
        # 1) boxes on targets
        box_frac = st.boxes_on_targets() / st.num_targets if st.num_targets else 0.0

        # 2) normalised total distance (using farthest‑target bound)
        targets = list(st.targets)
        max_possible = 0
        for (bx, by) in st.boxes:
            far = max(abs(bx - tx) + abs(by - ty) for (tx, ty) in targets) if targets else 0
            max_possible += far
        max_possible = max(1, max_possible)
        dist_score = 1.0 - (st.total_box_distance() / max_possible)

        # 3) pushability + proximity + distance reduction
        pushable_frac, proximity, max_reduction = pushability_proximity(st)

        # 4) alignment
        align_frac = alignment_fraction(st)

        # 5) mobility
        mob = mobility(st)

        # Weighted sum (weights sum to 1.0)
        return (0.45 * box_frac +
                0.20 * dist_score +
                0.15 * pushable_frac +
                0.10 * align_frac +
                0.05 * mob +
                0.05 * proximity)

    # ------------------------------------------------------------------
    # Helper 7 – one‑step look‑ahead scoring (includes scaled push bonus)
    # ------------------------------------------------------------------
    def lookahead_score(cur_state, action) -> float:
        trial = cur_state.clone()
        trial.apply_action(action)

        if is_deadlocked(trial):
            return -1.0

        # base heuristic
        h = heuristic(trial)

        # if a box moved, add a bonus proportional to distance reduction
        if trial.boxes != cur_state.boxes:
            # compute reduction for the moved box only (there is at most one)
            moved_box = next(iter(trial.boxes - cur_state.boxes))
            # previous position is the opposite of push direction
            # we can approximate by Manhattan distance reduction
            cur_dist = min(abs(moved_box[0] - tx) + abs(moved_box[1] - ty) for (tx, ty) in cur_state.targets)
            new_dist = min(abs(moved_box[0] - tx) + abs(moved_box[1] - ty) for (tx, ty) in trial.targets)
            reduction = cur_dist - new_dist
            max_dist = cur_state.height + cur_state.width
            # scale bonus to at most 0.07
            h += 0.07 * (reduction / max_dist)
        return h

    # ------------------------------------------------------------------
    # Rollout parameters
    # ------------------------------------------------------------------
    EPSILON = 0.15                         # lower randomisation, push‑biased only if improving
    # Adaptive horizon: longer when more boxes are already placed
    base_limit = 20
    adaptive = 4 * state.boxes_on_targets()
    ROLLOUT_LIMIT = min(max_depth, max(base_limit, base_limit + adaptive))

    sim_state = state.clone()
    depth = 0
    total_reward = 0.0
    best_h = heuristic(sim_state)

    while not sim_state.is_terminal() and depth < ROLLOUT_LIMIT:
        legal = sim_state.legal_actions()
        if not legal:
            break

        # ------------------- ε‑greedy -------------------
        if random.random() < EPSILON:
            # try to find a *useful* push (reduces distance)
            useful_pushes = []
            for a in legal:
                tmp = sim_state.clone()
                tmp.apply_action(a)
                if tmp.boxes != sim_state.boxes:
                    # ensure it reduces distance for the moved box
                    moved = next(iter(tmp.boxes - sim_state.boxes))
                    cur_dist = min(abs(moved[0] - tx) + abs(moved[1] - ty) for (tx, ty) in sim_state.targets)
                    new_dist = min(abs(moved[0] - tx) + abs(moved[1] - ty) for (tx, ty) in tmp.targets)
                    if new_dist < cur_dist:
                        useful_pushes.append(a)
            if useful_pushes:
                chosen = random.choice(useful_pushes)
            else:
                chosen = random.choice(legal)
        else:
            # Greedy look‑ahead over all legal actions
            best_score = -float("inf")
            chosen = None
            for a in legal:
                sc = lookahead_score(sim_state, a)
                if sc > best_score:
                    best_score = sc
                    chosen = a
            if chosen is None:
                chosen = random.choice(legal)

        # Apply chosen action
        prev_boxes = sim_state.boxes.copy()
        sim_state.apply_action(chosen)

        # Accumulate shaped reward for the new state (gives fine‑grained gradient)
        total_reward += sim_state.returns()[perspective_player]

        # Update best heuristic seen
        cur_h = heuristic(sim_state)
        if cur_h > best_h:
            best_h = cur_h

        # Early dead‑lock detection – abort with zero reward
        if is_deadlocked(sim_state):
            return 0.0

        depth += 1

        # Optional early exit: if a new box hit a target, we can return immediately
        if sim_state.boxes_on_targets() > state.boxes_on_targets():
            # give a strong signal for progress
            return 1.0

    # ------------------------------------------------------------------
    # Return value – combine best heuristic, average shaped reward and
    # a small push‑bonus (already reflected in the cumulative reward).
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        return sim_state.returns()[perspective_player]

    avg_shaped = total_reward / max(1, depth)
    reward = max(best_h, avg_shaped)          # choose the stronger signal
    return max(0.0, min(1.0, reward))
