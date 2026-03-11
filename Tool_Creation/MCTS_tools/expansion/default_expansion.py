"""
LLM-generated MCTS tool: expansion
Description: No changes needed; the draft implementation is correct and efficient.
Generated:   2026-03-11T15:44:53.533802
"""

def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions with additional
    diversification, richer dead‑lock detection and more nuanced
    distance‑based penalties.

    Improvements over the previous version:
      • Pre‑computed *dead‑square* map per root (cells that can never
        reach a target) – any action moving a box onto such a cell is
        discarded.
      • Frozen‑box detection: after a push, a non‑target box that has
        no legal push direction (considering dead squares) causes the
        state to be pruned.
      • 2×2 box block dead‑lock detection.
      • Distance‑increase penalty now only accounts for the box that
        was actually moved, not all boxes.
      • Farthest‑box distance change is multiplied by a discount factor
        (allowing temporary increases).
      • Dynamic weighting of total box distance based on remaining
        targets.
      • Stronger diversification via a random jitter added to the
        heuristic score, preventing the tree from over‑committing to the
        first “best” action.
    """
    import sys
    import random
    from collections import deque
    from mcts.node import MCTSNode

    # ------------------------------------------------------------------
    # Helper 1: BFS walk length (player moves only, boxes are static)
    # ------------------------------------------------------------------
    def bfs_walk_len(state, start, goal):
        """Shortest walk length from start to goal avoiding walls & boxes.
        Returns sys.maxsize if unreachable."""
        if start == goal:
            return 0
        walls = state.walls
        boxes = state.boxes
        h, w = state.height, state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque([(start[0], start[1], 0)])
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < h and 0 <= ny < w):
                    continue
                if (nx, ny) in walls or (nx, ny) in boxes:
                    continue
                if (nx, ny) == goal:
                    return nd
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, nd))
        return sys.maxsize

    # ------------------------------------------------------------------
    # Helper 2: Simple corner dead‑lock detection
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Wall‑line dead‑lock detection.
    # ------------------------------------------------------------------
    def is_wall_line_deadlocked(state, box_pos):
        walls = state.walls
        targets = state.targets
        h, w = state.height, state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in dirs:
            neighbor = (box_pos[0] + dx, box_pos[1] + dy)
            if neighbor not in walls:
                continue
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    return False        # a target beyond the wall → safe
                step += 1
            return True                 # no target found → dead‑locked
        return False

    # ------------------------------------------------------------------
    # Helper 4: 2×2 box block dead‑lock detection
    # ------------------------------------------------------------------
    def is_2x2_block_deadlocked(state):
        bset = state.boxes
        for bx, by in bset:
            if ((bx + 1, by) in bset and
                (bx, by + 1) in bset and
                (bx + 1, by + 1) in bset):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 5: Dead‑square map (cells that can never reach a target)
    # ------------------------------------------------------------------
    def compute_dead_squares(state):
        """Returns a set of floor cells that cannot reach any target
        when ignoring boxes (only walls block movement)."""
        h, w = state.height, state.width
        walls = state.walls
        targets = state.targets

        # flood‑fill from all targets
        q = deque(targets)
        reachable = set(targets)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in walls or (nr, nc) in reachable:
                    continue
                reachable.add((nr, nc))
                q.append((nr, nc))

        floor = {(r, c)
                 for r in range(h)
                 for c in range(w)
                 if (r, c) not in walls}
        dead = floor - reachable
        return dead

    # ------------------------------------------------------------------
    # Helper 6: Frozen‑box detection (no legal push direction)
    # ------------------------------------------------------------------
    def is_frozen_box_deadlocked(state, dead_squares):
        """A box is frozen if none of the four push directions is feasible
        (taking walls, other boxes and dead squares into account)."""
        walls = state.walls
        boxes = state.boxes
        h, w = state.height, state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for bx, by in boxes:
            if (bx, by) in state.targets:
                continue  # already solved
            can_move = False
            for dx, dy in dirs:
                dst = (bx + dx, by + dy)
                player_pos = (bx - dx, by - dy)
                if not (0 <= dst[0] < h and 0 <= dst[1] < w):
                    continue
                if not (0 <= player_pos[0] < h and 0 <= player_pos[1] < w):
                    continue
                if dst in walls or dst in boxes or dst in dead_squares:
                    continue
                if player_pos in walls or player_pos in boxes:
                    continue
                can_move = True
                break
            if not can_move:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 7: Aggregate dead‑lock test (corner + wall‑line + 2×2 +
    #              frozen + dead squares)
    # ------------------------------------------------------------------
    def is_deadlocked(state, dead_squares):
        if is_corner_deadlocked(state):
            return True
        for b in state.boxes:
            if b not in state.targets and is_wall_line_deadlocked(state, b):
                return True
        if is_2x2_block_deadlocked(state):
            return True
        if is_frozen_box_deadlocked(state, dead_squares):
            return True
        return False

    # ------------------------------------------------------------------
    # Helper 8: Closest‑target Manhattan distance for a single box
    # ------------------------------------------------------------------
    def closest_target_dist(box, targets):
        return min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets)

    # ------------------------------------------------------------------
    # Helper 9: Per‑root persistent dictionaries
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None
        default_expansion._dead_squares = {}

    # Find the root node (the one without a parent)
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    # Reset per‑root structures when we encounter a new root
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._dead_squares.clear()
        default_expansion._root_key = root_key

    visited = default_expansion._visited

    # Compute dead‑square set for this root if not yet done
    if root_key not in default_expansion._dead_squares:
        default_expansion._dead_squares[root_key] = compute_dead_squares(root.state)
    dead_squares = default_expansion._dead_squares[root_key]

    # ------------------------------------------------------------------
    # 0. Constants – tunable
    # ------------------------------------------------------------------
    ALPHA = 1.0        # walk‑distance weight
    BETA = 1.2         # moved‑box distance‑increase penalty weight
    GAMMA = 1.2        # target‑placement reward weight
    LAMBDA_FAR = 0.5   # farthest‑box distance change weight
    DECAY_FAR = 0.8    # discount for farthest‑box change (allows temporary increase)
    EPS_JITTER = 0.02  # random diversification term (added to h)
    RHO = 0.2          # penalty for pushing the same box twice in a row

    # ------------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # ------------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Pre‑compute old per‑box distances and farthest distance.
    # ------------------------------------------------------------------
    old_dist_per = {
        b: closest_target_dist(b, node.state.targets) for b in node.state.boxes
    }
    old_farthest = max(old_dist_per.values()) if old_dist_per else 0

    # direction vectors for macro‑pushes: UP, DOWN, LEFT, RIGHT
    drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # 3. Scan untried actions and compute heuristic scores.
    # ------------------------------------------------------------------
    scored = []                     # (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    # Determine which box was moved in the previous step (if any)
    prev_moved_box = None
    if node.parent_action is not None:
        prev_player, prev_dir = node.parent_action
        pdx, pdy = drc[prev_dir]
        prev_moved_box = (prev_player[0] + pdx, prev_player[1] + pdy)

    for action in list(node._untried_actions):
        player_pos, direction = action

        # a) walk distance to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # c) enriched dead‑lock detection (prune permanently)
        if is_deadlocked(next_state, dead_squares):
            deadlocked_to_remove.append(action)
            continue

        # d) visited‑pruning – keep only better‑g actions
        g_new = cur_steps + walk_len + 1          # macro‑push cost
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            continue

        # e) moved‑box distance increase (only for the box we just pushed)
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after = (moved_box_before[0] + dx, moved_box_before[1] + dy)

        before_dist = closest_target_dist(moved_box_before, node.state.targets)
        after_dist = closest_target_dist(moved_box_after, next_state.targets)
        delta = max(0, after_dist - before_dist)   # penalty only for worsening

        # f) farthest‑box distance change (discounted)
        new_dist_per = {
            b: closest_target_dist(b, next_state.targets) for b in next_state.boxes
        }
        new_farthest = max(new_dist_per.values()) if new_dist_per else 0
        far_diff = new_farthest - old_farthest    # can be negative
        far_term = LAMBDA_FAR * far_diff * DECAY_FAR

        # g) target placement bonus
        on_target = 1 if moved_box_after in next_state.targets else 0

        # h) recent‑box penalty (discourage back‑and‑forth)
        recent_pen = RHO if (prev_moved_box is not None and moved_box_before == prev_moved_box) else 0.0

        # i) dynamic weighting of total box distance (more emphasis later)
        remaining_targets = node.state.num_targets - node.state.boxes_on_targets()
        dist_weight = 1.0 + 0.2 * remaining_targets

        # j) final heuristic (lower is better)
        h = (dist_weight * next_state.total_box_distance()
             + ALPHA * walk_len
             + BETA * delta
             + far_term
             + recent_pen
             - GAMMA * on_target)

        # k) diversification jitter (adds a small random positive offset)
        h += random.random() * EPS_JITTER

        scored.append((h, walk_len, action, next_state, key, g_new))

    # ------------------------------------------------------------------
    # 4. Remove permanently dead‑locked actions from the node's pool.
    # ------------------------------------------------------------------
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 5. Choose the best action (lowest h, break ties with shorter walk).
    # ------------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        h, walk_len, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # Fallback: expand any remaining untried action (if any)
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            player_pos, _ = chosen_action
            walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            chosen_g = cur_steps + walk_len + 1
            if (chosen_key not in visited) or (visited[chosen_key] > chosen_g):
                visited[chosen_key] = chosen_g
        else:
            # No actions left – return any existing child (or self)
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # 6. Create the child node and register it.
    # ------------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
