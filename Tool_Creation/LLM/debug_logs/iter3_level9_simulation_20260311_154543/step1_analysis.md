# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:45:55 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 11.91s |
| Status     | success |
| Tokens     | 14318 |
| Validation | invalid — SyntaxError: invalid character '…' (U+2026) (<unknown>, line 1) |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: simulation).

PHASE: simulation
  • What it does: Rolls forward from a leaf state to estimate how promising it is. Returns a reward (e.g. 0–1) that flows into backpropagation.
  • Optimization goal: Produce REWARDS that reflect true state quality. Shaped partial progress helps MCTS distinguish good from bad actions.
  • Constraints: Must return a FLOAT. Reward MUST vary across states — flat rewards ≈ random play. Called thousands of times per move — keep it fast.
  • Good patterns: shaped score (boxes on targets, distance improvement), penalize deadlocks/loops/stagnation, prefer pushes over wandering, early termination when stuck.
  • Avoid: tree-level visit balancing, acceptance criteria for tools — this phase only scores rollouts.

APPROACH — 70 / 30 RULE:
  ~70% of iterations: INCREMENTAL OPTIMIZATION
    • Start from the CURRENT code.
    • Make targeted, gradual improvements (add a check,
      tweak weights, add a heuristic factor, etc.).
    • Prefer building on what works rather than replacing it.

  ~30% of iterations: PARADIGM SHIFT (when warranted)
    • If the current approach is fundamentally flawed or
      plateauing, you may propose a larger restructure.
    • Explain clearly WHY a rewrite is needed.
    • Even rewrites should keep proven components that work.

GENERAL RULES:
  • Write clean, well-structured code — as long as the
    heuristic needs to be (no artificial line limit).
  • Each iteration builds on the previous version.
  • Complex heuristics with multiple factors are encouraged
    when they improve play quality.

------------------------------------------------------------
GAME RULES
------------------------------------------------------------
Game: Sokoban (Macro-Push Variant)

== Overview ==
Sokoban is a single-player puzzle game. The player pushes boxes onto
target positions inside a grid-based warehouse. This variant uses
MACRO-PUSH actions: instead of single-step movement (UP/DOWN/LEFT/RIGHT),
each action represents a complete box push from any position the player
can walk to in the current state.

== Symbols ==
  #   wall (impassable)
  .   target position
  $   box
  *   box on a target
  @   player
  +   player standing on a target
  (space)  empty floor

== Rules ==
1. The player's reachable region is all floor cells connected to the
   player's current position by paths not blocked by walls or boxes
   (computed via BFS flood-fill).
2. An action is a (player_pos, direction) tuple where:
   - player_pos is a cell in the reachable region
   - direction is UP(0), DOWN(1), LEFT(2), or RIGHT(3)
   - The cell at player_pos + delta(direction) must contain a box
   - The cell at player_pos + 2*delta(direction) must be free
3. Applying an action:
   a. The player walks from current position to player_pos via BFS
      shortest path (costs len(path) steps).
   b. The player pushes the box one cell in the push direction
      (costs 1 step). Player ends at the box's original position.
4. Total step cost per action = walk_steps + 1.
5. The action space varies per state — different states may have
   different numbers of available pushes.
6. The puzzle is solved when ALL boxes are on target positions.
7. A game is lost when a box is in deadlock (e.g., corner not on target),
   or when max_steps is reached.

== Actions ==
The action space is a list of (player_pos, direction) tuples.
  player_pos = (row, col) — where the player must stand to push
  direction = 0(UP), 1(DOWN), 2(LEFT), 3(RIGHT) — push direction

Example: ((2, 3), 1) means "walk to (2,3) then push DOWN" — the box
at (3,3) moves to (4,3), player ends at (3,3).

== State Representation ==
Each state is described by:
  - The player's (row, col) position.
  - The set of (row, col) positions of all boxes.
  - A step counter (game terminates if max_steps is reached).
  - Derived metrics: boxes_on_targets count and total_box_distance
    (sum of Manhattan distance from each box to its nearest target).

== Reward ==
  - Solved (all boxes on targets): 1.0
  - Deadlocked or max_steps: 0.0

== GameState API ==
Public attributes (via properties):
  walls     : frozenset[tuple[int,int]]   – wall positions
  targets   : frozenset[tuple[int,int]]   – target positions
  boxes     : set[tuple[int,int]]         – current box positions
  player    : tuple[int,int]              – current player position
  height, width : int                     – grid dimensions
  num_targets   : int                     – number of targets
  steps, max_steps : int                  – current / maximum step count

Public methods:
  clone()            → new independent copy of the state
  legal_actions()    → list[tuple[tuple[int,int], int]]  (macro-push tuples)
  apply_action(a)    → None (mutates the state in-place)
  is_terminal()      → bool
  returns()          → list[float]
  current_player()   → int (always 0)
  state_key()        → str (hashable key for transposition)
  boxes_on_targets() → int
  total_box_distance() → int

== Key Strategic Concepts ==
  - Every action is a box push — think about WHICH box to push and
    from WHERE, not about walking directions.
  - Avoid pushes that create deadlocks (box in non-target corner,
    box against wall with no reachable target).
  - Plan push ORDER: placing one box may block paths needed for others.
  - Consider the cost of walking: a push requiring a long walk costs
    more steps than a nearby push.
  - Minimize total push count; fewer pushes usually means fewer
    opportunities for deadlock.


------------------------------------------------------------
MCTS TOOL FUNCTIONS (all 4 phases)
------------------------------------------------------------

--- selection ---
```python
"""
LLM-generated MCTS tool: selection
Description: Finalized improved selection heuristic
Generated:   2026-03-11T15:42:34.817346
"""

def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with corrected UCB exploration, richer dead‑lock
    detection, forward walk‑cost estimation and a dynamic target bonus.
    """
    import math
    from collections import deque

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0          # penalty for a dead‑locked state
    WALK_WEIGHT = 0.5                  # weight for future walk‑cost estimate
    NOVELTY_WEIGHT = 0.2               # encourages under‑visited nodes
    BASE_TARGET_BONUS = 5.0            # per‑remaining‑box multiplier

    # ----- Heuristic: total Manhattan distance from boxes to nearest targets -----
    def heuristic(state):
        return state.total_box_distance()

    # ----- Cheap corner dead‑lock detection -----
    def corner_deadlocked(state):
        walls = state.walls
        for bx, by in state.boxes:
            if (bx, by) in state.targets:
                continue
            if ((bx - 1, by) in walls and (bx, by - 1) in walls) or \
               ((bx - 1, by) in walls and (bx, by + 1) in walls) or \
               ((bx + 1, by) in walls and (bx, by - 1) in walls) or \
               ((bx + 1, by) in walls and (bx, by + 1) in walls):
                return True
        return False

    # ----- Wall‑line dead‑lock detection (same as expansion) -----
    def wall_line_deadlocked(state, box_pos):
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
                    return False          # a target beyond the wall → not dead‑locked
                step += 1
            return True                   # no target found in this direction
        return False

    # ----- Aggregate dead‑lock test (corner + wall‑line) -----
    def is_deadlocked(state):
        if corner_deadlocked(state):
            return True
        for b in state.boxes:
            if b not in state.targets and wall_line_deadlocked(state, b):
                return True
        return False

    # ----- Minimum walk distance to any legal push position -----
    def min_walk_to_any_push(state):
        """BFS from player to the closest push position; returns steps."""
        walls = state.walls
        boxes = state.boxes
        h, w = state.height, state.width
        start = state.player

        q = deque([start])
        dist = {start: 0}
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in walls or (nr, nc) in boxes:
                    continue
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = d + 1
                    q.append((nr, nc))

        min_d = None
        for push_pos, _ in state.legal_actions():
            if push_pos in dist:
                d = dist[push_pos]
                if (min_d is None) or (d < min_d):
                    min_d = d
        return min_d if min_d is not None else 0

    # ------------------------------------------------------------------
    # 2️⃣  Tree walk – pick child with minimal f‑value
    # ------------------------------------------------------------------
    node = root
    while not node.state.is_terminal():
        # A node is fully expanded when there are no untried actions left.
        fully_expanded = (len(node._untried_actions) == 0)

        if not fully_expanded:
            # Let the expansion phase create a fresh child.
            return node

        best_child = None
        best_f = math.inf
        best_q = -math.inf

        # Guard against division‑by‑zero when the parent has never been visited.
        parent_visits = node.visits if node.visits > 0 else 1

        # Parent info for target‑bonus scaling
        parent_on_target = node.state.boxes_on_targets()
        remaining_targets = node.state.num_targets - parent_on_target
        dynamic_target_bonus = BASE_TARGET_BONUS * max(remaining_targets, 1)

        for action, child in node.children.items():
            # ----- g: cumulative steps taken so far (already includes the walk for this action) -----
            g = getattr(child.state, "steps", 0)

            # ----- h components -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: reward for *new* boxes placed on targets
            child_on_target = child.state.boxes_on_targets()
            target_diff = child_on_target - parent_on_target
            target_bonus = dynamic_target_bonus * target_diff

            # Forward walk‑cost estimate from child to its next push
            walk_est = min_walk_to_any_push(child.state)

            h = h_base + dead_pen - target_bonus + WALK_WEIGHT * walk_est

            # ----- Exploration term (classic UCB: add, not subtract) -----
            if child.visits == 0:
                f = -math.inf          # forces selection of unvisited nodes
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                novelty = NOVELTY_WEIGHT / (child.visits + 1)
                f = g + h + explore - novelty

            # ----- Tie‑break on average value (q) -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        # Safety net – should never happen, but avoids an infinite loop.
        if best_child is None:
            break

        node = best_child

    return node
```

--- expansion ---
```python
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
```

--- simulation ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function is correct and efficient.
Generated:   2026-03-11T15:31:56.604000
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Improved leaf evaluation for MCTS simulation.

    Returns:
        1.0                                 if the state is solved,
        state.returns()[p]                  if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1] based on box distance,
                                              player walk cost, fraction of boxes placed,
                                              and dead‑lock awareness.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """Shortest‑path distances from `start` to every reachable floor cell."""
        q = deque([start])
        dist = {start: 0}
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        return dist

    # ------------------------------------------------------------------
    # Helper: simple corner dead‑lock detection.
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        """Return True if any non‑target box is stuck in a corner."""
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
    # Helper: wall‑line dead‑lock detection.
    # If a box sits next to a wall and there is no target beyond that wall
    # in the same line, the box can never be moved to a target.
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
            # look beyond the wall in the same direction
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a target exists beyond the wall → not dead‑locked
                    return False
                step += 1
            # no target found → dead‑locked
            return True
        return False

    # ------------------------------------------------------------------
    # Helper: aggregate dead‑lock test (corner + wall‑line).
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # 0️⃣ Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # state.returns() already contains the appropriate value for
        # dead‑lock or step‑limit termination.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Immediate dead‑lock check – heavily punish hopeless leaves.
    # ------------------------------------------------------------------
    if is_deadlocked(state):
        # a dead‑locked leaf should never be attractive.
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣ Box‑only Manhattan distance (h).
    # ------------------------------------------------------------------
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 3️⃣ Player walk cost to the *closest* legal push position (w).
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = bfs_distances(state.player, walls, boxes, height, width)

    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 4️⃣ Fraction of boxes already on targets (f).
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 5️⃣ Combine components.
    #    • ALPHA  – stronger walk‑cost penalty,
    #    • GAMMA  – larger bonus for placed boxes,
    #    • exponent – non‑linear scaling to spread values.
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walk distance (more influential)
    GAMMA = 0.9   # bonus for already‑placed boxes
    EXP   = 0.8   # exponent < 1 makes small h differences larger

    # Non‑linear base reward: higher when both h and walk are small.
    base = 1.0 / (1.0 + (h + ALPHA * w) ** EXP)

    # Add progress bonus.
    reward = base + GAMMA * f

    # Clamp to [0, 1] – solved states are handled explicitly below.
    reward = max(0.0, min(1.0, reward))

    # ------------------------------------------------------------------
    # 6️⃣ Perfect solution override.
    # ------------------------------------------------------------------
    if h == 0:
        reward = 1.0

    return reward
```

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Ensure solved bonus is applied undiscouted to the root and clean up minor redundancies.
Generated:   2026-03-11T15:32:44.153294
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a leaf reward up the tree while preserving useful
    signal for deep plans.

    Changes vs. previous version:
      • Use *leaf‑relative* depth discount instead of absolute depth.
        This lets the root receive a sizable fraction of the leaf
        reward while still favouring shallower ancestors.
      • Raise the discount factor γ to 0.95 (gentler decay).
      • Reduce step‑penalty to 0.001 – walk cost is already reflected in
        the simulation reward, so we avoid over‑penalising.
      • Add the solved‑state bonus **without discount** to the root node,
        guaranteeing a strong spike when a leaf is a solution.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be adjusted without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (gentler decay)
    SOLVED_BONUS = 1.0    # extra reward for a solved leaf (added to root)
    STEP_PENALTY = 0.001  # penalty per step taken in the leaf state

    # ------------------------------------------------------------------
    # Lazy‑import shared A* globals (avoids circular imports).
    # ------------------------------------------------------------------
    import importlib.util
    import sys
    from pathlib import Path

    _KEY = "astar_globals"
    if _KEY not in sys.modules:
        _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
        _s = importlib.util.spec_from_file_location(_KEY, str(_p))
        _m = importlib.util.module_from_spec(_s)
        sys.modules[_KEY] = _m
        assert _s.loader is not None
        _s.loader.exec_module(_m)
    _ag = sys.modules[_KEY]

    # ------------------------------------------------------------------
    # Gather leaf information once.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)          # absolute depth from root
    leaf_state = leaf_node.state

    # Detect solved terminal states.
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )

    # Apply a modest step‑cost penalty (clip at zero).
    steps_taken = getattr(leaf_state, "steps", 0)
    penalised_reward = reward - STEP_PENALTY * steps_taken
    if penalised_reward < 0.0:
        penalised_reward = 0.0

    base_reward = penalised_reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying *leaf‑relative* depth discount.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Current depth from the root.
        cur_depth = _ag.node_depth(cur)

        # Remaining depth to leaf (non‑negative by construction).
        remaining = leaf_depth - cur_depth

        # Discount factor for this ancestor.
        decay = GAMMA ** remaining

        # Discounted contribution of the leaf reward.
        discounted = decay * base_reward

        # Accumulate into the node's value.
        cur.value += discounted

        # If this node is the root and the leaf was a solution,
        # add the full solved bonus (undiscounted) to create a strong
        # signal for selection.
        if cur.parent is None and is_solved:
            cur.value += SOLVED_BONUS

        # ---- Synchronise A* g‑score (unchanged) ----
        key = cur.state.state_key()
        g = cur_depth
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent.
        cur = getattr(cur, "parent", None)
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function is correct and efficient.
Generated:   2026-03-11T15:31:56.604000
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Improved leaf evaluation for MCTS simulation.

    Returns:
        1.0                                 if the state is solved,
        state.returns()[p]                  if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1] based on box distance,
                                              player walk cost, fraction of boxes placed,
                                              and dead‑lock awareness.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """Shortest‑path distances from `start` to every reachable floor cell."""
        q = deque([start])
        dist = {start: 0}
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        return dist

    # ------------------------------------------------------------------
    # Helper: simple corner dead‑lock detection.
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        """Return True if any non‑target box is stuck in a corner."""
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
    # Helper: wall‑line dead‑lock detection.
    # If a box sits next to a wall and there is no target beyond that wall
    # in the same line, the box can never be moved to a target.
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
            # look beyond the wall in the same direction
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a target exists beyond the wall → not dead‑locked
                    return False
                step += 1
            # no target found → dead‑locked
            return True
        return False

    # ------------------------------------------------------------------
    # Helper: aggregate dead‑lock test (corner + wall‑line).
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # 0️⃣ Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # state.returns() already contains the appropriate value for
        # dead‑lock or step‑limit termination.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Immediate dead‑lock check – heavily punish hopeless leaves.
    # ------------------------------------------------------------------
    if is_deadlocked(state):
        # a dead‑locked leaf should never be attractive.
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣ Box‑only Manhattan distance (h).
    # ------------------------------------------------------------------
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 3️⃣ Player walk cost to the *closest* legal push position (w).
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = bfs_distances(state.player, walls, boxes, height, width)

    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 4️⃣ Fraction of boxes already on targets (f).
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 5️⃣ Combine components.
    #    • ALPHA  – stronger walk‑cost penalty,
    #    • GAMMA  – larger bonus for placed boxes,
    #    • exponent – non‑linear scaling to spread values.
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walk distance (more influential)
    GAMMA = 0.9   # bonus for already‑placed boxes
    EXP   = 0.8   # exponent < 1 makes small h differences larger

    # Non‑linear base reward: higher when both h and walk are small.
    base = 1.0 / (1.0 + (h + ALPHA * w) ** EXP)

    # Add progress bonus.
    reward = base + GAMMA * f

    # Clamp to [0, 1] – solved states are handled explicitly below.
    reward = max(0.0, min(1.0, reward))

    # ------------------------------------------------------------------
    # 6️⃣ Perfect solution override.
    # ------------------------------------------------------------------
    if h == 0:
        reward = 1.0

    return reward
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level9)
Timestamp:  2026-03-11T15:45:40.158367
Iterations: 500
Solved:     True
Steps:      12
Returns:    [1.0]

  Move 1: action=((3, 4), 0), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/4 | Total distance: 6
    Children: [((3, 4), 0)(v=499, avg=0.555), ((3, 4), 2)(v=1, avg=0.192)]
  Move 2: action=((2, 4), 2), total_visits=500
    State: Step 1/1000 | Boxes on target: 1/4 | Total distance: 5
    Children: [((2, 4), 2)(v=497, avg=0.585), ((3, 4), 2)(v=1, avg=0.438), ((3, 2), 3)(v=1, avg=0.434), ((1, 5), 1)(v=1, avg=0.433)]
  Move 3: action=((2, 3), 1), total_visits=500
    State: Step 2/1000 | Boxes on target: 1/4 | Total distance: 5
    Children: [((2, 3), 1)(v=493, avg=0.619), ((1, 2), 1)(v=1, avg=0.468), ((3, 4), 2)(v=1, avg=0.436), ((2, 6), 2)(v=1, avg=0.465), ((2, 1), 3)(v=1, avg=0.434), ((3, 2), 3)(v=1, avg=0.432), ((4, 3), 0)(v=1, avg=0.411), ((1, 5), 1)(v=1, avg=0.431)]
  Move 4: action=((1, 2), 1), total_visits=500
    State: Step 3/1000 | Boxes on target: 2/4 | Total distance: 4
    Children: [((4, 4), 2)(v=1, avg=0.692), ((4, 2), 3)(v=1, avg=0.692), ((1, 2), 1)(v=495, avg=0.651), ((2, 6), 2)(v=1, avg=0.735), ((2, 1), 3)(v=1, avg=0.691), ((1, 5), 1)(v=1, avg=0.688)]
  Move 5: action=((2, 2), 1), total_visits=500
    State: Step 7/1000 | Boxes on target: 2/4 | Total distance: 3
    Children: [((2, 2), 1)(v=494, avg=0.684), ((3, 1), 3)(v=1, avg=0.733), ((4, 4), 2)(v=1, avg=0.731), ((4, 2), 3)(v=1, avg=0.731), ((2, 6), 2)(v=1, avg=0.801), ((4, 2), 0)(v=1, avg=0.686), ((1, 5), 1)(v=1, avg=0.727)]
  Move 6: action=((2, 6), 2), total_visits=500
    State: Step 8/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((2, 6), 2)(v=499, avg=0.719), ((1, 5), 1)(v=1, avg=0.984)]
  Move 7: action=((2, 5), 2), total_visits=500
    State: Step 14/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((2, 5), 2)(v=499, avg=0.756), ((2, 3), 3)(v=1, avg=0.981)]
  Move 8: action=((2, 4), 2), total_visits=500
    State: Step 15/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((2, 4), 2)(v=498, avg=0.795), ((2, 2), 3)(v=1, avg=0.980), ((1, 3), 1)(v=1, avg=0.978)]
  Move 9: action=((1, 2), 1), total_visits=500
    State: Step 16/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((1, 2), 1)(v=499, avg=0.836), ((2, 1), 3)(v=1, avg=0.979)]
  Move 10: action=((3, 1), 3), total_visits=500
    State: Step 19/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((3, 1), 3)(v=500, avg=0.879)]
  Move 11: action=((3, 2), 3), total_visits=500
    State: Step 22/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((3, 2), 3)(v=499, avg=0.925), ((3, 4), 2)(v=1, avg=0.973)]
  Move 12: action=((2, 4), 1), total_visits=500
    State: Step 23/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((2, 4), 1)(v=497, avg=0.974), ((3, 3), 3)(v=1, avg=0.976), ((3, 5), 2)(v=1, avg=0.972), ((4, 4), 0)(v=1, avg=0.970)]

Final state:
Step 26/1000 | Boxes on target: 4/4 | Total distance: 0
########
#   *  #
#      #
#   @  #
# ***  #
########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level9
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level9 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 4 levels): 1.0000

Per-level best composites so far:
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=0.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level10', 'level4', 'level6', 'level7', 'level8']
Mastered levels: ['level2', 'level3', 'level5', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 2 [level3] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 3 [level9] [selection]: composite=1.0000, solve_rate=100%, eval_time=9.6s, desc=Finalized improved selection heuristic ★ BEST-for-level
  Iter 3 [level9] [expansion]: composite=1.0000, solve_rate=100%, eval_time=9.8s, desc=No changes needed; the draft implementation is correct and efficient. ← accepted

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'simulation'
heuristic code, and the gameplay traces above.

Produce a focused analysis with these sections:

1. KEY WEAKNESSES
   What are the main problems causing poor play?
   Cite specific move numbers, Q-value patterns, or state
   observations as evidence. Be specific. Rank by impact.

2. ROOT CAUSE
   WHY does the current code produce this behaviour?
   Point to specific logic or missing logic in the code.

3. PROPOSED APPROACH
   Choose one of these strategies:

   A) INCREMENTAL (~70% of cases): Describe targeted
      modifications to the 'simulation' function
      that address the top weakness(es). Build on and
      extend the current code.

   B) RESTRUCTURE (~30% of cases): If the current approach
      is fundamentally limited, describe a different
      strategy. Explain why incremental changes won't
      suffice. Keep proven components that work.

   State which strategy (A or B) you recommend and why.

Keep your analysis under 500 words. Do NOT write code.

---

## Response

**1. KEY WEAKNESSES**  

| Rank | Symptom (seen in the trace / general runs) | Why it hurts MCTS |
|------|-------------------------------------------|-------------------|
| **1** | *Dead‑lock blind‑spots.* The simulation only tests **corner** and **wall‑line** deadlocks. More subtle patterns (2 × 2 box blocks, frozen boxes, “dead‑square” cells) are ignored. When a rollout reaches such a state it still receives a non‑zero reward, so the back‑propagated value suggests the branch is promising even though it can never be solved. |
| **2** | *Loop/oscillation bias.* The reward does **not** penalise actions that merely undo the previous push (or any move that leaves the total box‑to‑target distance unchanged). Because the walk‑cost term only looks at the *closest* push, a back‑and‑forth push that costs a few steps but keeps *h* and *f* constant can still get a relatively high reward, wasting many simulations on cycles. |
| **3** | *Flat‑reward scaling.* The combination `1/(1+(h+α·w)^ε) + γ·f` with α = 0.5, ε = 0.8 produces a very narrow reward band (≈0.6‑0.9 for most non‑terminal states). This makes the simulation signal almost indistinguishable, so selection relies heavily on the noisy UCB term and requires many more iterations. |
| **4** | *No depth discount.* A leaf reached after many pushes is rewarded exactly the same as one after a single push, encouraging deep roll‑outs that are expensive to compute and that provide little additional information. |

**2. ROOT CAUSE**  

*Simulation* contains three helper checks:

```python
def is_corner_deadlocked(state): …
def is_wall_line_deadlocked(state, box_pos): …
def is_deadlocked(state): (corner or any wall‑line)
```

No other dead‑lock patterns are examined, even though the *expansion* phase already implements 2 × 2 blocks, frozen‑box detection, and a dead‑square map. Consequently, rollouts can happily continue into states that are provably unsolvable.

The reward formula never looks at **change** of the distance metric or at recent actions; it only uses the current `h`, the *minimum* walk distance `w`, and the fraction `f`. Thus a move that pushes a box forward and immediately pushes it back gets the same `h` and `f` as a useful move, with only a tiny increase in `w`. The exponent `ε = 0.8` compresses the range, producing the flat‑reward issue. Finally, the `max_depth` argument is ignored, so deep rollouts are not penalised.

**3. PROPOSED APPROACH – A (incremental)**  

We can keep the overall shape of the current simulation function (it is cheap and already supplies a useful baseline) and patch the identified deficiencies:

1. **Richer dead‑lock detection** – reuse the helpers already present in *expansion*:  
   * 2 × 2 block check, frozen‑box detection, dead‑square map (pre‑computed per‑root).  
   * If any of these detect a deadlock, return 0.0 immediately.  

2. **Stagnation / back‑track penalty** – add a tiny term that discourages moves that do not improve the distance metric:  
   * compute `delta = old_total_distance – new_total_distance`.  
   * If `delta ≤ 0` (no improvement) add `‑β·(1 + steps_since_last_improvement)` to the reward, where `β` is a small weight (≈0.02).  
   * Maintain a counter of steps since the last improvement in the state (can be derived from `state.steps` and the last improvement distance).  

3. **Depth discount** – use the optional `max_depth` argument (or simply `state.steps`). Multiply the base reward by `δ^(state.steps)` with `δ≈0.97`. This steadily reduces the value of very deep leafs without wiping out the signal for shallow, promising branches.

4. **Wider reward spread** – adjust the exponent to a value **≥ 1** (e.g., 1.0) and lower the walk‑weight α to 0.3. The new base becomes  

   ```
   base = 1.0 / (1.0 + (h + α·w))
   ```

   which yields a broader range (≈0.2‑0.9) and makes the simulation more discriminative.

All these changes are local additions to `default_simulation`; they do not affect the API, keep the cheap BFS walk already present, and can share the dead‑square cache used by expansion (a simple module‑level dict keyed by the root state). By enriching dead‑lock detection we eliminate misleading positive rewards, the stagnation penalty removes wasteful cycles, and the depth discount together with the revised scaling gives the MCTS a richer gradient to follow. This incremental refinement should raise solve rates on the still‑unmastered levels without incurring a noticeable runtime penalty.
