# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:49:15 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.80s |
| Status     | success |
| Tokens     | 15456 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: selection).

PHASE: selection
  • What it does: Walks down the tree from root to a leaf. Chooses which existing branch to explore next. Must balance exploration (UCB) and exploitation.
  • Optimization goal: Improve how we RANK existing nodes — favor promising branches, deprioritize dead ends. Your heuristic adjusts node scores used by UCB1.
  • Constraints: Called very often. Keep it CHEAP — no multi-step rollouts, no deep deadlock simulation. Rank nodes, don't simulate.
  • Good patterns: bonus for more boxes on targets, bonus for lower box distance, penalize obvious deadlocks, novelty bonus for under-visited nodes.
  • Avoid: expensive rollout logic, final reward shaping (that belongs in simulation).

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

--- selection ◀ TARGET ---
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

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: Fix wall‑line dead‑lock detection and clean up unused import.
Generated:   2026-03-11T15:46:35.650226
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enhanced leaf evaluation for MCTS simulation.

    Returns:
        1.0                                 if the state is solved,
        state.returns()[p]                  if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1] that reflects
        box‑to‑target distance, player walk cost, fraction of boxes placed,
        depth discount, and a richer dead‑lock detection suite.
    """
    # ------------------------------------------------------------------
    # Helpers – all defined locally to keep the function self‑contained.
    # ------------------------------------------------------------------
    from collections import deque

    # ---------- BFS distances for the player (ignoring boxes) ----------
    def bfs_distances(start, walls, boxes, height, width):
        """Return dict of shortest distances from `start` to every reachable
        floor cell, avoiding walls and boxes."""
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

    # ---------- Corner dead‑lock detection ----------
    def is_corner_deadlocked(st):
        walls = st.walls
        targets = st.targets
        for bx, by in st.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ---------- Wall‑line dead‑lock detection (fixed) ----------
    def is_wall_line_deadlocked(st, box_pos):
        """Return True if `box_pos` sits against a wall and there is **no**
        target further along that wall direction."""
        walls = st.walls
        targets = st.targets
        h, w = st.height, st.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in dirs:
            neighbor = (box_pos[0] + dx, box_pos[1] + dy)
            if neighbor not in walls:
                continue      # not a wall in this direction
            # walk beyond the wall looking for a target
            step = 2
            found_target = False
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    found_target = True
                    break
                step += 1
            if not found_target:
                # this wall direction offers no reachable target → dead‑locked
                return True
        return False

    # ---------- 2×2 block dead‑lock detection ----------
    def is_2x2_block_deadlocked(st):
        bset = st.boxes
        for bx, by in bset:
            if ((bx + 1, by) in bset and
                (bx, by + 1) in bset and
                (bx + 1, by + 1) in bset):
                return True
        return False

    # ---------- Dead‑square map (cells that cannot reach any target) ----------
    _dead_square_cache = getattr(default_simulation, "_dead_square_cache", {})

    def compute_dead_squares(st):
        """Cells that cannot reach a target when ignoring boxes."""
        key = st.state_key()
        if key in _dead_square_cache:
            return _dead_square_cache[key]

        h, w = st.height, st.width
        walls = st.walls
        targets = st.targets
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

        floor = {(r, c) for r in range(h) for c in range(w) if (r, c) not in walls}
        dead = floor - reachable
        _dead_square_cache[key] = dead
        setattr(default_simulation, "_dead_square_cache", _dead_square_cache)
        return dead

    # ---------- Frozen‑box detection (no legal push direction) ----------
    def is_frozen_box_deadlocked(st, dead_squares):
        walls = st.walls
        boxes = st.boxes
        h, w = st.height, st.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for bx, by in boxes:
            if (bx, by) in st.targets:
                continue
            can_move = False
            for dx, dy in dirs:
                dst = (bx + dx, by + dy)
                player = (bx - dx, by - dy)
                if not (0 <= dst[0] < h and 0 <= dst[1] < w):
                    continue
                if not (0 <= player[0] < h and 0 <= player[1] < w):
                    continue
                if dst in walls or dst in boxes or dst in dead_squares:
                    continue
                if player in walls or player in boxes:
                    continue
                can_move = True
                break
            if not can_move:
                return True
        return False

    # ---------- Aggregate dead‑lock test ----------
    def is_deadlocked(st):
        if is_corner_deadlocked(st):
            return True
        for b in st.boxes:
            if b not in st.targets and is_wall_line_deadlocked(st, b):
                return True
        if is_2x2_block_deadlocked(st):
            return True
        dead_squares = compute_dead_squares(st)
        if is_frozen_box_deadlocked(st, dead_squares):
            return True
        return False

    # ------------------------------------------------------------------
    # 0️⃣ Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Immediate dead‑lock check – heavily punish hopeless leaves.
    # ------------------------------------------------------------------
    if is_deadlocked(state):
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣ Core metrics.
    # ------------------------------------------------------------------
    h = state.total_box_distance()                     # sum of box‑to‑nearest‑target distances
    placed = state.boxes_on_targets()
    num_targets = state.num_targets
    f = placed / num_targets if num_targets > 0 else 0.0

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
    # 4️⃣ Additional penalties:
    #    • boxes sitting on dead‑square cells,
    #    • depth discount (encourages shallow rollouts).
    # ------------------------------------------------------------------
    dead_squares = compute_dead_squares(state)
    dead_square_penalty = sum(1 for b in state.boxes if b in dead_squares)
    DEAD_SQUARE_WEIGHT = 0.05          # modest penalty per dead‑square box

    # Depth discount – use either max_depth (if supplied) or steps taken.
    DEPTH_DISCOUNT = 0.97
    depth = state.steps if max_depth == 0 else max_depth - state.steps
    depth_factor = DEPTH_DISCOUNT ** max(depth, 0)

    # ------------------------------------------------------------------
    # 5️⃣ Reward composition.
    # ------------------------------------------------------------------
    ALPHA = 0.30          # walk‑cost weight
    GAMMA = 0.85          # progress bonus weight
    base = 1.0 / (1.0 + (h + ALPHA * w))          # broader spread without exponent

    reward = base + GAMMA * f
    reward -= DEAD_SQUARE_WEIGHT * dead_square_penalty
    reward *= depth_factor                       # apply depth discount

    # Clamp to the legal interval.
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
Description: Finalized incremental improvements to backpropagation.
Generated:   2026-03-11T15:48:04.408133
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up the tree.

    Improvements over the previous version:
      • Eliminates the leaf‑relative depth discount (the simulation already
        contains a depth‑aware reward). Nodes now receive the full reward.
      • Removes the per‑step penalty – walk‑cost is accounted for in the
        simulation reward, avoiding over‑penalisation of deep but correct
        solutions.
      • Adds a solved‑state bonus to **every** node on the path, not only the
        root, so the whole successful trajectory is reinforced.
      • Clips the propagated reward to [0, 1] before accumulation to keep node
        values bounded.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    SOLVED_BONUS = 2.0          # extra reward added to each ancestor of a solved leaf

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
    leaf_state = leaf_node.state

    # Detect solved terminal states.
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )

    # Ensure the propagated reward stays within the legal interval.
    base_reward = max(0.0, min(1.0, reward))

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards – no additional discounting.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # 1️⃣ Update visit count.
        cur.visits += 1

        # 2️⃣ Accumulate the (clipped) reward.
        cur.value += base_reward

        # 3️⃣ If the leaf was a solution, reinforce every ancestor.
        if is_solved:
            cur.value += SOLVED_BONUS

        # 4️⃣ Synchronise A* g‑score (unchanged).
        key = cur.state.state_key()
        cur_depth = _ag.node_depth(cur)   # absolute depth from the root
        if key not in visited or visited[key] > cur_depth:
            visited[key] = cur_depth

        # 5️⃣ Move to parent.
        cur = getattr(cur, "parent", None)
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (selection)
------------------------------------------------------------
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

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 6 levels): 0.8333

Per-level best composites so far:
  level1: best=1.0000 (baseline=1.0000) [MASTERED]
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level7: best=0.0000 (baseline=0.0000)
  level9: best=1.0000 (baseline=0.0000) [MASTERED]

Active levels (not yet mastered): ['level10', 'level4', 'level6', 'level7', 'level8']
Mastered levels: ['level1', 'level2', 'level3', 'level5', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 3 [level9] [simulation]: composite=1.0000, solve_rate=100%, eval_time=11.8s, desc=Fix wall‑line dead‑lock detection and clean up unused import. ← accepted
  Iter 3 [level9] [backpropagation]: composite=1.0000, solve_rate=100%, eval_time=9.2s, desc=Finalized incremental improvements to backpropagation. ← accepted
  Iter 4 [level1] [expansion]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (taken from the trace) | Why it matters | Evidence in the code |
|------|--------------------------------|----------------|----------------------|
| **1** | *No measurable progress after three macro‑pushes* (total distance goes 5 → 4 → 5 → 6). The selection routine is not steering the search toward states that actually reduce the box‑to‑target distance. | The heuristic used in selection (`h = h_base + dead_pen – target_bonus + WALK_WEIGHT * walk_est`) **ignores the sign of the distance change**; it treats a child with a *higher* total distance exactly the same as one that improves it, as long as the raw distance (`h_base`) is similar. Consequently the tree can wander into neutral or even worsening pushes. | `h_base = heuristic(child.state)` returns the absolute distance, not the *delta* from the parent. No term penalises `child.total_box_distance() – parent.total_box_distance()`. |
| **2** | *Over‑penalisation of the g‑value* (`g = child.state.steps`). Because each macro‑push carries a variable walk cost, the raw step count can dominate the UCB sum (`f = g + h + explore – novelty`). Deep but promising branches are discarded early. | In the trace the three moves already cost 6 steps; the growing `g` term pushes the algorithm to repeatedly expand shallow nodes rather than following a longer, more promising line. | `g = getattr(child.state, "steps", 0)` is added with weight 1.0, while `h` components are modest (walk weight 0.5, dead‑lock 1000 only when dead‑locked). |
| **3** | *Very weak novelty encouragement* (`NOVELTY_WEIGHT = 0.2`). The subtraction `‑ novelty` is tiny compared with the other terms, so rarely influences the ranking. | The tree keeps revisiting the same few children, which the trace shows – only three distinct actions ever appear. | `novelty = NOVELTY_WEIGHT / (child.visits + 1)` – with a single visit this is 0.1, negligible against `g` and `h`. |
| **4** | *Missing dynamic scaling of walk‑cost vs. distance‑improvement* – the walk‑estimate (`WALK_WEIGHT = 0.5`) is static, while the importance of moving a box closer grows as the puzzle nears completion. | Early in the game a long walk is acceptable if it reduces distance drastically, but the current formula does not reward that trade‑off. | No dependency on `remaining_targets` or on the magnitude of distance reduction. |

**2. ROOT CAUSE**  

- **Absence of a distance‑change term**: Selection only looks at the absolute Manhattan sum (`total_box_distance`). The expansion phase adds a penalty for increasing the moved‑box distance, but selection never inherits this information because it does not compare child vs. parent. Hence the most important guidance (“push a box closer to a target”) is missing.  

- **Unscaled g‑value**: By feeding the raw step count directly into the UCB score, deeper branches quickly acquire a large `g` that outweighs any heuristic advantage. The algorithm therefore prefers ever‑shorter horizons, preventing it from reaching the later pushes that are required on level 7.  

- **Undervalued novelty**: The novelty term is sub‑tracted rather than *added* (most UCB‑style implementations add an exploration bonus). Even when subtracted, its magnitude (0.2 / visits) is far too small to overcome the dominant `g + h` terms, so the tree does not explore under‑visited sub‑trees.  

These three issues together cause the tree to get stuck in a shallow “random walk” that never improves the distance metric, which is exactly what the trace shows.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental refinement** (the preferred route).  
We will keep the overall A*‑guided UCB framework – it already supplies dead‑lock detection and a walk‑estimate – but we add three targeted tweaks that directly address the weaknesses above.

1. **Introduce a distance‑delta term**  
   ```python
   delta = parent_state.total_box_distance() - child.state.total_box_distance()
   # positive = improvement, negative = regression
   DIST_IMPROVEMENT_WEIGHT = 2.0          # tunable
   distance_bonus = DIST_IMPROVEMENT_WEIGHT * max(delta, 0)
   distance_penalty = -DIST_IMPROVEMENT_WEIGHT * max(-delta, 0)
   h = h_base + dead_pen - target_bonus + WALK_WEIGHT*walk_est \
       + distance_bonus + distance_penalty
   ```
   This mirrors the expansion‑phase penalty but is applied already during selection, steering the tree toward pushes that actually bring boxes closer.

2. **Scale the g‑component**  
   Replace `g = child.state.steps` with a **scaled** version, e.g.  
   `g = STEP_WEIGHT * child.state.steps` where `STEP_WEIGHT` ≈ 0.1–0.2. This prevents raw step counts from drowning out heuristic signals while still keeping a bias toward shallower nodes.

3. **Make novelty an *additive* exploration bonus** and increase its magnitude.  
   ```python
   NOVELTY_WEIGHT = 0.8   # larger
   novelty = NOVELTY_WEIGHT / math.sqrt(child.visits + 1)
   f = g + h + explore + novelty   # note the + instead of –
   ```  
   Using the square‑root denominator mirrors classic UCB and yields a stronger boost for rarely‑visited children, encouraging broader coverage.

These three modifications can be inserted directly into the existing selection loop without altering its overall structure, preserving the dead‑lock checks, walk‑estimate, and dynamic target‑bonus that already work. The changes are cheap (constant‑time arithmetic) and reuse data already available in the loop (parent state, child state, visit counts).  

By rewarding genuine distance improvement, dampening the raw step cost, and amplifying novelty, the selection phase will rank branches much more in line with Sokoban‑specific progress, giving the MCTS a realistic chance to solve level 7 while staying within the required cheap‑heuristic budget.

------------------------------------------------------------
TASK — IMPROVE THE HEURISTIC
------------------------------------------------------------
Improve the 'selection' function above based on your
analysis. Decide which approach fits best:

OPTION A — Incremental optimization (~70% of the time):
  Start from the CURRENT code and make targeted improvements.
  Examples:
  • Add a deadlock check (box in corner with no target)
  • Add distance-based scoring factors
  • Switch from random rollout to greedy lookahead
  • Add early-termination for stuck states
  • Tune weights or scoring formula
  • Combine multiple heuristic signals

OPTION B — Paradigm shift (~30% of the time):
  If the current approach is fundamentally limited, propose
  a different strategy. Keep any components that work well.
  State clearly in DESCRIPTION why a restructure is needed.

How the 'selection' phase works in MCTS:
  - Receives (root, exploration_weight). Returns the chosen child index.
  - Adjusts UCB scores or selection policy to favor promising branches.
  - Called at every level during tree descent — keep it FAST.
  - Must produce a valid child index; selection drives which branch gets expanded next.

CONSTRAINTS:
  • Same function signature as the current code.
  • Standalone function, standard library only.
  • SELF-CONTAINED: define ALL helper functions inline.
    Do NOT use undefined names (e.g. _ag, _bfs_distance).
  • node is MCTSNode (uses __slots__). Use ONLY node.state,
    node.parent, node.children, node._untried_actions,
    node.visits, node.value. Do NOT assign new attributes.
  • Code can be as long as the heuristic requires — no
    artificial line limit. Prioritize quality over brevity.

You MUST format your response EXACTLY as follows:

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line: what you changed and why>
```python
<your complete function code here>
```

Rules for the header fields:
- ACTION must be either 'create' (brand new tool) or 'modify'
  (improving the existing tool shown above).
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Added distance‑delta term, scaled step cost, and stronger additive novelty bonus to better prioritize progress and exploration.
```python
def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with corrected UCB exploration, richer dead‑lock
    detection, forward walk‑cost estimation and a dynamic target bonus.
    Incremental improvements:
      • distance‑delta term rewards pushes that reduce total box‑to‑target distance;
      • step‑cost is scaled down to keep deeper promising branches alive;
      • novelty is an additive +UCB bonus (larger weight, sqrt denominator).
    """
    import math
    from collections import deque

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0          # penalty for a dead‑locked state
    WALK_WEIGHT = 0.5                  # weight for future walk‑cost estimate
    STEP_WEIGHT = 0.15                 # scaling for raw step count (g)
    NOVELTY_WEIGHT = 0.8               # additive novelty bonus magnitude
    BASE_TARGET_BONUS = 5.0            # per‑remaining‑box multiplier
    DIST_IMPROVEMENT_WEIGHT = 2.0      # weight for distance‑delta term

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

        # Parent info for target‑bonus scaling and distance‑delta
        parent_state = node.state
        parent_on_target = parent_state.boxes_on_targets()
        parent_total_dist = heuristic(parent_state)
        remaining_targets = parent_state.num_targets - parent_on_target
        dynamic_target_bonus = BASE_TARGET_BONUS * max(remaining_targets, 1)

        for action, child in node.children.items():
            # ----- g: scaled cumulative steps taken so far -----
            g = STEP_WEIGHT * getattr(child.state, "steps", 0)

            # ----- h components -----
            child_state = child.state
            h_base = heuristic(child_state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child_state) else 0.0

            # Target bonus: reward for *new* boxes placed on targets
            child_on_target = child_state.boxes_on_targets()
            target_diff = child_on_target - parent_on_target
            target_bonus = dynamic_target_bonus * target_diff

            # Forward walk‑cost estimate from child to its next push
            walk_est = min_walk_to_any_push(child_state)

            # Distance‑delta term (positive if distance improves)
            child_total_dist = heuristic(child_state)
            delta = parent_total_dist - child_total_dist   # >0 => improvement
            distance_term = DIST_IMPROVEMENT_WEIGHT * delta

            h = (h_base + dead_pen
                 - target_bonus
                 + WALK_WEIGHT * walk_est
                 + distance_term)

            # ----- Exploration term (classic UCB) -----
            if child.visits == 0:
                # Force selection of unvisited nodes
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                novelty = NOVELTY_WEIGHT / math.sqrt(child.visits + 1)
                f = g + h + explore + novelty  # note the + novelty

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
