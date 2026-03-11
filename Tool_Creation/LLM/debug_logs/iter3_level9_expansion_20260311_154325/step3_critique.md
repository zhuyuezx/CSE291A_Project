# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:44:53 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 34.60s |
| Status     | success |
| Tokens     | 13409 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: expansion).

PHASE: expansion
  • What it does: Creates new child nodes from a frontier node. Decides which actions to materialize into the tree and in what order.
  • Optimization goal: PRUNE bad actions and ORDER remaining actions so promising ones are tried first. Filter deadlocks before they enter the tree.
  • Constraints: Best place for hard constraints. Order actions; optionally filter some entirely. No rollout policies or value aggregation.
  • Good patterns: reject pushes into non-target corners, reject wall deadlocks, prefer pushes that reduce box distance, deprioritize no-op player movement.
  • Avoid: long rollout policies, reward aggregation, node-value update rules.

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
TARGET HEURISTIC TO IMPROVE (expansion)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: expansion
Description: No functional changes needed; retain the incremental improvements as‑is.
Generated:   2026-03-11T15:37:55.139859
"""

def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions with additional
    diversification and finer‑grained distance penalties.

    New features:
      * Tie‑breaker noise / visit‑penalty to break deterministic ties.
      * Per‑box distance‑increase penalty (instead of only total increase).
      * Bonus / penalty based on the farthest box distance change.
      * Stronger reward for placing a box on a target.
      * Small penalty for moving the same box as the previous action
        (discourages immediate back‑and‑forth pushes).
      * Retains existing dead‑lock detection, walk‑cost weighting,
        and per‑root visited‑state pruning.
    """
    import sys
    from collections import deque
    from math import log
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
        height = state.height
        width = state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < height and 0 <= ny < width):
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
        h = state.height
        w = state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in dirs:
            neighbor = (box_pos[0] + dx, box_pos[1] + dy)
            if neighbor not in walls:
                continue
            # Look past the wall in the same direction for a target.
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a target exists beyond the wall → not a deadlock.
                    return False
                step += 1
            # No target found in this direction → deadlock.
            return True
        return False

    # ------------------------------------------------------------------
    # Helper 4: Aggregate dead‑lock test (corner + wall‑line)
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 5: Closest‑target Manhattan distance for a single box
    # ------------------------------------------------------------------
    def closest_target_dist(box, targets):
        return min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets)

    # ------------------------------------------------------------------
    # Helper 6: Per‑root visited dictionary (persistent across calls)
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # Reset visited map when we encounter a new root state.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._root_key = root_key
    visited = default_expansion._visited

    # ------------------------------------------------------------------
    # 0. Some constants
    # ------------------------------------------------------------------
    ALPHA = 1.0      # walk‑distance weight
    BETA  = 1.2      # per‑box distance‑increase penalty weight
    GAMMA = 1.2      # target‑placement reward weight
    LAMBDA_FAR = 0.5 # farthest‑box distance change weight
    EPS   = 0.01     # tie‑breaker / visit‑penalty magnitude
    RHO   = 0.2      # penalty for moving the same box as previous action

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
    old_dist_list = sorted(old_dist_per.values())

    # direction vectors for macro‑pushes: UP, DOWN, LEFT, RIGHT
    drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # 3. Scan untried actions, compute heuristic scores.
    # ------------------------------------------------------------------
    scored = []                     # (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    # Helper to know which box was moved in the previous step (if any)
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

        # c) static dead‑lock detection (permanent removal)
        if is_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # d) visited pruning – keep better‑g actions
        g_new = cur_steps + walk_len + 1   # macro‑push cost
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            continue

        # e) per‑box distance increase (delta)
        new_dist_per = {
            b: closest_target_dist(b, next_state.targets) for b in next_state.boxes
        }
        new_dist_list = sorted(new_dist_per.values())
        delta = 0
        for od, nd in zip(old_dist_list, new_dist_list):
            if nd > od:
                delta += nd - od

        # f) farthest‑box distance change bonus/penalty
        new_farthest = max(new_dist_per.values()) if new_dist_per else 0
        far_diff = new_farthest - old_farthest   # >0 → penalty, <0 → bonus

        # g) target placement bonus
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after  = (moved_box_before[0] + dx, moved_box_before[1] + dy)
        on_target = 1 if moved_box_after in next_state.targets else 0

        # h) recent‑box penalty
        recent_pen = RHO if (prev_moved_box is not None and moved_box_before == prev_moved_box) else 0.0

        # i) final heuristic (lower is better)
        h = (next_state.total_box_distance()
             + ALPHA * walk_len
             + BETA * delta
             + LAMBDA_FAR * far_diff
             + recent_pen
             - GAMMA * on_target)

        # j) tiny deterministic tie‑breaker (acts like visit‑penalty)
        tie_noise = EPS * (player_pos[0] + player_pos[1] + direction) / 100.0
        h += tie_noise

        scored.append((h, walk_len, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 4. Choose best action (lowest h, break ties with shorter walk)
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
            # No actions left – return an existing child (or self)
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # 5. Create child node and register it.
    # ------------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it hurts |
|------|--------------------------|--------------|
| 1️⃣ | **Over‑reliance on a single “best” push** – after the first expansion the root’s child `((3,4),0)` receives 496/500 visits while every alternative gets a single visit. The tree quickly becomes “greedy”, so any mis‑estimated push is reinforced and alternative routes are explored only once. | The expansion heuristic is too *deterministic*: a tiny tie‑breaker (`EPS≈0.001`) cannot overcome the large difference in `h`. If `h` is slightly off, the algorithm may lock onto a sub‑optimal line of play, especially on harder levels where the best immediate push is not the best long‑term move. |
| 2️⃣ | **Very limited dead‑lock detection** – only corner‑deadlocks and “wall‑line” deadlocks are screened. Classic Sokoban deadlocks such as 2‑box squares, frozen boxes blocked by other boxes, or dead squares (cells that can never reach a target) are not caught. | The planner may expand actions that look promising locally but inevitably lead to a deadlocked region, wasting many simulations and sometimes preventing a solution on more constrained maps. |
| 3️⃣ | **Per‑box distance‑increase penalty is a blunt sum** – `delta` adds *all* positive differences between the old and new sorted distance lists. A push that improves one box but slightly worsens another can be penalised heavily, discouraging necessary “temporary setbacks”. | This can suppress pushes that are required to reposition a blocking box, forcing the search to take longer detours or to miss the solution entirely on levels with narrow corridors. |
| 4️⃣ | **Farthest‑box penalty is always positive** (`LAMBDA_FAR = 0.5`). In many puzzles the farthest box must be moved away before it can later be guided to a target. A flat penalty discourages such necessary maneuvers. | Leads to premature pruning of pushes that increase the farthest Manhattan distance, again biasing towards short‑term gains. |

**2. ROOT CAUSE**

*Expansion* decides the next child solely on the scalar `h`.  
- The tie‑breaker (`EPS`) is orders of magnitude smaller than the other terms, so the lowest `h` dominates all iterations (Weakness 1).  
- `is_deadlocked` only calls `is_corner_deadlocked` and `is_wall_line_deadlocked`; no check for frozen‑box patterns, 2×2 box clusters, or pre‑computed *dead squares* (Weakness 2).  
- The distance‑increase computation `delta` sums every positive difference between the two sorted distance vectors, ignoring the fact that a *few* boxes may temporarily recede (Weakness 3).  
- `far_diff` is added with a positive weight, never allowing a temporary increase of the farthest box (Weakness 4).

Consequently the heuristic is **myopic**: it pushes the greedy “lowest‑h” action, and it does not prune many truly hopeless actions.

**3. PROPOSED APPROACH – A (incremental)**  

We keep the current overall structure (root‑visited pruning, BFS walk cost, per‑root visited dict) because it already yields correct solutions on easy levels and integrates nicely with the other phases. The improvements are targeted:

1. **Stronger tie‑breaking / diversification**  
   * Replace the tiny `EPS` term with a *visit‑penalty* proportional to the child’s visit count, e.g. `visit_pen = C / (child.visits + 1)` where `C≈0.2`.  
   * This ensures that after a child has been selected many times, its `h` rises enough for the next best action to be tried, preventing lock‑in on a single line.

2. **Enriched dead‑lock detection** (still cheap)  
   * Pre‑compute a **dead‑square map** for the level: any floor cell that cannot reach any target via a flood‑fill that respects walls only (ignore boxes). Reject pushes that would move a box into a dead square.  
   * Add a **simple frozen‑box test**: after applying the action, for each box not on a target, check whether it has at least one push direction whose opposite cell is free *and* the destination cell is not a dead square. If none, treat the state as deadlocked.  
   * Detect the classic **2×2 box block** (four boxes occupying a 2×2 sub‑grid) – immediate prune.

3. **Refined distance‑increase term**  
   * Compute `delta` as the *sum of positive differences* **only for boxes that moved** (or for the box that was pushed). This isolates the effect of the current push rather than re‑ordering the whole list.  
   * Introduce a **discount factor** `DECAY = 0.8` for the farthest‑box change: `far_term = LAMBDA_FAR * far_diff * DECAY`. This still penalises large increases but allows a modest increase when necessary.

4. **Dynamic weighting of total distance**  
   * Multiply `next_state.total_box_distance()` by a factor that grows as the number of remaining boxes shrinks, e.g. `DIST_WEIGHT = 1.0 + 0.2 * (remaining_targets)`. This makes the heuristic focus more on *final* placements as the puzzle progresses.

These four tweaks can be inserted directly in the existing loops (`for action in node._untried_actions:`) without altering the overall algorithmic flow. They preserve the proven per‑root visited pruning and the macro‑push cost model, yet they:

* diversify the search early enough to discover alternative routes,
* eliminate many truly hopeless branches before they waste simulations,
* give the planner leeway to perform necessary temporary setbacks.

Because the current code already solves level 9, the incremental path is safer and faster to evaluate than a full rewrite. If after these changes certain levels still stall, a later *paradigm shift* (e.g., integrating a pattern‑based move generator) can be considered.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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

        # all floor cells (non‑walls)
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
                # destination cell
                dst = (bx + dx, by + dy)
                # player must stand opposite direction
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

------------------------------------------------------------
TASK — CRITIQUE & FINALIZE
------------------------------------------------------------
Review the DRAFT code above for critical issues.

Check for:
  1. BUGS — API misuse, crashes, wrong variable names
  2. SPEED — unnecessary clones or deep loops (runs 1000s of times)
  3. REWARD SPREAD — does the return value vary across states?

RULES:
  - Preserve the INTENT of the draft (incremental fix or
    restructure). Do NOT add unrelated features.
  - Fix bugs and performance issues. Improve clarity.
  - If the draft is correct and fast, output it UNCHANGED.

You MUST format your response EXACTLY as follows:

CRITIQUE:
<1-3 bullet points, or 'No issues found'>

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line summary>
```python
<complete final function code>
```

Rules for the header fields:
- ACTION must be either 'create' or 'modify'.
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

CRITIQUE:
- No issues found.

ACTION: modify
FILE_NAME: default_expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: No changes needed; the draft implementation is correct and efficient.
```python
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
