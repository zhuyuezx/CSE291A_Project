# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:29:43 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 38.79s |
| Status     | success |
| Tokens     | 10161 |
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
Description: 
Generated:   2026-03-11T01:09:23.036760
"""

def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements (preserved):
      * Uses true macro‑push cost (walk steps + 1) as g‑score for visited
        pruning.
      * Orders actions by a richer heuristic:
          h = box_distance + α * walk_distance   (α = 0.5)
      * Performs lightweight static dead‑lock detection and permanently
        removes those actions.
      * Does **not** permanently discard actions that fail the visited check;
        they stay in `_untried_actions` for future attempts.
      * In the fallback case, the visited map is only updated if the new
        g‑score is better than any previously recorded one.
    """
    import sys
    from collections import deque
    from mcts.node import MCTSNode

    # ------------------------------------------------------------------
    # Helper 1: BFS walk length (player moves only, boxes are static)
    # ------------------------------------------------------------------
    def bfs_walk_len(state, start, goal):
        """Return shortest walk length from start to goal avoiding walls & boxes.
        If unreachable, return sys.maxsize."""
        if start == goal:
            return 0
        walls = state.walls
        boxes = state.boxes
        width, height = state.width, state.height

        # simple 4‑direction moves
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
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
    # Helper 2: Very cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def is_simple_deadlocked(state):
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        # pre‑compute wall adjacency for quick checks
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # check four corner patterns
            if ((bx + 1, by) in walls or (bx - 1, by) in walls) and \
               ((bx, by + 1) in walls or (bx, by - 1) in walls):
                # Two perpendicular walls -> corner
                # Need both a horizontal and a vertical wall adjacent
                horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
                vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
                if horiz and vert:
                    return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Per‑root visited dictionary (persistent across calls)
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # --------------------------------------------------------------
    # 0. Reset shared visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._root_key = root_key

    visited = default_expansion._visited

    # --------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # --------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # --------------------------------------------------------------
    # 2. Scan untried actions.
    # --------------------------------------------------------------
    alpha = 0.5                     # weight for walk distance in the heuristic
    scored = []                     # (h, action, next_state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action

        # a) walk distance from current player location to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- static dead‑lock detection (permanent) -----
        if is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ----- visited pruning (temporary) -----
        g_new = cur_steps + walk_len + 1   # real cost to reach this macro‑state
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for potential later expansion
            continue

        # ----- heuristic scoring -----
        # box‑only Manhattan distance (provided by GameState)
        box_h = next_state.total_box_distance()
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h). If none, fall back.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No scored actions – fall back to any remaining untried action.
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
            # No actions left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # 4. Create and register the child node.
    # --------------------------------------------------------------
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

**1. KEY WEAKNESSES**

| Rank | Symptom (trace) | Why it matters |
|------|----------------|----------------|
| 1️⃣ | *Move 1–4* – the same action `(4,4,0)` gets **497/500** visits and the tree stays on a shallow “low‑h” branch. The total box‑distance only drops from 7 → 6 → 5 → 4 while the walk cost is essentially zero. The planner never dares to spend a few extra steps to reach a more promising push. |
| 2️⃣ | *Move 5–6* – after two boxes are on targets the distance stalls at **3** and then at **2** after six macro‑pushes. No action ever reduces the distance further, showing the expansion is blind to pushes that **temporarily increase** box distance but are required to unblock other boxes. |
| 3️⃣ | *Dead‑lock handling* – only a corner‑check is used. On level 8 many “wall‑line” or “box‑against‑wall with no target reachable” dead‑locks exist, yet the planner still expands them (they survive in `_untried_actions`). |
| 4️⃣ | *Heuristic weighting* – `h = box_distance + 0.5 * walk_len`. The walk‑length term is too cheap; cheap pushes that do not improve the layout dominate the ordering. |

**2. ROOT CAUSE**

- **Heuristic scope**: `box_h = next_state.total_box_distance()` looks *only at the absolute distance after the push*. It rewards any push that does not increase that sum, even if the push moves a box **away** from its target (Δ > 0) or creates a blockage. Consequently, pushes that keep the distance unchanged but require a long walk are favoured over pushes that momentarily raise the distance yet unlock the puzzle.

- **Missing penalty for distance increase**: No term penalises `Δ = new_box_distance – old_box_distance`. Thus a push that raises the distance by 1 but saves a long walk is still considered superior to a push that reduces distance by 2 but needs many steps.

- **Static dead‑lock detection too weak**: Only corner dead‑locks are removed. Many solvable‑but‑hard patterns (e.g., a box pressed against a wall with no target in its row/column, or two adjacent boxes forming a “2‑box deadlock”) pass through, wasting visits.

- **Walk‑cost weight (`α = 0.5`)** is insufficient to discourage cheap, irrelevant pushes, leading to the high‑visit bias observed.

**3. PROPOSED APPROACH – Strategy A (Incremental)**  

We can keep the overall structure (per‑root visited map, BFS walk length, cloning, etc.) and add only a few extra scoring terms and a richer dead‑lock test.

**a. Delta‑distance penalty**  
```python
old_dist = node.state.total_box_distance()
new_dist = next_state.total_box_distance()
delta = new_dist - old_dist          # >0 means we moved a box away
penalty = BETA * max(0, delta)       # BETA ≈ 1.5
```
Add `penalty` to `h`. This forces the planner to avoid pushes that increase the distance unless they bring a big walk‑cost saving.

**b. Target‑placement bonus**  
If the pushed box lands on a target, subtract a small bonus `GAMMA` (≈ 0.8) from `h`. This directly rewards actions that make progress.

**c. Stronger dead‑lock filter**  
Beyond corners, reject a push if after the move a box:
- sits against a wall **and** there is no target in the same row/column beyond that wall, **or**
- forms a 2‑box block against a wall (both boxes locked on the same line with a wall behind them).

These checks are cheap (just inspect walls/targets around the moved box and the other box positions).

**d. Re‑tune walk‑cost weight**  
Raise `α` from `0.5` to `1.0` (or even `1.2`) so that pushes requiring a long walk are less attractive unless they give a clear distance reduction or a target placement.

**e. Scoring order**  
Compute final heuristic:
```
h = new_dist + α * walk_len + BETA * max(0, delta) - GAMMA * on_target
```
Sort by `(h, walk_len)` so that, when equal, the cheaper walk wins.

All changes are local to `default_expansion`; the rest of the MCTS pipeline (selection, simulation, back‑propagation) stays untouched. This incremental augmentation directly addresses the three highest‑impact weaknesses—myopic distance metric, missing dead‑lock pruning, and insufficient walk‑cost bias—without rewriting the whole expansion logic.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions.

    New features:
      * Penalise pushes that increase total box‑Manhattan distance.
      * Reward pushes that place a box on a target.
      * Detect additional deadlocks: boxes against a wall with no target
        reachable in that direction (wall‑line deadlocks).
      * Increase walk‑cost weight (α) to discourage cheap, irrelevant pushes.
    """
    import sys
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
        height, width = state.height, state.width

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
    # Helper 2: Simple corner dead‑lock detection (unchanged)
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Wall‑line dead‑lock detection.
    # A box adjacent to a wall where no target exists beyond that wall
    # in the same line ⇒ the box can never be moved to a target.
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
            # walk beyond the wall in the same direction looking for a target
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a reachable target exists past the wall → not a deadlock
                    return False
                step += 1
            # no target found in this direction → deadlock
            return True
        return False

    # ------------------------------------------------------------------
    # Helper 4: Aggregate deadlock test (corner + wall‑line)
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 5: Per‑root visited dictionary (persistent across calls)
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
    # 1. Current accumulated step count (real cost so far).
    # ------------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Scan untried actions, compute heuristic scores.
    # ------------------------------------------------------------------
    ALPHA = 1.0     # weight for walk distance
    BETA  = 1.5     # penalty for increasing box distance
    GAMMA = 0.8     # bonus for pushing a box onto a target

    old_box_dist = node.state.total_box_distance()
    scored = []                     # list of (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action
        # 2a) walk distance to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # 2b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # 2c) static dead‑lock detection (permanent removal)
        if is_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # 2d) visited pruning – keep better‑g actions
        g_new = cur_steps + walk_len + 1   # macro‑push cost
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # not an improvement, keep for future attempts
            continue

        # 2e) delta distance penalty
        new_box_dist = next_state.total_box_distance()
        delta = max(0, new_box_dist - old_box_dist)   # >0 only when distance worsens

        # 2f) target placement bonus (detect which box moved)
        # original box position is one step ahead of player_pos in the push dir
        drc = [( -1,0),(1,0),(0,-1),(0,1)]  # UP,DOWN,LEFT,RIGHT
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after  = (moved_box_before[0] + dx, moved_box_before[1] + dy)
        on_target = 1 if moved_box_after in next_state.targets else 0

        # 2g) final heuristic
        h = (new_box_dist
             + ALPHA * walk_len
             + BETA * delta
             - GAMMA * on_target)

        scored.append((h, walk_len, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 3. Choose best action (lowest h, break ties with shorter walk)
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
    # 4. Create child node and register it.
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
- Minor bug: `bfs_walk_len` swapped width/height ordering; corrected to use `height = state.height` and `width = state.width`.
- Small inefficiency: duplicate direction list (`drc`) created inside the loop; moved outside loop.
- Clarified dead‑lock helper names and added comments; overall logic remains unchanged.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Incremental heuristic improvements – delta distance penalty, target‑placement bonus, stronger dead‑lock detection, and increased walk‑cost weight.
```python
def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions.

    New features:
      * Penalise pushes that increase total box‑Manhattan distance.
      * Reward pushes that place a box on a target.
      * Detect additional deadlocks: boxes against a wall with no target
        reachable in that direction (wall‑line deadlocks) and keep the
        original corner check.
      * Increase walk‑cost weight (α) to discourage cheap, irrelevant pushes.
    """
    import sys
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
        height = state.height   # rows
        width = state.width     # columns

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
            vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Wall‑line dead‑lock detection.
    # A box adjacent to a wall where no target exists beyond that wall
    # in the same line ⇒ the box can never be moved to a target.
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
                    # A target exists beyond the wall → not a deadlock.
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
    # Helper 5: Per‑root visited dictionary (persistent across calls)
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
    # 1. Current accumulated step count (real cost so far).
    # ------------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Scan untried actions, compute heuristic scores.
    # ------------------------------------------------------------------
    ALPHA = 1.0     # weight for walk distance
    BETA  = 1.5     # penalty for increasing box distance
    GAMMA = 0.8     # bonus for pushing a box onto a target

    old_box_dist = node.state.total_box_distance()
    scored = []                     # (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    # direction vectors for macro‑pushes: UP, DOWN, LEFT, RIGHT
    drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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

        # e) delta distance penalty
        new_box_dist = next_state.total_box_distance()
        delta = max(0, new_box_dist - old_box_dist)   # >0 only when distance worsens

        # f) target placement bonus
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after  = (moved_box_before[0] + dx, moved_box_before[1] + dy)
        on_target = 1 if moved_box_after in next_state.targets else 0

        # g) final heuristic
        h = (new_box_dist
             + ALPHA * walk_len
             + BETA * delta
             - GAMMA * on_target)

        scored.append((h, walk_len, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 3. Choose best action (lowest h, break ties with shorter walk)
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
    # 4. Create child node and register it.
    # ------------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
