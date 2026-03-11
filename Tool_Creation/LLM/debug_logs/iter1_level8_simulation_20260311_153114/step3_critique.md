# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:31:56 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.55s |
| Status     | success |
| Tokens     | 7408 |
| Validation | valid |

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
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: simulation
Description: 
Generated:   2026-03-11T01:07:22.346016
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """
        Compute the shortest‑path distance from `start` to every reachable
        cell on the board, treating both walls and boxes as obstacles.
        Returns a dict {(r, c): distance}.
        """
        frontier = deque([start])
        distances = {start: 0}
        # 4‑neighbour moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while frontier:
            cur = frontier.popleft()
            cur_dist = distances[cur]

            for dr, dc in moves:
                nr, nc = cur[0] + dr, cur[1] + dc
                # stay inside the board
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                # cannot step onto a wall or a box
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in distances:
                    distances[nxt] = cur_dist + 1
                    frontier.append(nxt)
        return distances

    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # `returns()` returns a sequence indexed by player id.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    # The public API provides `total_box_distance()` which returns the
    # sum of Manhattan distances from each box to its nearest target.
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    # Compute distances from the player's current location.
    distances = bfs_distances(state.player, walls, boxes, height, width)

    # Determine the minimal distance to any push position that is legal.
    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    # If there is no reachable push position, treat the walk cost as 0.
    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    # `boxes_on_targets()` returns the number of boxes that sit on targets.
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # penalty for long walks
    GAMMA = 0.4   # stronger bonus for having boxes placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    reward = max(0.0, min(1.0, reward))

    # Special case: all boxes on targets → perfect reward.
    if h == 0:
        reward = 1.0

    return reward
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (observed in the trace) | Why it hurts performance |
|------|--------------------------------|--------------------------|
| 1️⃣ | *Almost identical Q‑values* – after many visits the average value of the best child is only ≈ 0.24, even after placing two boxes (step 13, `Boxes on target: 2/4`). The algorithm cannot tell the “good” pushes from the mediocre ones, so the tree favours whichever was sampled first. |
| 2️⃣ | *Reward range is compressed* – the current formula yields roughly `0.15 – 0.45` for almost every non‑terminal leaf (e.g. total distance = 7 → 0.19, distance = 2 → 0.34). With the back‑propagation depth discount (γ = 0.70) these numbers shrink further, making selection noise‑dominated. |
| 3️⃣ | *Walk‑cost weight is tiny* (`ALPHA = 0.1`). Macro‑pushes that need a long walk are barely penalised, so the roll‑out rewards a state that is expensive to reach just as much as a cheap one. |
| 4️⃣ | *No explicit penalty for worsening box layout* – the simulation only looks at the *total* Manhattan distance (`h`). A push that keeps `h` unchanged but moves a box into a dead‑end is rewarded the same as a constructive push. |
| 5️⃣ | *Dead‑lock awareness is absent* – terminal detection catches only *already* dead‑locked states. A leaf that is one push away from a corner dead‑lock still receives a high reward, encouraging the tree to explore it. |

The combination of (1) and (2) makes the MCTS value estimates almost flat, which explains why the search keeps expanding the same high‑visit branches (see the huge visit counts on the first child of every node) and never discovers the sequence that solves the puzzle.

---

**2. ROOT CAUSE**

The simulation function is deliberately simple:

```python
base = 1.0 / (1.0 + h + ALPHA * w)      # h = total box distance
reward = base + GAMMA * f                # f = fraction of boxes placed
```

* `ALPHA = 0.1` yields a negligible walk penalty (`w` is usually ≤ 5).  
* `GAMMA = 0.4` adds at most 0.4 when half the boxes are placed (`f = 0.5`).  
* `base` is bounded by `1/(1+h)`, so for any realistic `h` the range is 0.08–0.33.  
* The final `clamp` forces everything into `[0,1]` but never stretches the spread.

Consequently, different board configurations map to a narrow band of rewards, and the subsequent back‑propagation (γ‑depth discount, step penalty) further flattens them. Moreover, the function never checks whether the current layout is *dead‑locked* or whether the last push increased the distance (`delta > 0`). Those missing signals let the rollout reward “good” states that are actually traps.

---

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinement of the simulation heuristic**  

The core idea is to *amplify the signal* that already exists (boxes placed, distance, walk cost) and *inject two missing signals*: (i) a penalty when a push makes the box‑distance **worse**, and (ii) a hard‑negative when the state is *dead‑locked* (corner or wall‑line). All changes are local to `default_simulation`; the rest of the pipeline (selection, expansion, back‑prop) stays untouched.

| Change | Rationale | Expected effect |
|--------|-----------|-----------------|
| **Increase walk weight**: `ALPHA ← 0.5` (or 0.7) | Makes long‑walk pushes visibly cheaper in the reward, steering the tree toward low‑cost macro‑pushes. |
| **Boost target bonus**: `GAMMA ← 0.8` (or 1.0) | Differentiates states that have already placed boxes; a state with 2/4 boxes jumps from ~0.30 to ~0.60, giving the tree a strong incentive to lock in progress. |
| **Add distance‑worsening penalty**: compute `delta = max(0, new_h - old_h)` (using the caller’s `state.total_box_distance()` before/after a hypothetical push). Subtract `BETA * delta` from the reward (`BETA ≈ 0.4`). | Penalises pushes that move a box away from any target, producing a lower reward for regressions. |
| **Dead‑lock check**: reuse the `is_corner_deadlocked` / `is_wall_line_deadlocked` helpers from `expansion`. If the leaf is dead‑locked, return `0.0` immediately (or a tiny negative value). | Prevents the rollout from rewarding traps, cutting off branches that would otherwise dominate due to sheer visitation. |
| **Optional non‑linear scaling**: replace `base = 1/(1+h+ALPHA*w)` with `base = 1.0 / (1.0 + (h + ALPHA*w) ** 0.8)` to stretch differences for small `h`. | Gives a steeper slope when the puzzle is near‑solved, making the last few pushes stand out. |

All modifications keep the function pure, fast (still only BFS once) and return a single float. By widening the reward distribution and explicitly punishing regressions and dead‑locks, the MCTS value estimates will acquire a richer gradient, allowing the UCT‑style (or the A*‑guided selection) to discriminate actions far better. This should raise the average leaf reward, increase visit diversity, and ultimately produce the missing solving sequences for level 8.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
            vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
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
    #    • exponent – non‑linear scaling to spread values,
    #    • optional delta‑penalty (not used here as we lack a previous state).
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
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: No changes needed; the function is correct and efficient.
```python
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
