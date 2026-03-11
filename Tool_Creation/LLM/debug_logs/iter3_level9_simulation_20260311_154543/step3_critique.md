# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:46:35 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 20.67s |
| Status     | success |
| Tokens     | 9669 |
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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

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

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
    import math

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

    # ---------- Wall‑line dead‑lock detection ----------
    def is_wall_line_deadlocked(st, box_pos):
        walls = st.walls
        targets = st.targets
        h, w = st.height, st.width
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
                    return False            # a target beyond the wall → safe
                step += 1
            return True                     # no target found → dead‑locked
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
        # store cache back on the function object for persistence across calls
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
    # A discount factor < 1 reduces the reward for deep leaves.
    DEPTH_DISCOUNT = 0.97
    depth = state.steps if max_depth == 0 else max_depth - state.steps
    depth_factor = DEPTH_DISCOUNT ** max(depth, 0)

    # ------------------------------------------------------------------
    # 5️⃣ Reward composition.
    #    • Wider spread: exponent = 1.0, lower walk weight.
    #    • Base term inversely related to (h + α·w).
    #    • Progress bonus (γ·f).
    #    • Subtract dead‑square penalty, then apply depth discount.
    # ------------------------------------------------------------------
    ALPHA = 0.30          # walk‑cost weight (smaller → less compression)
    GAMMA = 0.85          # progress bonus weight
    base = 1.0 / (1.0 + (h + ALPHA * w))          # range roughly (0,1)

    reward = base + GAMMA * f
    reward -= DEAD_SQUARE_WEIGHT * dead_square_penalty

    # Apply depth discount (keeps reward positive)
    reward *= depth_factor

    # Clamp to the legal interval.
    reward = max(0.0, min(1.0, reward))

    # ------------------------------------------------------------------
    # 6️⃣ Perfect solution override (covers any numeric imprecision).
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
- `is_wall_line_deadlocked` returns after the first wall neighbor, potentially missing dead‑lock detection in other directions.
- Unused import `math`.
- Minor redundant call to `compute_dead_squares` (cached, but fine).

ACTION: modify
FILE_NAME: simulation_heuristic.py
FUNCTION_NAME: default_simulation
DESCRIPTION: Fix wall‑line dead‑lock detection and clean up unused import.
```python
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
