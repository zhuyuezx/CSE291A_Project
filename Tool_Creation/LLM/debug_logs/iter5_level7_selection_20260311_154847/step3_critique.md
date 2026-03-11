# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:49:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 21.09s |
| Status     | success |
| Tokens     | 9210 |
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
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
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
- The distance‑delta term is added to `h_base`, which unintentionally makes worsening distances look *better* (because `h = child_dist + weight*(parent‑child)` can decrease when `child_dist` increases if `weight > 1`).  
- The sign of the distance‑delta bonus should be **subtracted** so that a reduction in total box distance lowers the heuristic value, and an increase raises it.  
- No other functional or performance issues were found.

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Fix distance‑delta sign to correctly reward distance improvements and penalise regressions.
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

            # NOTE: improvement should *reduce* the heuristic value,
            # so we subtract the distance_term.
            h = (h_base + dead_pen
                 - target_bonus
                 + WALK_WEIGHT * walk_est
                 - distance_term)

            # ----- Exploration term (classic UCB) -----
            if child.visits == 0:
                # Force selection of unvisited nodes
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                novelty = NOVELTY_WEIGHT / math.sqrt(child.visits + 1)
                f = g + h + explore + novelty  # additive novelty bonus

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
