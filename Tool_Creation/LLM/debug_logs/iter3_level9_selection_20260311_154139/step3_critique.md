# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:42:34 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 20.13s |
| Status     | success |
| Tokens     | 8225 |
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
Description: 
Generated:   2026-03-11T15:35:45.173827
"""

def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    Parameters
    ----------
    root : MCTSNode
        The current node from which the walk starts (may be any node in the tree).
    exploration_weight : float, optional
        Weight applied to the UCB exploration term (default is √2 ≈ 1.41).

    Returns
    -------
    MCTSNode
        The selected node for expansion or simulation.
    """
    import math

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0   # large penalty to discourage dead‑locked states
    TARGET_BONUS    = 10.0       # reward for moving a box onto a target

    def heuristic(state):
        """
        Base heuristic: total Manhattan distance from every box to the
        nearest target (the GameState already provides this via
        `total_box_distance`).
        """
        # The method exists in the public API.
        return state.total_box_distance()

    def is_deadlocked(state):
        """
        Very cheap dead‑lock detection: a box that is not on a target
        and is stuck in a corner formed by two orthogonal walls.
        """
        walls = state.walls
        for box in state.boxes:
            if box in state.targets:
                continue
            x, y = box
            # check four possible corner configurations
            if ((x - 1, y) in walls and (x, y - 1) in walls) or \
               ((x - 1, y) in walls and (x, y + 1) in walls) or \
               ((x + 1, y) in walls and (x, y - 1) in walls) or \
               ((x + 1, y) in walls and (x, y + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # 2️⃣  Tree walk – always pick the child with minimal f‑value
    # ------------------------------------------------------------------
    node = root
    while not node.state.is_terminal():
        # A node is **fully expanded** when there are no untried actions left.
        fully_expanded = (len(node._untried_actions) == 0)

        if not fully_expanded:
            # Let the expansion phase create a fresh child.
            return node

        best_child = None
        best_f = math.inf
        best_q = -math.inf

        # Guard against division‑by‑zero when the parent has never been visited.
        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: heuristic + dead‑lock penalty – target bonus -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: boxes placed on targets relative to the parent.
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- Exploration term (UCB) -----
            if child.visits == 0:
                # Unvisited nodes are given the most attractive (lowest) f.
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore  # lower f is better

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

| # | Symptom (from the trace or aggregate stats) | Why it hurts the search |
|---|---------------------------------------------|--------------------------|
| 1️⃣ | **Selection ignores walk‑cost** – `g = child.state.steps` is used directly, but the *incremental* cost of the *next* macro‑push (walk + push) is not reflected in the f‑value.  Nodes that require a long walk get a huge `g` and are instantly deem‑preferred out, even though the walk may be unavoidable.  This forces the tree to stay in “cheap‑walk” zones and repeatedly expands the same short‑reach moves, leading to the 0 % solve‑rate on most levels. |
| 2️⃣ | **Very weak target incentive** – `TARGET_BONUS = 10` is dwarfed by the typical `h = total_box_distance` (often ≥ 20) and by the massive `DEADLOCK_PENALTY = 1000`.  The algorithm therefore does not prioritize pushes that place a box on a goal; it often explores side‑bars that keep the distance unchanged. |
| 3️⃣ | **Sparse dead‑lock pruning** – the dead‑lock test only catches corner deadlocks.  Many Sokoban dead‑locks are wall‑line or “freeze‑deadlock” patterns.  Because selection never sees them, it wastes visits on branches that can never lead to a solution. |
| 4️⃣ | **No novelty/under‑visited bonus** – the UCB term is *subtracted* from `g+h`.  When a child has been visited a few times, `explore` grows, but the subtraction makes the node *less* attractive (higher f).  This is the opposite of classic UCB (which *adds* an exploration bonus).  Consequently the search quickly collapses onto a single branch and does not explore alternative push orders. |
| 5️⃣ | **Static exploration weight** – a fixed √2 works for many domains but, combined with the subtraction‑style UCB, it is too weak to rescue exploration when `g+h` differences are large (as they are with walk‑cost ignored). |

The trace itself shows that once a few actions receive hundreds of visits (e.g., the first child at Move 1 has 496/500 visits), the algorithm effectively *locks* onto them, even though later moves (e.g., Move 5‑7) needed different pushes to finish the puzzle. The over‑exploitation explains why only a single lucky run solved level 9 while the average remains zero.

---

**2. ROOT CAUSE**

*Selection code* builds `f = g + h - explore`.  
- **Walk‑cost omission**: `g` is the *cumulative* step count, not the *incremental* cost of reaching the child. A child that required a long walk already has a high `g`, so `f` becomes large and the branch is discarded early.  
- **Exploration term sign**: subtracting `explore` makes unvisited nodes artificially *best* (`-inf`), but after the first visit the term *reduces* the node’s attractiveness instead of encouraging further exploration. This contradicts the purpose of UCB and leads to rapid over‑exploitation.  
- **Heuristic scaling**: `TARGET_BONUS` is a constant that does not adapt to the remaining number of boxes or distance magnitudes, so its influence is negligible.  
- **Dead‑lock detection limited to corners** – many dead‑locked configurations are missed, so selection continues down futile branches.

Together these design choices cause the tree to gravitate toward cheap‑walk, high‑visit nodes and ignore promising but costlier pushes.

---

**3. PROPOSED APPROACH (Strategy A – Incremental)**  

We can remedy the dominant problems with a small set of targeted tweaks that keep the overall A*‑guided, UCB‑based selection intact:

1. **Add a *forward walk‑cost* estimate to the heuristic**  
   ```python
   walk_est = min_walk_to_any_push(child.state)   # cheap BFS from child.player to nearest legal push position
   h = h_base + dead_pen - target_bonus + α * walk_est
   ```
   A light weight `α≈0.5` makes pushes that require long walks slightly less attractive, but they are still considered when they dramatically reduce box distance.

2. **Flip the UCB sign (classic formulation)**  
   Replace `f = g + h - explore` with  
   ```python
   f = g + h + explore      # lower f = more promising
   ```
   This restores the intended exploration pressure. Keep the `-inf` shortcut for completely unvisited children.

3. **Scale the target bonus dynamically**  
   ```python
   TARGET_BONUS = 5.0 * (state.num_targets - parent_state.boxes_on_targets())
   ```
   The bonus grows when few boxes remain, making the final pushes decisive.

4. **Enrich dead‑lock detection** – embed the *wall‑line* test already present in the expansion phase (the `is_wall_line_deadlocked` helper).  It is cheap (O(#boxes)) and dramatically reduces wasted visits.

5. **Introduce a tiny novelty term** proportional to the inverse of the child’s visit count:  
   ```python
   novelty = β / (child.visits + 1)   # β≈0.2
   f = g + h + explore - novelty      # novelty lowers f, encouraging under‑visited branches
   ```
   This works together with the corrected UCB to keep the search breadth.

All five modifications are computationally cheap (a few BFS distance look‑ups, a constant‑time penalty, and reuse of existing dead‑lock logic). They directly address the three highest‑impact weaknesses (walk‑cost omission, wrong exploration sign, and weak target incentive) while preserving the overall A*‑guided selection framework that already proved capable of solving a level when luck aligned. Incremental testing can tune `α`, `β`, and the dynamic `TARGET_BONUS` without a full rewrite, adhering to the 70 % rule.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
                    return False          # a target beyond the wall -> not dead‑locked
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
        # pre‑compute reachable distances
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

        # find minimal distance to a push position
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

        # parent info for target‑bonus scaling
        parent_on_target = node.state.boxes_on_targets()
        remaining_targets = node.state.num_targets - parent_on_target
        dynamic_target_bonus = BASE_TARGET_BONUS * max(remaining_targets, 1)

        # iterate over children together with the action that created them
        for action, child in node.children.items():
            # ----- g: cumulative steps taken so far (already includes the walk for this action) -----
            g = getattr(child.state, "steps", 0)

            # ----- h components -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # target bonus: reward for *new* boxes placed on targets
            child_on_target = child.state.boxes_on_targets()
            target_diff = child_on_target - parent_on_target
            target_bonus = dynamic_target_bonus * target_diff

            # forward walk‑cost estimate from child to its next push
            walk_est = min_walk_to_any_push(child.state)

            h = h_base + dead_pen - target_bonus + WALK_WEIGHT * walk_est

            # ----- Exploration term (classic UCB: add, not subtract) -----
            if child.visits == 0:
                explore = math.inf  # ensures unvisited nodes are chosen first
                f = -math.inf        # lowest possible f forces selection
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                # novelty term encourages under‑visited nodes
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
- No issues found

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Finalized improved selection heuristic
```python
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
