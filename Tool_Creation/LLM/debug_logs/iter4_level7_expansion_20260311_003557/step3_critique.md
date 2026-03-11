# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:36:58 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 28.92s |
| Status     | success |
| Tokens     | 10254 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: expansion).

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
Description: Fixed the undefined `_ag` reference by introducing module‑level globals (`_EXPANSION_STATE`) that store the current root key and visited table, added a local `node_depth` helper, and replaced heuristic calls with the public `total_box_distance` method. All other logic (corner dead‑lock pruning, monotonicity, walk‑cost computation, A*‑style scoring) remains unchanged.
Generated:   2026-03-11T00:09:03.499832
"""

# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine.
# This replaces the previous undefined `_ag` global that was used for:
#   * root‑key tracking & visited‑table reset
#   * node depth lookup
#   * heuristic functions
# --------------------------------------------------------------------------- #
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}

def default_expansion(node):
    """
    Expand the best untried action using A*‑style scoring while respecting
    actual walk costs and simple dead‑lock / monotonicity pruning.

    The implementation mirrors the original strategy but no longer relies on
    the undefined `_ag` object.  All required information is obtained from the
    public `GameState` API or from the module‑level `_EXPANSION_STATE`.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    # Find the root of the current tree
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        # New search – start with a clean visited dict
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    from collections import deque

    def bfs_distance(start, goal, walls, boxes):
        """Shortest walk distance (ignoring pushes) using Manhattan BFS."""
        if start == goal:
            return 0
        q = deque([start])
        visited = {start}
        dist = {start: 0}
        while q:
            cur = q.popleft()
            d = dist[cur]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in walls or nxt in boxes or nxt in visited:
                    continue
                if nxt == goal:
                    return d + 1
                visited.add(nxt)
                dist[nxt] = d + 1
                q.append(nxt)
        # Unreachable – assign a large penalty
        return 10 ** 6

    def simple_corner_deadlock(state):
        """Detect obvious corner deadlocks (box stuck in a wall corner)."""
        walls = state.walls
        targets = state.targets
        for b in state.boxes:
            if b in targets:
                continue
            r, c = b
            if ((r - 1, c) in walls and (r, c - 1) in walls) or \
               ((r - 1, c) in walls and (r, c + 1) in walls) or \
               ((r + 1, c) in walls and (r, c - 1) in walls) or \
               ((r + 1, c) in walls and (r, c + 1) in walls):
                return True
        return False

    def node_depth(n):
        """Number of pushes from the root to this node."""
        d = 0
        while n.parent is not None:
            d += 1          # each edge corresponds to one push
            n = n.parent
        return d

    # ------------------------------------------------------------------ #
    # 1. Parent information (depth & heuristic)
    # ------------------------------------------------------------------ #
    parent_depth = node_depth(node)

    # Heuristic: use the public `total_box_distance` method (sum of Manhattan
    # distances from boxes to the nearest targets).  It is available on every
    # GameState object according to the provided API.
    try:
        parent_h = node.state.total_box_distance()
    except Exception:
        # Fallback – if for some reason the method is missing, use 0.
        parent_h = 0

    visited = _EXPANSION_STATE["visited"]

    scored = []   # (f, walk, action, next_state, key, total_g)
    pruned = []   # actions we permanently discard from the node's pool

    # ------------------------------------------------------------------ #
    # 2. Examine every untried action
    # ------------------------------------------------------------------ #
    for action in list(node._untried_actions):
        # -------------------------------------------------------------- #
        # a) Generate successor state
        # -------------------------------------------------------------- #
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------- #
        # b) Simple corner deadlock pruning
        # -------------------------------------------------------------- #
        if simple_corner_deadlock(next_state):
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # c) Existing deadlock check (if the state implements it)
        # -------------------------------------------------------------- #
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # d) Monotonicity: discard actions that increase the heuristic
        # -------------------------------------------------------------- #
        try:
            next_h = next_state.total_box_distance()
        except Exception:
            next_h = 0
        if next_h > parent_h:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # e) Compute walk distance from current player to the required push cell
        # -------------------------------------------------------------- #
        player_start = node.state.player
        push_pos = action[0]                     # cell where player must stand
        walk = bfs_distance(player_start, push_pos,
                            node.state.walls, node.state.boxes)

        # -------------------------------------------------------------- #
        # f) Total g‑score for visited table (push count + walk + 1)
        # -------------------------------------------------------------- #
        total_g = parent_depth + walk + 1

        # -------------------------------------------------------------- #
        # g) Visited‑table pruning
        # -------------------------------------------------------------- #
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # h) Score by full A* f = g + h
        # -------------------------------------------------------------- #
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose the best remaining action (minimal f, then minimal walk)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        # Remove the selected action from the untried list
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ------------------------------------------------------------------
        # Fallback: expand the first remaining untried action (if any)
        # ------------------------------------------------------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = bfs_distance(node.state.player, chosen_action[0],
                                node.state.walls, node.state.boxes)
            total_g = parent_depth + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create the child node, register it and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
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

| Rank | Symptom (trace) | Evidence |
|------|-----------------|----------|
| **1** | **Over‑aggressive monotonicity pruning** – after the third push the algorithm discards every action that makes *total_box_distance* rise, even if the increase is only temporary. | In Move 2 the children `((1,3),3)` and `((1,5),2)` have a fairly high average (≈ 0.24), but they are later removed because their successor states raise the distance from 4 → 5 before a later push can bring it down. By Move 4 the node has **only one child** left, all other possibilities eliminated, and the planner can no longer reposition the remaining two boxes. |
| **2** | **Depth‑only g‑score** – `total_g = parent_depth + walk + 1` counts only pushes, ignoring the walk steps already taken in ancestor nodes. This under‑estimates the true cost of a long‑walk push, causing the visited‑table to treat a cheap‑looking state as already explored and prune it. | Walk distances of 5‑10 steps appear early (e.g., `walk = 5` for the first action) but the stored `total_g` only adds the current walk, not the accumulated walk from the root; later states with the same `total_g` are discarded even though they are far more expensive. |
| **3** | **Very simple dead‑lock detection** – only corner deadlocks are filtered. More subtle patterns (e.g., a box against a wall on a line without any target) survive, filling the tree with futile branches and wasting the limited 500 iterations. | No evidence of obvious corner dead‑locks in the trace, yet the algorithm spends most visits on low‑value children (`avg≈0.003`). |
| **4** | **Heuristic only box‑distance** – `total_box_distance` is used both for pruning and for f‑score. It is blind to player‑reachability and to the fact that a box might need to be pulled back later, so it often mis‑ranks useful pushes. | The promising actions with avg ≈ 0.24 are those that bring a box nearer to its target, but the algorithm later throws them away because the heuristic spikes after a necessary detour. |

The dominant problem is **(1)** – the monotonicity filter collapses the branching factor after a few pushes, preventing the planner from exploring necessary “back‑track” moves and from reaching the final solution.

---

**2. ROOT CAUSE**

The expansion routine contains:

```python
next_h = next_state.total_box_distance()
if next_h > parent_h:
    pruned.append(action)
    continue
```

`total_box_distance` is **not monotone** in Sokoban; moving a box away from its nearest target can be required to clear a corridor or to position another box. The strict > test therefore throws away exactly the actions needed for multi‑box coordination. Because the pruning is applied *before* any other scoring, the child list quickly shrinks to a single line of pushes, as seen in Move 4 where only one child remains and the search stalls.

The g‑score used for the visited table also ignores accumulated walk cost, which together with monotonicity makes the visited‑pruning overly aggressive.

---

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinements (recommended).**  
The core architecture (A*‑style f = g + h, dead‑lock filter, BFS walk cost) works and should be retained. We only need to soften the pruning and improve cost estimation.

1. **Relax monotonicity**  
   * Replace the hard `>` test with a tolerance: `if next_h > parent_h + ε:` where `ε` is a small integer (e.g., 1 or 2).  
   * Optionally allow a limited number `k` of non‑monotonic expansions per node (track a counter in the node). This preserves exploratory moves while still discouraging large regressions.

2. **Account for accumulated walk cost in g**  
   * Compute `total_g = node.state.steps + walk + 1` (or store cumulative steps in the node).  
   * Use this `total_g` for the visited‑table comparison, so states reached via long walks are not mistakenly considered “already seen cheaply”.

3. **Weighted heuristic**  
   * Blend box‑distance with a reachability term: `h = α * total_box_distance + β * player_unreachable_boxes`, where `player_unreachable_boxes` is the count of boxes not currently reachable by the player. This penalises pushes that isolate boxes and gives the f‑score a better sense of feasibility.

4. **Enhanced dead‑lock patterns (light)**  
   * Add a quick “wall‑line without target” check: if a box lies on a straight wall segment and no target exists on that line in the same direction, treat it as dead‑locked. This adds negligible overhead but removes obvious futile branches.

These changes are small, keep the existing A*‑inspired expansion logic, and directly address the most damaging weakness (monotonicity pruning). They should increase the branching factor, allow the planner to consider necessary detours, and improve the quality of the f‑score, leading to higher solve rates on level 7 without a full redesign.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
Improved expansion for Sokoban‑Macro MCTS.

Key upgrades:
  • **Relaxed monotonicity** – allow a small increase (ε) in the
    total‑box‑distance heuristic instead of discarding every non‑monotone
    move. This keeps necessary back‑track pushes.
  • **Accurate g‑score** – use the state's cumulative step count
    (`state.steps`) plus the current walk cost, so the visited‑pruning
    table respects the true cost of long walks.
  • **Weighted heuristic** – combine box‑distance with a penalty for
    boxes that are currently unreachable by the player, encouraging
    pushes that keep boxes reachable.
  • **Extra dead‑lock check** – simple wall‑line dead‑locks (box on a wall
    with no target on that line) are filtered out early.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple, List, Set

# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine (unchanged interface).
# --------------------------------------------------------------------------- #
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}


def _bfs_distance(start: Tuple[int, int],
                  goal: Tuple[int, int],
                  walls: Set[Tuple[int, int]],
                  boxes: Set[Tuple[int, int]]) -> int:
    """Shortest walk distance ignoring pushes (Manhattan BFS)."""
    if start == goal:
        return 0
    q = deque([start])
    visited = {start}
    dist = {start: 0}
    while q:
        cur = q.popleft()
        d = dist[cur]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            if nxt == goal:
                return d + 1
            visited.add(nxt)
            dist[nxt] = d + 1
            q.append(nxt)
    # Unreachable – assign a large penalty
    return 10 ** 6


def _simple_corner_deadlock(state) -> bool:
    """Detect obvious corner deadlocks (box stuck in a wall corner)."""
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in targets:
            continue
        r, c = b
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def _wall_line_deadlock(state) -> bool:
    """
    Detect simple wall‑line deadlocks:
      * box on a wall (left/right/top/bottom)
      * no target exists on the same line beyond the wall direction
    This catches cases like a box pressed against a wall where it can
    never be moved onto a target.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in state.targets:
            continue
        r, c = b
        # Left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Top wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
        # Bottom wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
    return False


def _reachable_boxes(state) -> int:
    """Number of boxes not reachable by the player (ignoring pushes)."""
    walls = state.walls
    boxes = state.boxes
    start = state.player
    q = deque([start])
    visited = {start}
    while q:
        cur = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)
    # boxes outside the reachable set are “unreachable”
    return sum(1 for b in boxes if b not in visited)


def _heuristic(state, alpha: float = 0.7, beta: float = 0.3) -> float:
    """
    Weighted heuristic:
        h = α * total_box_distance  +  β * unreachable_boxes
    Both components are integer counts; the weights keep them on a comparable scale.
    """
    box_dist = state.total_box_distance()
    unreachable = _reachable_boxes(state)
    return alpha * box_dist + beta * unreachable


def default_expansion(node):
    """
    Expand the best untried action using an A*‑style f = g + h scoring,
    with improved pruning and cost estimation.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    visited = _EXPANSION_STATE["visited"]

    # ------------------------------------------------------------------ #
    # 1. Gather parent information
    # ------------------------------------------------------------------ #
    parent_steps = getattr(node.state, "steps", 0)          # cumulative steps so far
    parent_h = _heuristic(node.state)                      # weighted heuristic

    # ------------------------------------------------------------------ #
    # 2. Scan untried actions
    # ------------------------------------------------------------------ #
    scored = []          # tuples of (f, walk, action, next_state, key, total_g)
    pruned = []          # actions permanently removed

    # tolerance for non‑monotone moves (ε)
    EPSILON = 2

    for action in list(node._untried_actions):
        # ---- a) generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- b) dead‑lock checks ----------------------------------------
        if _simple_corner_deadlock(next_state) or _wall_line_deadlock(next_state):
            pruned.append(action)
            continue

        # ---- c) heuristic monotonicity (relaxed) -----------------------
        next_h = _heuristic(next_state)
        if next_h > parent_h + EPSILON:
            pruned.append(action)
            continue

        # ---- d) walk distance to push position -------------------------
        player_start = node.state.player
        push_pos = action[0]                 # cell where player must stand
        walk = _bfs_distance(player_start, push_pos,
                             node.state.walls, node.state.boxes)

        # ---- e) g‑score: accumulate real steps (push count + walk) -----
        total_g = parent_steps + walk + 1    # +1 for the push itself

        # ---- f) visited‑table pruning ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # ---- g) compute f = g + h ---------------------------------------
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose best action (minimal f, then minimal walk, then deterministic)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ---------- fallback when everything was pruned -----------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = _bfs_distance(node.state.player, chosen_action[0],
                                 node.state.walls, node.state.boxes)
            total_g = parent_steps + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create child node, register, and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
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
- No issues found – the function correctly follows the intended incremental improvements, uses the public GameState API, and avoids undefined references or performance‑critical overhead.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
```python
"""
Improved expansion for Sokoban‑Macro MCTS.

Key upgrades:
  • **Relaxed monotonicity** – allow a small increase (ε) in the
    total‑box‑distance heuristic instead of discarding every non‑monotone
    move. This keeps necessary back‑track pushes.
  • **Accurate g‑score** – use the state's cumulative step count
    (`state.steps`) plus the current walk cost, so the visited‑pruning
    table respects the true cost of long walks.
  • **Weighted heuristic** – combine box‑distance with a penalty for
    boxes that are currently unreachable by the player, encouraging
    pushes that keep boxes reachable.
  • **Extra dead‑lock check** – simple wall‑line dead‑locks (box on a wall
    with no target on that line) are filtered out early.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple, Set

# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine (unchanged interface).
# --------------------------------------------------------------------------- #
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}


def _bfs_distance(start: Tuple[int, int],
                  goal: Tuple[int, int],
                  walls: Set[Tuple[int, int]],
                  boxes: Set[Tuple[int, int]]) -> int:
    """Shortest walk distance ignoring pushes (Manhattan BFS)."""
    if start == goal:
        return 0
    q = deque([start])
    visited = {start}
    dist = {start: 0}
    while q:
        cur = q.popleft()
        d = dist[cur]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            if nxt == goal:
                return d + 1
            visited.add(nxt)
            dist[nxt] = d + 1
            q.append(nxt)
    # Unreachable – assign a large penalty
    return 10 ** 6


def _simple_corner_deadlock(state) -> bool:
    """Detect obvious corner deadlocks (box stuck in a wall corner)."""
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in targets:
            continue
        r, c = b
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def _wall_line_deadlock(state) -> bool:
    """
    Detect simple wall‑line deadlocks:
      * box on a wall (left/right/top/bottom)
      * no target exists on the same line beyond the wall direction
    This catches cases like a box pressed against a wall where it can
    never be moved onto a target.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in state.targets:
            continue
        r, c = b
        # Left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Top wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
        # Bottom wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
    return False


def _reachable_boxes(state) -> int:
    """Number of boxes not reachable by the player (ignoring pushes)."""
    walls = state.walls
    boxes = state.boxes
    start = state.player
    q = deque([start])
    visited = {start}
    while q:
        cur = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)
    # boxes outside the reachable set are “unreachable”
    return sum(1 for b in boxes if b not in visited)


def _heuristic(state,
               alpha: float = 0.7,
               beta: float = 0.3) -> float:
    """
    Weighted heuristic:
        h = α * total_box_distance  +  β * unreachable_boxes
    Both components are integer counts; the weights keep them on a comparable scale.
    """
    box_dist = state.total_box_distance()
    unreachable = _reachable_boxes(state)
    return alpha * box_dist + beta * unreachable


def default_expansion(node):
    """
    Expand the best untried action using an A*‑style f = g + h scoring,
    with improved pruning and cost estimation.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    visited = _EXPANSION_STATE["visited"]

    # ------------------------------------------------------------------ #
    # 1. Gather parent information
    # ------------------------------------------------------------------ #
    parent_steps = getattr(node.state, "steps", 0)          # cumulative steps so far
    parent_h = _heuristic(node.state)                      # weighted heuristic

    # ------------------------------------------------------------------ #
    # 2. Scan untried actions
    # ------------------------------------------------------------------ #
    scored = []          # tuples of (f, walk, action, next_state, key, total_g)
    pruned = []          # actions permanently removed

    EPSILON = 2          # tolerance for non‑monotone moves

    for action in list(node._untried_actions):
        # ---- a) generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- b) dead‑lock checks ----------------------------------------
        if _simple_corner_deadlock(next_state) or _wall_line_deadlock(next_state):
            pruned.append(action)
            continue

        # ---- c) heuristic monotonicity (relaxed) -----------------------
        next_h = _heuristic(next_state)
        if next_h > parent_h + EPSILON:
            pruned.append(action)
            continue

        # ---- d) walk distance to push position -------------------------
        player_start = node.state.player
        push_pos = action[0]                 # cell where player must stand
        walk = _bfs_distance(player_start, push_pos,
                             node.state.walls, node.state.boxes)

        # ---- e) g‑score: accumulate real steps (push count + walk) -----
        total_g = parent_steps + walk + 1    # +1 for the push itself

        # ---- f) visited‑table pruning ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # ---- g) compute f = g + h ---------------------------------------
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose best action (minimal f, then minimal walk, then deterministic)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ---------- fallback when everything was pruned -----------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = _bfs_distance(node.state.player, chosen_action[0],
                                 node.state.walls, node.state.boxes)
            total_g = parent_steps + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create child node, register, and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
