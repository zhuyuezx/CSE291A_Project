# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:59:58 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 13.53s |
| Status     | success |
| Tokens     | 7993 |
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
A*-guided MCTS expansion.

Mirrors A*'s inner loop:
    for action in legal_actions(state):
        if deadlocked(next): skip
        if visited[key] <= g_new: skip      ← visited check
        visited[key] = g_new                ← record best g
        heappush(pq, (g+h, ...))            ← pick lowest-h action

Permanently removes pruned actions (deadlock / already-visited-better) from
_untried_actions so they are never reconsidered. Among surviving actions
picks the one with the lowest h (smallest distance-to-goal).
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_expansion(node):
    """
    Expand the best untried action using A* visited + heuristic filtering.

    Steps:
      0. If this tree has a new root (new MCTS move), reset shared visited.
      1. Compute g_new = depth(node) + 1.
      2. For every untried action: deadlock → prune; visited[key] <= g_new → prune;
         else score by h_sokoban(next_state).
      3. Pick action with lowest h; write visited[key] = g_new.
      4. Fallback: if all pruned, expand first remaining untried action.
    """
    # Reset A* visited at the start of each MCTS search (each move). Otherwise
    # we prune states in the current tree using g-scores from previous trees.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    g_new   = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    scored  = []   # (h, action, next_state, key)
    pruned  = []   # actions to remove permanently

    for action in list(node._untried_actions):
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ── A*: deadlock prune ────────────────────────────────────────
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # ── A*: visited check ─────────────────────────────────────────
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            pruned.append(action)
            continue

        # Use box-only heuristic to match sokoban_astar.ipynb A* ordering
        scored.append((_ag.h_sokoban_box_only(next_state), action, next_state, key))

    # Remove all permanently-pruned actions
    for a in pruned:
        node._untried_actions.remove(a)

    if scored:
        # Pick action with min h; break ties by action order to match A* determinism
        scored.sort(key=lambda x: (x[0], x[1]))
        _, chosen_action, chosen_state, chosen_key = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # All actions pruned — fallback: expand first remaining action
        # (avoids stall when entire frontier is already visited)
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state  = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key    = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # Nothing left — return existing child (safety net)
            if node.children:
                return next(iter(node.children.values()))
            # Absolute fallback: return the node itself (terminal-like)
            return node

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

| Rank | Symptom (evidence from trace) | Why it hurts the solver |
|------|------------------------------|--------------------------|
| 1️⃣ | *Many pushes are explored that do not bring any box closer to a target.*  In the trace the algorithm repeatedly expands actions whose **h‑value stays the same** (e.g. moves 1‑2 keep total distance = 5→4, but the walk cost of the macro‑push is hidden).  This wastes visits and prevents the tree from focusing on truly promising pushes. |
| 2️⃣ | *Only a single dead‑lock test (`_is_deadlocked`) is consulted.*  Simple corner‑dead‑locks (box pushed into a non‑target corner) are not caught, so large sub‑trees are created that can never lead to a solution.  The trace shows many low‑value children (average ≈ 0.14) that are never useful. |
| 3️⃣ | *Visited pruning uses node **depth** (`node_depth`) as the g‑score.*  Depth is merely “one push deeper”, while the real cost is **walk‑steps + 1**.  An action that requires a long walk gets the same g as a cheap one, leading to premature pruning of cheaper‑cost paths and permanent removal of the action from `_untried_actions`. |
| 4️⃣ | *Scoring is pure box‑only Manhattan (`h_sokoban_box_only`).*  It ignores the *direction* of change (increase vs. decrease) and the player‑walk distance, so the expansion order can be “nearest‑target” but far‑away to reach.  This creates the high‑visit bias seen in the first child of every node. |

**2. ROOT CAUSE**  

The expansion routine:

```python
g_new = _ag.node_depth(node) + 1          # ← uses depth, not real step cost
...
scored.append((_ag.h_sokoban_box_only(next_state), ...))
```

* No walk‑cost is considered when computing `g_new` or the ranking score.  
* The dead‑lock predicate is only `next_state._is_deadlocked()`, which in the current code detects *only* fully blocked configurations, not the classic corner‑dead‑lock.  
* After an action is pruned because `visited[key] <= g_new`, it is **removed permanently** from `_untried_actions`.  Since `g_new` is depth‑based, a later cheaper path can never resurrect the action.  

Together these choices make the expansion too myopic and overly aggressive in pruning, resulting in many wasted visits and low solve rates.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement** (keep the existing A*‑guided framework, add smarter filters and a cost‑aware score).

| Change | Where & How (conceptual) | Expected effect |
|--------|--------------------------|-----------------|
| **a. Walk‑cost aware g** | Compute `g_new = node.state.steps + walk_cost(action) + 1` (the real steps taken to reach `next_state`).  Store this in the shared visited table instead of depth. | Prunes only when a state has already been reached with *equal or lower* true step cost; cheap pushes are no longer discarded. |
| **b. Delta‑distance heuristic** | For each candidate compute `Δh = h_sokoban_box_only(next) - h_sokoban_box_only(current)`.  Score actions with a weighted sum `score = α * walk_cost + β * (h_next)`.  Prefer actions that **reduce** the box‑only distance (negative Δh) and heavily penalise those that increase it. | Pushes that move a box away from any target are de‑prioritised, focusing the tree on progress‑making moves. |
| **c. Simple corner dead‑lock filter** | Before scoring, check the box that will occupy `push_target = box_pos + delta`.  If `push_target` is not a target **and** both orthogonal neighbours are walls, mark the action as dead‑locked and add to `pruned`. | Immediately removes the huge class of hopeless pushes that currently inflate the child list. |
| **d. Preserve pruned‑by‑visited actions** | Instead of `node._untried_actions.remove(a)` for visited‑based pruning, keep the action but skip it in this expansion pass.  When a later node reaches the same state with a *lower* real `g`, the action will be reconsidered. | Guarantees that a previously discarded action can be resurrected if a cheaper path appears, mitigating the “once‑pruned‑forever” problem caused by depth‑based g. |
| **e. Tie‑break on Δh** | When multiple actions share the same `score`, break ties by smallest `Δh` (prefer greater distance reduction). | Gives deterministic, progress‑biased ordering without extra randomness. |

These modifications are small, isolated, and can be added to the existing `default_expansion` without rewriting the whole MCTS pipeline.  All proven components (A*‑guided selection, simulation, back‑propagation) remain untouched, preserving the overall A*‑style convergence while markedly improving the quality of the frontier that MCTS expands.  

**Why not a full restructure?**  
The current system already integrates A* visited‑pruning, depth‑first expansion, and heuristic‑based simulation—all of which produce the fast 7‑step solution shown in the trace.  The failures stem from missing cost information and coarse dead‑lock detection, not from a fundamentally broken architecture.  Therefore an incremental set of targeted enhancements (Strategy A) is sufficient and far less risky than redesigning the expansion phase from scratch.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS expansion – incremental improvement.

Enhancements:
  • Use real walk‑cost + push (g) instead of pure depth.
  • Simple corner dead‑lock detection (box in non‑target corner).
  • Score actions by a combination of walk cost, box‑only heuristic and
    a penalty for increasing the heuristic (Δh).
  • Do **not** permanently discard actions that fail the visited‑check;
    they remain in `_untried_actions` and may be revisited if a cheaper
    path appears later.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from collections import deque
from typing import Set, Tuple, List

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def _bfs_distance(start: Tuple[int, int],
                  goal: Tuple[int, int],
                  walls: Set[Tuple[int, int]],
                  boxes: Set[Tuple[int, int]]) -> int:
    """
    Return the shortest walk distance from start to goal avoiding walls
    and boxes.  If unreachable, return a large number (treated as dead‑lock).
    """
    if start == goal:
        return 0
    queue = deque([start])
    visited = {start}
    dist = 0
    while queue:
        for _ in range(len(queue)):
            cur = queue.popleft()
            if cur == goal:
                return dist
            r, c = cur
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nxt in walls or nxt in boxes or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        dist += 1
    # unreachable
    return sys.maxsize


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """
    Detect classic corner dead‑lock: a box in a non‑target cell that is
    adjacent to walls on two orthogonal sides.
    """
    if pos in targets:
        return False
    r, c = pos
    up = (r - 1, c) in walls
    down = (r + 1, c) in walls
    left = (r, c - 1) in walls
    right = (r, c + 1) in walls
    # corner if (up or down) AND (left or right)
    return (up or down) and (left or right)


def default_expansion(node):
    """
    Expand the most promising untried action.

    Improvements over the baseline:
      * g‑score = real steps taken (walk + push) instead of tree depth.
      * Corner dead‑lock pruning.
      * Scoring = walk_cost + h_box_only + penalty for Δh > 0.
      * Actions that fail the visited‑check are kept for possible later use.
    """
    # ---- Reset visited info for a new MCTS search (once per root) ----
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    visited = _ag.get_visited()

    # Pre‑compute current heuristic for Δh calculations
    h_curr = _ag.h_sokoban_box_only(node.state)

    # Containers
    scored: List[Tuple[float, Tuple[Tuple[int, int], int],
                       object, str, int]] = []   # (score, action, next_state, key, g_new)
    deadlocked_actions = []  # permanently remove

    # -----------------------------------------------------------------
    # Examine each untried action
    # -----------------------------------------------------------------
    for action in list(node._untried_actions):
        push_pos, direction = action          # push_pos = player location to push from
        # ----- walk cost ------------------------------------------------
        walk_cost = _bfs_distance(
            start=node.state.player,
            goal=push_pos,
            walls=node.state.walls,
            boxes=node.state.boxes,
        )
        if walk_cost == sys.maxsize:
            # unreachable – treat as dead‑lock for this branch
            deadlocked_actions.append(action)
            continue

        # ----- generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- corner dead‑lock test ------------------------------------
        # Determine the new box position (box was at push_pos + delta)
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        new_box = (push_pos[0] + dr, push_pos[1] + dc)
        if _is_corner_deadlock(new_box, next_state.walls, next_state.targets):
            deadlocked_actions.append(action)
            continue

        # ----- A* visited check (using real g) -------------------------
        g_new = node.state.steps + walk_cost + 1   # +1 for the push itself
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # Do **not** remove permanently; simply skip this expansion pass
            continue

        # ----- heuristic scoring ----------------------------------------
        h_next = _ag.h_sokoban_box_only(next_state)
        delta_h = h_next - h_curr            # >0 means we moved away from goal
        penalty = 0.0
        if delta_h > 0:
            penalty = 5.0 * delta_h           # heavy penalty for worsening
        score = walk_cost + h_next + penalty

        scored.append((score, action, next_state, key, g_new))

    # ---- permanently prune corner‑deadlocked actions -------------------
    for a in deadlocked_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ---- Choose best scored action ------------------------------------
    if scored:
        scored.sort(key=lambda x: x[0])   # lowest composite score first
        best_score, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        # Remove the chosen action from the pool
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No promising actions left – fallback to any remaining untried action
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)

            # compute walk cost for consistency of visited table
            walk_cost = _bfs_distance(
                start=node.state.player,
                goal=chosen_action[0],
                walls=node.state.walls,
                boxes=node.state.boxes,
            )
            chosen_g = node.state.steps + walk_cost + 1
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = chosen_g
        else:
            # No actions at all – return an existing child if any, else node
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ---- Create child node --------------------------------------------
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
- Corner dead‑lock detection uses the wrong cell (`push_pos + delta`) which is the original box location, not the box’s new location after the push. This lets pushes that land a box in a corner slip through.
- Minor: tie‑break on `Δh` could be added for deterministic ordering, but not a bug.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Fix corner dead‑lock detection to evaluate the box’s new position after the push and keep the incremental improvements.
```python
def default_expansion(node):
    """
    Expand the most promising untried action.

    Improvements over the baseline:
      * g‑score = real steps taken (walk + push) instead of tree depth.
      * Corner dead‑lock pruning (now checks the box’s *new* location).
      * Scoring = walk_cost + h_box_only + penalty for Δh > 0.
      * Actions that fail the visited‑check are kept for possible later use.
    """
    # ---- Reset visited info for a new MCTS search (once per root) ----
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    visited = _ag.get_visited()

    # Pre‑compute current heuristic for Δh calculations
    h_curr = _ag.h_sokoban_box_only(node.state)

    # Containers
    scored = []   # (score, action, next_state, key, g_new)
    deadlocked_actions = []  # permanently removed

    # -----------------------------------------------------------------
    # Examine each untried action
    # -----------------------------------------------------------------
    for action in list(node._untried_actions):
        push_pos, direction = action          # player location to push from

        # ----- walk cost ------------------------------------------------
        walk_cost = _bfs_distance(
            start=node.state.player,
            goal=push_pos,
            walls=node.state.walls,
            boxes=node.state.boxes,
        )
        if walk_cost == sys.maxsize:
            # unreachable – treat as dead‑lock for this branch
            deadlocked_actions.append(action)
            continue

        # ----- generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- corner dead‑lock test ------------------------------------
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        # new box location after the push = player_pos + 2*delta
        new_box = (push_pos[0] + 2 * dr, push_pos[1] + 2 * dc)
        if _is_corner_deadlock(new_box, next_state.walls, next_state.targets):
            deadlocked_actions.append(action)
            continue

        # ----- A* visited check (using real g) -------------------------
        g_new = node.state.steps + walk_cost + 1   # +1 for the push itself
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for possible reconsideration later
            continue

        # ----- heuristic scoring ----------------------------------------
        h_next = _ag.h_sokoban_box_only(next_state)
        delta_h = h_next - h_curr            # >0 means we moved away from goal
        penalty = 5.0 * delta_h if delta_h > 0 else 0.0
        score = walk_cost + h_next + penalty

        scored.append((score, action, next_state, key, g_new))

    # ---- permanently prune corner‑deadlocked actions -------------------
    for a in deadlocked_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ---- Choose best scored action ------------------------------------
    if scored:
        # lowest composite score first; tie‑break by smaller Δh (implicit via score)
        scored.sort(key=lambda x: x[0])
        best_score, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]

        # Remove the chosen action from the pool and record its cost
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No promising actions left – fallback to any remaining untried action
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)

            walk_cost = _bfs_distance(
                start=node.state.player,
                goal=chosen_action[0],
                walls=node.state.walls,
                boxes=node.state.boxes,
            )
            chosen_g = node.state.steps + walk_cost + 1
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = chosen_g
        else:
            # No actions at all – return an existing child if any, else node
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ---- Create child node --------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

