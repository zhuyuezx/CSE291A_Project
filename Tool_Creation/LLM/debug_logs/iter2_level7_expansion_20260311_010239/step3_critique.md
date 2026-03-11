# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:03:30 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 18.34s |
| Status     | success |
| Tokens     | 8855 |
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

**1. KEY WEAKNESSES**

| Rank | Symptom (observed in Trace #1) | Evidence |
|------|--------------------------------|----------|
| 1️⃣ | **Myopic ordering – the same push dominates** – the child “((3,3),0)” receives 495 of 500 visits while every other legal macro‑push gets only a single visit. | The children list after Move 1 shows one action with v = 495, avg ≈ 0.73 and all others v = 1. The tree never explores alternative boxes despite them being available. |
| 2️⃣ | **Heuristic blind to player‑walk cost** – pushes that are cheap for the boxes but require a long walk are treated as equally good. | Expansion scores only with `_ag.h_sokoban_box_only`. No term for the distance the player must walk to the pushing position, even though the simulation already computes a “walk‑cost” term. |
| 3️⃣ | **No simple dead‑lock detection for corner pushes** – pushes that place a box against a wall/corner (non‑target) are not filtered, leading to a state with total distance = 1 but still unsolvable after 4 moves. | After four pushes the board shows a box wedged next to a wall (`* $ @`) with only one empty target left; the algorithm never back‑tracks because the dead‑lock predicate `_is_deadlocked` is either missing or only catches *global* deadlocks, not these local corner cases. |
| 4️⃣ | **Over‑pruning via visited‑check** – once a state is visited with the cheapest g, all other actions that lead to the same state (even via a cheaper walk) are permanently removed from `_untried_actions`. | The visited check (`if key in visited and visited[key] <= g_new: prune`) removes *all* other actions after the first expansion, preventing the algorithm from trying a different ordering of pushes that could reach the same state with lower overall cost. |

The combination of (1) and (2) creates a narrow, depth‑first search that repeatedly pushes the same box because it has the lowest raw Manhattan distance, ignoring that a different push might reduce the walk distance or unblock other boxes. Consequently the search stalls with two boxes on target and cannot finish the level.

---

**2. ROOT CAUSE**

The expansion routine scores **only** `h_sokoban_box_only(next_state)`. This heuristic:

* counts each box’s Manhattan distance to the nearest target,
* ignores the player’s reachable region and the cost to get to the push,
* gives identical scores to many distinct pushes (e.g., pushes that move a box one step toward its target versus pushes that move a different box the same Manhattan amount).

Because ties are broken deterministically by the raw action tuple, the algorithm consistently selects the same action, leading to the visitation imbalance seen in the trace.  

The dead‑lock filter relies on `next_state._is_deadlocked()`. In the macro‑push variant this method typically checks *global* deadlocks (e.g., two boxes forming a 2‑by‑2 block) but does **not** catch simple corner deadlocks, which are frequent in Sokoban. Hence pushes that create a box‑in‑corner (non‑target) slip through.

Lastly, the visited‑pruning logic is sound for A* but, when used inside MCTS expansion, it becomes *over‑aggressive*: after a single expansion, all alternative actions that would lead to the same state (perhaps via a cheaper walk) are removed, reducing exploration breadth and preventing the tree from discovering a better ordering of pushes.

---

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement (≈ 70 % effort).**  
We keep the overall A*‑guided structure (reset handling, visited table, permanent removal of proven deadlocks) because it already provides useful pruning. The modifications focus on the scoring function and on a cheap local dead‑lock filter.

| Modification | Rationale |
|--------------|-----------|
| **a. Enrich the expansion heuristic** – compute `h = α * box_distance + β * walk_cost + γ * corner_penalty`. <br>* `box_distance = _ag.h_sokoban_box_only(state)` (existing). <br>* `walk_cost` = BFS distance from the current player position to the required `player_pos` of the macro‑push (same BFS used in the simulation). <br>* `corner_penalty` = large constant (e.g., 100) if the pushed box would end up in a non‑target corner or against a wall with no target reachable along that line. | Directly addresses weaknesses 1 & 2 by preferring pushes that are cheap for the player and that do not create obvious corner deadlocks. |
| **b. Add a fast “corner‑deadlock” test** – before scoring, check whether the destination cell of the pushed box is a non‑target corner (i.e., two orthogonal neighbours are walls). If true, treat the action as pruned (add to `pruned`). | Captures the missing local dead‑lock detection that caused the final unsolvable state. |
| **c. Slightly relax the visited‑prune** – only prune when `visited[key] < g_new` (strictly better cost), otherwise keep the action in `_untried_actions` but still record the better g. This lets the tree revisit an action that reaches the same state with a *lower* walk cost later. | Prevents over‑pruning and restores some breadth without sacrificing A*‑style dominance. |
| **d. Tie‑breaker that favors lower walk cost** – when two actions have identical `h`, choose the one with smaller `walk_cost` before falling back to tuple order. | Ensures deterministic but better ordering when Manhattan distances are equal. |

All four changes are **local** to `default_expansion`; they reuse the BFS routine already present in the simulation (no extra heavy computation) and preserve the existing A*‑guided reset/visited logic. The weighted coefficients (`α, β, γ`) can be set to modest values (e.g., α = 1.0, β = 0.2, γ = 100) and later tuned experimentally.

**Why not a full restructure?**  
The current expansion already integrates A*‑style g‑tracking and dead‑lock pruning, which are advantageous. The principal failure stems from an *over‑simplified heuristic* and a missing corner‑deadlock filter, not from a conceptual flaw in the expansion pipeline. Incremental enrichment therefore promises the biggest gain with minimal risk and maintains compatibility with the rest of the MCTS pipeline.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
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

Improvements:
  * Adds player‑walk distance to the scoring function.
  * Detects simple corner dead‑locks (box pushed into a non‑target corner)
    and prunes those actions early.
  * Relaxes the visited‑prune: actions are only pruned when a strictly
    better g‑score already exists, preventing over‑pruning of equally‑good
    alternatives.
  * Tie‑breaks by walk distance before falling back to action ordering.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from collections import deque
from typing import Tuple, Set

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def _bfs_distances(state) -> dict[Tuple[int, int], int]:
    """
    BFS from the player over free cells (walls and boxes block movement).
    Returns a dict mapping each reachable cell to its distance (steps)
    from the player position.
    """
    walls: Set[Tuple[int, int]] = state.walls
    boxes: Set[Tuple[int, int]] = state.boxes
    start = state.player

    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls or (nr, nc) in boxes:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
    return dist


def _is_corner_deadlock(state, dest: Tuple[int, int]) -> bool:
    """
    Detects a simple corner dead‑lock:
    - destination cell is not a target
    - at least two orthogonal neighbours are walls (or the grid edge)
    """
    if dest in state.targets:
        return False
    walls = state.walls
    r, c = dest
    # helper to test wall/out‑of‑bounds
    def blocked(pos):
        return pos in walls

    # check the four orthogonal neighbours
    up = (r - 1, c)
    down = (r + 1, c)
    left = (r, c - 1)
    right = (r, c + 1)

    # corner if (up or down) and (left or right) are walls
    if (blocked(up) or blocked(down)) and (blocked(left) or blocked(right)):
        return True
    return False


def default_expansion(node):
    """
    Expand the best untried action using A* visited + enhanced heuristic.

    Steps:
      0. Reset shared visited when a new MCTS search starts.
      1. Compute g_new = depth(node) + 1.
      2. For each untried action:
         - generate next_state
         - prune if it creates a corner dead‑lock
         - prune only if a *strictly* better g already exists in visited
         - score by: h = box_distance + WALK_WEIGHT * walk_cost
      3. Choose the action with minimal (h, walk_cost, action) tuple.
      4. Record visited cost and create the child node.
      5. Fallback: if everything was pruned, expand the first remaining
         untried action (ensuring the tree never stalls).
    """
    # ----- 0. reset visited for a fresh MCTS search --------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    g_new = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    # ----- pre‑compute player walk distances for scoring -----------------
    walk_dist = _bfs_distances(node.state)

    scored = []   # (h_total, walk_cost, action, next_state, key)
    pruned = []   # actions to permanently remove from _untried_actions

    WALK_WEIGHT = 0.2   # empirical weight for walk distance
    # ------------------------------------------------------------------
    for action in list(node._untried_actions):
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- a) corner dead‑lock detection --------------------------------
        # Determine where the pushed box ends up.
        (pr, pc), direction = action
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        box_start = (pr + dr, pc + dc)          # original box location
        box_dest = (box_start[0] + dr, box_start[1] + dc)  # after push
        if _is_corner_deadlock(next_state, box_dest):
            pruned.append(action)
            continue

        # ---- b) visited check (relaxed) ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep action for possible alternative ordering; only prune if
            # we already have a *strictly* better g‑score.
            if visited[key] < g_new:
                pruned.append(action)
                continue
            # visited[key] == g_new → keep action (do not prune)

        # ---- c) heuristic scoring -----------------------------------------
        h_box = _ag.h_sokoban_box_only(next_state)

        # walk distance = distance from current player to the required push position
        walk_cost = walk_dist.get(action[0], 10 ** 6)   # large if somehow unreachable

        h_total = h_box + WALK_WEIGHT * walk_cost

        scored.append((h_total, walk_cost, action, next_state, key))

    # ----- remove permanently pruned actions -----------------------------
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ----- choose best scored action ------------------------------------
    if scored:
        # sort by (h_total, walk_cost, action) – deterministic tie‑break
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_action, chosen_state, chosen_key = scored[0]
        # remove the chosen action from the pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # ----- fallback when everything was pruned -----------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # No actions left – return an existing child or self as safety net
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ----- create the child node ----------------------------------------
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
- The corner‑deadlock check only treats walls as blockers; it ignores grid edges, so pushes into a map border without an explicit wall aren’t pruned.
- Minor type‑annotation issue: using `dict[Tuple[int, int], int]` requires Python 3.9+, but the project may run on older versions. Using `Dict` from `typing` is safer.
- No other functional bugs; performance is fine.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Improve corner dead‑lock detection to treat out‑of‑bounds as walls and use broader typing for compatibility.
```python
"""
A*-guided MCTS expansion with enhanced heuristic and corner dead‑lock detection.

Improvements:
  * Adds player‑walk distance to the scoring function.
  * Detects simple corner dead‑locks (box pushed into a non‑target corner or
    out‑of‑bounds) and prunes those actions early.
  * Relaxes the visited‑prune: actions are only pruned when a strictly better
    g‑score already exists, preventing over‑pruning of equally‑good alternatives.
  * Tie‑breaks by walk cost before falling back to action ordering.
"""

from __future__ import annotations

import sys
import importlib.util
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Set

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def _bfs_distances(state) -> Dict[Tuple[int, int], int]:
    """
    BFS from the player over free cells (walls and boxes block movement).
    Returns a dict mapping each reachable cell to its distance (steps)
    from the player position.
    """
    walls: Set[Tuple[int, int]] = state.walls
    boxes: Set[Tuple[int, int]] = state.boxes
    start = state.player

    dist: Dict[Tuple[int, int], int] = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls or (nr, nc) in boxes:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
    return dist


def _is_corner_deadlock(state, dest: Tuple[int, int]) -> bool:
    """
    Detect a simple corner dead‑lock:
      * destination is not a target, and
      * at least two orthogonal neighbours are walls or outside the map.
    """
    if dest in state.targets:
        return False

    rows, cols = state.height, state.width

    def blocked(pos: Tuple[int, int]) -> bool:
        r, c = pos
        # out‑of‑bounds counts as a wall
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return True
        return pos in state.walls

    r, c = dest
    up = (r - 1, c)
    down = (r + 1, c)
    left = (r, c - 1)
    right = (r, c + 1)

    # Corner if (up or down) *and* (left or right) are blocked
    if (blocked(up) or blocked(down)) and (blocked(left) or blocked(right)):
        return True
    return False


def default_expansion(node):
    """
    Expand the best untried action using A* visited + enhanced heuristic.

    Steps:
      0. Reset shared visited when a new MCTS search starts.
      1. Compute g_new = depth(node) + 1.
      2. For each untried action:
         - generate next_state
         - prune if it creates a corner dead‑lock
         - prune only if a *strictly* better g already exists in visited
         - score by: h = box_distance + WALK_WEIGHT * walk_cost
      3. Choose the action with minimal (h, walk_cost, action) tuple.
      4. Record visited cost and create the child node.
      5. Fallback: if everything was pruned, expand the first remaining
         untried action (ensuring the tree never stalls).
    """
    # ----- 0. reset visited for a fresh MCTS search --------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    g_new = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    # ----- pre‑compute player walk distances for scoring -----------------
    walk_dist = _bfs_distances(node.state)

    scored = []   # (h_total, walk_cost, action, next_state, key)
    pruned = []   # actions to permanently remove from _untried_actions

    WALK_WEIGHT = 0.2   # empirical weight for walk distance

    for action in list(node._untried_actions):
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- a) corner dead‑lock detection --------------------------------
        (pr, pc), direction = action
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        box_start = (pr + dr, pc + dc)          # original box location
        box_dest = (box_start[0] + dr, box_start[1] + dc)  # after push
        if _is_corner_deadlock(next_state, box_dest):
            pruned.append(action)
            continue

        # ---- b) visited check (relaxed) ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            if visited[key] < g_new:          # strictly better path already known
                pruned.append(action)
                continue
            # visited[key] == g_new → keep the action

        # ---- c) heuristic scoring -----------------------------------------
        h_box = _ag.h_sokoban_box_only(next_state)

        # walk distance = distance from current player to the required push position
        walk_cost = walk_dist.get(action[0], 10 ** 6)   # large if somehow unreachable

        h_total = h_box + WALK_WEIGHT * walk_cost

        scored.append((h_total, walk_cost, action, next_state, key))

    # ----- remove permanently pruned actions -----------------------------
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ----- choose best scored action ------------------------------------
    if scored:
        # sort by (h_total, walk_cost, action) – deterministic tie‑break
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, chosen_action, chosen_state, chosen_key = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # ----- fallback when everything was pruned -----------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # No actions left – return an existing child or self as safety net
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ----- create the child node ----------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
