# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:06:19 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 16.36s |
| Status     | success |
| Tokens     | 8424 |
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

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it matters |
|------|---------------------------|----------------|
| **1** | After a good push → `move 7` (box placed on a target) the algorithm immediately undoes progress: `move 8‑10` push a different box away, raising the total distance again (3 → 4 → 3 → 2). | The expansion stage is **blind to the cost of walking** to the push position, so a cheap‑looking *box‑distance* reduction wins even when it requires a long walk‑step (high step‑cost) and “undoes” earlier progress. |
| **2** | Boxes already on a target are moved off again (e.g. the box on target after move 7 is never protected). | The **box‑only heuristic** treats a box on a target as distance 0; moving it off simply adds the new distance, which may still be the smallest among the remaining actions, so the algorithm does not bias against removing a solved box. |
| **3** | Dead‑lock pruning relies solely on `state._is_deadlocked()`. Simple “push a box off a target into a corner” deadlocks are **not caught**, so the tree may expand into states that become impossible to solve later. |
| **4** | Actions that are *temporarily* pruned because a state was previously visited with a better *g* are **removed permanently** from `_untried_actions`. If the same push becomes useful after other boxes move, it is lost, narrowing exploration dramatically (e.g. after move 6 the only remaining child is the repeatedly‑chosen push, ignoring alternatives). |

**2. ROOT CAUSE**

The expansion function scores every candidate solely with  

```python
_ag.h_sokoban_box_only(next_state)
```  

which sums the Manhattan distance of each box to its nearest target.  
Consequences:

* **No walk‑cost factor** – the algorithm cannot distinguish a push that needs 1 step from one that needs 15 steps; both are judged only by how the *boxes* move.
* **No “target‑protection” term** – pushing a box off a target merely increases the heuristic; there is no extra penalty, so the algorithm may freely undo solved boxes.
* **Dead‑lock detection is only the explicit `_is_deadlocked` method**, which in the provided code checks only classic corner‑deadlocks; it does not consider “box‑off‑target‑into‑corner” situations.
* **Permanent removal of pruned actions** (`node._untried_actions.remove(a)`) ties the pruning to a *single* g‑value check. When the same state is reachable later with a different context (different surrounding boxes), the action is gone forever, limiting breadth.

Because the expansion step is the only place where actions are ordered, these omissions steer the search toward low‑h, high‑walk‑cost pushes that discard progress.

**3. PROPOSED APPROACH – A (incremental)**  

The current framework works (A*‑style visited table, deterministic ordering) and should be kept. We only need a richer **action‑ranking score** and a milder pruning policy.

| Modification | Rationale |
|--------------|-----------|
| **a. Add walk‑cost term** – compute `walk = bfs_distance(state.player, push_pos)` (the length of the shortest reachable path to the player position required for the push). Include it in the score, e.g. `score = α·h_box + β·walk`. Choose `β` > 0 (e.g. 0.5) so long walks are penalised. | Prevents the algorithm from “chasing” distant pushes that look good box‑wise but cost many steps, addressing weakness 1. |
| **b. Protect solved boxes** – if the action would move a box that is currently on a target, add a large penalty `γ` (e.g. +5) to the score. Detect by `if box_pos in state.targets and next_box_pos not in state.targets:`. | Keeps already‑solved boxes in place, mitigating weakness 2. |
| **c. Light‑weight corner‑after‑push check** – after applying the action, run a cheap test: for every box not on a target, if it is in a corner (two orthogonal walls) **or** if it is on a target and the new position is a corner, treat the state as deadlocked and prune the action. This is a one‑line addition to the existing deadlock block. | Catches the most common “push off target into a corner” deadlocks, fixing weakness 3 without heavy computation. |
| **d. Revise permanent pruning** – keep the visited‑g check, but **do not remove the action** from `_untried_actions`. Instead, just skip it for this expansion (`continue`). Only actions that are *intrinsically* deadlocked (a‑c) are removed. This preserves breadth for later re‑evaluation when the surrounding context changes. | Allows previously‑skipped pushes to become viable later, fixing weakness 4. |
| **e. Tie‑breaker refinement** – when `score` ties, prefer the action with the **smaller walk cost** (or the one that keeps a target‑box stationary). | Further biases toward cheap, safe pushes. |

All changes fit inside the current `default_expansion` body: after generating `next_state`, compute `walk`, `h_box`, add the two penalty terms, store `(score, action, next_state, key)` in `scored`, and later `scored.sort(key=lambda x: x[0])`. The rest of the A*‑style visited handling and child creation stays unchanged.

**Why incremental is enough:**  
The fundamental problem is *how actions are evaluated*, not the overall A*‑guided architecture. By enriching the scoring function and loosening the permanent‑prune rule, we keep the deterministic, low‑overhead expansion loop while giving the tree the information it currently lacks. This should immediately reduce wasteful long‑walk pushes, keep solved boxes stable, and broaden exploration, yielding higher returns on level 6 without a full redesign.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_expansion(node):
    """
    Expand the best untried action using richer scoring and safer pruning.

    Improvements over the original version:
      * Incorporates the player‑walk distance to the push position
        (penalised with WALK_WEIGHT) so long‑range pushes are deprioritised.
      * Adds a heavy penalty when a push would move a box off a target
        (TARGET_MOVE_PENALTY) – protects already‑solved boxes.
      * Performs a cheap corner‑dead‑lock test on the resulting state;
        pushes that create a box in a non‑target corner are pruned.
      * Visited‑g pruning no longer permanently removes the action from
        ``_untried_actions`` – actions are only skipped for this expansion,
        preserving breadth for later reconsideration.

    The rest of the A*‑style visited handling and deterministic ordering
    remains unchanged.
    """
    # ------------------------------------------------------------------
    # Reset shared A* visited table if we have entered a new MCTS search.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper utilities – everything defined here to keep the function
    # self‑contained.
    # ------------------------------------------------------------------
    from collections import deque

    # Directions: (dr, dc) matching the action encoding 0=UP,1=DOWN,2=LEFT,3=RIGHT
    DIRS = [( -1, 0), ( 1, 0), ( 0,-1), ( 0, 1)]

    WALK_WEIGHT = 0.5          # importance of walk‑cost in the final score
    TARGET_MOVE_PENALTY = 5.0  # penalty for moving a box off a target
    CORNER_PENALTY = 1e9       # effectively infinite – we prune such actions

    def bfs_distances(state):
        """Return a dict mapping each reachable cell to its shortest walk distance."""
        walls = state.walls
        boxes = state.boxes
        start = state.player
        visited = {start: 0}
        q = deque([start])
        while q:
            r, c = q.popleft()
            d = visited[(r, c)] + 1
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in walls or (nr, nc) in boxes:
                    continue
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = d
                    q.append((nr, nc))
        return visited

    def is_corner(pos, walls_set):
        """Detect a static corner (two orthogonal walls)."""
        r, c = pos
        # four possible corner configurations
        return ((r + 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r + 1, c) in walls_set and (r, c - 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c - 1) in walls_set)

    # ------------------------------------------------------------------
    # Pre‑compute walk distances once for the current state (before any push).
    # ------------------------------------------------------------------
    walk_map = bfs_distances(node.state)

    g_new   = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    scored  = []   # list of tuples (score, action, next_state, key)
    to_remove = [] # actions that are truly deadlocked and must be deleted

    for action in list(node._untried_actions):
        push_pos, direction = action
        dr, dc = DIRS[direction]

        # ------------------------------------------------------------------
        # Walk cost: distance from player to the required push position.
        # If unreachable (should not happen for legal actions) we skip.
        # ------------------------------------------------------------------
        walk_cost = walk_map.get(push_pos, None)
        if walk_cost is None:
            # unreachable – treat as deadlocked for safety
            to_remove.append(action)
            continue

        # ------------------------------------------------------------------
        # Apply the action to obtain the successor state.
        # ------------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ------------------------------------------------------------------
        # 1️⃣  Dead‑lock pruning – original check (if any) plus cheap corner test.
        # ------------------------------------------------------------------
        dead = False
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            dead = True
        else:
            # cheap corner test on all boxes after the push
            for b in next_state.boxes:
                if b in next_state.targets:
                    continue          # target boxes are safe
                if is_corner(b, next_state.walls):
                    dead = True
                    break
        if dead:
            to_remove.append(action)
            continue

        # ------------------------------------------------------------------
        # 2️⃣  Visited‑g check – only *skip* for this expansion, do NOT delete.
        # ------------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # skip this action for now; keep it for possible later use
            continue

        # ------------------------------------------------------------------
        # 3️⃣  Scoring – combine box‑only heuristic, walk cost and target‑move penalty.
        # ------------------------------------------------------------------
        h_box = _ag.h_sokoban_box_only(next_state)

        # Detect "moving a box off a target"
        box_before = (push_pos[0] + dr, push_pos[1] + dc)   # original box location
        box_after  = (box_before[0] + dr, box_before[1] + dc)  # where it ends up
        target_move_pen = 0.0
        if box_before in node.state.targets and box_after not in node.state.targets:
            target_move_pen = TARGET_MOVE_PENALTY

        score = h_box + WALK_WEIGHT * walk_cost + target_move_pen

        scored.append((score, action, next_state, key))

    # ------------------------------------------------------------------
    # Remove actions that are certainly deadlocked.
    # ------------------------------------------------------------------
    for a in to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # Choose the best scored action, if any.
    # ------------------------------------------------------------------
    if scored:
        # lower score → more promising
        scored.sort(key=lambda x: x[0])
        _, chosen_action, chosen_state, chosen_key = scored[0]
        # permanently remove the chosen action from the untried pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # No action survived scoring (all skipped by visited check or deadlocked).
        # Fallback: expand the first remaining untried action, if any.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # Nothing left – return an existing child if we have one,
            # otherwise return the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # Create the child node and attach it.
    # ------------------------------------------------------------------
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
- No bugs detected; the function uses the GameState API correctly and handles all edge cases.
- Performance is reasonable: BFS distances are computed once per expansion and the added heuristics are lightweight.
- The scoring function now varies with walk cost and target‑move penalties, providing better action discrimination.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: No changes needed; the draft implementation is correct and efficient.
```python
def default_expansion(node):
    """
    Expand the best untried action using richer scoring and safer pruning.

    Improvements over the original version:
      * Incorporates the player‑walk distance to the push position
        (penalised with WALK_WEIGHT) so long‑range pushes are deprioritised.
      * Adds a heavy penalty when a push would move a box off a target
        (TARGET_MOVE_PENALTY) – protects already‑solved boxes.
      * Performs a cheap corner‑dead‑lock test on the resulting state;
        pushes that create a box in a non‑target corner are pruned.
      * Visited‑g pruning no longer permanently removes the action from
        ``_untried_actions`` – actions are only skipped for this expansion,
        preserving breadth for later reconsideration.

    The rest of the A*‑style visited handling and deterministic ordering
    remains unchanged.
    """
    # ------------------------------------------------------------------
    # Reset shared A* visited table if we have entered a new MCTS search.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper utilities – everything defined here to keep the function
    # self‑contained.
    # ------------------------------------------------------------------
    from collections import deque

    # Directions: (dr, dc) matching the action encoding 0=UP,1=DOWN,2=LEFT,3=RIGHT
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    WALK_WEIGHT = 0.5          # importance of walk‑cost in the final score
    TARGET_MOVE_PENALTY = 5.0  # penalty for moving a box off a target
    # (no explicit corner penalty needed; we simply prune such actions)

    def bfs_distances(state):
        """Return a dict mapping each reachable cell to its shortest walk distance."""
        walls = state.walls
        boxes = state.boxes
        start = state.player
        visited = {start: 0}
        q = deque([start])
        while q:
            r, c = q.popleft()
            d = visited[(r, c)] + 1
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in walls or (nr, nc) in boxes:
                    continue
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = d
                    q.append((nr, nc))
        return visited

    def is_corner(pos, walls_set):
        """Detect a static corner (two orthogonal walls)."""
        r, c = pos
        return ((r + 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r + 1, c) in walls_set and (r, c - 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c - 1) in walls_set)

    # ------------------------------------------------------------------
    # Pre‑compute walk distances once for the current state (before any push).
    # ------------------------------------------------------------------
    walk_map = bfs_distances(node.state)

    g_new   = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    scored      = []  # list of tuples (score, action, next_state, key)
    dead_actions = []  # actions that are truly deadlocked and must be deleted

    for action in list(node._untried_actions):
        push_pos, direction = action
        dr, dc = DIRS[direction]

        # ------------------------------------------------------------------
        # Walk cost: distance from player to the required push position.
        # If unreachable (should not happen for legal actions) we skip.
        # ------------------------------------------------------------------
        walk_cost = walk_map.get(push_pos, None)
        if walk_cost is None:
            # unreachable – treat as deadlocked for safety
            dead_actions.append(action)
            continue

        # ------------------------------------------------------------------
        # Apply the action to obtain the successor state.
        # ------------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ------------------------------------------------------------------
        # 1️⃣  Dead‑lock pruning – original check (if any) plus cheap corner test.
        # ------------------------------------------------------------------
        dead = False
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            dead = True
        else:
            # cheap corner test on all boxes after the push
            for b in next_state.boxes:
                if b in next_state.targets:
                    continue          # target boxes are safe
                if is_corner(b, next_state.walls):
                    dead = True
                    break
        if dead:
            dead_actions.append(action)
            continue

        # ------------------------------------------------------------------
        # 2️⃣  Visited‑g check – only *skip* for this expansion, do NOT delete.
        # ------------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # skip this action for now; keep it for possible later use
            continue

        # ------------------------------------------------------------------
        # 3️⃣  Scoring – combine box‑only heuristic, walk cost and target‑move penalty.
        # ------------------------------------------------------------------
        h_box = _ag.h_sokoban_box_only(next_state)

        # Detect "moving a box off a target"
        box_before = (push_pos[0] + dr, push_pos[1] + dc)   # original box location
        box_after  = (box_before[0] + dr, box_before[1] + dc)  # where it ends up
        target_move_pen = TARGET_MOVE_PENALTY if (
            box_before in node.state.targets and box_after not in node.state.targets
        ) else 0.0

        score = h_box + WALK_WEIGHT * walk_cost + target_move_pen

        scored.append((score, action, next_state, key))

    # ------------------------------------------------------------------
    # Remove actions that are certainly deadlocked.
    # ------------------------------------------------------------------
    for a in dead_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # Choose the best scored action, if any.
    # ------------------------------------------------------------------
    if scored:
        # lower score → more promising
        scored.sort(key=lambda x: x[0])
        _, chosen_action, chosen_state, chosen_key = scored[0]
        # permanently remove the chosen action from the untried pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # No action survived scoring (all skipped by visited check or deadlocked).
        # Fallback: expand the first remaining untried action, if any.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # Nothing left – return an existing child if we have one,
            # otherwise return the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # Create the child node and attach it.
    # ------------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

