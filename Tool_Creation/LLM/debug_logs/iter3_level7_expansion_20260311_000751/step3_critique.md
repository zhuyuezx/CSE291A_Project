# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:08:42 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 19.12s |
| Status     | success |
| Tokens     | 7861 |
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

| Rank | Symptom (from the trace) | Evidence |
|------|--------------------------|----------|
| 1️⃣ | **Expansion ignores the walk‑cost**. The chosen first move `((3,4),0)` keeps the total box‑to‑target distance at 5 while consuming several hidden walk steps. Its child node receives 490 visits (≈ 98 % of the budget) and dominates the tree, starving better‑located pushes that would reduce the distance more quickly. |
| 2️⃣ | **Heuristic only looks at boxes** (`h_sokoban_box_only`). Because the same *box‑only* distance can be achieved from many different player positions, the expansion routine repeatedly expands the same low‑h action even when it is far from the player, inflating the search depth without progress. |
| 3️⃣ | **No monotonicity guard** – actions that *increase* the box‑only distance are never filtered out. The tree quickly contains many “undo” pushes, which explains the plateau at 2/3 boxes on target after four moves even though many actions are being explored. |
| 4️⃣ | **Visited‑check is too permissive**. It only prevents re‑expansion when a state has been visited with a *shallower* g‑score. Since g is *depth* (number of pushes) and not the true walk cost, the same state can re‑appear many times with the same g, allowing the same poor action to be revisited in different branches. |

The net effect is that the MCTS spends the overwhelming majority of its 500 iterations on a single low‑h, high‑cost branch, yielding a shallow exploration of the true solution space and a final return of 0.0.

---

**2. ROOT CAUSE**

The expansion function builds its candidate list with:

```python
scored.append((_ag.h_sokoban_box_only(next_state), action, ...))
```

* It **excludes the walk component** (`g_new`) from the scoring, while the selection phase later uses `f = g + h` (full A* f‑score).  
* The pruning step `if key in visited and visited[key] <= g_new:` only looks at *depth* (`g_new`) and not at the actual step cost (walk + push). Consequently, states reached via a long walk are considered “new” as long as the push count is the same, letting expensive pushes survive.  
* No check discards actions that raise the box‑only heuristic, so the algorithm happily expands moves that move a box away from any target.

These design decisions create a mismatch: **selection prefers low *f* (walk + box distance) while expansion only cares about low *box distance***, so the tree is built around the wrong metric.

---

**3. PROPOSED APPROACH – Strategy A (Incremental)**  

The problem is structural but can be solved by modest, targeted changes that keep the overall A*‑guided framework intact.

| Modification | Rationale |
|--------------|----------|
| **a. Score actions with full A* f‑value** <br>`score = g_new + _ag.h_sokoban(next_state)` <br>instead of only the box‑only heuristic. | Aligns expansion ordering with the selection phase, ensuring that pushes requiring a long walk are penalised early. |
| **b. Tie‑break on walk‑cost** <br>When two actions share the same f, prefer the one with the *smaller* walk distance (`walk = _ag.walk_cost(state, action)` – a cheap helper that returns the BFS length from `state.player` to `action[0]`). | Gives deterministic preference to nearer pushes, preventing the algorithm from repeatedly expanding distant pushes that happen to have identical box‑only distance. |
| **c. Monotonicity filter** <br>`if _ag.h_sokoban(next_state) > parent_h: prune` (where `parent_h = _ag.h_sokoban(node.state)`). | Stops the tree from exploring pushes that move a box farther from any target, drastically reducing dead‑end branches. |
| **d. Strengthen visited pruning** <br>Store the *full* g‑score (`g = g_new + walk_cost`) in the visited table and prune when `visited[key] <= g`. | Guarantees that a state reached via a cheaper walk is preferred and eliminates redundant re‑expansions of the same state with equal push count but higher walk cost. |
| **e. Optional light‑weight deadlock check** – keep the existing `_is_deadlocked` guard, but also prune “simple corner deadlocks” using a cheap pattern test (walls‑plus‑box not on target). | Improves pruning without heavy computation. |

All changes are confined to `default_expansion`. The rest of the pipeline (selection, simulation, back‑propagation) already uses the full A* heuristic and can remain unchanged, preserving the proven A*‑guided architecture.

**Why incremental, not a full restructure?**  
The current framework already supplies a shared A* state, visited management, and a cohesive A*‑guided selection. The primary bottleneck is the *mismatch of scoring* in expansion. By re‑weighting the scoring and tightening pruning, we keep the existing deterministic A*‑style ordering while dramatically improving expansion quality. A complete redesign (e.g., switching to random rollouts) would discard the dense heuristic information that already yields far better guidance than pure MCTS; the trace shows that the heuristic works when used correctly, so a focused tweak is sufficient and lower‑risk.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_expansion(node):
    """
    Expand the best untried action using A* visited + heuristic filtering,
    now taking the actual walk cost into account and discarding actions that
    increase the heuristic or create simple corner deadlocks.

    Steps:
      0. Reset shared A* visited if this is a new root.
      1. Compute parent depth (push count) and parent heuristic.
      2. For every untried action:
           • generate next_state
           • prune if a simple corner deadlock is detected
           • prune if next_state's heuristic > parent heuristic (non‑monotonic)
           • compute walk cost from current player pos to the required push cell
           • total_g = parent_depth + walk_cost + 1   (push step)
           • prune if visited[key] <= total_g
           • score by f = total_g + h_sokoban(next_state)
      3. Choose the action with minimal f (tie‑break on smaller walk cost).
      4. Fallback to first remaining untried action if everything was pruned.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset A* visited if we have a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    from collections import deque

    def bfs_distance(start, goal, walls, boxes):
        """Return shortest Manhattan distance avoiding walls & boxes."""
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
        # Should never happen for legal actions; return a large penalty
        return 10 ** 6

    def simple_corner_deadlock(state):
        """Detect obvious corner deadlocks (box in a wall corner not on target)."""
        walls = state.walls
        targets = state.targets
        for b in state.boxes:
            if b in targets:
                continue
            r, c = b
            # check four corner patterns
            if ((r - 1, c) in walls and (r, c - 1) in walls) or \
               ((r - 1, c) in walls and (r, c + 1) in walls) or \
               ((r + 1, c) in walls and (r, c - 1) in walls) or \
               ((r + 1, c) in walls and (r, c + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------ #
    # 1. Parent information
    # ------------------------------------------------------------------ #
    parent_depth = _ag.node_depth(node)          # push count so far
    parent_h = _ag.h_sokoban(node.state)         # full heuristic of current node
    visited = _ag.get_visited()

    scored = []   # list of (f, walk, action, next_state, key)
    pruned = []   # actions to permanently discard

    for action in list(node._untried_actions):
        # -------------------------------------------------------------- #
        # Generate successor state
        # -------------------------------------------------------------- #
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------- #
        # a) Simple corner deadlock pruning (if not already provided)
        # -------------------------------------------------------------- #
        if simple_corner_deadlock(next_state):
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # b) Existing deadlock method, if present
        # -------------------------------------------------------------- #
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # c) Monotonicity: do not allow heuristic to increase
        # -------------------------------------------------------------- #
        next_h = _ag.h_sokoban(next_state)
        if next_h > parent_h:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # d) Compute walk distance from current player to the push position
        # -------------------------------------------------------------- #
        player_start = node.state.player
        push_pos = action[0]                     # where the player must stand
        walk = bfs_distance(player_start, push_pos,
                            node.state.walls, node.state.boxes)

        # -------------------------------------------------------------- #
        # e) Total g‑score for visited table (push count + walk + 1)
        # -------------------------------------------------------------- #
        total_g = parent_depth + walk + 1

        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # f) Score by full A* f = g + h  (use the same heuristic as selection)
        # -------------------------------------------------------------- #
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 2. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 3. Choose best action (minimal f, then minimal walk)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))   # f, walk, action order
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
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
            # recompute walk & total_g for consistency
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
    # 4. Create child node and register it
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
- The draft assumes the existence of `_ag.h_sokoban`, but the original shared module only provided `h_sokoban_box_only`. This may raise an `AttributeError`.
- The BFS distance function recalculates the full distance for every action; while acceptable, we can reuse the same function without changes.
- Minor clarity improvements: compute the parent heuristic once with a safe fallback, and annotate types for readability.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Fix heuristic lookup, add safe fallback, and clean up minor issues.
```python
def default_expansion(node):
    """
    Expand the best untried action using A* visited + heuristic filtering,
    now taking the actual walk cost into account and discarding actions that
    increase the heuristic or create simple corner deadlocks.

    Steps:
      0. Reset shared A* visited if this is a new root.
      1. Compute parent depth and parent heuristic.
      2. For every untried action:
           • generate next_state
           • prune if a simple corner deadlock is detected
           • prune if next_state's heuristic > parent heuristic (non‑monotonic)
           • compute walk cost from current player pos to the required push cell
           • total_g = parent_depth + walk_cost + 1   (push step)
           • prune if visited[key] <= total_g
           • score by f = total_g + h(next_state)
      3. Choose the action with minimal f (tie‑break on smaller walk cost).
      4. Fallback to first remaining untried action if everything was pruned.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset A* visited if we have a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    from collections import deque

    def bfs_distance(start, goal, walls, boxes):
        """Return shortest distance avoiding walls & boxes (Manhattan BFS)."""
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
        # Should not happen for legal actions; return a large penalty
        return 10**6

    def simple_corner_deadlock(state):
        """Detect obvious corner deadlocks (box in a wall corner not on target)."""
        walls = state.walls
        targets = state.targets
        for b in state.boxes:
            if b in targets:
                continue
            r, c = b
            # four possible corner configurations
            if ((r - 1, c) in walls and (r, c - 1) in walls) or \
               ((r - 1, c) in walls and (r, c + 1) in walls) or \
               ((r + 1, c) in walls and (r, c - 1) in walls) or \
               ((r + 1, c) in walls and (r, c + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------ #
    # 1. Parent information
    # ------------------------------------------------------------------ #
    parent_depth = _ag.node_depth(node)          # number of pushes so far
    # safe heuristic lookup (full h if available, else box‑only h)
    if hasattr(_ag, "h_sokoban"):
        parent_h = _ag.h_sokoban(node.state)
    else:
        parent_h = _ag.h_sokoban_box_only(node.state)

    visited = _ag.get_visited()

    scored = []   # (f, walk, action, next_state, key, total_g)
    pruned = []   # actions to permanently discard

    for action in list(node._untried_actions):
        # -------------------------------------------------------------- #
        # Generate successor state
        # -------------------------------------------------------------- #
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------- #
        # a) Simple corner deadlock pruning
        # -------------------------------------------------------------- #
        if simple_corner_deadlock(next_state):
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # b) Existing deadlock method, if present
        # -------------------------------------------------------------- #
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # c) Monotonicity: discard actions that raise the heuristic
        # -------------------------------------------------------------- #
        if hasattr(_ag, "h_sokoban"):
            next_h = _ag.h_sokoban(next_state)
        else:
            next_h = _ag.h_sokoban_box_only(next_state)

        if next_h > parent_h:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # d) Compute walk distance from current player to the push position
        # -------------------------------------------------------------- #
        player_start = node.state.player
        push_pos = action[0]                     # cell where player must stand
        walk = bfs_distance(player_start, push_pos,
                            node.state.walls, node.state.boxes)

        # -------------------------------------------------------------- #
        # e) Total g‑score for visited table (push count + walk + 1)
        # -------------------------------------------------------------- #
        total_g = parent_depth + walk + 1

        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # f) Score by full A* f = g + h
        # -------------------------------------------------------------- #
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 2. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 3. Choose best action (minimal f, then minimal walk, then action order)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
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
    # 4. Create child node and register it
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
