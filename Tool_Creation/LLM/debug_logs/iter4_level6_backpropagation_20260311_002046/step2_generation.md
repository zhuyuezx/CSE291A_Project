# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:21:06 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.24s |
| Status     | success |
| Tokens     | 8367 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

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
MCTS TOOL FUNCTIONS (all 4 phases)
------------------------------------------------------------

--- selection ---
```python
"""
A*-guided MCTS selection.

Replaces UCB1 with best-first (min f = g + h) node selection, mirroring
A*'s heapq.heappop(pq). Detects the start of each new MCTS search and
resets the shared visited table.
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state (one instance shared across all 4 phase files) ──
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk: at each level select the child with min f = g + h.

    Mirrors A*'s heapq.heappop(pq) — always descend toward the lowest-cost
    frontier node rather than the highest UCB1 score.

    Tie-breaking: among equal f-scores, prefer the child with fewer visits
    to encourage coverage of equally-promising branches.

    Also detects when a new MCTS search begins (root state key changed) and
    resets the shared visited dict for the new search.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    # Walk to the actual root of the tree passed in.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    # Reset when:
    #   • visits == 0  → brand-new root (first iteration of a fresh search,
    #                     including after smoke-test pollution)
    #   • key changed  → different game/level
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ── Walk tree: min-f child selection ─────────────────────────────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best      = None
        best_f    = math.inf
        best_q    = -math.inf  # avg value — tie-break when f is equal

        for child in node.children.values():
            g = _ag.node_depth(child)
            h = _ag.h_sokoban(child.state)
            f = g + h
            q = child.value / child.visits if child.visits > 0 else 0.0
            if f < best_f or (f == best_f and q > best_q):
                best   = child
                best_f = f
                best_q = q

        if best is None:
            break
        node = best

    return node
```

--- expansion ---
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

--- simulation ---
```python
"""
A* heuristic evaluation — no rollout.

In A*, there is no simulation: the heuristic h(state) IS the leaf evaluation.
This phase replaces the random rollout with a direct heuristic score:

    reward = 1.0 / (1.0 + h)     h = sum(box_to_nearest_target)  [box-only, match notebook]

Values are in (0, 1]:  solved state → 1.0,  farther state → closer to 0.
The shaped reward gives MCTS a dense gradient toward the goal instead of
the sparse {0, 1} terminal signal from random rollouts.
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


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A* heuristic evaluation at the leaf node — no rollout.

    Returns:
        1.0                    if the state is solved (h = 0)
        state.returns()[p]     if terminal (deadlock / step-limit)
        1.0 / (1.0 + h)        otherwise  (h = A* Manhattan heuristic)
    """
    if state.is_terminal():
        return state.returns()[perspective_player]

    # Use box-only heuristic to match sokoban_astar.ipynb and expansion ordering
    h = _ag.h_sokoban_box_only(state)
    if h == 0:
        return 1.0
    return 1.0 / (1.0 + h)
```

--- backpropagation ◀ TARGET ---
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with A* g-score sync:
for every node on the path from leaf to root, update

    visited[state_key] = min(g, existing)

so that the expansion phase's visited check always reflects the shallowest
depth at which each state has been reached — matching A*'s invariant that
visited[key] holds the best known path cost.
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


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root + sync A* visited g-scores.

    For each node on the walk:
      • node.visits += 1
      • node.value  += reward  (sign-flipped at opponent nodes for 2-player games;
                                 Sokoban is single-player so sign never flips)
      • visited[state_key] = min(current_g, known_best_g)
    """
    # Determine root player perspective (same convention as default_backpropagation)
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    visited = _ag.get_visited()

    while node is not None:
        node.visits += 1

        mover = node.parent.state.current_player() if node.parent else perspective
        node.value += reward if mover == perspective else -reward

        # ── A* g-sync ─────────────────────────────────────────────────
        key = node.state.state_key()
        g   = _ag.node_depth(node)
        if key not in visited or visited[key] > g:
            visited[key] = g

        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with A* g-score sync:
for every node on the path from leaf to root, update

    visited[state_key] = min(g, existing)

so that the expansion phase's visited check always reflects the shallowest
depth at which each state has been reached — matching A*'s invariant that
visited[key] holds the best known path cost.
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


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root + sync A* visited g-scores.

    For each node on the walk:
      • node.visits += 1
      • node.value  += reward  (sign-flipped at opponent nodes for 2-player games;
                                 Sokoban is single-player so sign never flips)
      • visited[state_key] = min(current_g, known_best_g)
    """
    # Determine root player perspective (same convention as default_backpropagation)
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    visited = _ag.get_visited()

    while node is not None:
        node.visits += 1

        mover = node.parent.state.current_player() if node.parent else perspective
        node.value += reward if mover == perspective else -reward

        # ── A* g-sync ─────────────────────────────────────────────────
        key = node.state.state_key()
        g   = _ag.node_depth(node)
        if key not in visited or visited[key] > g:
            visited[key] = g

        node = node.parent
```

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level6
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level6 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 4 levels): 0.6667

Per-level best composites so far:
  level10: best=0.6667 (baseline=0.6667)
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level3', 'level4', 'level6', 'level7', 'level9']
Mastered levels: ['level5', 'level8']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level5] [simulation]: composite=1.0000, solve_rate=100%, eval_time=0.2s, desc=Added missing deadlock detection helper (`_simple_corner_deadlock`) and ensured required imports are present. The heuristic logic remains unchanged. ← accepted
  Iter 2 [level10] [simulation]: composite=0.0000, solve_rate=0%, eval_time=0.2s, desc=No changes required; the function is correct and efficient. ✗ rejected
  Iter 3 [level8] [simulation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|--------------------------|--------------------------|
| 1️⃣ | *All children are visited 500 times with the same tiny average value (≈0.167).* The tree never branches – every iteration re‑expands the same single line of pushes. | The algorithm prunes almost every alternative action during **expansion**; once a state is marked “visited” the next g‑score is considered *better* and the action is discarded forever. |
| 2️⃣ | *`visited` is updated with `g = _ag.node_depth(node)`* (push‑count only) while **expansion** stores `total_g = parent_depth + walk + 1` (push‑count + walk cost). | The pruning test `if key in visited and visited[key] <= total_g: prune` is overly aggressive because `visited` contains a *smaller* cost (pushes only) than the true cost used for comparison. Many legitimate successors are rejected simply because they require a longer walk, even if they lead to a better heuristic. |
| 3️⃣ | *Back‑propagation does not influence selection:* selection uses `f = g + h` and never looks at `node.value`. | Even if the value were propagated correctly, it would not steer the search; the bottleneck is the premature pruning caused by the mismatched g‑scores. |
| 4️⃣ | *Reward is added unchanged for every node on the path.* This inflates early nodes’ values but, again, values are ignored by the A*‑guided selector. | No measurable effect on the observed behaviour, but it wastes a tiny amount of work. |

**2. ROOT CAUSE**

The **core defect** is the inconsistency between the *cost metric* used for pruning in **expansion** and the *cost metric* stored during **back‑propagation**:

*Expansion* computes  
```python
total_g = parent_depth + walk + 1          # pushes + walking steps
```
and saves `visited[key] = total_g`.

*Back‑propagation* writes  
```python
g = _ag.node_depth(node)                   # pushes only
visited[key] = min(visited.get(key, ∞), g)
```

Because `node_depth` ≤ `total_g`, the visited table almost always contains a *lower* cost than the one being compared later, causing `visited[key] <= total_g` to be true and the action to be pruned permanently. The search therefore collapses to the first branch it discovers and never explores alternatives – exactly what the trace shows.

**3. PROPOSED APPROACH – A (incremental)**  

Fix the mismatch by making back‑propagation store the *same* cost metric that expansion uses. The `GameState` already tracks the cumulative step count (`state.steps`), which equals the sum of all walk steps plus one per push. Therefore:

1. **Compute the true g‑score** for each node as the *total number of steps taken from the root*:
   ```python
   g = node.state.steps - root.state.steps   # or simply node.state.steps if root.steps == 0
   ```
2. **Update the visited table with that g‑score** instead of `node_depth`:
   ```python
   if key not in visited or visited[key] > g:
       visited[key] = g
   ```
3. (Optional but cheap) expose a helper `node_total_cost(node)` that returns `node.state.steps` so the code stays readable.

*Why this works:*  
Now `visited[key]` contains the same metric (`pushes + walks`) that expansion compares against `total_g`. An action will be pruned only if a *cheaper* path (in total steps) to the same state has already been found, exactly mimicking A*’s closed‑list invariant. This eliminates the systematic over‑pruning, allows many previously discarded pushes to be explored, and should break the “single‑line” pattern seen in the trace.

**Additional low‑cost refinements (still incremental):**

* **Depth‑aware reward discount** – multiply `reward` by `γ**node.state.steps` (γ≈0.99) before back‑propagating. Although selection ignores values, a more meaningful value can help any future switch back to a UCB‑based selector without harming the current A*‑guided path.
* **Store `node.total_g` on creation** (e.g., `node.g = parent.g + walk + 1`) so that back‑propagation can read it directly, avoiding reliance on `state.steps` if the environment ever changes.

These tweaks keep the proven A*‑guided selection and simulation intact, only repairing the pruning invariant that is currently crippling exploration on level 6.

------------------------------------------------------------
TASK — IMPROVE THE HEURISTIC
------------------------------------------------------------
Improve the 'backpropagation' function above based on your
analysis. Decide which approach fits best:

OPTION A — Incremental optimization (~70% of the time):
  Start from the CURRENT code and make targeted improvements.
  Examples:
  • Add a deadlock check (box in corner with no target)
  • Add distance-based scoring factors
  • Switch from random rollout to greedy lookahead
  • Add early-termination for stuck states
  • Tune weights or scoring formula
  • Combine multiple heuristic signals

OPTION B — Paradigm shift (~30% of the time):
  If the current approach is fundamentally limited, propose
  a different strategy. Keep any components that work well.
  State clearly in DESCRIPTION why a restructure is needed.

How the 'backpropagation' phase works in MCTS:
  - Called from a LEAF node, receives a game state.
  - Must return a FLOAT reward backpropagated up the tree.
  - Reward MUST vary across states so MCTS can distinguish
    good from bad actions. Flat rewards ≈ random play.
  - Called thousands of times per move — keep it FAST.

CONSTRAINTS:
  • Same function signature as the current code.
  • Standalone function, standard library only.
  • Code can be as long as the heuristic requires — no
    artificial line limit. Prioritize quality over brevity.

You MUST format your response EXACTLY as follows:

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line: what you changed and why>
```python
<your complete function code here>
```

Rules for the header fields:
- ACTION must be either 'create' (brand new tool) or 'modify'
  (improving the existing tool shown above).
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: Fixed cost mismatch by using total steps (push + walk) as the A* g‑score and added optional depth‑discounting of the reward.
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with A* g-score sync:
for every node on the path from leaf to root, update

    visited[state_key] = min(g, existing)

where *g* is the true accumulated step cost (pushes + walk steps) stored
in ``state.steps``.  This matches the ``total_g`` metric used during
expansion, preventing over‑pruning.  Additionally the propagated reward
is optionally discounted by a small factor γ per step to give deeper
nodes a slightly lower contribution (useful if a value‑based selector
is ever re‑introduced).
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Optional

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_backpropagation(node, reward: float, gamma: Optional[float] = 0.99) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1/(1+h) or terminal return).
    gamma : float, optional
        Discount factor per step.  If supplied, the reward is multiplied by
        ``gamma ** depth`` where *depth* is the number of steps taken from
        the root to the current node.  This makes deeper nodes contribute
        slightly less, which can improve value‑based selection without
        affecting the current A*‑guided selector.

    The function updates:
        * node.visits
        * node.value   (adds discounted reward)
        * _ag.visited[state_key]  – the minimum total step cost ``g`` seen
          for that state.
    """
    # ------------------------------------------------------------------
    # 1. Determine the root of the tree (needed for perspective & g‑calc)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    visited = _ag.get_visited()

    # cache root step count once – normally zero but keep generic
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        # Determine which player made the move that led to *node*.
        # For single‑player Sokoban this is always the same player,
        # but we keep the generic formulation.
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # Apply optional discount based on total steps from the root.
        # ``node.state.steps`` holds pushes + walk steps.
        node_steps = getattr(node.state, "steps", 0)
        depth = node_steps - root_steps                      # total steps so far
        discounted_reward = reward * (gamma ** depth) if gamma else reward

        node.value += discounted_reward if mover == perspective else -discounted_reward

        # ---- A* g‑score sync -------------------------------------------
        # Use the same metric that expansion records: total steps (pushes+walks).
        # ``node.state.steps`` already equals that total cost.
        g = node_steps - root_steps   # cost from root to this node

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # move up the tree
        node = node.parent
```
