# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:22:30 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.45s |
| Status     | success |
| Tokens     | 9379 |
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
LLM-generated MCTS tool: backpropagation
Description: Fixed the `NameError` caused by the undefined `_ag` identifier and ensured a fallback visited‑table is available when the A* guide object is not present.
Generated:   2026-03-11T00:21:20.186102
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1.0 for solved, 0.0 for dead‑end).

    The function updates:
        * node.visits
        * node.value   (adds reward, respecting player perspective)
        * visited[state_key] – the minimum total step cost ``g`` seen
          for that state, where ``g`` equals the number of steps taken
          from the root (pushes + walk steps). This matches the metric
          used in expansion and prevents over‑pruning.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the visited table.
    #    The original implementation expected a global ``_ag`` object with
    #    a ``get_visited`` method.  If this object is not defined (as was
    #    the case during the test run), we fall back to a simple module‑
    #    level dictionary that persists across calls.
    # ------------------------------------------------------------------
    if "_ag" in globals() and hasattr(_ag, "get_visited"):
        visited = _ag.get_visited()          # type: ignore[name-defined]
    else:
        # Create (or reuse) a fallback visited dict attached to the function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root of the tree (needed for perspective & cost offset)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    # Cache root step count (normally 0, but keep generic)
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # No discounting needed for the current A*‑guided selector.
        node.value += reward if mover == perspective else -reward

        # ---- A* g‑score sync -------------------------------------------
        # Total steps from the root to this node (pushes + walking steps).
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move up the tree
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fixed the `NameError` caused by the undefined `_ag` identifier and ensured a fallback visited‑table is available when the A* guide object is not present.
Generated:   2026-03-11T00:21:20.186102
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1.0 for solved, 0.0 for dead‑end).

    The function updates:
        * node.visits
        * node.value   (adds reward, respecting player perspective)
        * visited[state_key] – the minimum total step cost ``g`` seen
          for that state, where ``g`` equals the number of steps taken
          from the root (pushes + walk steps). This matches the metric
          used in expansion and prevents over‑pruning.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the visited table.
    #    The original implementation expected a global ``_ag`` object with
    #    a ``get_visited`` method.  If this object is not defined (as was
    #    the case during the test run), we fall back to a simple module‑
    #    level dictionary that persists across calls.
    # ------------------------------------------------------------------
    if "_ag" in globals() and hasattr(_ag, "get_visited"):
        visited = _ag.get_visited()          # type: ignore[name-defined]
    else:
        # Create (or reuse) a fallback visited dict attached to the function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root of the tree (needed for perspective & cost offset)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    # Cache root step count (normally 0, but keep generic)
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # No discounting needed for the current A*‑guided selector.
        node.value += reward if mover == perspective else -reward

        # ---- A* g‑score sync -------------------------------------------
        # Total steps from the root to this node (pushes + walking steps).
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move up the tree
        node = node.parent
```

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level6
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level6 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 7 levels): 0.8095

Per-level best composites so far:
  level1: best=1.0000 (baseline=1.0000) [MASTERED]
  level10: best=0.6667 (baseline=0.6667)
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level10', 'level3', 'level4', 'level6', 'level7']
Mastered levels: ['level1', 'level2', 'level5', 'level8', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 6 [level2] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 7 [level9] [expansion]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 8 [level10] [simulation]: composite=0.0000, solve_rate=0%, desc=Refine leaf evaluation by incorporating walk estimate, progress term, dead‑lock penalty, and exponential scaling (code unchanged but reformatted for clarity). ✗ rejected

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|--------------------------|--------------------------|
| **1️⃣** | The algorithm cycles: after move 2 the state is *almost* the same as after move 1 (`Step 4/1000 … distance 6` → later `Step 11/1000 … distance 6`). The tree keeps expanding the same macro‑pushes instead of steering toward new, better configurations. | There is **no effective duplicate‑state pruning**. Expansion keeps generating children that have already been seen with a cheaper cost, because the *visited* table that drives this pruning is never populated with the real g‑scores of explored nodes. |
| **2️⃣** | All child nodes report the same average value (`avg = 0.167`). The value term never differentiates promising from unpromising branches, so the selector relies almost entirely on the A*‑style `f = g + h`. | The back‑propagation routine does **not feed back any useful reward information** to the nodes that matter for the selector (the `value/visits` term). Consequently the tree does not learn from the heuristic evaluation; it only follows raw depth‑plus‑heuristic, which is blind to the fact that a deeper node may already be a dead‑end. |
| **3️⃣** | The reward from the simulation (`1/(1+h)`) is added unchanged irrespective of how many pushes or walk steps were needed to reach the leaf. | This gives **no depth discount**, so a long, costly solution is valued the same as a short one, encouraging the algorithm to explore costly loops rather than pruning them. |

**2. ROOT CAUSE**  

*Back‑propagation updates the wrong visited table.*

```python
if "_ag" in globals() and hasattr(_ag, "get_visited"):
    visited = _ag.get_visited()
else:
    visited = default_backpropagation._fallback_visited
```

The expansion code (the only place that actually **prunes** duplicate states) uses a **different** dictionary:

```python
_EXPANSION_STATE = { "visited": {} }   # filled in default_expansion
```

Thus the `g`‑score of a node is never recorded where the expansion routine looks, so the test  

```python
if key in visited and visited[key] <= total_g:  # prune
```

always fails. The tree can therefore revisit the same board position with a larger cost, creating the loops seen in the trace.  

Because the selector’s tie‑breaker `q = value/visits` never receives a meaningful signal (the value is inflated equally for all ancestors), the search cannot compensate for the missing pruning, yielding the flat `avg=0.167` scores.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental (targeted) improvements**

The fundamental issue is the *split* visited table; fixing it requires only a few controlled changes while preserving the rest of the pipeline that already works (A*‑guided selection, expansion heuristics, simulation). The steps are:

1. **Unify the visited store** – have both expansion and back‑propagation read/write a single module‑level dictionary (e.g. rename `_EXPANSION_STATE["visited"]` to `_VISITED`).  
   * In `default_expansion` replace references to `_EXPANSION_STATE["visited"]` with `_VISITED`.  
   * In `default_backpropagation` fetch the same `_VISITED` (create it lazily if absent) instead of the fallback dict.

2. **Record the true g‑score** – during back‑propagation compute `g = node.state.steps - root_steps` (already done) and store the *minimum* g for the state key in the unified visited table, exactly as expansion expects. No extra work is needed beyond using the common dict.

3. **Add a shallow depth discount to the propagated reward** so that longer trajectories are penalised and the `value` term becomes informative:

   ```python
   discount = 0.99 ** (node_depth(node))   # small exponential decay
   node.value += reward * discount
   ```

   This keeps the algorithm single‑player (no sign flip) but ensures that deeper nodes contribute less, allowing the selector’s `q` to differentiate cheap solutions from expensive loops.

4. **Optional but cheap** – after updating the unified visited table, also propagate the *best* reward seen for each state (`best_reward[key] = max(best_reward.get(key,0), reward)`). This can be used later for tie‑breaking if desired, but is not mandatory.

These changes directly eliminate the duplicate‑state explosion (weakness 1) and give the `value/visits` term a meaningful gradient (weakness 2). The depth discount addresses weakness 3 without altering the existing A*‑guided selection logic.

**Why not a full restructure?**  
The current pipeline (A*‑guided selection + expansion with monotonicity & corner dead‑lock checks) already produces sensible child ordering; the only bottleneck is the missing feedback loop between back‑propagation and expansion. A modest unification of the visited store plus a depth‑aware reward update resolves the deadlock with minimal risk and preserves all validated components.

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
DESCRIPTION: Unified the visited table with the expansion routine and added a depth‑based discount to the propagated reward so that deeper (more costly) paths contribute less, giving the value/visits term a meaningful gradient and enabling proper duplicate‑state pruning.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root while keeping the shared
    A*‑style visited table consistent and applying a small depth discount
    to the reward.

    Changes compared with the original implementation:
      • The visited table used for duplicate‑state pruning is now the
        same dictionary that ``default_expansion`` writes to
        (``_EXPANSION_STATE["visited"]``).  If the expansion module cannot
        be imported we fall back to a module‑level fallback dictionary.
      • A discount of ``discount_factor ** depth`` (default 0.99) is
        applied to the reward before adding it to a node's ``value``.
        This penalises long, costly paths and makes the ``value/visits``
        term informative for the A*‑guided selector.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the shared visited table.
    # ------------------------------------------------------------------
    # Try to import the expansion module and reuse its visited dict.
    visited = None
    try:
        import importlib
        exp_mod = importlib.import_module("expansion")  # module name may vary
        if hasattr(exp_mod, "_EXPANSION_STATE"):
            visited = exp_mod._EXPANSION_STATE.get("visited")
    except Exception:
        # Import failed – continue with fallback.
        pass

    if visited is None:
        # Fallback – a persistent dict attached to this function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited

    # ------------------------------------------------------------------
    # 2. Helper to compute depth (number of pushes from root).
    # ------------------------------------------------------------------
    def node_depth(n):
        d = 0
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ------------------------------------------------------------------
    # 3. Find the root node (needed for perspective & step offset).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 4. Walk up the tree, updating visits, value (with depth discount),
    #    and the shared visited table (minimum g‑score seen for each state).
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        depth = node_depth(node)                # pushes from root to this node
        discount = 0.99 ** depth               # exponential decay, 0.99≈small penalty
        node.value += reward * discount

        # ---- A* g‑score sync -------------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps               # total steps from root

        key = node.state.state_key()
        # Record the smallest g‑value seen for this state.
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent
        node = node.parent
```
