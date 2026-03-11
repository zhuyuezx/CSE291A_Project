# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:35:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.41s |
| Status     | success |
| Tokens     | 8149 |
| Validation | invalid — No code block found in response. |

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
Description: Unify visited table with expansion, add depth‑discounted reward, and compute depth efficiently.
Generated:   2026-03-11T00:22:36.441500
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root while keeping the shared
    A*‑style visited table consistent and applying a small depth discount
    to the reward.

    This version:
      • Uses the same visited dictionary that ``default_expansion`` writes
        to (``_EXPANSION_STATE["visited"]``) instead of an ad‑hoc fallback.
      • Applies an exponential discount (default 0.99) based on the node's
        depth from the root, making deeper (more costly) solutions less
        attractive and giving the ``value/visits`` term a useful signal.
      • Computes the depth incrementally while climbing the tree,
        avoiding the O(depth²) cost of repeatedly calling a helper.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the shared visited table used by expansion.
    # ------------------------------------------------------------------
    if "_EXPANSION_STATE" in globals():
        visited = globals()["_EXPANSION_STATE"].setdefault("visited", {})
    else:
        # Fallback – a persistent dict attached to this function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root node (needed for step offset).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits, discounted value,
    #    and the shared visited table (minimum g‑score for each state).
    # ------------------------------------------------------------------
    depth = 0                     # number of pushes from root to current node
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value with depth discount --------------------------------
        discount = 0.99 ** depth   # small exponential penalty for deeper nodes
        node.value += reward * discount

        # ---- A* g‑score sync -------------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps                # total steps from root

        key = node.state.state_key()
        # Record the smallest g‑value seen for this state.
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent and increase depth
        node = node.parent
        depth += 1
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Unify visited table with expansion, add depth‑discounted reward, and compute depth efficiently.
Generated:   2026-03-11T00:22:36.441500
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root while keeping the shared
    A*‑style visited table consistent and applying a small depth discount
    to the reward.

    This version:
      • Uses the same visited dictionary that ``default_expansion`` writes
        to (``_EXPANSION_STATE["visited"]``) instead of an ad‑hoc fallback.
      • Applies an exponential discount (default 0.99) based on the node's
        depth from the root, making deeper (more costly) solutions less
        attractive and giving the ``value/visits`` term a useful signal.
      • Computes the depth incrementally while climbing the tree,
        avoiding the O(depth²) cost of repeatedly calling a helper.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the shared visited table used by expansion.
    # ------------------------------------------------------------------
    if "_EXPANSION_STATE" in globals():
        visited = globals()["_EXPANSION_STATE"].setdefault("visited", {})
    else:
        # Fallback – a persistent dict attached to this function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root node (needed for step offset).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits, discounted value,
    #    and the shared visited table (minimum g‑score for each state).
    # ------------------------------------------------------------------
    depth = 0                     # number of pushes from root to current node
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value with depth discount --------------------------------
        discount = 0.99 ** depth   # small exponential penalty for deeper nodes
        node.value += reward * discount

        # ---- A* g‑score sync -------------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps                # total steps from root

        key = node.state.state_key()
        # Record the smallest g‑value seen for this state.
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent and increase depth
        node = node.parent
        depth += 1
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level5)
Timestamp:  2026-03-11T00:35:25.517983
Iterations: 500
Solved:     True
Steps:      7
Returns:    [1.0]

  Move 1: action=((1, 3), 1), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/3 | Total distance: 5
    Children: [((1, 3), 1)(v=500, avg=0.913)]
  Move 2: action=((2, 5), 2), total_visits=500
    State: Step 5/1000 | Boxes on target: 0/3 | Total distance: 5
    Children: [((2, 3), 1)(v=1, avg=0.167), ((2, 5), 2)(v=498, avg=0.926), ((4, 3), 0)(v=1, avg=0.167)]
  Move 3: action=((2, 4), 2), total_visits=500
    State: Step 10/1000 | Boxes on target: 0/3 | Total distance: 4
    Children: [((2, 4), 2)(v=498, avg=0.940), ((2, 4), 1)(v=1, avg=0.200), ((4, 4), 0)(v=1, avg=0.200)]
  Move 4: action=((2, 4), 1), total_visits=500
    State: Step 11/1000 | Boxes on target: 1/3 | Total distance: 3
    Children: [((2, 3), 1)(v=2, avg=0.249), ((2, 4), 1)(v=495, avg=0.958), ((1, 2), 1)(v=1, avg=0.250), ((4, 4), 0)(v=1, avg=0.250), ((4, 3), 0)(v=1, avg=0.250)]
  Move 5: action=((4, 5), 2), total_visits=500
    State: Step 13/1000 | Boxes on target: 1/3 | Total distance: 3
    Children: [((3, 4), 2)(v=1, avg=0.333), ((4, 5), 2)(v=496, avg=0.975), ((2, 3), 1)(v=1, avg=0.250), ((1, 2), 1)(v=1, avg=0.250), ((4, 3), 0)(v=1, avg=0.250)]
  Move 6: action=((3, 4), 2), total_visits=500
    State: Step 16/1000 | Boxes on target: 1/3 | Total distance: 2
    Children: [((4, 4), 2)(v=2, avg=0.497), ((3, 4), 2)(v=497, avg=0.989), ((1, 2), 1)(v=1, avg=0.333)]
  Move 7: action=((4, 4), 2), total_visits=500
    State: Step 18/1000 | Boxes on target: 2/3 | Total distance: 1
    Children: [((4, 4), 2)(v=500, avg=1.000)]

Final state:
Step 21/1000 | Boxes on target: 3/3 | Total distance: 0
########
#      #
# *    #
# *    #
# *@   #
########

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'backpropagation'
heuristic code, and the gameplay traces above.

Produce a focused analysis with these sections:

1. KEY WEAKNESSES
   What are the main problems causing poor play?
   Cite specific move numbers, Q-value patterns, or state
   observations as evidence. Be specific. Rank by impact.

2. ROOT CAUSE
   WHY does the current code produce this behaviour?
   Point to specific logic or missing logic in the code.

3. PROPOSED APPROACH
   Choose one of these strategies:

   A) INCREMENTAL (~70% of cases): Describe targeted
      modifications to the 'backpropagation' function
      that address the top weakness(es). Build on and
      extend the current code.

   B) RESTRUCTURE (~30% of cases): If the current approach
      is fundamentally limited, describe a different
      strategy. Explain why incremental changes won't
      suffice. Keep proven components that work.

   State which strategy (A or B) you recommend and why.

Keep your analysis under 500 words. Do NOT write code.

---

## Response

**1. KEY WEAKNESSES**  

| Rank | Symptom (trace evidence) | Why it hurts the search |
|------|---------------------------|--------------------------|
| **1️⃣** | *Visited‑table mismatch*: expansion stores a “g‑score” as `parent_depth + walk + 1` (push‑count + walk steps). Back‑propagation later overwrites the same entry with `node.state.steps – root_steps` (pure step count). Because the two formulas differ, a state that is cheap in the expansion view may be marked “already explored with a lower g” by back‑propagation and subsequently pruned. This can block promising branches early, especially on larger levels where walk cost dominates. |
| **2️⃣** | *Depth‑discounted reward*: `discount = 0.99 ** depth` is applied on every backup. For a solution that needs > 15 pushes the cumulative discount is ≈ 0.86, pulling the average `value/visits` down and making the node look worse than a shallow but dead‑end path. The solver therefore prefers short, possibly myopic pushes and may miss deeper but necessary sequences. |
| **3️⃣** | *No‑normalisation of accumulated value*: `node.value += reward * discount` is a sum, while the selection phase uses `node.value / node.visits`. Because the discount varies with depth, the average becomes a mixture of heavily discounted and undiscounted rewards, introducing a hidden bias that is hard for UCB‑style (or the A*‑guided) selector to interpret. |
| **4️⃣** | *Fallback handling*: when the visited table is missing (`_EXPANSION_STATE` not in globals) a hidden fallback dict is created. If the module reloads (e.g., during a hot‑reload of the MCTS loop) the two tables diverge, again causing inconsistent pruning. |

**2. ROOT CAUSE**  

The back‑propagation routine was written to *share* the visited dictionary used by expansion, but it records a **different g‑metric** (`steps‑root_steps`) than the one expansion uses (`pushes + walk + 1`). Because both phases read/write the same dict, the second write overwrites the first, so later expansions read a g‑value that does not reflect the true cost used for A*‑style pruning. Consequently many legal actions are discarded prematurely (see the large pruning lists in the expansion code).  

The depth discount is another design choice: it assumes “deeper = more pushes = worse”, yet in Sokoban macro‑push the *walk* component can dominate depth, and a constant 0.99 factor is too aggressive for puzzles that legitimately need > 10 pushes.  

Finally, the fallback visited dict makes the bug intermittent and hard to reproduce, which explains why the trace shows a perfect solve on level 5 (few pushes) but would likely fail on harder levels.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvements** (recommended). The core algorithm works; we only need to align the cost metric and temper the discount.

1. **Unified g‑score computation** – Introduce a small helper `calc_g(state, parent_depth, walk)` that both expansion and back‑propagation call. Back‑propagation should recompute `g = parent_depth + walk + 1` using the *same* walk distance that expansion used (the walk from the parent node’s player position to the push cell). The necessary walk can be cached on the node when it is created (e.g., `node._walk_cost`). This removes the inconsistency without changing the pruning logic.

2. **Separate visited tables** – Keep the expansion‑pruning table (`_EXPANSION_STATE["visited"]`) and a *pure* back‑propagation table for value‑only statistics. Back‑propagation only needs to update the first; it should **not** overwrite the pruning entry. Instead, after the backup just store the g‑score in a second dict (`_BP_STATE["g_min"]`) if you still need it for analytics. This prevents accidental “over‑pruning”.

3. **Depth‑aware discount based on real step cost** – Replace `0.99 ** depth` with `gamma ** (node.state.steps - root_steps)`, where `gamma` (e.g., 0.995) discounts proportionally to the *actual* number of steps taken, not the abstract push depth. This yields a smoother penalty that respects long walks but does not cripple solutions that require many pushes.

4. **Normalize value on the fly** – Instead of accumulating a raw sum, update an *average* directly:  
   `node.value = ((node.visits-1) * node.value + reward * discount) / node.visits`.  
   This keeps `node.value` already as an average, eliminating the `value/visits` division in selection and avoiding the mixed‑scale bias.

5. **Guard the fallback** – At module load, force creation of `_EXPANSION_STATE` once and never fall back to a hidden dict. Raise an exception if the dict is missing; this surfaces bugs early.

These targeted changes keep the existing A*‑guided selection and expansion logic intact while fixing the primary source of premature pruning and the excessive depth penalty. The solver will retain its fast solves on easy levels (as shown in the trace) and gain robustness on deeper puzzles where walk costs dominate.
