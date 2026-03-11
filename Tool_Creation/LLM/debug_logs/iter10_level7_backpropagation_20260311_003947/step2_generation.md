# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:40:08 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 5.91s |
| Status     | success |
| Tokens     | 8861 |
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
Description: Fixed the `NameError` by embedding the heuristic logic directly inside `default_expansion` (ensuring it is always defined) and aligned the code with the public `GameState` API.
Generated:   2026-03-11T00:38:15.500868
"""

def default_expansion(node):
    """
    Expand the best untried action using an A*‑style f = g + h scoring,
    with improved pruning and cost estimation.

    Fixes applied:
      • The private helper ``_heuristic`` caused a ``NameError`` because it
        was not guaranteed to be defined before this function was executed.
        The heuristic calculation is now performed inline (identical
        behaviour) to remove the dependency.
      • Uses the public ``GameState`` API (`player`, `steps`, `walls`,
        `boxes`, `total_box_distance`, `state_key`) consistently.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    # Find the root node of the current search tree.
    root = node
    while getattr(root, "parent", None) is not None:
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

    # ---- inline heuristic (replacement for the missing _heuristic) ----
    def _inline_heuristic(state, alpha: float = 0.7, beta: float = 0.3) -> float:
        """
        Weighted heuristic:
            h = α * total_box_distance  +  β * unreachable_boxes
        """
        # total_box_distance is a method of GameState
        box_dist = state.total_box_distance()
        # count boxes that the player cannot reach without pushing
        walls = state.walls
        boxes = state.boxes
        start = state.player
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in walls or nxt in boxes or nxt in seen:
                    continue
                seen.add(nxt)
                q.append(nxt)
        unreachable = sum(1 for b in boxes if b not in seen)
        return alpha * box_dist + beta * unreachable

    parent_h = _inline_heuristic(node.state)               # weighted heuristic

    # ------------------------------------------------------------------ #
    # 2. Scan untried actions
    # ------------------------------------------------------------------ #
    scored = []          # (f, walk, action, next_state, key, total_g)
    pruned = []          # actions removed permanently

    EPSILON = 2          # tolerance for non‑monotone moves

    # ``_untried_actions`` is the internal container used by the existing node
    # implementation; we keep the same attribute name to stay compatible.
    for action in list(getattr(node, "_untried_actions", [])):
        # ---- a) generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- b) dead‑lock checks ----------------------------------------
        if _simple_corner_deadlock(next_state) or _wall_line_deadlock(next_state):
            pruned.append(action)
            continue

        # ---- c) heuristic monotonicity (relaxed) -----------------------
        next_h = _inline_heuristic(next_state)
        if next_h > parent_h + EPSILON:
            pruned.append(action)
            continue

        # ---- d) walk distance to push position -------------------------
        player_start = node.state.player                       # tuple (r, c)
        push_pos = action[0]                                   # cell where player must stand
        walk = _bfs_distance(player_start, push_pos,
                             node.state.walls, node.state.boxes)

        # ---- e) g‑score: accumulate real steps (push count + walk) -----
        total_g = parent_steps + walk + 1                       # +1 for the push itself

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
        if hasattr(node, "_untried_actions") and a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose best action (minimal f, then minimal walk, then deterministic)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        if hasattr(node, "_untried_actions"):
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ---------- fallback when everything was pruned -----------------
        if hasattr(node, "_untried_actions") and node._untried_actions:
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
            if getattr(node, "children", None):
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create child node, register, and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    if getattr(node, "children", None) is not None:
        node.children[chosen_action] = child
    else:
        node.children = {chosen_action: child}
    return child
```

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: Fixed missing imports (`sys`, `deque`) and aligned variable names with the public `GameState` API.
Generated:   2026-03-11T00:39:40.315107
"""

from __future__ import annotations

import importlib.util
import math
import sys                     # ← added
from collections import deque  # ← added
from pathlib import Path

# ── Load shared A* state (may be unused after the rewrite) ─────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag  # retained for compatibility; not used directly


def _bfs_reachable(start, walls, boxes):
    """
    BFS from ``start`` avoiding walls and boxes.
    Returns a dict mapping each reachable cell to its distance.
    """
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        d = dist[cur] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in dist:
                continue
            dist[nxt] = d
            q.append(nxt)
    return dist


def _simple_corner_deadlock(state) -> bool:
    """
    Fast detection of obvious corner dead‑locks:
    a non‑target box that is adjacent to walls on two orthogonal sides.
    """
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


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for Sokoban‑macro MCTS.

    Signals combined:
      • Manhattan distance of boxes to nearest targets.
      • Count of boxes unreachable by the player (without pushing).
      • Estimated minimal walk distance to any legal push position.
      • Heavy penalty for obvious corner dead‑locks.

    The combined heuristic ``h`` is transformed into a smooth reward
    ``exp(-λ·h)`` (λ≈0.3).  Terminal states return the game‑provided reward,
    but obvious dead‑locks are given a small negative value to separate them
    from ordinary step‑limit terminations.
    """
    # ---- Terminal handling -------------------------------------------------
    if state.is_terminal():
        ret = state.returns()[perspective_player]   # 1.0 for solved, 0.0 otherwise
        # Penalise only clear dead‑locks; keep step‑limit terminations at 0.0
        if ret == 0.0 and _simple_corner_deadlock(state):
            return -0.1
        return ret

    # ---- 1️⃣ Box‑to‑target Manhattan distance --------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ---- 2️⃣ Unreachable‑box count -------------------------------------------
    walls = state.walls
    boxes = state.boxes
    reachable = _bfs_reachable(state.player, walls, boxes)
    unreachable_cnt = sum(1 for b in boxes if b not in reachable)

    # ---- 3️⃣ Minimal walk distance to a legal push position ------------------
    min_walk = None
    for action in state.legal_actions():
        push_pos = action[0]                       # cell where player must stand
        if push_pos in reachable:
            d = reachable[push_pos]
            if min_walk is None or d < min_walk:
                min_walk = d
    if min_walk is None:                           # no push possible (should be terminal)
        min_walk = 10.0

    # ---- 4️⃣ Corner dead‑lock penalty ----------------------------------------
    deadlock_penalty = 5.0 if _simple_corner_deadlock(state) else 0.0

    # ---- 5️⃣ Weighted combination --------------------------------------------
    alpha = 1.0   # box distance weight
    beta  = 2.0   # unreachable boxes weight
    gamma = 0.5   # walk distance weight
    delta = 1.0   # dead‑lock penalty scaling
    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) + (delta * deadlock_penalty)

    # ---- 6️⃣ Reward shaping --------------------------------------------------
    lam = 0.3
    reward = math.exp(-lam * h)

    # Guard against numerical underflow
    if reward < 1e-8:
        reward = 0.0
    return reward
```

--- backpropagation ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: No changes needed; the draft implementation is correct and efficient.
Generated:   2026-03-11T00:35:53.297559
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Improvements over the previous version:
      • Do **not** overwrite the A*‑style visited table used during
        expansion. The table stores a g‑score based on pushes + walk
        cost, while the leaf node only knows the total step count.
        Updating it here caused mismatched pruning.  We therefore
        leave the visited dict untouched (it is already maintained
        by ``default_expansion``).
      • Apply an exponential discount based on the *actual number of
        steps* taken from the root, not on abstract push depth.  This
        yields a smoother penalty (γ‑discount) that respects long
        walks but does not overly punish deep but necessary solutions.
      • Keep ``node.value`` as a running **average** instead of a raw
        sum.  This removes the hidden bias introduced by mixing
        discounted and undiscounted rewards and allows the selector
        to use ``node.value`` directly.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Discount factor per step (tuned for Sokoban macro‑push)
    _GAMMA = 0.995

    # ------------------------------------------------------------------
    # 1. Locate the root node to obtain the step offset.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits, discounted average value.
    #    The visited‑pruning table is deliberately left unchanged.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- depth‑based discount --------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        step_depth = max(0, node_steps - root_steps)   # >= 0
        discount = _GAMMA ** step_depth

        # ---- running average of discounted reward ---------------------
        #   new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward * discount) / node.visits

        # Move toward the root
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: backpropagation
Description: No changes needed; the draft implementation is correct and efficient.
Generated:   2026-03-11T00:35:53.297559
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Improvements over the previous version:
      • Do **not** overwrite the A*‑style visited table used during
        expansion. The table stores a g‑score based on pushes + walk
        cost, while the leaf node only knows the total step count.
        Updating it here caused mismatched pruning.  We therefore
        leave the visited dict untouched (it is already maintained
        by ``default_expansion``).
      • Apply an exponential discount based on the *actual number of
        steps* taken from the root, not on abstract push depth.  This
        yields a smoother penalty (γ‑discount) that respects long
        walks but does not overly punish deep but necessary solutions.
      • Keep ``node.value`` as a running **average** instead of a raw
        sum.  This removes the hidden bias introduced by mixing
        discounted and undiscounted rewards and allows the selector
        to use ``node.value`` directly.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Discount factor per step (tuned for Sokoban macro‑push)
    _GAMMA = 0.995

    # ------------------------------------------------------------------
    # 1. Locate the root node to obtain the step offset.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits, discounted average value.
    #    The visited‑pruning table is deliberately left unchanged.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- depth‑based discount --------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        step_depth = max(0, node_steps - root_steps)   # >= 0
        discount = _GAMMA ** step_depth

        # ---- running average of discounted reward ---------------------
        #   new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward * discount) / node.visits

        # Move toward the root
        node = node.parent
```

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 8 levels): 0.8750

Per-level best composites so far:
  level10: best=1.0000 (baseline=1.0000) [MASTERED]
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=0.0000) [MASTERED]
  level7: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level6', 'level7']
Mastered levels: ['level10', 'level2', 'level3', 'level4', 'level5', 'level8', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 7 [level9] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 8 [level7] [simulation]: composite=0.0000, solve_rate=0%, eval_time=0.2s, desc=Fixed missing imports (`sys`, `deque`) and aligned variable names with the public `GameState` API. ← accepted
  Iter 9 [level3] [expansion]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (trace evidence) | Why it hurts the search |
|------|--------------------------|--------------------------|
| 1️⃣ | All children in the first three moves have **avg ≈ 0.0** despite thousands of visits (e.g. `((3,4),0) v=229, avg=0.000`). The only non‑zero averages are tiny (0.036‑0.065) and they belong to actions that do **not** lead to a solution (the final state still has only 1/3 boxes on targets). | The back‑propagated reward that reaches the root is essentially zero, so the selector has no signal to prefer the few promising pushes. |
| 2️⃣ | Simulation returns a **smooth exponential** (`exp(-0.3 · h)`) for non‑terminal states. For a typical heuristic `h≈5‑8` this yields **<0.2**. When averaged with the many *0.0* leaves (dead‑locks, step‑limit) the value collapses to ≈0. | The value function becomes indistinguishable from the default “no information” value, so the A*‑guided selector cannot break ties on quality. |
| 3️⃣ | Back‑propagation **discounts** the leaf reward by `_GAMMA**step_depth` (0.995^d). Even with depth ≈ 20 the discount ≈ 0.90, trimming the already tiny reward further. | The discount amplifies the loss of signal and makes deep but promising branches look worse than shallow dead‑ends. |
| 4️⃣ | The node stores only a **single average** (`node.value`). A single solved leaf (reward = 1.0) is quickly drowned by many low‑reward visits, because the average is *linear* in the number of visits. | The algorithm never recognises that a branch has *ever* reached a solution, so it keeps exploring sub‑optimal branches. |

**2. ROOT CAUSE**

The back‑propagation routine is mathematically correct but **mis‑aligned with the needs of a deterministic puzzle**:

* It treats every leaf as a stochastic payoff, averaging away the *binary* win signal that is most informative for Sokoban.
* The exponential discount (per step) is unnecessary – the simulation already encodes distance‑to‑goal through `h`. Discounting only attenuates the already‑small heuristic‑derived reward.
* By storing only the mean, the backup discards the *maximum* (or win count) information that would allow the selector to prefer branches that have ever produced a win, even if they have been visited many times without reward.

Consequently, the selector (which selects by `f = g + h` and uses `value/visits` only as a tie‑breaker) receives essentially a flat value landscape, leading to the observed uniform visitation and failure to solve level 7.

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement** (the problem is fixable without rewriting the whole MCTS pipeline).

We will augment the back‑propagation routine with three lightweight changes that directly address the weaknesses above while preserving the existing A*‑guided selection and expansion code.

| Change | Rationale |
|--------|-----------|
| **a. Keep a “wins” counter** (`node.wins += 1` when `reward == 1.0`). | Gives each node a binary success metric that survives averaging. |
| **b. Replace the discounted average with a *max‑reward* backup** (`node.best = max(node.best, reward)`). Keep the running average for compatibility, but let the selector (via tie‑break) also see `node.best`. | Guarantees that a branch that has ever solved the puzzle retains a high value, preventing it from being drowned out. |
| **c. Remove the step‑based discount** (`discount = 1.0`). | The simulation’s heuristic already penalises distance; extra discount only erodes signal. |
| **d. Optional – small epsilon boost for wins** (e.g., `reward = 1.0 + 1e‑3` when solved). | Guarantees that a solved leaf’s contribution outranks any non‑solved leaf even after averaging. |

Implementation sketch (no code required):

```text
while node:
    node.visits += 1
    if reward == 1.0:          # solved leaf
        node.wins   = getattr(node, "wins",   0) + 1
        node.best   = max(getattr(node, "best", 0.0), 1.0)
        reward_adj = 1.0 + 1e-3          # tiny boost
    else:
        reward_adj = reward
    # running average (unchanged)
    node.value = ((node.visits-1)*node.value + reward_adj) / node.visits
    node = node.parent
```

**Why this works**

* The **wins / best** fields give the selector a higher‑resolution signal when `f` scores are equal; nodes that have ever led to a solution will dominate tie‑breaks, steering search toward the proven path.
* Removing the discount stops the systematic attenuation of the already‑small heuristic reward, so a solved leaf propagates a stronger signal.
* The epsilon boost ensures that a solved leaf’s contribution to the mean never collapses to zero, preserving its influence even after many non‑solved rollouts.

These changes are **local** (only the back‑propagation file), keep the current A*‑guided selection/exansion untouched, and directly address the observed weakness (near‑zero averages and loss of win information). The expected outcome is a sharper value landscape, faster identification of promising pushes, and ultimately a higher solve rate on level 7 and the remaining unsolved levels.

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
DESCRIPTION: Added win / best tracking, removed step‑based discount, and boosted solved rewards so solved leaves retain a strong signal during averaging.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements:
      • Remove per‑step discounting (the simulation already encodes distance).
      • Keep a per‑node ``wins`` counter and ``best`` value so that any node
        that has ever produced a solved leaf retains a strong signal.
      • Add a tiny epsilon boost to solved rewards to prevent them from being
        washed out by many low‑reward visits when computing the running average.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Small boost for a solved leaf – guarantees it stays above any
    # non‑solved averaged value even after many visits.
    _SOLVED_EPS = 1e-3

    # ------------------------------------------------------------------
    # 1. Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- win / best tracking ---------------------------------------
        if reward == 1.0:                     # leaf is a solved state
            # Increment win count (use getattr to avoid AttributeError)
            node.wins = getattr(node, "wins", 0) + 1
            # Record the best reward seen at this node
            node.best = max(getattr(node, "best", 0.0), 1.0)
            # Apply epsilon boost for the averaging step
            reward_adj = 1.0 + _SOLVED_EPS
        else:
            # No win – keep existing best if any
            node.best = max(getattr(node, "best", 0.0), reward)
            reward_adj = reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the root
        node = node.parent
```
