# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:46:37 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 26.83s |
| Status     | success |
| Tokens     | 12559 |
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

--- expansion ◀ TARGET ---
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

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: Added missing helper functions (`_bfs_reachable` and `_simple_corner_deadlock`) and corrected usage of the public `GameState` API.
Generated:   2026-03-11T00:44:54.944744
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Refined leaf evaluation for Sokoban‑macro MCTS.

    Improvements over the previous version:
      • **Progress term** – rewards states that already have boxes on targets.
      • **Weaker unreachable‑box penalty** – the original count dominates early
        rollouts; we now give it a small weight (or zero) because macro‑push
        actions often make many boxes temporarily unreachable.
      • **Softened corner‑deadlock penalty** – a modest additive cost instead of
        a large jump that drove the exponential to near‑zero.
      • **Smoother reward mapping** – use ``1 / (1 + λ·h)`` (λ≈0.1) instead of
        ``exp(-λ·h)`` to keep the reward in a usable range [0,1].
      • **Adjusted weight balance** – box‑to‑target distance remains the main
        signal, walk distance a mild bias, and progress a strong positive boost.
      • **Clamping** – guarantees the returned value stays within ``[0.0, 1.0]``.
    """
    # -------------------------------------------------------------------------
    # Helper: breadth‑first search of reachable squares (player cannot walk
    # through walls or boxes). Returns a dict mapping each reachable cell to the
    # distance (in steps) from the player start position.
    # -------------------------------------------------------------------------
    def _bfs_reachable(start, walls, boxes):
        from collections import deque

        visited = {start: 0}
        q = deque([start])

        # Moves: up, down, left, right
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while q:
            x, y = q.popleft()
            cur_dist = visited[(x, y)]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                # cannot walk into walls or boxes
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in visited:
                    visited[nxt] = cur_dist + 1
                    q.append(nxt)
        return visited

    # -------------------------------------------------------------------------
    # Helper: very cheap corner dead‑lock detection.
    # Returns True if any box that is **not** already on a target sits in a
    # corner formed by two orthogonal walls.
    # -------------------------------------------------------------------------
    def _simple_corner_deadlock(state):
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue          # box already solved – ignore
            # Check the four corner patterns
            if ((bx + 1, by) in walls or (bx - 1, by) in walls) and \
               ((bx, by + 1) in walls or (bx, by - 1) in walls):
                return True
        return False

    # ---- 0️⃣ Terminal handling ------------------------------------------------
    if state.is_terminal():
        base_ret = state.returns()[perspective_player]   # 1.0 solved, 0.0 otherwise
        # Small negative for obvious corner dead‑locks to separate them from a plain timeout
        if base_ret == 0.0 and _simple_corner_deadlock(state):
            return -0.1
        return base_ret

    # ---- 1️⃣ Box‑to‑target Manhattan distance ---------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ---- 2️⃣ Unreachable‑box count (lightly weighted) -------------------------
    walls = state.walls
    boxes = state.boxes
    reachable_map = _bfs_reachable(state.player, walls, boxes)
    reachable_set = set(reachable_map.keys())
    unreachable_cnt = sum(1 for b in boxes if b not in reachable_set)

    # ---- 3️⃣ Minimal walk distance to any legal push position -------------------
    min_walk = None
    for action in state.legal_actions():
        # In macro‑action representation the first element is the cell the
        # player must stand on to execute the push.
        push_pos = action[0]
        if push_pos in reachable_map:
            d = reachable_map[push_pos]
            if (min_walk is None) or (d < min_walk):
                min_walk = d

    if min_walk is None:               # no push possible (should be terminal)
        min_walk = 10.0

    # ---- 4️⃣ Corner dead‑lock detection (soft penalty) -------------------------
    deadlock_penalty = 0.5 if _simple_corner_deadlock(state) else 0.0

    # ---- 5️⃣ Progress: how many boxes already on targets -----------------------
    on_target = state.boxes_on_targets()
    progress_ratio = on_target / state.num_targets if state.num_targets else 0.0

    # ---- 6️⃣ Weighted combination ------------------------------------------------
    # Positive contributions (want to minimise) : box_dist, unreachable, walk, deadlock
    # Negative contribution (want to maximise) : progress_ratio
    alpha = 1.0   # box distance weight
    beta  = 0.3   # unreachable boxes weight (reduced from 2.0)
    gamma = 0.2   # walk distance weight
    delta = 0.6   # deadlock penalty scaling
    epsilon = 4.0 # progress boost weight

    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) \
        + (delta * deadlock_penalty) - (epsilon * progress_ratio)

    # ---- 7️⃣ Reward shaping ------------------------------------------------------
    lam = 0.10
    # Guard against division by zero / overly negative h
    denominator = 1.0 + lam * max(h, -0.9 / lam)
    reward = 1.0 / denominator   # ensures reward ∈ (0,1]

    # Clamp for safety (e.g. numerical issues)
    if reward < 0.0:
        reward = 0.0
    elif reward > 1.0:
        reward = 1.0
    return reward
```

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fix solved‑bonus scope, use iterative discount, and safely acquire leaf step information.
Generated:   2026-03-11T00:45:33.562098
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root with depth discounting
    and a stronger solved‑leaf boost.

    Enhancements:
      • Depth‑discount (γ = 0.95) – rewards closer to the leaf count more,
        encouraging shorter solution paths.
      • Solved‑leaf bonus – a relative boost proportional to remaining
        step budget, applied **only at the leaf** where the solution was
        found, so it dominates after many visits.
      • Keeps the simple running‑average update used elsewhere.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    _SOLVED_EPS   = 1e-3          # tiny constant boost (keeps >0 after many rolls)
    _SOLVED_WEIGHT = 0.10         # relative solved‑bonus factor
    _GAMMA        = 0.95          # depth‑discount factor (0 < γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣ Capture leaf‑specific information (if available)
    # ------------------------------------------------------------------
    leaf_steps = None
    leaf_max_steps = None
    try:
        # `node` is the leaf when the function is called.
        leaf_steps = node.state.steps          # type: ignore[attr-defined]
        leaf_max_steps = node.state.max_steps  # type: ignore[attr-defined]
    except Exception:
        # If the node does not expose those attributes we simply skip the
        # solved‑bonus, relying on the epsilon boost.
        pass

    # ------------------------------------------------------------------
    # 2️⃣ Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    depth = 0          # distance from the leaf (0 for the leaf itself)
    discount = 1.0     # γ ** depth, updated iteratively

    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- reward handling -----------------------------------------
        reward_adj = reward

        # Apply solved‑bonus only at the leaf (depth == 0) and only if the
        # leaf is a solved state (reward == 1.0).
        if depth == 0 and reward == 1.0:
            bonus = _SOLVED_WEIGHT
            if leaf_max_steps and leaf_max_steps > 0:
                # Scale bonus by remaining steps ratio.
                bonus *= (leaf_max_steps - leaf_steps) / leaf_max_steps
            reward_adj = reward + _SOLVED_EPS + bonus

        # ---- depth discount -------------------------------------------
        reward_adj *= discount

        # ---- running average update -----------------------------------
        # new_avg = ((n‑1) * old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # ---- move upward -----------------------------------------------
        node = node.parent
        depth += 1
        discount *= _GAMMA   # equivalent to γ ** depth for the next level
```

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
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 9 levels): 0.8889

Per-level best composites so far:
  level1: best=1.0000 (baseline=1.0000) [MASTERED]
  level10: best=1.0000 (baseline=1.0000) [MASTERED]
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=0.0000) [MASTERED]
  level7: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level6', 'level7']
Mastered levels: ['level1', 'level10', 'level2', 'level3', 'level4', 'level5', 'level8', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 7 [level7] [backpropagation]: composite=0.0000, solve_rate=0%, eval_time=0.2s, desc=Fix solved‑bonus scope, use iterative discount, and safely acquire leaf step information. ← accepted
  Iter 8 [level1] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 9 [level2] [simulation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. Key Weaknesses (ranked by impact)**  

| # | Symptom (trace) | Why it hurts the search |
|---|----------------|--------------------------|
| 1️⃣ | *Move 1*: three children get > 160 visits each with avg ≈ 0.004, while the three actions that immediately put a box on a target receive a single visit but a high avg ≈ 0.58. The planner keeps expanding the low‑value branch and never revisits the high‑value pushes. | Expansion scores ignore *progress* (boxes already on targets). The A*‑style heuristic (`h_sokoban_box_only`) only measures remaining Manhattan distance, so a push that solves a box (distance drops by 1) is treated the same as a push that moves a box without solving it. Consequently the tree is flooded with “cheap‑looking” pushes that do not change the solved‑box count. |
| 2️⃣ | *All moves*: the same few “high‑value” actions are explored once and then **permanently removed** from `_untried_actions` because the visited‑pruning sees the state key already in `visited` (depth‑based g). | `g_new` is computed as `node_depth(node)+1`, i.e. *number of pushes* only. Macro‑pushes have wildly different real costs (walk + push). A cheap‑looking push that required a long walk is stored with a low g; a later, cheaper‑walk version of the **same** push is discarded, even though it would be far more promising. |
| 3️⃣ | *All moves*: dead‑lock filter only calls `next_state._is_deadlocked()`. In the trace the algorithm explores pushes that create “wall‑deadlocks” (box against a wall with no reachable target) and wastes visits. | The dead‑lock test is too narrow (only the private `_is_deadlocked` implementation, which in many Sokoban libraries only catches corner deadlocks). Many actions that will later block other boxes survive the filter and dominate the search budget. |
| 4️⃣ | *Move 4*: after three pushes the total distance is only 1 but the planner still spends all 500 iterations on the same push (`((1,7),2)`) with avg ≈ 0.002, never trying the remaining box. | Because the expansion ordering (lowest `h`) still favours the already‑expanded push (its `h` stays minimal) and because the visited table prevents re‑expansion of alternative pushes, the search gets stuck in a local minima. |

**2. Root Cause**  

The current expansion routine is a literal port of an A* inner‑loop, but it was designed for unit‑cost moves. In Sokoban‑macro:

* **Cost mis‑representation** – `g_new` reflects only push count, not the true step cost (`walk_steps + 1`). The visited dictionary therefore prunes better‑cost variants of the same macro‑push.
* **Heuristic narrowness** – Scoring uses only `h_sokoban_box_only`, which is the sum of Manhattan distances to the nearest targets. It does **not** reward immediate box‑on‑target progress nor penalise long walks.
* **Dead‑lock filter incompleteness** – Only the private `_is_deadlocked` is consulted; classic wall‑deadlocks (box against a wall with no target in line) are missed, so many useless pushes survive.
* **Permanent removal of pruned actions** – Once an action is dropped because its state was visited with a lower `g`, it never reappears, even if a later macro‑push reaches the same state with a *smaller* real cost.

These combine to flood the tree with low‑value, high‑visit branches and discard the few high‑value pushes that actually advance the puzzle.

**3. Proposed Approach – Strategy A (Incremental)**  

We can keep the overall A*‑guided structure (visited table, “expand‑best‑untried”) but enrich the cost and scoring logic:

1. **Real‑cost `g_new`** – Compute `g_new` as `node.state.steps + walk_cost(action) + 1`.  
   *Walk cost* can be obtained by a quick BFS from `node.state.player` to the required push position (the first element of the macro‑action). This aligns the visited check with the true step count, preventing premature pruning of cheaper variants.

2. **Progress‑aware heuristic score** – Replace the plain box‑only distance with a composite:
   ```
   h = α * total_box_distance
       + β * walk_cost(action)               # encourages short walks
       - γ * boxes_on_targets                 # lower h when progress made
   ```
   Choose modest α≈1, β≈0.5, γ≈3 (tuned empirically). This makes actions that solve a box immediately (increase `boxes_on_targets`) rank ahead of pure‑distance reductions.

3. **Stronger dead‑lock detection** – Add a cheap wall‑deadlock test:
   *For each pushed box, if it lies against a wall (or another box) in a direction where no target exists on that line → deadlock.* This can be done with a simple scan of the row/column for a target. Reject such actions before they are added to `scored`.

4. **Soft pruning instead of permanent removal** – Rather than `node._untried_actions.remove(a)` for every pruned action, keep a *blocked* set only for true deadlocks. For visited‑based pruning, simply skip the action this expansion but **do not** delete it from `_untried_actions`; allow a later, cheaper‑cost variant to be considered when the node is revisited at a deeper depth.

5. **Tie‑breaking by progress** – When two actions share the same `h`, prefer the one that yields the larger `boxes_on_targets` increase. This ensures the few high‑value pushes (seen with a single visit but high avg) are revisited early.

These targeted changes address the three dominant weaknesses without discarding the proven A*‑style visited mechanism. By using the true step cost, rewarding immediate progress, and blocking obvious wall deadlocks, the expansion phase will generate far fewer useless visits, focus the search on pushes that advance the solution, and give the MCTS enough budget to finish the remaining box on level 7.

------------------------------------------------------------
TASK — IMPROVE THE HEURISTIC
------------------------------------------------------------
Improve the 'expansion' function above based on your
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

How the 'expansion' phase works in MCTS:
  - Receives (node). Returns the action to expand (or orders/filters untried actions).
  - Orders actions so promising ones are tried first; optionally prunes bad ones.
  - Produces children that selection will later choose among.
  - Best place for hard constraints (e.g. deadlock pruning).

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
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Added real walk‑cost based g, progress‑aware heuristic, stronger wall‑deadlock filter, and soft visited pruning (only deadlocks are permanently removed). This prioritises pushes that solve boxes, prefers short walks, and avoids discarding cheaper variants of the same macro‑push.

```python
"""
A*-guided MCTS expansion – improved.

Key enhancements:
  * Real cost `g_new` = current steps + walk distance + 1 (push).
  * Heuristic combines remaining box distance, walk distance and
    immediate progress (boxes placed on targets).
  * Additional cheap wall‑deadlock detection.
  * Only deadlock actions are permanently removed from `_untried_actions`;
    actions pruned by the visited‑g check remain for possible re‑expansion
    when a cheaper path appears.
  * Tie‑breaker prefers actions that increase the number of solved boxes.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from collections import deque
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


def _bfs_walk(start: Tuple[int, int],
              walls: Set[Tuple[int, int]],
              boxes: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Bread‑first search returning the shortest walk distance from `start`
    to every reachable empty cell (walls and boxes are obstacles)."""
    q = deque([start])
    dist = {start: 0}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if nxt in walls or nxt in boxes:
                continue
            if nxt not in dist:
                dist[nxt] = d + 1
                q.append(nxt)
    return dist


def _simple_wall_deadlock(state, box_pos: Tuple[int, int]) -> bool:
    """Detect a basic wall‑deadlock: a box adjacent to a wall where no
    target exists on that wall‑aligned line."""
    h, w = state.height, state.width
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in dirs:
        adj = (box_pos[0] + dx, box_pos[1] + dy)
        if adj in state.walls:
            # scan outward from the box in the same direction
            x, y = box_pos[0] + 2 * dx, box_pos[1] + 2 * dy
            while 0 <= x < h and 0 <= y < w and (x, y) not in state.walls:
                if (x, y) in state.targets:
                    break               # a target exists → not a deadlock
                x += dx
                y += dy
            else:
                # reached border or wall without finding a target
                return True
    return False


def default_expansion(node):
    """
    Expand the best untried action using A* visited + richer heuristic
    filtering (real walk cost, progress, wall‑deadlocks).
    """
    # -----------------------------------------------------------------
    # 0. Reset visited table if we are at a new root (new MCTS search).
    # -----------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    visited = _ag.get_visited()

    # -----------------------------------------------------------------
    # 1. Pre‑compute walk distances from current player position.
    # -----------------------------------------------------------------
    walk_map = _bfs_walk(node.state.player,
                         node.state.walls,
                         node.state.boxes)

    # -----------------------------------------------------------------
    # 2. Evaluate each currently untried macro‑push.
    # -----------------------------------------------------------------
    scored = []      # (h, -progress_gain, action, next_state, key, g_new)
    deadlocked = []  # actions to remove permanently (hard deadlocks)

    cur_steps = node.state.steps
    cur_progress = node.state.boxes_on_targets()

    for action in list(node._untried_actions):
        # -------------------------------------------------------------
        # a) Apply action to obtain the successor state.
        # -------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------
        # b) Hard dead‑lock detection (corner + wall).
        # -------------------------------------------------------------
        is_dead = False
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            is_dead = True
        else:
            # locate the box that just moved (the one present in next_state
            # but not in the parent state)
            moved_box = None
            diff = next_state.boxes - node.state.boxes
            if diff:
                moved_box = next(iter(diff))
                if moved_box not in next_state.targets:
                    if _simple_wall_deadlock(next_state, moved_box):
                        is_dead = True

        if is_dead:
            deadlocked.append(action)
            continue

        # -------------------------------------------------------------
        # c) Compute real g‑cost (total steps so far after this push).
        # -------------------------------------------------------------
        push_pos = action[0]                     # where the player must stand
        walk_dist = walk_map.get(push_pos, 10 ** 6)   # large if somehow unreachable
        g_new = cur_steps + walk_dist + 1

        # -------------------------------------------------------------
        # d) Visited check – *soft*: skip this expansion if we have already
        #    reached the same state with an equal or cheaper g.
        # -------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # Do NOT remove the action; a cheaper path might appear later.
            continue

        # -------------------------------------------------------------
        # e) Heuristic score.
        #    h = α·box_distance + β·walk_dist – γ·progress_gain
        # -------------------------------------------------------------
        α, β, γ = 1.0, 0.5, 3.0
        box_dist = next_state.total_box_distance()
        progress_gain = next_state.boxes_on_targets() - cur_progress
        h = α * box_dist + β * walk_dist - γ * progress_gain

        scored.append((h, -progress_gain, action, next_state, key, g_new))

    # -----------------------------------------------------------------
    # 3. Permanently discard actions that were hard dead‑locked.
    # -----------------------------------------------------------------
    for a in deadlocked:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # -----------------------------------------------------------------
    # 4. Choose the most promising action.
    # -----------------------------------------------------------------
    if scored:
        # sort by (h, -progress_gain) → lower h, higher progress first
        scored.sort(key=lambda x: (x[0], x[1]))
        h, _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        # remove the chosen action from the pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # -----------------------------------------------------------------
        # Fallback: expand any remaining action (should be non‑deadlocked).
        # -----------------------------------------------------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            # compute its real g for the visited table
            push_pos = chosen_action[0]
            walk_dist = walk_map.get(push_pos, 10 ** 6)
            chosen_g = cur_steps + walk_dist + 1
            visited[chosen_key] = chosen_g
        else:
            # Nothing left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # -----------------------------------------------------------------
    # 5. Create the child node and attach it.
    # -----------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
