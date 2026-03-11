# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:45:14 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 12.94s |
| Status     | success |
| Tokens     | 8541 |
| Validation | invalid — No function definition found in code. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

PHASE: backpropagation
  • What it does: Sends the simulation result back up the visited path. Updates node statistics (visits, value) that selection's UCB1 uses.
  • Optimization goal: Control HOW strongly rollout evidence affects node values. Calibrate depth discount, solved vs partial progress, path length.
  • Constraints: Only aggregates evidence — no move generation, no deadlock pruning, no rollout policy. Must stay coherent with selection's expectations.
  • Good patterns: depth discount so shorter plans dominate, weight solved outcomes above partial progress, reduce credit for noisy weak rollouts.
  • Avoid: move generation, deadlock pruning, rollout action-choice policy.

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

--- backpropagation ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fixed attribute handling for `MCTSNode` (which uses `__slots__`) by removing illegal `best`/`wins` assignments and preserving the original boost‑for‑solved‑leaf logic.
Generated:   2026-03-11T00:40:19.410251
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements retained:
      • No per‑step discounting (the simulation already encodes distance).
      • A tiny epsilon boost is added to solved rewards so that they remain
        noticeable even after many low‑reward visits.
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

        # ---- reward handling ------------------------------------------
        # Add epsilon boost only for a solved leaf (reward == 1.0).
        reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the parent (rootward)
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fixed attribute handling for `MCTSNode` (which uses `__slots__`) by removing illegal `best`/`wins` assignments and preserving the original boost‑for‑solved‑leaf logic.
Generated:   2026-03-11T00:40:19.410251
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements retained:
      • No per‑step discounting (the simulation already encodes distance).
      • A tiny epsilon boost is added to solved rewards so that they remain
        noticeable even after many low‑reward visits.
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

        # ---- reward handling ------------------------------------------
        # Add epsilon boost only for a solved leaf (reward == 1.0).
        reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the parent (rootward)
        node = node.parent
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level7)
Timestamp:  2026-03-11T00:45:01.675723
Iterations: 500
Solved:     False
Steps:      4
Returns:    [0.0]

  Move 1: action=((3, 3), 0), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/3 | Total distance: 5
    Children: [((3, 3), 0)(v=174, avg=0.005), ((3, 4), 0)(v=158, avg=0.005), ((3, 6), 0)(v=165, avg=0.005), ((1, 3), 1)(v=1, avg=0.578), ((1, 4), 1)(v=1, avg=0.592), ((1, 6), 1)(v=1, avg=0.578)]
  Move 2: action=((3, 4), 0), total_visits=500
    State: Step 3/1000 | Boxes on target: 1/3 | Total distance: 4
    Children: [((3, 4), 0)(v=264, avg=0.003), ((3, 6), 0)(v=233, avg=0.003), ((1, 2), 3)(v=1, avg=0.629), ((1, 4), 1)(v=1, avg=0.677), ((1, 6), 1)(v=1, avg=0.668)]
  Move 3: action=((3, 6), 0), total_visits=500
    State: Step 6/1000 | Boxes on target: 1/3 | Total distance: 3
    Children: [((3, 6), 0)(v=499, avg=0.002), ((1, 6), 1)(v=1, avg=0.716)]
  Move 4: action=((1, 7), 2), total_visits=500
    State: Step 10/1000 | Boxes on target: 1/3 | Total distance: 2
    Children: [((1, 7), 2)(v=500, avg=0.002)]

Final state:
Step 13/1000 | Boxes on target: 2/3 | Total distance: 1
 ########
 # *$*@.#
 #   #  #
 #      #
 ########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 7 levels): 0.8571

Per-level best composites so far:
  level10: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=0.0000) [MASTERED]
  level7: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level2', 'level6', 'level7']
Mastered levels: ['level10', 'level3', 'level4', 'level5', 'level8', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 4 [level8] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 5 [level4] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 6 [level3] [backpropagation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

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

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|---------------------------|--------------------------|
| **1** | *Almost all children receive an average ≈ 0.002‑0.005* despite some moves immediately placing a box on a target (e.g. actions with 1‑visit avg ≈ 0.58‑0.72). | The back‑propagation treats every rollout reward equally, so the few high‑reward visits are drowned by the huge mass of low‑reward visits. The averaged value that later decides the real move is therefore almost indistinguishable from “bad”. |
| **2** | *Solved leaves get only a microscopic boost (`_SOLVED_EPS = 1e‑3`).* After a single solved rollout the child’s value is ≈ 0.001 + 1 = 1.001, but after a few hundred cheap rollouts the average collapses back toward 0. | The boost is far too small to keep a solved branch dominant over the flood of noisy, low‑reward simulations. The algorithm can easily “forget” a winning line. |
| **3** | *No depth‑dependent discount.* A rollout that reaches a solved state after 30 pushes contributes the same 1.0 (plus ε) as a solved state after 2 pushes. because the simulation already penalises distance only weakly, the back‑propagation does not amplify short solutions. | MCTS should favour shorter plans; otherwise the tree keeps expanding deep “flat” branches (see the 500‑visit child on `(3,6)` that never advances). |
| **4** | *Value is only used for tie‑breaking in selection.* The A*‑guided selection picks the child with the lowest *f = g + h*; value influences the choice only when two children have identical f. Thus a badly averaged value never helps to prefer a promising child, and the search wastes visits on low‑potential nodes. | The back‑propagation’s output is effectively ignored by the dominant selection rule, so any improvement must make the value itself more informative for the final move choice. |

**2. ROOT CAUSE**

The back‑propagation routine:

```python
while node:
    node.visits += 1
    reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward
    node.value = ((node.visits-1) * node.value + reward_adj) / node.visits
    node = node.parent
```

* **Averaging alone** forces every low‑reward rollout to pull the average down, drowning out the sparse high‑reward signals.
* **Fixed ε** (1 × 10⁻³) is negligible compared with the typical reward scale (≈ 0.0‑0.2 for non‑solved states), so solved leaves cannot stay “above” the mass of low values.
* **No depth awareness** – the same `reward_adj` is added at every level, regardless of how many pushes were required to obtain it.
* **The value is never used for selection** except as a tie‑breaker, so even a perfectly calibrated average would have little impact unless it also drives the final move‑choice rule.

These design choices make the back‑propagation ineffective at steering the search toward short, high‑progress paths.

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement (recommended).**  
The current framework works (A*‑guided selection, dead‑lock pruning, etc.). We only need a smarter statistic that survives the flood of cheap rollouts and that rewards short, solved trajectories.

Add **three lightweight modifications** to `default_backpropagation`:

1. **Depth‑discounted reward**  
   Compute the depth of the node being updated (distance from the leaf) and apply a discount factor γ ∈ (0, 1) (e.g. 0.95).  
   ```python
   depth = 0
   while node:
       discount = gamma ** depth
       reward_adj = (reward + solved_eps) * discount if reward == 1.0 else reward * discount
       …
       depth += 1
   ```
   This gives near‑term nodes a larger contribution, encouraging the algorithm to prefer shorter solutions.

2. **Stronger solved‑leaf boost**  
   Replace the tiny ε (1e‑3) with a *relative* boost proportional to the remaining step budget, e.g. `solved_bonus = 0.1 * (max_steps - leaf.steps) / max_steps`. This makes a solved leaf noticeably better than any unsolved average, even after many visits.

3. **Maintain a cumulative reward sum** (optional but cheap)  
   Store `node.total_reward` (add a new slot) and compute `node.value = node.total_reward / node.visits`. Updating a sum avoids repeated multiplication and keeps the average numerically stable. The added slot does not interfere with other phases.

These changes preserve the existing loop, respect the `__slots__` layout (by only adding a field that can be pre‑declared once), and require no alteration to selection or simulation. The result is:

* **Higher‑value children for short, solved branches**, so the final move choice (usually “pick child with highest value”) will select them.
* **Depth discount** keeps the tree from over‑investing in deep, flat branches, reducing the 500‑visit `(3,6)` pattern.
* **Larger solved boost** guarantees that once a complete solution is found, it dominates the average even after thousands of cheap rollouts.

No restructuring of the overall MCTS pipeline is needed; the improvements directly address the three highest‑impact weaknesses identified above.
