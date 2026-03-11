# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:01:55 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 7.58s |
| Status     | success |
| Tokens     | 8250 |
| Validation | valid |

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
Description: No changes needed; preserve the improved heuristic implementation.
Generated:   2026-03-11T01:01:37.159080
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from collections import deque

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
    Heuristic leaf evaluation for Sokoban‑Macro.

    Improvements over the original version:
      * Adds an estimate of the player’s walking cost to the cheapest
        reachable push (penalises distant pushes).
      * Rewards partial progress by counting boxes already on targets.
      * Uses a smoother scaling so that moderate‑to‑large heuristic values
        still produce distinguishable rewards.

    Returns:
        1.0                – solved states (h == 0)
        state.returns()[p] – terminal dead‑lock / step‑limit
        reward             – shaped value for non‑terminal states
    """
    # Terminal states keep their original return value (0.0 for dead‑lock, 1.0 for win)
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣  Base box‑only Manhattan distance (same as original A* heuristic)
    # ------------------------------------------------------------------
    h_box = _ag.h_sokoban_box_only(state)          # sum of box‑to‑nearest‑target distances

    # ------------------------------------------------------------------
    # 2️⃣  Estimate minimal walk distance the player must perform to execute any legal macro‑push.
    # ------------------------------------------------------------------
    # Compute BFS distances from the player over free cells (walls and boxes block movement).
    walls = state.walls
    boxes = state.boxes
    start = state.player

    # BFS over floor cells not occupied by walls or boxes.
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

    # Determine the cheapest push position (player must stand on the given cell).
    min_walk = None
    for (player_pos, _dir) in state.legal_actions():
        d = dist.get(player_pos)
        if d is not None:
            if min_walk is None or d < min_walk:
                min_walk = d

    # If for any reason no reachable push exists (should be caught by is_terminal),
    # fall back to a large penalty.
    if min_walk is None:
        min_walk = 0  # dead‑lock handling will already have returned above.

    # ------------------------------------------------------------------
    # 3️⃣  Combine the components.
    # ------------------------------------------------------------------
    WALK_WEIGHT = 0.1   # weight for walk cost (empirically small)
    BETA = 0.5          # scaling factor for smoother reward curve

    # Augmented heuristic: box distance + penalise long walks
    h_aug = h_box + WALK_WEIGHT * min_walk

    # Progress term: how many boxes are already on targets
    boxes_done = state.boxes_on_targets()          # 0 … num_targets

    # Smooth reward: (progress + 1) / (progress + 1 + β * h_aug)
    reward = (boxes_done + 1) / (boxes_done + 1 + BETA * h_aug)

    # Clamp to the valid range [0.0, 1.0] (protect against tiny floating errors)
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    return float(reward)
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
Current level: level5
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level5 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 1 levels): 1.0000

Per-level best composites so far:
  level5: best=1.0000 (baseline=0.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level3', 'level4', 'level6', 'level7', 'level8', 'level9']
Mastered levels: ['level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level5] [selection]: composite=0.0000, solve_rate=0%, desc=Remove unused import and keep existing logic intact. ✗ rejected
  Iter 1 [level5] [expansion]: composite=0.0000, solve_rate=0%, desc=Fix corner dead‑lock detection to evaluate the box’s new position after the push and keep the incremental improvements. ✗ rejected
  Iter 1 [level5] [simulation]: composite=1.0000, solve_rate=100%, eval_time=0.4s, desc=No changes needed; preserve the improved heuristic implementation. ★ BEST-for-level

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (observed in trace or likely on other levels) | Why it hurts play |
|------|------------------------------------------------------|-------------------|
| **1** | **No depth‑discount on the back‑propagated reward** – every ancestor receives the *full* leaf reward, even if the solution required many pushes.  In the trace the first three moves receive > 0.96 average value despite the leaf being 7 pushes away.  This inflates the value of deep branches and can make the search favour longer, higher‑cost plans on harder levels. |
| **2** | **Uniform treatment of solved vs. merely‑good roll‑outs** – the reward coming from `default_simulation` already differentiates progress, but back‑propagation does not amplify a *fully solved* leaf.  Consequently a solved rollout (reward ≈ 1.0) and a high‑but‑incomplete rollout (reward ≈ 0.9) are propagated identically, making the algorithm less eager to terminate early when a solution is found. |
| **3** | **Redundant sign‑flip logic for a single‑player game** – the `mover`/`perspective` gymnastics add overhead and obscure the core update.  While not fatal, it increases the chance of a bug (e.g., using the wrong player when the tree is re‑rooted). |
| **4** | **No explicit handling of “partial‑progress” credit beyond the simulation** – once a rollout is back‑propagated, nodes that already have several boxes on targets do not get extra boost proportional to that progress; the information is lost in the averaging of `value / visits`. |

The most damaging of these is **(1) the lack of a depth‑discount**, because it directly conflicts with the “shorter plans dominate” pattern recommended for back‑propagation.  

---

**2. ROOT CAUSE**  

The current `default_backpropagation` performs:

```python
while node is not None:
    node.visits += 1
    node.value  += reward   # possibly sign‑flipped
    visited[key] = min(g, visited.get(key, INF))
    node = node.parent
```

* It adds the *raw* `reward` to every ancestor, irrespective of how many edges away the leaf is (`g` is only used for the A* visited sync, not for the value update).  
* It does not examine the leaf’s `state.is_terminal()` flag, so a solved leaf (`reward == 1.0`) is treated the same as a leaf that merely has a high heuristic reward.  
* The sign‑flip code is kept for a two‑player template, but Sokoban is single‑player, so the extra branching is unnecessary and can hide bugs when the tree is re‑rooted after a move.  

Thus the algorithm propagates *undiscounted, undifferentiated* evidence, which biases the average value toward deep, noisy rollouts and blurs the distinction between “solved” and “almost solved”.

---

**3. PROPOSED APPROACH**  

**Strategy A – Incremental refinement** (recommended).  
The back‑propagation step can be improved with two small, well‑contained changes that directly address weaknesses 1 and 2 while preserving the proven A*‑g‑sync and the rest of the pipeline.

1. **Depth‑discounted reward**  
   *Compute the distance* `d` from the current node to the leaf (e.g., `leaf_depth – node_depth`).  
   *Apply a discount* `γ^d` (`γ` ∈ (0, 1), e.g., 0.95) before adding to `node.value`.  
   This makes each push contribute less to the value the farther it is from the successful outcome, encouraging the tree to prefer short, low‑cost solutions.

2. **Solved‑leaf amplification**  
   Detect whether the leaf state is terminal and solved (`leaf.state.boxes_on_targets() == leaf.state.num_targets`).  
   If so, boost the reward (e.g., `reward = 1.0 + SOLVED_BONUS` with `SOLVED_BONUS` ≈ 0.2) **before** discounting.  
   The bonus propagates (still discounted) so ancestors of a genuine solution receive a stronger signal than ancestors of merely‑good rollouts.

3. **Simplify sign handling**  
   Since `current_player()` is always `0`, drop the `mover` logic and always add `reward` (or its discounted version).  This reduces noise and clarifies intent.

4. **Leave A* visited sync untouched** – it already correctly records the shallowest `g`.

*Implementation sketch (no code required):*  

* At the start, store `leaf_depth = _ag.node_depth(leaf_node)` (the incoming `node`).  
* Walk upward, compute `d = leaf_depth - _ag.node_depth(node)`.  
* `discount = γ ** d`.  
* If `leaf_is_solved`, set `base = 1.0 + SOLVED_BONUS` else `base = reward`.  
* Update `node.value += discount * base`.  

All other lines (visits increment, visited‑sync) remain unchanged.  

These tweaks are lightweight, keep the existing A*‑guided selection/expansion intact, and directly target the most harmful bias observed in the trace and likely on harder levels.

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
  - Receives (node, reward). Returns None; updates node.value and node.visits up the tree.
  - Propagates the simulation result to ancestors; selection's UCB1 uses these values.
  - Controls how strongly rollout evidence affects node values (depth discount, etc.).
  - Only aggregates evidence — no move generation or rollout policy.

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
DESCRIPTION: Added depth‑discounted rewards and a solved‑leaf bonus, and removed unnecessary sign‑flip logic to let rollouts favor short, truly solved plans.
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with:
  • Depth‑discounted reward propagation – rewards diminish with distance
    from the leaf, encouraging shorter solution plans.
  • Solved‑leaf bonus – when the leaf state is a solved terminal, a small
    extra bonus is added before discounting, making genuine solutions
    stand out from high‑but‑incomplete rollouts.
  • Simplified sign handling (Sokoban is single‑player).

In addition, the A* visited‑g‑score sync is preserved so that expansion
continues to respect the shallowest‑known path cost.
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
    Backpropagate a rollout reward from a leaf node up to the root.

    Enhancements:
      * depth_discount (γ) reduces the impact of evidence the farther
        the ancestor is from the leaf (γ ≈ 0.95 by default).
      * solved_bonus adds a small extra reward for truly solved leaf
        states, ensuring they are preferred over merely‑good rollouts.
    """
    # ------------------------------------------------------------------
    # Configuration parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (0 < γ < 1)
    SOLVED_BONUS = 0.20   # extra reward for a leaf that actually solves the puzzle

    # ------------------------------------------------------------------
    # Determine the depth of the leaf node once – used for discounting.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)

    # Detect whether the leaf is a *solved* terminal state.
    leaf_state = leaf_node.state
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )
    # Base reward for this leaf (may be boosted if solved).
    base_reward = reward + SOLVED_BONUS if is_solved else reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying depth discount and updating A* g‑scores.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Compute how far this node is from the leaf (0 for leaf itself).
        cur_depth = _ag.node_depth(cur)
        distance_from_leaf = leaf_depth - cur_depth

        # Apply exponential discount.
        discounted_reward = (GAMMA ** distance_from_leaf) * base_reward

        # Sokoban is single‑player → always add (no sign flip).
        cur.value += discounted_reward

        # ── A* g‑score sync (unchanged) ─────────────────────────────────
        key = cur.state.state_key()
        g = cur_depth
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent.
        cur = cur.parent
```
