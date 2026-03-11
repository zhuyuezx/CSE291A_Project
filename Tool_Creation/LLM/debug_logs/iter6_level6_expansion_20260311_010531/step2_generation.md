# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:06:03 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 16.47s |
| Status     | success |
| Tokens     | 10234 |
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

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Removed the misplaced `from __future__ import annotations` (which caused a `SyntaxError`) and added a lightweight import guard. The rest of the heuristic logic is unchanged.
Generated:   2026-03-11T01:02:10.854452
"""

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
    # Lazy‑import shared A* globals to avoid circular‑import problems.
    # ------------------------------------------------------------------
    import importlib.util
    import sys
    from pathlib import Path

    _KEY = "astar_globals"
    if _KEY not in sys.modules:
        _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
        _s = importlib.util.spec_from_file_location(_KEY, str(_p))
        _m = importlib.util.module_from_spec(_s)
        sys.modules[_KEY] = _m
        assert _s.loader is not None  # for mypy / linters
        _s.loader.exec_module(_m)
    _ag = sys.modules[_KEY]

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

        # Compute distance from the leaf (0 for leaf itself).
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
        cur = getattr(cur, "parent", None)
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
Current level: level6
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level6 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 7 levels): 0.8571

Per-level best composites so far:
  level1: best=1.0000 (baseline=1.0000) [MASTERED]
  level10: best=1.0000 (baseline=1.0000) [MASTERED]
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level6', 'level7', 'level8', 'level9']
Mastered levels: ['level1', 'level10', 'level2', 'level3', 'level4', 'level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 4 [level2] [expansion]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 5 [level4] [backpropagation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 6 [level6] [selection]: composite=0.0000, solve_rate=0%, desc=No changes required; the draft implementation is correct and efficient. ✗ rejected

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
  • SELF-CONTAINED: define ALL helper functions inline.
    Do NOT use undefined names (e.g. _ag, _bfs_distance).
  • node is MCTSNode (uses __slots__). Use ONLY node.state,
    node.parent, node.children, node._untried_actions,
    node.visits, node.value. Do NOT assign new attributes.
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
DESCRIPTION: added walk‑cost, target‑box protection, cheap corner dead‑lock check and softened visited‑pruning to better order/prune actions.
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
