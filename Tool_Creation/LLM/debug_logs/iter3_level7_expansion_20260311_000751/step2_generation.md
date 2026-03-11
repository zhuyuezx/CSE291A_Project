# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:08:23 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 18.95s |
| Status     | success |
| Tokens     | 9537 |
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
Aggregate best (avg across 3 levels): 0.3333

Per-level best composites so far:
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level7: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8', 'level9']
Mastered levels: ['level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level5] [backpropagation]: composite=0.0000, solve_rate=0%, desc=No changes needed; the draft implementation is correct and efficient. ✗ rejected
  Iter 2 [level6] [selection]: composite=0.0000, solve_rate=0%, desc=Fix missing heuristic function, add deterministic tie‑breaker, and improve clarity. ✗ rejected

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
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Added walk‑cost aware scoring, monotonicity pruning, stronger visited check, and a lightweight corner deadlock filter to align expansion with the A* f‑score used elsewhere.
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
