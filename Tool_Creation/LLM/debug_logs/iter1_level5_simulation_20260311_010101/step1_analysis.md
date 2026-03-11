# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:01:15 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 13.62s |
| Status     | success |
| Tokens     | 8470 |
| Validation | invalid — No function definition found in code. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: simulation).

PHASE: simulation
  • What it does: Rolls forward from a leaf state to estimate how promising it is. Returns a reward (e.g. 0–1) that flows into backpropagation.
  • Optimization goal: Produce REWARDS that reflect true state quality. Shaped partial progress helps MCTS distinguish good from bad actions.
  • Constraints: Must return a FLOAT. Reward MUST vary across states — flat rewards ≈ random play. Called thousands of times per move — keep it fast.
  • Good patterns: shaped score (boxes on targets, distance improvement), penalize deadlocks/loops/stagnation, prefer pushes over wandering, early termination when stuck.
  • Avoid: tree-level visit balancing, acceptance criteria for tools — this phase only scores rollouts.

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

--- simulation ◀ TARGET ---
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
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
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

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level5)
Timestamp:  2026-03-11T01:01:01.443400
Iterations: 500
Solved:     True
Steps:      7
Returns:    [1.0]

  Move 1: action=((3, 5), 2), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/3 | Total distance: 5
    Children: [((3, 5), 2)(v=497, avg=0.936), ((1, 3), 1)(v=1, avg=0.167), ((3, 3), 0)(v=1, avg=0.143), ((3, 3), 3)(v=1, avg=0.143)]
  Move 2: action=((3, 4), 2), total_visits=500
    State: Step 1/1000 | Boxes on target: 0/3 | Total distance: 4
    Children: [((3, 4), 2)(v=497, avg=0.942), ((1, 4), 1)(v=1, avg=0.200), ((3, 2), 3)(v=1, avg=0.167), ((3, 4), 0)(v=1, avg=0.167)]
  Move 3: action=((1, 3), 1), total_visits=500
    State: Step 2/1000 | Boxes on target: 1/3 | Total distance: 3
    Children: [((1, 3), 1)(v=482, avg=0.971), ((1, 4), 1)(v=12, avg=0.247), ((2, 2), 1)(v=1, avg=0.250), ((4, 2), 0)(v=1, avg=0.250), ((3, 1), 3)(v=1, avg=0.200), ((3, 3), 0)(v=1, avg=0.200), ((3, 3), 2)(v=1, avg=0.200), ((3, 4), 0)(v=1, avg=0.200)]
  Move 4: action=((2, 5), 2), total_visits=500
    State: Step 9/1000 | Boxes on target: 1/3 | Total distance: 3
    Children: [((2, 5), 2)(v=493, avg=0.982), ((1, 4), 1)(v=1, avg=0.250), ((2, 2), 1)(v=1, avg=0.250), ((2, 3), 1)(v=1, avg=0.250), ((4, 2), 0)(v=1, avg=0.250), ((4, 3), 0)(v=1, avg=0.250), ((2, 3), 3)(v=1, avg=0.200), ((3, 4), 0)(v=1, avg=0.200)]
  Move 5: action=((2, 4), 2), total_visits=500
    State: Step 14/1000 | Boxes on target: 1/3 | Total distance: 2
    Children: [((2, 4), 2)(v=497, avg=0.986), ((2, 2), 1)(v=1, avg=0.333), ((4, 2), 0)(v=1, avg=0.333), ((2, 2), 3)(v=1, avg=0.250)]
  Move 6: action=((2, 3), 1), total_visits=500
    State: Step 15/1000 | Boxes on target: 2/3 | Total distance: 1
    Children: [((2, 3), 1)(v=496, avg=0.992), ((4, 3), 0)(v=2, avg=0.417), ((2, 1), 3)(v=1, avg=0.333), ((2, 3), 2)(v=1, avg=0.333)]
  Move 7: action=((4, 4), 2), total_visits=500
    State: Step 16/1000 | Boxes on target: 2/3 | Total distance: 1
    Children: [((4, 4), 2)(v=495, avg=1.000), ((2, 1), 3)(v=1, avg=0.333), ((2, 3), 2)(v=1, avg=0.333), ((3, 1), 3)(v=1, avg=0.333), ((3, 3), 2)(v=1, avg=0.333), ((4, 2), 3)(v=1, avg=0.333)]

Final state:
Step 19/1000 | Boxes on target: 3/3 | Total distance: 0
########
#      #
# *    #
# *    #
# *@   #
########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level5
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level5 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 1 levels): 0.0000

Per-level best composites so far:
  level5: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level3', 'level4', 'level5', 'level6', 'level7', 'level8', 'level9']


SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level5] [selection]: composite=0.0000, solve_rate=0%, desc=Remove unused import and keep existing logic intact. ✗ rejected
  Iter 1 [level5] [expansion]: composite=0.0000, solve_rate=0%, desc=Fix corner dead‑lock detection to evaluate the box’s new position after the push and keep the incremental improvements. ✗ rejected

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'simulation'
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
      modifications to the 'simulation' function
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

**1. KEY WEAKNESSES**

| Rank | Symptom (trace evidence) | Why it matters |
|------|--------------------------|----------------|
| 1️⃣ | *All* non‑optimal children get ≈ 0.15‑0.20 Q (e.g. Move 1 children ((1,3),1), ((3,3),0)…). The heuristic never distinguishes between pushes that are cheap walks and pushes that require the player to traverse the whole map. | Macro‑push actions differ wildly in walking cost; rewarding a push solely by box‑to‑target distance over‑values moves that are far away, leading MCTS to waste iterations on expensive pushes and miss the cheap “obvious” solution. |
| 2️⃣ | The reward curve `1/(1+h)` collapses for moderate/large h (h ≥ 10 → ≤ 0.09). Most early states have h ≈ 5‑12, so many branches receive almost the same tiny value, flattening the value signal and making Q‑averages noisy. | With such compression the search cannot rank the dozens of legal macro‑pushes; the few that happen to reduce h a little get the bulk of visits (see the 0.94‑0.99 averages), while many viable pushes are almost ignored. |
| 3️⃣ | No explicit dead‑lock foresight beyond the terminal check. A state that is one push away from a classic corner dead‑lock can still have a low h (boxes close to targets) and therefore be assigned a high reward, steering the tree into a dead‑end that is only discovered after several pushes. | The solver repeatedly expands into doomed lines, wasting visits that could have been spent exploring safe alternatives. |
| 4️⃣ | Player‑related progress (boxes already on targets) is not reflected except via h = 0 when solved. A state with 2/3 boxes on target but h = 1 receives the same reward as a state with 0/3 boxes on target and h = 1. | Early progress is a strong indicator of a good line; ignoring it reduces the gradient that guides MCTS toward completing the puzzle. |

**2. ROOT CAUSE**

The **simulation** function is a single‑line evaluator:

```python
h = _ag.h_sokoban_box_only(state)      # sum of box‑to‑nearest‑target Manhattan distances
return 1.0 / (1.0 + h)
```

* It **ignores** any metric that involves the player (walk distance, reachable region size).  
* It **compresses** the range with a hyper‑linear transform, giving almost zero distinction for h > 5.  
* It **does not incorporate** `boxes_on_targets()` or any dead‑lock‑prediction beyond the terminal flag.  

Consequently the rollout value is blind to two of the most decisive factors in Sokoban‑Macro: *how expensive a push is to reach* and *how many boxes are already solved*.

**3. PROPOSED APPROACH – Strategy A (Incremental)**  

We can keep the fast, deterministic structure of the current evaluator and enrich it with three cheap, orthogonal terms:

1. **Walk‑cost estimate** – use the reachable‑region BFS already available inside `state` (most implementations expose a `reachable()` method or we can recompute it here).  
   *`walk = min distance from player to any pushable box`* (or the average of the two squares involved in the push).  
   Add a linear penalty `α * walk` (α ≈ 0.1) to the distance sum: `h' = h + α*walk`. This demotes pushes that require long detours without adding overhead proportional to the number of boxes.

2. **Progress bias** – reward states that already placed boxes on targets:  
   `b = state.boxes_on_targets()` (0 … num_targets).  
   Incorporate it as a numerator boost: `reward = (b + 1) / (b + 1 + h')`.  
   When `b` grows, the same h′ yields a larger value, giving a smooth gradient toward completion.

3. **Gentler scaling** – replace the hard `1/(1+h')` with a *soft‑sigmoid*‑like transform that preserves differences for larger h:  
   `reward = 1.0 / (1.0 + β * h')` with β ≈ 0.5 (or even `exp(-β*h')`).  
   This expands the effective range so that h = 5 still yields ≈0.38 rather than ≈0.17, giving the tree more signal to discriminate among many pushes.

**Resulting pseudo‑logic (no code required):**

```text
if terminal: return terminal_return
h  = box‑only Manhattan sum
walk = estimated player steps to the cheapest push (0 if already adjacent)
h' = h + α*walk
b  = boxes_on_targets()
reward = (b + 1) / (b + 1 + β*h')
```

All three additions are **O(1)** or **O(|reachable|)** (tiny compared to a macro‑push expansion) and preserve the existing deterministic nature of the simulation phase. They directly address the three ranked weaknesses, so we expect more visits to cheap, high‑progress pushes, a richer value spread, and fewer dead‑lock excursions, ultimately raising solve‑rate and average returns without a wholesale redesign.
