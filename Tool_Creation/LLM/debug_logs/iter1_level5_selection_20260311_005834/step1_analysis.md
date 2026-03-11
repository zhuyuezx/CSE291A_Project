# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:58:49 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.38s |
| Status     | success |
| Tokens     | 8007 |
| Validation | invalid — No function definition found in code. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: selection).

PHASE: selection
  • What it does: Walks down the tree from root to a leaf. Chooses which existing branch to explore next. Must balance exploration (UCB) and exploitation.
  • Optimization goal: Improve how we RANK existing nodes — favor promising branches, deprioritize dead ends. Your heuristic adjusts node scores used by UCB1.
  • Constraints: Called very often. Keep it CHEAP — no multi-step rollouts, no deep deadlock simulation. Rank nodes, don't simulate.
  • Good patterns: bonus for more boxes on targets, bonus for lower box distance, penalize obvious deadlocks, novelty bonus for under-visited nodes.
  • Avoid: expensive rollout logic, final reward shaping (that belongs in simulation).

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

--- selection ◀ TARGET ---
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
TARGET HEURISTIC TO IMPROVE (selection)
------------------------------------------------------------
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

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level5)
Timestamp:  2026-03-11T00:58:34.689755
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
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'selection'
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
      modifications to the 'selection' function
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

| Rank | Symptom (from the trace) | Why it hurts |
|------|---------------------------|--------------|
| 1 | After the first push, one child gets **≈ 500 visits** while the other legal pushes receive only **1 visit** (e.g. Move 1 children). The tree collapses onto a single line. | The selection routine is *purely greedy*: it never injects exploration, so once a child looks a little better its f‑value dominates forever. On harder levels this prevents the algorithm from ever trying alternative pushes that may avoid a later dead‑lock. |
| 2 | `g` is computed as **node depth (number of pushes)**, but the true cost of a macro‑push is `walk_steps + 1`. Long‑walk pushes are treated as cheap. | The f‑score (`g + h`) systematically under‑estimates the cost of pushes that require many player steps, leading the search to favour pushes that look good heuristically but are expensive in reality. |
| 3 | `exploration_weight` is accepted as a parameter but never used; the classic UCB term is absent. | Same as 1 – no statistical incentive to visit low‑visit children, so the algorithm behaves like a deterministic A* rather than a balanced MCTS. |
| 4 | Selection never checks for **obvious dead‑locks** (cornered boxes not on targets). The expansion phase prunes dead‑locked actions, but a node that *already* contains a dead‑locked box can be repeatedly selected because its f‑value may still be low. | Leads to wasted simulations and can cause the algorithm to become trapped in a subtree that can never solve the puzzle. |
| 5 | Tie‑breaking uses the **average value (q)** of a child, while the classic UCB tie‑break prefers the **least‑visited** node. | When many children have identical f, the current rule pushes the already‑explored child further, reinforcing the imbalance observed in 1. |

**2. ROOT CAUSE**  

The selection function implements:

```python
g = _ag.node_depth(child)                # push count only
h = _ag.h_sokoban(child.state)           # box‑distance heuristic
f = g + h
```

and then picks the child with the smallest `f`. No term involving `child.visits` appears, so the `exploration_weight` argument is dead code. Consequently the algorithm behaves like a deterministic best‑first search that ignores the Monte‑Carlo exploration principle. Moreover, `g` does not reflect the actual macro‑push cost (`walk_steps + 1`), so the ordering is skewed toward actions with cheap‐looking pushes regardless of their true step cost.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement (≈ 70 % effort).**  
The current framework already provides the tree structure and cheap heuristics; we only need to enrich the scoring function while keeping the overall A*‑guided flow.

*Key modifications to `default_selection`:*  

1. **Add an exploration bonus** (UCB‑style) to the f‑score:  

   ```
   N  = node.visits            # visits of the parent (current node)
   n  = child.visits
   explore = exploration_weight * sqrt(log(N + 1) / (n + 1))
   f = g + h - explore          # lower f is better → exploration lowers f
   ```

   This re‑uses the existing `exploration_weight` argument and restores the balance between exploitation (low g + h) and exploration (high uncertainty).

2. **Replace depth‑only `g` with an estimated true cost**. The macro‑push action stored in `child.parent_action` can be examined for its walk length (most environments expose it via `action_cost` or we can compute a cheap BFS distance from `parent.state.player` to `action[0]`). Use:

   ```
   g = parent_g + walk_cost(parent.state, child.parent_action) + 1
   ```

   Store the cumulative `g` in the node (e.g., `node.g_cost`) at creation (expansion already knows the cost of the chosen action).

3. **Dead‑lock penalty** – a tiny additive term to `h` for any box that sits in a corner without being a target (a pre‑computed dead‑lock mask is cheap).  

   ```
   dead_penalty = 5 if _ag.is_deadlocked_box(child.state) else 0
   h = _ag.h_sokoban(child.state) + dead_penalty
   ```

4. **Tie‑break on visits, not on average value**. When `f` values are equal, choose the child with the **fewest visits** (or random tie‑break) to keep the search diversified.

These four tweaks keep the overall “A*‑guided” philosophy (still descending by a best‑first score) but inject the missing Monte‑Carlo exploration component, a more realistic cost estimate, and a cheap dead‑lock awareness. They are lightweight (constant‑time per child) and compatible with the existing shared `astar_globals` utilities.

**Why not a full restructure?**  
The present code already integrates A* heuristics, a shared visited table, and a clean expansion/back‑prop pipeline. The dominant failure mode is the absence of exploration and an inaccurate `g`. Adding these terms directly into the selection routine resolves the most harmful behavior (over‑commitment to a single child) without discarding the proven A*‑driven pruning that accelerates search on Sokoban macro‑pushes. Hence an **incremental (Strategy A)** solution is sufficient and low‑risk.
