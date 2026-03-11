# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:38:27 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.64s |
| Status     | success |
| Tokens     | 7887 |
| Validation | invalid — No function definition found in code. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: simulation).

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
Game:       Sokoban_Macro (level7)
Timestamp:  2026-03-11T00:38:16.994445
Iterations: 500
Solved:     False
Steps:      4
Returns:    [0.0]

  Move 1: action=((3, 4), 0), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/3 | Total distance: 5
    Children: [((3, 4), 0)(v=197, avg=0.003), ((3, 6), 0)(v=132, avg=0.003), ((3, 3), 0)(v=171, avg=0.003)]
  Move 2: action=((3, 6), 0), total_visits=500
    State: Step 2/1000 | Boxes on target: 0/3 | Total distance: 4
    Children: [((3, 3), 0)(v=1, avg=0.246), ((3, 6), 0)(v=497, avg=0.001), ((1, 3), 3)(v=1, avg=0.241), ((1, 5), 2)(v=1, avg=0.239)]
  Move 3: action=((1, 7), 2), total_visits=500
    State: Step 6/1000 | Boxes on target: 0/3 | Total distance: 3
    Children: [((1, 7), 2)(v=169, avg=0.003), ((3, 3), 0)(v=167, avg=0.003), ((1, 3), 3)(v=164, avg=0.003)]
  Move 4: action=((3, 3), 0), total_visits=500
    State: Step 9/1000 | Boxes on target: 1/3 | Total distance: 2
    Children: [((3, 3), 0)(v=500, avg=0.001)]

Final state:
Step 15/1000 | Boxes on target: 2/3 | Total distance: 1
 ########
 # *$* .#
 # @ #  #
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
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=0.0000) [MASTERED]
  level7: best=0.0000 (baseline=0.0000)
  level8: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=1.0000 (baseline=1.0000) [MASTERED]

Active levels (not yet mastered): ['level1', 'level3', 'level6', 'level7']
Mastered levels: ['level10', 'level2', 'level4', 'level5', 'level8', 'level9']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 5 [level10] [simulation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 6 [level2] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 7 [level9] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

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

| Rank | Symptom (from Trace #1) | Evidence |
|------|--------------------------|----------|
| 1️⃣ | **Heuristic blind to dead‑locks / unreachable boxes** – the search keeps pushing boxes that lower the *pure* Manhattan sum but later leads to a stuck configuration (2/3 boxes on targets after 15 steps, no progress). | After Move 3 the total distance dropped from 5 → 3, yet the chosen push at Move 4 placed a box into a corner (the only remaining box is now isolated). The Q‑values of all children stay around 0.003, showing the simulation cannot discriminate the bad continuation. |
| 2️⃣ | **No awareness of walk cost** – the simulation reward ignores the *g* (walk + push) cost that the selection phase uses (f = g + h). A child that requires a long walk may look as good as a cheap one, leading the selector to expand sub‑optimal pushes. | Move 2 uses a push that required a walk of several steps (the state reports “Step 2/1000”), yet the simulation still returns ≈0.2 (h = 4). The same h value would be given to a push reachable in one step, so the planner cannot prefer the cheaper action. |
| 3️⃣ | **Compressed reward range** – `1/(1+h)` varies only from 1.0 (h = 0) down to ≈0.14 (h = 6) on typical levels, giving a shallow gradient. The selector’s tie‑breaker (average Q) therefore receives almost identical signals, producing the near‑uniform Q≈0.003 seen for all children. | All children after Move 1 have avg≈0.003 despite differing h values (5 vs 4). The tiny differences are lost in Monte‑Carlo averaging over 500 iterations. |
| 4️⃣ | **No terminal‑specific shaping** – dead‑locked states receive the generic “0.0” from `state.returns()`, but the simulation never checks *before* expanding a node whether a dead‑lock is imminent. Consequently the tree wastes visits on hopeless branches. | The expansion already prunes some dead‑locks, but the simulation does not penalise states that are *one* push away from a dead‑lock, so the search still explores them. |

**2. ROOT CAUSE**

The current `default_simulation` computes a single scalar:

```python
h = _ag.h_sokoban_box_only(state)   # sum of Manhattan distances to nearest targets
return 1.0 / (1.0 + h)               # solved → 1.0, otherwise small value
```

* It ignores **player‑reachability** (unreachable boxes are not reflected in `h`).  
* It ignores **walk/push cost** (`g`) even though selection uses `f = g + h`.  
* It provides a **linear‑in‑h inverse** mapping, giving very low resolution for the range of h typical in Sokoban macro‑pushes.  
* It treats all non‑terminal states uniformly, offering no extra penalty for states that are already dead‑locked or near‑dead‑locked; `state.is_terminal()` only catches the final dead‑lock, not “dangerous” configurations.

Because selection expands nodes based on `f = g + h` while simulation evaluates only `h`, the two phases are out‑of‑sync: a node with a low `f` (cheap walk) may receive a mediocre simulation reward, and vice‑versa. This mismatch yields the flat Q‑values and the inability to steer away from unsafe pushes.

**3. PROPOSED APPROACH**

**Strategy A – Incremental enhancement of the simulation heuristic**

The deficiencies are localized to the heuristic’s composition and scaling; we can keep the overall “no‑rollout” design (which is fast and already integrated) and simply enrich the score.

**Targeted modifications**

1. **Add an *unreachable‑box* term**  
   – Run a BFS from `state.player` ignoring boxes, count boxes not in the reachable floor set.  
   – Weight this count (β ≈ 0.4) and add to the heuristic: `h' = α*box_dist + β*unreachable`.

2. **Include an estimate of *walk cost* for the next push**  
   – Approximate the minimal walk distance to any push position (e.g., the Manhattan distance from the player to the nearest box that can be pushed).  
   – Weight it (γ ≈ 0.2) and add: `h'' = h' + γ*min_walk`.

3. **Dead‑lock proximity penalty**  
   – Detect *simple corner* and *wall‑line* dead‑locks (already used in expansion). If the current state contains a box that would become dead‑locked after any push (e.g., a box adjacent to a wall with no target in that direction), add a large constant penalty δ (e.g., +5) to the heuristic.

4. **Rescale to a richer reward curve**  
   – Use a **soft‑max‑like** mapping: `reward = exp(-λ * h'')` with λ ≈ 0.3. This yields a smoothly decreasing reward from 1.0 (h'' = 0) down toward 0, preserving a noticeable gradient even for higher h'' values.

5. **Tie‑breaker for terminal vs non‑terminal**  
   – If `state.is_terminal()` and `state.returns()[p]` == 0 (dead‑lock), return a *negative* reward (e.g., ‑0.1) instead of 0, so that the average Q for dead‑locked leaves is clearly lower than any viable leaf.

These changes still produce a single scalar, keep the “no‑rollout” speed, and align the simulation more closely with the A*‑guided selection (both now consider walk distance and dead‑lock risk). The added terms are cheap O(|grid|) BFS operations already used elsewhere, so performance impact is negligible.

**Why not a full restructure?**  
The current framework (A*‑guided selection + deterministic simulation) works well on levels where the box‑only distance alone is discriminative (e.g., mastered levels). The observed failure on level 7 stems from missing information rather than a fundamentally flawed paradigm. Incrementally enriching the heuristic addresses the root cause while preserving the proven selection and expansion components, satisfying the 70 % “incremental” rule.
