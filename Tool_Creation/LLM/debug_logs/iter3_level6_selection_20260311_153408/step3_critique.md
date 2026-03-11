# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:35:06 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 25.51s |
| Status     | success |
| Tokens     | 8018 |
| Validation | valid |

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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| # | Observation (from the trace) | Why it hurts performance |
|---|------------------------------|--------------------------|
| 1 | The same macro‑pushes are revisited thousands of times (e.g., `((2,4),2)` appears with v≈500 in many successive moves). | The selection routine keeps descending the same low‑f branch, never giving a chance to explore alternative pushes that might break the dead‑end. |
| 2 | The tree‑walk ignores the `exploration_weight` argument completely. | No UCB‑style term → pure greedy A* (min g + h). When several children share an identical f‑score the tie‑breaker (`fewest visits`) is too weak; the algorithm falls into depth‑first “tunnel” behaviour. |
| 3 | `g` is the pure node depth, not the accumulated **real** cost (walk + push). | A cheap‑looking shallow node may hide a huge walk cost, so the algorithm prefers pushes that look close in depth but actually waste many steps, stalling progress toward the goal. |
| 4 | `h_sokoban` is a static Manhattan‑distance heuristic; it does **not** penalise obvious dead‑locks (corner, wall‑line) or reward a box landing on a target. | The f‑value does not distinguish a push that creates a dead‑lock from one that makes progress, so the greedy descent often walks into dead‑ends (see moves 9–12 where a box is placed on a target but the algorithm keeps wobbling around it). |
| 5 | No novelty/visit‑bonus beyond the simple “fewest visits” tie‑break. | Nodes that have been visited many times still receive the same f, so the algorithm spends many iterations on an already‑exhausted branch. |

The combined effect is **excessive exploitation of a single cheap‑looking line**, leading to a long but unproductive search (76 steps, only 1/2 boxes placed, return 0).

---

**2. ROOT CAUSE**  

The selection code implements a pure best‑first walk:

```python
g = _ag.node_depth(child)          # depth only
h = _ag.h_sokoban(child.state)     # static distance heuristic
f = g + h
```

*No exploration term* (`exploration_weight` is never used).  
*Depth* is not the true accumulated cost, so `f` does not reflect the macro‑push cost.  
`h_sokoban` lacks dead‑lock penalties and target‑placement bonuses, so f‐values of dead‑ended and promising children can be identical, making the tie‑breaker (`fewest visits`) insufficient. Consequently the algorithm repeatedly selects the same child, inflating its visit count while never opening alternative branches.

---

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement (≈70 % of effort).**  
We keep the overall A*‑guided framework (it already provides a useful heuristic backbone) but augment it with three lightweight, high‑impact tweaks:

1. **Add an explicit exploration component** to the score, preserving the supplied `exploration_weight`.  
   - Compute `explore = exploration_weight * sqrt(log(parent.visits) / child.visits)` (or the classic UCB term).  
   - New score: `f = g + h - explore` (lower is better).  
   This restores a balanced trade‑off without discarding the A* intuition.

2. **Replace pure depth with the real accumulated step cost** (`child.state.steps` or the macro‑push cost).  
   - `g = child.state.steps` (or the stored `g` from the shared A* table).  
   - This makes `f` penalise long walks, steering the search toward pushes that actually advance the solution.

3. **Enrich the heuristic `h` locally** (still cheap, no extra roll‑outs).  
   - Pull the dead‑lock detection already used in expansion/simulation (`is_deadlocked`) and add a small penalty (`+ deadlock_penalty`) when `h_sokoban` would otherwise be identical.  
   - Add a tiny **target‑placement bonus** (`‑ target_bonus` if a push puts a box on a target).  
   - Both constants can be tiny (e.g., 0.1) just enough to break ties in favour of progress.

The modified loop would still iterate over children, compute `g`, `h`, `explore`, and select the child with the smallest `f`. Tie‑breaking can stay “fewest visits” as a secondary fallback.

These three changes are O(1) per child, keep the code simple, and directly address the highest‑impact weaknesses: lack of exploration, inaccurate cost estimation, and indistinguishable dead‑ends. No restructuring of the whole MCTS pipeline is required, and the existing A* globals remain usable.

*Why not a full restructure?*  
A complete switch back to classic UCB1 would discard the valuable domain‑specific distance heuristic already proven useful in other levels (selection performed perfectly on level 3). The problem is not the idea of A*‑guided selection but its overly greedy instantiation; an incremental fix restores exploration while preserving the beneficial distance bias.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS selection with exploration and richer heuristic.

Improvements:
  • Uses the actual accumulated step cost (state.steps) as g instead of
    pure tree depth.
  • Adds a UCB‑style exploration term (weighted by exploration_weight)
    to avoid over‑exploitation of a single low‑f branch.
  • Enriches the heuristic h with a small dead‑lock penalty and a bonus
    for pushes that increase the number of boxes on targets.
  • Keeps the original A*‑style best‑first walk and root‑search reset.
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
    A*-guided tree walk with UCB exploration and richer heuristic.
    Descends the tree, at each level picking the child with the smallest
    score:

        f = g + h - exploration

    where
      g = real accumulated step cost (state.steps)
      h = static Manhattan distance + dead‑lock penalty – target bonus
      exploration = exploration_weight * sqrt(log(parent_visits) / child_visits)

    Ties are broken by preferring the child with fewer visits (more novel).

    Also detects the start of a new MCTS search and resets the shared
    visited dictionary.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Reset shared A* state when a new root is encountered.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper: dead‑lock detection (corner + wall‑line) – cheap O(#boxes).
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    def is_wall_line_deadlocked(state, box):
        walls = state.walls
        targets = state.targets
        h, w = state.height, state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in dirs:
            neighbor = (box[0] + dx, box[1] + dy)
            if neighbor not in walls:
                continue
            step = 2
            while True:
                check = (box[0] + dx * step, box[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a target beyond the wall ⇒ not a dead‑lock in this direction
                    return False
                step += 1
            # no target found in this direction ⇒ dead‑lock
            return True
        return False

    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for b in state.boxes:
            if b not in state.targets and is_wall_line_deadlocked(state, b):
                return True
        return False

    # ------------------------------------------------------------------
    # Constants for the enriched heuristic.
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 5.0   # sizable enough to push dead‑ends up in f
    TARGET_BONUS = 0.5       # reward placing a box on a target

    # ------------------------------------------------------------------
    # 2️⃣ Walk down the tree, always picking the child with minimal f.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion create a fresh child

        best_child = None
        best_f = math.inf
        best_q = -math.inf   # avg value, used only when f ties

        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: base Manhattan distance + dead‑lock / target tweaks -----
            h_base = _ag.h_sokoban(child.state)

            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: difference in boxes on targets compared to parent
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- exploration term (UCB) -----
            if child.visits == 0:
                # Encourage first‑visit nodes strongly
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore

            # ----- tie‑break on average value -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        if best_child is None:
            break   # no children (should not happen)

        node = best_child

    return node
```

------------------------------------------------------------
TASK — CRITIQUE & FINALIZE
------------------------------------------------------------
Review the DRAFT code above for critical issues.

Check for:
  1. BUGS — API misuse, crashes, wrong variable names
  2. SPEED — unnecessary clones or deep loops (runs 1000s of times)
  3. REWARD SPREAD — does the return value vary across states?

RULES:
  - Preserve the INTENT of the draft (incremental fix or
    restructure). Do NOT add unrelated features.
  - Fix bugs and performance issues. Improve clarity.
  - If the draft is correct and fast, output it UNCHANGED.

You MUST format your response EXACTLY as follows:

CRITIQUE:
<1-3 bullet points, or 'No issues found'>

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line summary>
```python
<complete final function code>
```

Rules for the header fields:
- ACTION must be either 'create' or 'modify'.
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

CRITIQUE:
- Helper functions (`is_corner_deadlocked`, `is_wall_line_deadlocked`, `is_deadlocked`) and constant parameters are re‑defined on **every** call to `default_selection`, adding unnecessary overhead in a hot path.
- The `exploration_weight` argument is accepted but not used in the original draft’s signature; the function now uses it correctly, but its default value should be documented.
- Minor style issue: using `-math.inf` for unvisited children works but makes the selection order deterministic (first unvisited child). This is acceptable for now.

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Move cheap deadlock helpers and heuristic constants out of the hot‑path, keeping the enhanced A*‑guided selection logic.
```python
"""
A*-guided MCTS selection with exploration and richer heuristic.

Improvements:
  • Uses the actual accumulated step cost (state.steps) as g instead of
    pure tree depth.
  • Adds a UCB‑style exploration term (weighted by exploration_weight)
    to avoid over‑exploitation of a single low‑f branch.
  • Enriches the heuristic h with a small dead‑lock penalty and a bonus
    for pushes that increase the number of boxes on targets.
  • Keeps the original A*‑style best‑first walk and root‑search reset.
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state (one instance shared across all phase files) ──
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag

# ── Heuristic tuning constants ────────────────────────────────────────
DEADLOCK_PENALTY = 5.0   # pushes a dead‑locked node higher in f
TARGET_BONUS = 0.5       # reward for each additional box placed on a target


def _is_corner_deadlocked(state) -> bool:
    """Simple corner dead‑lock detection (ignores targets)."""
    walls = state.walls
    targets = state.targets
    for bx, by in state.boxes:
        if (bx, by) in targets:
            continue
        horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
        vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
        if horiz and vert:
            return True
    return False


def _is_wall_line_deadlocked(state, box) -> bool:
    """
    Detect wall‑line dead‑locks: a box adjacent to a wall where, looking
    past that wall in the same direction, no target exists.
    """
    walls = state.walls
    targets = state.targets
    h, w = state.height, state.width
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in dirs:
        neighbor = (box[0] + dx, box[1] + dy)
        if neighbor not in walls:
            continue        # not a wall in this direction
        step = 2
        while True:
            check = (box[0] + dx * step, box[1] + dy * step)
            if not (0 <= check[0] < h and 0 <= check[1] < w):
                break
            if check in walls:
                break
            if check in targets:
                # a target beyond the wall – not a dead‑lock this way
                return False
            step += 1
        # No target found before hitting another wall/out‑of‑bounds → dead‑lock
        return True
    return False


def _is_deadlocked(state) -> bool:
    """Combine corner and wall‑line checks."""
    if _is_corner_deadlocked(state):
        return True
    for b in state.boxes:
        if b not in state.targets and _is_wall_line_deadlocked(state, b):
            return True
    return False


def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    At each level the child minimizing

        f = g + h - exploration

    is selected, where
        g = real accumulated step cost (state.steps)
        h = Manhattan distance + dead‑lock penalty – target bonus
        exploration = exploration_weight * sqrt(log(parent_visits) / child_visits)

    Ties are broken by preferring the child with the higher average value
    (more promising) and, finally, by fewer visits.

    The function also detects a new root (new search) and resets the shared
    A* globals.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Detect a new search and reset shared A* state.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # 2️⃣ Walk down the tree, always picking the child with minimal f.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion create a fresh child

        best_child = None
        best_f = math.inf
        best_q = -math.inf   # average value, used only when f ties

        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: base Manhattan distance + dead‑lock / target tweaks -----
            h_base = _ag.h_sokoban(child.state)
            dead_pen = DEADLOCK_PENALTY if _is_deadlocked(child.state) else 0.0

            # Target bonus: difference in boxes on targets compared to parent
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- exploration term (UCB) -----
            if child.visits == 0:
                # Strongly encourage unvisited nodes
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore

            # ----- tie‑break on average value -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        if best_child is None:
            break   # safety net; should not happen

        node = best_child

    return node
```
