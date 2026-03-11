# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:39:02 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 18.21s |
| Status     | success |
| Tokens     | 7212 |
| Validation | valid |

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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

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

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
from __future__ import annotations

import sys
import importlib.util
import math
from collections import deque
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


def _bfs_reachable(start, walls, boxes):
    """
    Returns a dict mapping each reachable cell (ignoring boxes as obstacles)
    to its shortest path length from ``start``.
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
    Very fast corner dead‑lock detection:
    a box that is not on a target and has walls on two orthogonal sides.
    """
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


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for Sokoban‑macro MCTS.

    Combines several inexpensive signals:
      • total Manhattan distance of boxes to nearest targets
      • count of boxes unreachable by the player (without pushing)
      • estimated walk distance to the cheapest push position
      • a heavy penalty for obvious corner dead‑locks
    The combined metric ``h`` is turned into a smooth reward
    ``exp(-λ·h)`` (λ≈0.3) so differences in the range of typical
    Sokoban states are more pronounced than the original 1/(1+h).

    Returns:
        1.0                              if the state is solved,
        -0.1                             if terminal dead‑locked,
        exp(-λ·h)                        otherwise.
    """
    # Terminal check – use the game‑provided returns, but depress dead‑lock.
    if state.is_terminal():
        ret = state.returns()[perspective_player]
        # dead‑lock states return 0.0; give them a small negative value
        return -0.1 if ret == 0.0 else ret

    # ------------------------------------------------------------------
    # 1️⃣  Box‑to‑target Manhattan distance (already provided by the API)
    # ------------------------------------------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ------------------------------------------------------------------
    # 2️⃣  Unreachable‑box count (player cannot reach without pushing)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    reachable = _bfs_reachable(state.player, walls, boxes)
    unreachable_cnt = sum(1 for b in boxes if b not in reachable)

    # ------------------------------------------------------------------
    # 3️⃣  Minimal walk distance to any legal push position
    # ------------------------------------------------------------------
    # Build a distance map once (already in ``reachable``).  Legal actions
    # are macro‑pushes, each requiring the player to stand on ``action[0]``.
    min_walk = None
    for action in state.legal_actions():
        push_pos = action[0]                     # cell where player must stand
        if push_pos in reachable:
            d = reachable[push_pos]
            if (min_walk is None) or (d < min_walk):
                min_walk = d
    # If no push is possible (should be caught as terminal elsewhere) fall back
    if min_walk is None:
        min_walk = 10  # arbitrary large cost

    # ------------------------------------------------------------------
    # 4️⃣  Dead‑lock proximity penalty (simple corner detection)
    # ------------------------------------------------------------------
    deadlock_penalty = 5.0 if _simple_corner_deadlock(state) else 0.0

    # ------------------------------------------------------------------
    # 5️⃣  Weighted combination → heuristic h
    # ------------------------------------------------------------------
    alpha = 1.0   # box distance weight
    beta  = 2.0   # unreachable boxes weight (more severe)
    gamma = 0.5   # walk cost weight
    delta = 1.0   # scale for deadlock penalty (already large)
    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) + (delta * deadlock_penalty)

    # ------------------------------------------------------------------
    # 6️⃣  Reward shaping – exponential decay gives a smoother gradient
    # ------------------------------------------------------------------
    lam = 0.3
    reward = math.exp(-lam * h)

    # Guard against numerical underflow (very bad states)
    if reward < 1e-8:
        reward = 0.0
    return reward
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
- The terminal handling treats *any* zero-return state (including max‑step termination) as a dead‑lock and returns a negative reward, which overstates the penalty for step‑limit terminations.
- Minor unused import (`sys`) and unused variable (`_ag` after removal of original heuristic) could be cleaned but are not harmful.
- Overall logic is sound; only the dead‑lock penalty condition needs refinement.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: Refine terminal reward handling to penalize only genuine dead‑locks while keeping other improvements.

```python
from __future__ import annotations

import importlib.util
import math
from collections import deque
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
