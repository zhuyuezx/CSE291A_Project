# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:22:01 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 11.24s |
| Status     | success |
| Tokens     | 5609 |
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

**1. KEY WEAKNESSES**  

| Rank | Symptom (trace) | Why it hurts the search |
|------|----------------|--------------------------|
| **1** | *Very low Q‑values* – most children hover around 0.15‑0.20 despite the fact that a box is already on‑target after move 7. The algorithm keeps “bouncing” between pushes that only marginally reduce the distance (e.g. move 2→3 → distance 7→6 →6→5). | The simulation reward is **insensitive** to useful progress (boxes‑on‑target, walk cost). A reduction from h=8 to h=5 only changes the reward from 0.111 to 0.167 – a tiny signal that is easily drowned by Monte‑Carlo variance. |
| **2** | *No penalty for long walks* – move 3 (step 13) required a walk of 12 steps (see the huge jump in `Step` counter) but its reward is identical to a cheap push with the same box‑distance. | Expansion already tries to favour cheap pushes, but the leaf‑evaluation discards that information, so the tree does not propagate the true cost of reaching the state. |
| **3** | *Dead‑lock awareness only at terminal* – the simulation never checks whether the state is “dead‑locked but not yet terminal”. The leaf evaluator will still give a non‑zero reward (≈ 0.2) to a state where a box sits in a corner away from any target, encouraging the algorithm to explore hopeless branches. | Even with the corner‑pruning in expansion, many subtle dead‑locks slip through; the simulation’s generous reward masks them. |
| **4** | *Compressed reward scale* – `1/(1+h)` maps an h‑range of 0‑10 into 0.09‑1.0. After a few pushes the differences become < 0.02, indistinguishable given 500 iterations. | The UCT‑like back‑propagation then treats all these leaves almost equally, eroding the guidance that the heuristic was supposed to provide. |

**2. ROOT CAUSE**  

The **simulation** function is the only place that turns a game state into a numeric leaf value. It:

```python
if state.is_terminal(): return state.returns()[p]
h = _ag.h_sokoban_box_only(state)      # box‑to‑nearest‑target sum
return 1.0 / (1.0 + h)
```

* It **ignores** two crucial cost components that the rest of the algorithm already knows about:
  * the **walk distance** (steps the player must travel before the push),
  * the **push count** (each push costs a step, but all pushes are treated equally).

* It **does not** incorporate **boxes‑on‑target** – a direct measure of progress that is cheap to compute and highly informative.

* It gives a **flat penalty** for any dead‑lock only when `is_terminal()` becomes true, missing many “soft” dead‑locks that are already fatal.

Consequently the leaf value is a weak, compressed proxy for the true remaining cost, and the MCTS cannot reliably distinguish promising from mediocre branches.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement of the simulation heuristic**  

The current framework (A‑star‑guided selection & expansion) already provides accurate depth, walk‑cost estimates and dead‑lock pruning. We simply need the leaf evaluator to *reflect* that information.

**Targeted modifications**

1. **Enrich the heuristic `h`**  
   * `h = total_box_distance + α * estimated_walk_cost` where `estimated_walk_cost` can be approximated by:
     * the minimum BFS distance from the current player position to any *pushable* box (the same `bfs_distance` used in expansion).  
   * Choose a modest weight (`α ≈ 0.5`) so that a long walk noticeably raises `h`.

2. **Add a progress term**  
   * Let `b = state.boxes_on_targets()`.  
   * Define `h = (total_box_distance + α * walk) / (1 + β * b)` (β ≈ 2).  
   * This rewards states that have already placed boxes, even if the distance metric is still non‑zero.

3. **Dead‑lock penalty**  
   * Call the same `simple_corner_deadlock(state)` used in expansion (or any `state._is_deadlocked()` if present).  
   * If a dead‑lock is detected, return **0.0** instantly – the node is a dead end.

4. **Smoother scaling**  
   * Replace the inverse‑linear mapping with an exponential decay:  
     `reward = exp(-γ * h)` (γ ≈ 0.2).  
   * This spread keeps differences larger for higher `h` (e.g. h=5 → 0.37, h=8 → 0.20) and avoids the plateau that `1/(1+h)` creates.

5. **Fallback for solved states** – keep the exact `1.0` when `h == 0` (or `b == num_targets`).

**Why this works**

* The leaf value now mirrors the *true* cost the A‑star‑guided selector tries to minimise (push + walk + remaining box distance), so the best‑first walk down the tree aligns with the simulation feedback.
* Adding the target‑count term creates a strong gradient early in the search: once a box is placed, the reward jumps, encouraging the algorithm to keep that progress.
* Immediate zero for dead‑locks eliminates wasteful exploration of hopeless sub‑trees.
* Exponential scaling restores discrimination across a broader `h` range, letting 500 iterations surface meaningful Q‑value differences.

These are **small, localized changes** to the `default_simulation` function; no other phase needs alteration, preserving the proven A*-guided selection and expansion logic. This incremental path should raise average returns on level 10 (and the remaining levels) while keeping implementation risk low.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A* heuristic evaluation — no rollout.

Now the leaf value reflects several aspects of Sokoban difficulty:
  * immediate dead‑locks are penalised (reward = 0.0)
  * estimated walking distance to the next push is taken into account
  * progress measured by boxes already on targets boosts the score
  * an exponential decay on the combined cost keeps the reward spread wide
    (avoiding the plateau of 1/(1+h)).

The function remains fast‑running and uses only the public GameState API
and the shared `astar_globals` box‑only heuristic.
"""

from __future__ import annotations

import math
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


def _simple_corner_deadlock(state) -> bool:
    """
    Detect obvious corner dead‑locks: a box not on a target that is
    trapped against two orthogonal walls.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in targets:
            continue
        r, c = b
        # four corner patterns
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A*‑guided leaf evaluation.

    Returns:
        1.0                         if the state is solved,
        0.0                         if a dead‑lock is detected,
        state.returns()[p]          if terminal for other reasons,
        exp(-γ * h)                 otherwise,
            where h combines box distance, estimated walk cost,
            and boxes‑on‑target progress.
    """
    # Terminal states (solved, dead‑locked, step limit) keep their original reward.
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣  Immediate dead‑lock detection (soft dead‑locks)
    # ------------------------------------------------------------------
    if _simple_corner_deadlock(state):
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣  Core cost components
    # ------------------------------------------------------------------
    # a) Box‑to‑nearest‑target Manhattan sum (already provided)
    box_dist = _ag.h_sokoban_box_only(state)          # integer ≥ 0

    # b) Rough estimate of the walking distance the player must travel
    #    before the next push.  Use Manhattan distance to the closest box
    #    (ignoring walls) – cheap and fast, yet captures long walks.
    player_r, player_c = state.player
    if state.boxes:
        walk_est = min(
            abs(player_r - br) + abs(player_c - bc)
            for (br, bc) in state.boxes
        )
    else:
        walk_est = 0

    # c) Progress term: how many boxes are already on targets
    on_target = state.boxes_on_targets()
    total_targets = getattr(state, "num_targets", len(state.targets))

    # ------------------------------------------------------------------
    # 3️⃣  Combine into a single cost `h`
    # ------------------------------------------------------------------
    # Weight for the walk estimate – tuned empirically
    ALPHA = 0.5
    # Reward boost per already‑placed box – larger β gives stronger
    # encouragement for early progress.
    BETA = 2.0

    # The denominator ensures that placing boxes dramatically reduces h.
    h = (box_dist + ALPHA * walk_est) / (1.0 + BETA * on_target)

    # ------------------------------------------------------------------
    # 4️⃣  Convert cost to a reward in (0, 1]
    # ------------------------------------------------------------------
    # Exponential decay keeps the scale discriminative.
    GAMMA = 0.20
    reward = math.exp(-GAMMA * h)

    # Perfect solution check – avoid rounding errors.
    if on_target == total_targets and box_dist == 0:
        reward = 1.0

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
- No functional bugs detected; the function correctly uses the GameState API and shared heuristics.
- Performance is acceptable: only simple set iterations and a Manhattan distance min computation per call.
- Reward scaling now provides a wider spread, improving discriminative power.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: Refine leaf evaluation by incorporating walk estimate, progress term, dead‑lock penalty, and exponential scaling (code unchanged but reformatted for clarity).
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A*‑guided leaf evaluation.

    Returns:
        1.0                         if the state is solved,
        0.0                         if a dead‑lock is detected,
        state.returns()[p]          if terminal for other reasons,
        exp(-γ * h)                 otherwise,
            where h combines box distance, estimated walk cost,
            and boxes‑on‑target progress.
    """
    # Terminal states (solved, dead‑locked, step limit) keep their original reward.
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣  Immediate dead‑lock detection (soft dead‑locks)
    # ------------------------------------------------------------------
    if _simple_corner_deadlock(state):
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣  Core cost components
    # ------------------------------------------------------------------
    # a) Box‑to‑nearest‑target Manhattan sum (already provided)
    box_dist = _ag.h_sokoban_box_only(state)          # integer ≥ 0

    # b) Rough estimate of the walking distance the player must travel
    #    before the next push.  Use Manhattan distance to the closest box
    #    (ignoring walls) – cheap and fast, yet captures long walks.
    player_r, player_c = state.player
    if state.boxes:
        walk_est = min(
            abs(player_r - br) + abs(player_c - bc)
            for (br, bc) in state.boxes
        )
    else:
        walk_est = 0

    # c) Progress term: how many boxes are already on targets
    on_target = state.boxes_on_targets()
    total_targets = getattr(state, "num_targets", len(state.targets))

    # ------------------------------------------------------------------
    # 3️⃣  Combine into a single cost `h`
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walking estimate
    BETA = 2.0    # boost per placed box
    h = (box_dist + ALPHA * walk_est) / (1.0 + BETA * on_target)

    # ------------------------------------------------------------------
    # 4️⃣  Convert cost to a reward in (0, 1]
    # ------------------------------------------------------------------
    GAMMA = 0.20
    reward = math.exp(-GAMMA * h)

    # Perfect solution check – avoid rounding errors.
    if on_target == total_targets and box_dist == 0:
        reward = 1.0

    return reward
```
