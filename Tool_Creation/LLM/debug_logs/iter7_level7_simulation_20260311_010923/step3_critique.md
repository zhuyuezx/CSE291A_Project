# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:09:57 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.69s |
| Status     | success |
| Tokens     | 6104 |
| Validation | valid |

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

**1. KEY WEAKNESSES**

| Rank | Symptom (trace #1) | Why it matters | Evidence |
|------|-------------------|----------------|----------|
| 1 | **Flat‑shaped reward** – `reward = 1/(1+h)` where *h* is only the sum of Manhattan distances from each box to the *nearest* target. | Ignores how many boxes are already on targets, how costly the push was (walk + push steps), and any emerging dead‑lock. The algorithm therefore cannot distinguish a cheap push that brings a box a little closer from a long walk that sacrifices future mobility. | After 500 visits the tree concentrates on action `(3,6),UP` (v=483, avg 0.424) even though it yields only 2/3 boxes on target after 17 steps; other actions with similar *h* receive far lower visit counts. |
| 2 | **No depth / step penalty** – the simulation never discounts rewards for deeper leaves. | MCTS will gladly explore long “stagnant” branches that keep *h* unchanged, because the reward stays the same. This inflates visit counts on actions that loop without progress. | The final leaf has `total distance = 1` but reward ≈ 0.5 (since `1/(1+1)=0.5`). A deeper leaf with the same *h* would receive the identical value, giving it no disadvantage. |
| 3 | **Dead‑lock detection only at terminal** – `state.is_terminal()` is checked, but many dead‑locks are *non‑terminal* (e.g., a box stuck in a corner that is not a target). | The heuristic still assigns a positive reward to a state that is actually unsolvable, causing the search to waste visits on hopeless branches. | The trace ends with 2/3 boxes on target but the remaining box sits in a corner; the heuristic still returns `1/(1+0)=1.0` for that box if it were the *nearest* target, misleading the value estimate. |
| 4 | **No explicit encouragement for boxes already on targets** – the reward does not scale with `boxes_on_targets`. | A state with 2/3 boxes placed and *h* = 1 gets the same numerical reward as a state with 0/3 boxes placed and *h* = 1 (both 0.5). The planner therefore undervalues progress toward the goal. | After move 3 the child with 1/3 boxes on target has avg ≈ 0.475, barely above children with 0 boxes; the algorithm does not push the remaining box aggressively. |

**2. ROOT CAUSE**

The *simulation* phase is a one‑liner heuristic:

```python
if state.is_terminal():
    return state.returns()[p]
h = _ag.h_sokoban_box_only(state)
return 1.0 / (1.0 + h)
```

* It uses only `_ag.h_sokoban_box_only`, i.e. the **box‑only Manhattan distance**.  
* It never looks at `state.steps`, `state.boxes_on_targets()`, or any dead‑lock predicate (`_is_deadlocked`).  
* The `max_depth` argument is ignored, so no depth discount is applied.  

Consequently the reward is *purely distance‑based* and constant across depths, giving the tree no signal to prefer cheap pushes, avoid dead‑locks, or reward already‑placed boxes.

**3. PROPOSED APPROACH – A (incremental)**  

We can keep the existing structure (fast, no rollout) and enrich the score with a few cheap, high‑impact terms:

1. **Progress bonus** – add a linear component proportional to the fraction of boxes already on targets.  
   `progress = boxes_on_targets / num_targets`

2. **Step‑cost penalty** – subtract a small factor proportional to the total steps taken so far (or to the “g‑cost” of the leaf).  
   `step_pen = α * state.steps` with `α ≈ 0.01`

3. **Depth discount** – optionally multiply by `γ ** depth`, where depth can be approximated by `state.steps` (or by a passed‑in `max_depth`). A modest `γ ≈ 0.95` gently penalises very deep leaves.

4. **Early dead‑lock abort** – if the state provides `_is_deadlocked` and it returns True, return `0.0` immediately.

5. **Combine** – a simple formula that remains fast:

```
if state.is_terminal():
    return state.returns()[p]

if hasattr(state, "_is_deadlocked") and state._is_deadlocked():
    return 0.0

h = _ag.h_sokoban_box_only(state)          # distance term
progress = state.boxes_on_targets() / state.num_targets
base = (1.0 + progress) / (1.0 + h)        # lifts reward when more boxes placed
penalised = max(0.0, base - α * getattr(state, "steps", 0))
return penalised * (γ ** getattr(state, "steps", 0))
```

These modifications are **local** to the simulation file, preserve the deterministic, zero‑rollout nature, and add only a handful of arithmetic operations, keeping per‑iteration overhead negligible. By rewarding *progress* and penalising *expensive* or *dead‑locked* leaves, the tree will shift visits toward actions that genuinely advance the puzzle, improving both average return and solve rate on the currently failing levels (especially level 7).

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A* heuristic evaluation — enriched reward.

The original simulation returned a flat 1/(1+h) based only on the
box‑only Manhattan distance.  That ignored useful signals such as:
  • how many boxes are already placed on targets,
  • how many steps have been spent (long walks should be penalised),
  • whether the state is already dead‑locked,
  • and that deeper leaves should be less valuable.

This version keeps the zero‑rollout speed but mixes several cheap
factors into a single scalar reward:

    * progress      = boxes_on_targets / num_targets
    * distance term = h = box‑only Manhattan distance
    * base reward   = (1 + PROGRESS_WEIGHT * progress) / (1 + h)
    * step penalty  = STEP_PENALTY * steps
    * depth discount= GAMMA ** steps   (steps ≈ depth in macro‑push variant)

If a dead‑lock is detected via the optional `_is_deadlocked` method the
reward is immediately 0.0.  Terminal states (solved / max‑steps / loss)
still return the environment‑provided returns.

The resulting reward stays in [0, 1] (or slightly above 1 for a solved
state before clamping) and varies smoothly with progress, encouraging
MCTS to favour cheap, goal‑making pushes while avoiding wasteful loops.
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
    Enriched heuristic evaluation for a leaf node.

    Returns:
        * state.returns()[p]                         if the state is terminal
        * 0.0                                         if a dead‑lock is detected
        * ((1 + w * progress) / (1 + h) - α * steps) * γ**steps
          otherwise, where
            - h = box‑only Manhattan distance (sum of nearest‑target distances)
            - progress = boxes_on_targets / num_targets
            - w, α, γ are tunable constants (default w=0.5, α=0.01, γ=0.97)

    The formula yields values in (0, 1] for non‑terminal,
    non‑dead‑locked states and heavily penalises deep, costly leaves.
    """
    # ------------------------------------------------------------------
    # Tunable hyper‑parameters – adjust if needed without breaking API.
    # ------------------------------------------------------------------
    PROGRESS_WEIGHT = 0.5   # boost for each fraction of boxes already placed
    STEP_PENALTY = 0.01     # linear penalty per macro‑push step taken
    GAMMA = 0.97            # depth discount factor (γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣  Terminal handling (solved, max‑steps, loss)
    # ------------------------------------------------------------------
    if state.is_terminal():
        # The environment already encodes solved/dead‑lock/step‑limit rewards.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 2️⃣  Early dead‑lock detection (optional helper on the state)
    # ------------------------------------------------------------------
    if hasattr(state, "_is_deadlocked") and callable(state._is_deadlocked):
        try:
            if state._is_deadlocked():
                return 0.0
        except Exception:
            # Defensive: if the dead‑lock check itself fails, ignore it.
            pass

    # ------------------------------------------------------------------
    # 3️⃣  Core heuristic components
    # ------------------------------------------------------------------
    # Box‑only Manhattan distance (same as used by A* expansion)
    h = _ag.h_sokoban_box_only(state)

    # Fraction of boxes already on their targets (0.0 … 1.0)
    progress = (
        state.boxes_on_targets() / state.num_targets
        if state.num_targets > 0 else 0.0
    )

    # Base reward mixes distance and progress.
    # When h == 0 (all boxes on targets) the formula yields >1;
    # we will clamp later.
    base_reward = (1.0 + PROGRESS_WEIGHT * progress) / (1.0 + h)

    # ------------------------------------------------------------------
    # 4️⃣  Apply step‑cost penalty (steps = macro‑push count so far)
    # ------------------------------------------------------------------
    steps = getattr(state, "steps", 0)
    penalised = base_reward - STEP_PENALTY * steps
    if penalised < 0.0:
        penalised = 0.0

    # ------------------------------------------------------------------
    # 5️⃣  Depth discount – deeper leaves become less attractive.
    # ------------------------------------------------------------------
    # Use the same `steps` value as a proxy for depth.  If `max_depth`
    # is supplied (non‑zero) we cap the exponent to avoid under‑flow.
    depth = steps
    if max_depth > 0:
        depth = min(depth, max_depth)

    discounted = penalised * (GAMMA ** depth)

    # ------------------------------------------------------------------
    # 6️⃣  Final clamping – keep reward in a sensible range.
    # ------------------------------------------------------------------
    # Solved states (h == 0) should ideally return 1.0.
    if h == 0:
        return 1.0

    # Ensure we never exceed 1.0 after all adjustments.
    if discounted > 1.0:
        discounted = 1.0

    return discounted
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
- No issues found. The function correctly handles terminals, dead‑locks, adds progress, step penalty, and depth discount without unnecessary overhead.

ACTION: modify
FILE_NAME: default_simulation.py
FUNCTION_NAME: default_simulation
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched heuristic evaluation for a leaf node.

    Returns:
        * state.returns()[p]                         if the state is terminal
        * 0.0                                         if a dead‑lock is detected
        * ((1 + w * progress) / (1 + h) - α * steps) * γ**steps
          otherwise, where
            - h = box‑only Manhattan distance (sum of nearest‑target distances)
            - progress = boxes_on_targets / num_targets
            - w, α, γ are tunable constants (default w=0.5, α=0.01, γ=0.97)

    The formula yields values in (0, 1] for non‑terminal,
    non‑dead‑locked states and heavily penalises deep, costly leaves.
    """
    # ------------------------------------------------------------------
    # Tunable hyper‑parameters – adjust if needed without breaking API.
    # ------------------------------------------------------------------
    PROGRESS_WEIGHT = 0.5   # boost for each fraction of boxes already placed
    STEP_PENALTY = 0.01     # linear penalty per macro‑push step taken
    GAMMA = 0.97            # depth discount factor (γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣  Terminal handling (solved, max‑steps, loss)
    # ------------------------------------------------------------------
    if state.is_terminal():
        # The environment already encodes solved/dead‑lock/step‑limit rewards.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 2️⃣  Early dead‑lock detection (optional helper on the state)
    # ------------------------------------------------------------------
    if hasattr(state, "_is_deadlocked") and callable(state._is_deadlocked):
        try:
            if state._is_deadlocked():
                return 0.0
        except Exception:
            # Defensive: if the dead‑lock check itself fails, ignore it.
            pass

    # ------------------------------------------------------------------
    # 3️⃣  Core heuristic components
    # ------------------------------------------------------------------
    # Box‑only Manhattan distance (same as used by A* expansion)
    h = _ag.h_sokoban_box_only(state)

    # Fraction of boxes already on their targets (0.0 … 1.0)
    progress = (
        state.boxes_on_targets() / state.num_targets
        if getattr(state, "num_targets", 0) > 0 else 0.0
    )

    # Base reward mixes distance and progress.
    # When h == 0 (all boxes on targets) the formula yields >1;
    # we will clamp later.
    base_reward = (1.0 + PROGRESS_WEIGHT * progress) / (1.0 + h)

    # ------------------------------------------------------------------
    # 4️⃣  Apply step‑cost penalty (steps = macro‑push count so far)
    # ------------------------------------------------------------------
    steps = getattr(state, "steps", 0)
    penalised = base_reward - STEP_PENALTY * steps
    if penalised < 0.0:
        penalised = 0.0

    # ------------------------------------------------------------------
    # 5️⃣  Depth discount – deeper leaves become less attractive.
    # ------------------------------------------------------------------
    # Use the same `steps` value as a proxy for depth.  If `max_depth`
    # is supplied (non‑zero) we cap the exponent to avoid under‑flow.
    depth = steps
    if max_depth > 0:
        depth = min(depth, max_depth)

    discounted = penalised * (GAMMA ** depth)

    # ------------------------------------------------------------------
    # 6️⃣  Final clamping – keep reward in a sensible range.
    # ------------------------------------------------------------------
    # Solved states (h == 0) should ideally return 1.0.
    if h == 0:
        return 1.0

    # Ensure we never exceed 1.0 after all adjustments.
    if discounted > 1.0:
        discounted = 1.0

    return discounted
```
