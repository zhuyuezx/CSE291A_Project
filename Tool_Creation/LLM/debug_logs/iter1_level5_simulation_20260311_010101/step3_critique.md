# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:01:37 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.97s |
| Status     | success |
| Tokens     | 6093 |
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

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
        min_walk = 0  # treat as zero – the dead‑lock will be handled by terminal check.

    # ------------------------------------------------------------------
    # 3️⃣  Combine the components.
    # ------------------------------------------------------------------
    # Weight for walk cost (tuned empirically – small enough to keep box distance dominant)
    WALK_WEIGHT = 0.1
    # Scaling factor for the final reward denominator (softens compression of large h)
    BETA = 0.5

    # Augmented heuristic: box distance + penalise long walks
    h_aug = h_box + WALK_WEIGHT * min_walk

    # Progress term: how many boxes are already on targets
    boxes_done = state.boxes_on_targets()          # 0 … num_targets

    # Smooth reward: (progress + 1) / (progress + 1 + β * h_aug)
    # Adding 1 to the numerator avoids division by zero when both are zero.
    reward = (boxes_done + 1) / (boxes_done + 1 + BETA * h_aug)

    # Ensure reward stays in (0, 1] and clamp tiny floating errors
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    return float(reward)
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
- No issues found – the function correctly uses the GameState API, adds the intended heuristic components, and avoids unnecessary overhead.

ACTION: modify
FILE_NAME: default_simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: No changes needed; preserve the improved heuristic implementation.
```python
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
