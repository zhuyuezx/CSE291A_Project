# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:20:45 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 13.68s |
| Status     | success |
| Tokens     | 6317 |
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
LLM-generated MCTS tool: simulation
Description: Added missing deadlock detection helper (`_simple_corner_deadlock`) and ensured required imports are present. The heuristic logic remains unchanged.
Generated:   2026-03-11T00:20:09.491905
"""

import math
from typing import Any

def default_simulation(
    state: Any,
    perspective_player: int,
    max_depth: int = 0,
) -> float:
    """
    Heuristic leaf evaluation for MCTS.

    Returns a floating‑point reward that reflects both distance to the goal
    and the cost already incurred (walk steps + pushes). The reward is:

        * 1.0                               – if the state is solved
        * state.returns()[p]                – if the state is terminal
        * exp(-γ * blended_cost)            – otherwise

    where ``blended_cost = h_box + λ * g_steps``.
    ``h_box``  – sum of Manhattan distances from each box to its nearest target
                 (available via ``state.total_box_distance()``).
    ``g_steps`` – total steps taken so far (walk + pushes), available via
                 ``state.steps``.
    ``γ`` and ``λ`` are small positive constants that shape the reward curve.
    """
    # ------------------------------------------------------------------- #
    # Helper: simple corner dead‑lock detection
    # ------------------------------------------------------------------- #
    def _simple_corner_deadlock(s) -> bool:
        """
        Detects the classic corner dead‑lock: a box that is not already on a target
        and is positioned in a corner formed by two orthogonal walls (or a wall
        and another box that is itself a wall‑adjacent corner). This is a cheap
        approximation and suffices for the default simulation.
        """
        walls = s.walls
        boxes = s.boxes
        targets = s.targets

        # Directions that form a corner (dx, dy) pairs
        corner_dirs = [((-1, 0), (0, -1)),  # left & up
                       ((-1, 0), (0, 1)),   # left & down
                       ((1, 0), (0, -1)),   # right & up
                       ((1, 0), (0, 1))]    # right & down

        for bx, by in boxes:
            if (bx, by) in targets:
                continue  # already solved, not a dead‑lock
            for (dx1, dy1), (dx2, dy2) in corner_dirs:
                neighbor1 = (bx + dx1, by + dy1)
                neighbor2 = (bx + dx2, by + dy2)
                if (neighbor1 in walls or neighbor1 in boxes) and \
                   (neighbor2 in walls or neighbor2 in boxes):
                    return True
        return False

    # ------------------------------------------------------------------- #
    # 1) Terminal / solved handling (unchanged)
    # ------------------------------------------------------------------- #
    if state.is_terminal():
        # Terminal includes dead‑lock or max‑step limit; the GameState
        # supplies the appropriate return vector.
        return float(state.returns()[perspective_player])

    if state.boxes_on_targets() == state.num_targets:
        # Perfect solution – maximal reward.
        return 1.0

    # ------------------------------------------------------------------- #
    # 2) Early dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------- #
    if _simple_corner_deadlock(state):
        # Treat as a dead‑locked terminal state.
        return float(state.returns()[perspective_player])

    # ------------------------------------------------------------------- #
    # 3) Compute blended cost:
    #    h  = box‑only Manhattan distance (lower is better)
    #    g  = steps already spent (walk + push cost)
    #    blended = h + λ * g
    # ------------------------------------------------------------------- #
    try:
        h = state.total_box_distance()
    except Exception:
        # Fallback to the box‑only heuristic from the shared module if
        # the public method is unavailable.
        from astar_globals import h_sokoban_box_only as _h_box_only
        h = _h_box_only(state)

    g = getattr(state, "steps", 0)  # total steps taken so far

    # Weight for the step component – empirically chosen to give
    # comparable magnitude to the distance term.
    LAMBDA = 0.5
    blended_cost = h + LAMBDA * g

    # ------------------------------------------------------------------- #
    # 4) Shape the reward.
    #    Using an exponential yields values in (0, 1] and provides a
    #    steeper gradient for small differences in blended_cost.
    # ------------------------------------------------------------------- #
    GAMMA = 0.2
    reward = math.exp(-GAMMA * blended_cost)

    # Guard against numerical underflow for extremely large costs.
    if reward < 1e-12:
        reward = 0.0

    return float(reward)
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (trace #1) | Evidence |
|------|-------------------|----------|
| **1** | **Progress on boxes is not rewarded** – after move 6 a box is placed on a target (1/4), but the child node’s average value collapses from ~0.22 to ~0.07. | Q‑values of children after the first successful push (move 6) are dramatically lower than earlier branches, despite the state being objectively closer to the goal. |
| **2** | **Reward saturation** – the exponential `exp(-γ·(h+λ·g))` produces values in a very narrow band (≈0.15–0.23) for most non‑terminal nodes, giving the tree little gradient to separate good from mediocre pushes. | All leaf returns lie between 0.06 and 0.23; the selection step therefore relies almost entirely on the A*‑guided f‑score, not on the simulation feedback. |
| **3** | **Step‑cost weight is too low** – λ = 0.5 makes the accumulated walk‑cost (g ≈ 27) contribute only ~13 to the blended cost, while the Manhattan distance (h ≈ 5) dominates. The heuristic therefore prefers cheap pushes that keep distance low, even if they waste many steps and do not advance boxes. | After move 7 the total steps jump from 8 to 16, yet the blended cost changes only from ~8.5 to ~12.5, barely affecting the reward. |
| **4** | **Dead‑lock detection is minimal** – only simple corner patterns are checked. More subtle dead‑locks (boxes against walls with no reachable target line) remain unpenalised, allowing the search to waste iterations on hopeless branches. | No explicit dead‑lock penalties appear in the trace, and the tree explores deep into a region that later becomes a dead‑end (final state with 2/4 boxes). |

**2. ROOT CAUSE**

The simulation function evaluates a leaf state with  

```
reward = exp( -γ * ( h + λ * g ) )
```

- **No box‑on‑target term** → progress that reduces `h` only marginally (e.g., a box already on a target) does not raise the reward appreciably.  
- **Exponential compression** → the range of `exp(-γ·x)` for realistic `x` (≈5‑15) is <0.3, providing little discrimination for the UCT‑style back‑propagation used later.  
- **Low λ** → steps already taken contribute far less than distance, so the heuristic ignores the costly walks that the macro‑push model charges.  
- **Dead‑lock handling** is limited to corner detection; the simulation simply returns the terminal‐state payoff if `_simple_corner_deadlock` fires, otherwise it treats the state as normal, missing many “soft” dead‑locks that should be heavily penalised.

Because back‑propagation aggregates these compressed rewards, the value estimates for promising nodes stay low, causing the selection phase to drift toward cheap, distance‑only pushes instead of moves that increase `boxes_on_targets`.

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinement of the simulation heuristic**

The core issue is the *reward formulation*. A modest set of additions can make the simulation much more informative while preserving the existing infrastructure:

1. **Add a target‑completion bonus**  
   ```python
   progress = state.boxes_on_targets() / state.num_targets
   bonus = B * progress            # B ≈ 0.4 … 0.6
   ```
   Incorporate `bonus` additively (or multiplicatively) with the exponential term so that each box placed on a goal raises the leaf value substantially.

2. **Replace the exponential with a scaled linear‑plus‑exponential blend**  
   Preserve the smooth decay but keep a larger dynamic range:
   ```python
   base = 1.0 - α * (h / h_max)    # α ≈ 0.4
   step_penalty = β * g            # β ≈ 0.02
   reward = (base - step_penalty) + bonus
   reward = max(0.0, min(1.0, reward))
   ```
   This yields rewards roughly between 0.0 and 1.0, giving the back‑propagation step a richer signal.

3. **Increase the step‑weight λ** (or incorporate β above) so that long walks are discouraged proportionally to their true cost in macro‑pushes. A value around 1.0–1.5 aligns the magnitude of `g` with `h`.

4. **Stronger dead‑lock penalty**  
   When `_simple_corner_deadlock` is true, return **0.0** (instead of delegating to `state.returns()`, which may be 0 anyway) and optionally add a small negative bias for states that are *almost* dead‑locked (e.g., a box adjacent to a wall with no target in that row/column).

These four tweaks keep the existing API, require only changes inside `default_simulation`, and directly address the top‑ranked weaknesses: they give the leaf evaluation a clear signal for boxes placed, expand the reward range, penalise wasted steps, and block deeper exploration of dead‑ends.

**Why not a full restructure?**  
The current MCTS pipeline (A*‑guided selection, expansion with monotonicity pruning, back‑propagation) already provides a solid search skeleton. The only bottleneck is the leaf evaluator’s lack of granularity, which can be remedied with the targeted modifications above. A complete redesign would be unnecessary and risk discarding the proven A*‑guided guidance.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
import math
from typing import Any

def default_simulation(
    state: Any,
    perspective_player: int,
    max_depth: int = 0,
) -> float:
    """
    Refined leaf evaluation for MCTS.

    • Returns 1.0 for solved states.
    • Returns the terminal payoff for any terminal (dead‑locked or max‑step) state.
    • Detects simple corner dead‑locks early and returns 0.0 (hard penalty).
    • Otherwise computes a blended cost based on:
        – h : sum of Manhattan distances from each box to its nearest target
        – g : total steps taken so far (walk + pushes)
      and combines it with a progress bonus proportional to the fraction of
      boxes already placed on targets.
    The final reward is clamped to [0.0, 1.0] to give the back‑propagation
    a rich, discriminative signal.
    """
    # --------------------------------------------------------------------- #
    # Helper: simple corner dead‑lock detection
    # --------------------------------------------------------------------- #
    def _simple_corner_deadlock(s) -> bool:
        """
        Detect classic corner dead‑locks: a non‑target box trapped by two
        orthogonal obstacles (walls or other boxes). This cheap check catches
        the most common unsolvable patterns.
        """
        walls = s.walls
        boxes = s.boxes
        targets = s.targets

        corner_dirs = [((-1, 0), (0, -1)),  # up & left
                       ((-1, 0), (0, 1)),   # up & right
                       ((1, 0), (0, -1)),   # down & left
                       ((1, 0), (0, 1))]    # down & right

        for bx, by in boxes:
            if (bx, by) in targets:
                continue
            for (dx1, dy1), (dx2, dy2) in corner_dirs:
                n1 = (bx + dx1, by + dy1)
                n2 = (bx + dx2, by + dy2)
                if (n1 in walls or n1 in boxes) and (n2 in walls or n2 in boxes):
                    return True
        return False

    # --------------------------------------------------------------------- #
    # 1) Terminal / solved handling
    # --------------------------------------------------------------------- #
    if state.is_terminal():
        # Terminal includes dead‑lock or max‑step limit; rely on GameState's
        # return vector.
        return float(state.returns()[perspective_player])

    if state.boxes_on_targets() == state.num_targets:
        # All boxes are on goals – highest possible reward.
        return 1.0

    # --------------------------------------------------------------------- #
    # 2) Early dead‑lock detection (hard penalty)
    # --------------------------------------------------------------------- #
    if _simple_corner_deadlock(state):
        # Immediate dead‑lock – treat as the worst possible leaf.
        return 0.0

    # --------------------------------------------------------------------- #
    # 3) Core heuristic components
    # --------------------------------------------------------------------- #
    # h – box‑to‑target Manhattan distance (lower is better)
    try:
        h = state.total_box_distance()
    except Exception:
        # Fallback to the shared box‑only heuristic if the method is absent.
        from astar_globals import h_sokoban_box_only as _h_box_only
        h = _h_box_only(state)

    # g – total steps already spent (walk + pushes)
    g = getattr(state, "steps", 0)

    # progress – fraction of boxes already placed on targets
    progress = (
        state.boxes_on_targets() / state.num_targets
        if state.num_targets > 0
        else 0.0
    )

    # --------------------------------------------------------------------- #
    # 4) Parameterised blend
    # --------------------------------------------------------------------- #
    # Weight for step cost – increased to give walks comparable influence.
    LAMBDA = 1.0      # step‑cost multiplier
    # Exponential decay factor – smaller γ yields a wider value range.
    GAMMA = 0.15
    # Bonus weight for each box already on a target.
    BONUS_WEIGHT = 0.4

    blended_cost = h + LAMBDA * g
    exp_term = math.exp(-GAMMA * blended_cost)

    # Add a linear progress bonus (0 … BONUS_WEIGHT)
    reward = exp_term + BONUS_WEIGHT * progress

    # Clamp to the valid range.
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
- No issues found.

ACTION: modify
FILE_NAME: default_simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: No changes required; the function is correct and efficient.
```python
import math
from typing import Any

def default_simulation(
    state: Any,
    perspective_player: int,
    max_depth: int = 0,
) -> float:
    """
    Refined leaf evaluation for MCTS.

    • Returns 1.0 for solved states.
    • Returns the terminal payoff for any terminal (dead‑locked or max‑step) state.
    • Detects simple corner dead‑locks early and returns 0.0 (hard penalty).
    • Otherwise computes a blended cost based on:
        – h : sum of Manhattan distances from each box to its nearest target
        – g : total steps taken so far (walk + pushes)
      and combines it with a progress bonus proportional to the fraction of
      boxes already placed on targets.
    The final reward is clamped to [0.0, 1.0] to give the back‑propagation
    a rich, discriminative signal.
    """
    # --------------------------------------------------------------------- #
    # Helper: simple corner dead‑lock detection
    # --------------------------------------------------------------------- #
    def _simple_corner_deadlock(s) -> bool:
        """
        Detect classic corner dead‑locks: a non‑target box trapped by two
        orthogonal obstacles (walls or other boxes). This cheap check catches
        the most common unsolvable patterns.
        """
        walls = s.walls
        boxes = s.boxes
        targets = s.targets

        corner_dirs = [((-1, 0), (0, -1)),  # up & left
                       ((-1, 0), (0, 1)),   # up & right
                       ((1, 0), (0, -1)),   # down & left
                       ((1, 0), (0, 1))]    # down & right

        for bx, by in boxes:
            if (bx, by) in targets:
                continue
            for (dx1, dy1), (dx2, dy2) in corner_dirs:
                n1 = (bx + dx1, by + dy1)
                n2 = (bx + dx2, by + dy2)
                if (n1 in walls or n1 in boxes) and (n2 in walls or n2 in boxes):
                    return True
        return False

    # --------------------------------------------------------------------- #
    # 1) Terminal / solved handling
    # --------------------------------------------------------------------- #
    if state.is_terminal():
        # Terminal includes dead‑lock or max‑step limit; rely on GameState's
        # return vector.
        return float(state.returns()[perspective_player])

    if state.boxes_on_targets() == state.num_targets:
        # All boxes are on goals – highest possible reward.
        return 1.0

    # --------------------------------------------------------------------- #
    # 2) Early dead‑lock detection (hard penalty)
    # --------------------------------------------------------------------- #
    if _simple_corner_deadlock(state):
        # Immediate dead‑lock – treat as the worst possible leaf.
        return 0.0

    # --------------------------------------------------------------------- #
    # 3) Core heuristic components
    # --------------------------------------------------------------------- #
    # h – box‑to‑target Manhattan distance (lower is better)
    try:
        h = state.total_box_distance()
    except Exception:
        # Fallback to the shared box‑only heuristic if the method is absent.
        from astar_globals import h_sokoban_box_only as _h_box_only
        h = _h_box_only(state)

    # g – total steps already spent (walk + pushes)
    g = getattr(state, "steps", 0)

    # progress – fraction of boxes already placed on targets
    progress = (
        state.boxes_on_targets() / state.num_targets
        if state.num_targets > 0
        else 0.0
    )

    # --------------------------------------------------------------------- #
    # 4) Parameterised blend
    # --------------------------------------------------------------------- #
    # Weight for step cost – increased to give walks comparable influence.
    LAMBDA = 1.0      # step‑cost multiplier
    # Exponential decay factor – smaller γ yields a wider value range.
    GAMMA = 0.15
    # Bonus weight for each box already on a target.
    BONUS_WEIGHT = 0.4

    blended_cost = h + LAMBDA * g
    exp_term = math.exp(-GAMMA * blended_cost)

    # Add a linear progress bonus (0 … BONUS_WEIGHT)
    reward = exp_term + BONUS_WEIGHT * progress

    # Clamp to the valid range.
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    return float(reward)
```
