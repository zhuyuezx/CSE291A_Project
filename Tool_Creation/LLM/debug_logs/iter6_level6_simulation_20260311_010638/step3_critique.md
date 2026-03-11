# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:07:13 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 7.89s |
| Status     | success |
| Tokens     | 6057 |
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

| Rank | Symptom (trace evidence) | Why it hurts MCTS |
|------|--------------------------|-------------------|
| **1** | The leaf reward never exceeds **0.5** when a single box is one step from a target (h = 1 ⇒ reward ≈ 0.5). After move 12 the search still favours the *0.47–0.50*‑range children and never commits to the final push. The gradient is too shallow to make the last‑step action look decisive. |
| **2** | The heuristic ignores **player walking cost** and **push cost**. Two states with identical box‑to‑target sums can differ wildly in the number of steps required to reach the push; the simulator treats them as equal, so MCTS wastes visits on expensive pushes that do not improve the distance. |
| **3** | No **penalty for distance regression**. Actions that increase the total box‑to‑target distance receive the same 1/(1+h) value as a neutral move (e.g. h = 4 → 0.2). The search therefore does not actively avoid pushes that make the puzzle harder, leading to loops and stagnation observed in moves 4‑6 where the total distance rose from 5→6 repeatedly. |
| **4** | The only “terminal” shortcut is dead‑lock / step‑limit detection; there is no early‑exit when the state is *almost* solved (e.g. all but one box placed). Consequently the simulation continues to evaluate shallow rewards instead of recognising a near‑goal and pushing harder. |

Overall, the simulation provides a **single‑dimensional, low‑resolution score** that cannot discriminate the crucial final pushes from mediocre ones, so the tree “thinks” all actions are roughly equal and never concentrates enough visits on the winning line.

---

**2. ROOT CAUSE**  

The current `default_simulation` does:

```python
if state.is_terminal(): return state.returns()[p]
h = h_sokoban_box_only(state)          # sum of Manhattan distances of boxes to nearest targets
return 1.0 / (1.0 + h)
```

* It uses **only the box‑only Manhattan sum** (`h_sokoban_box_only`).  
* It never looks at **player‑reachable distance**, **push cost**, or **whether the action reduced the total distance**.  
* The mapping `1/(1+h)` yields a **max of 0.5** for h = 1, which is insufficient to create a clear “win” signal.  
* No extra term for **boxes already on targets** or for **distance improvement** exists, so the reward curve is flat across many qualitatively different states.  

Thus the heuristic is *too coarse* and *insensitive* to the strategic factors that drive Sokoban success.

---

**3. PROPOSED APPROACH**  

**Strategy A – Incremental refinement of the simulation heuristic**  

Why incremental?  
* The existing pipeline (selection, expansion, back‑prop) already leverages A*‑style pruning and depth‑aware updates; those components are solid.  
* The only missing piece is a richer leaf evaluation. Adding a few cheap terms will preserve speed while dramatically sharpening the reward signal, addressing the highest‑impact weakness (rank 1).  

**Targeted modifications (conceptual, no code):**

1. **Hybrid distance term** – compute `walk_cost = state.player_to_nearest_push()` (the shortest BFS distance from the player to any cell from which a legal push can be made).  
   *New reward base*: `base = 1.0 / (1.0 + h_box + α * walk_cost)`, with a small weight α (e.g. 0.1) so that a long walk slightly reduces the score.

2. **Progress bias** – pass the parent state’s `total_box_distance` (or keep it in a temporary cache) and reward a *decrease* in that distance:  
   `progress = β * (prev_dist - cur_dist) / prev_dist` (β ≈ 0.2). Add this to the base reward. This makes a move that reduces distance noticeably better than a neutral move.

3. **Target‑completion boost** – give a bonus proportional to the fraction of boxes already on targets:  
   `target_bonus = γ * (boxes_on_targets / num_targets)` (γ ≈ 0.15). This lifts states where most boxes are placed, pushing the final steps into a higher‑reward region.

4. **Steeper final‑step signal** – after applying the three terms, clamp the reward so that a state with `h = 1` and a positive progress bonus exceeds **0.8**. This can be achieved by scaling: `reward = min(1.0, reward * scale)` where `scale` is determined empirically (e.g., 2.0). The important thing is to make the final push *clearly* more attractive than any earlier action.

5. **Optional dead‑lock early exit** – keep the existing `state.is_terminal()` check but also return a very low value (e.g., 0.01) for states where `h` increased compared to the parent, discouraging regressions.

All added terms are **O(1)** or O(F) where F is the number of reachable floor cells (already computed during BFS for legal actions), so the per‑simulation cost stays tiny relative to the thousands of rollouts.

**Resulting effect**  
* The reward gradient will span the full (0, 1] range, making the last‑step push stand out.  
* Moves that cost the player many steps or that do not improve distance will be demoted, focusing the tree on efficient push sequences.  
* Near‑goal states receive a noticeable bump, encouraging the planner to finish the puzzle rather than wander.

If after a few iterations the solver still fails on level 6, a **paradigm shift** (Strategy B) could involve a mini‑rollout that performs a few deterministic “greedy‑push” steps, but the incremental changes above directly address the dominant weakness and preserve the fast, heuristic‑only simulation already proven to work on the mastered levels.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A* heuristic evaluation — enriched simulation.

The original version used only the box‑only Manhattan sum:
    reward = 1 / (1 + h)

That signal is too shallow for Sokoban: a state with h=1 only yields 0.5,
making the final push indistinguishable from earlier moves.  We now
combine three cheap, informative factors:

1. **Box distance (h)** – the classic sum of Manhattan distances from each
   box to its nearest target (via the shared A* heuristic).

2. **Player walk cost (w)** – the shortest number of steps the player must
   walk to reach a push position.  A small weight (α) penalises actions that
   require long walks.

3. **Target completion fraction (f)** – proportion of boxes already on
   their targets, giving a bonus (γ) that grows as we near the solution.

The final reward is:

    r = 1 / (1 + h + α·w) + γ·f
    r = min(1.0, r)                # never exceed perfect score

Terminal states (solved / dead‑locked / step‑limit) keep the original
behaviour.

All calculations are O(|floor|) and use only the public GameState API,
so the function remains fast enough for thousands of rollouts.
"""

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


def _bfs_distances(start, walls, boxes, height, width):
    """
    Bread‑first search returning a dict mapping each reachable floor cell
    to its Manhattan step distance from *start* (player position).  Walls
    and boxes are treated as impassable.
    """
    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in walls or (nr, nc) in boxes:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
    return dist


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    h = _ag.h_sokoban_box_only(state)   # sum of distances box→nearest target

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    # Gather all reachable cells and their distances from the player.
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = _bfs_distances(state.player, walls, boxes, height, width)

    # Find the minimum distance among all legal push positions.
    min_walk = None
    for (push_pos, _dir) in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            # This shouldn't happen – the action would be illegal – but skip.
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    # If, for any reason, no legal action is reachable (should be terminal),
    # treat walk cost as 0 to avoid division by zero.
    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    if num_targets > 0:
        f = state.boxes_on_targets() / num_targets
    else:
        f = 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # how much a long walk penalises the reward
    GAMMA = 0.2   # bonus for having boxes already placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    # Special case: h == 0 (all boxes on targets) should be perfect.
    if h == 0:
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
- The reward for a state with `h = 1` (only one step from solution) caps around 0.7 because the target‑completion bonus (γ) is too small; the final push does not stand out clearly.
- Minor performance note: the BFS distance map is recomputed for every simulation even when there are no legal actions, but this cost is negligible compared with thousands of rollouts and keeps the function self‑contained.
- No bugs that would cause crashes; the API usage is correct.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: Increase the target‑completion bonus to widen the reward spread, ensuring near‑solved states receive a clearly higher reward.
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    h = _ag.h_sokoban_box_only(state)   # sum of distances box→nearest target

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = _bfs_distances(state.player, walls, boxes, height, width)

    min_walk = None
    for (push_pos, _dir) in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    f = state.boxes_on_targets() / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # penalty for long walks
    GAMMA = 0.4   # stronger bonus for having boxes placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    # Special case: all boxes on targets → perfect reward.
    if h == 0:
        reward = 1.0

    return reward
```
