# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:18:56 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 45.61s |
| Status     | success |
| Tokens     | 10491 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban' (phase: simulation).

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
Game: Sokoban

== Overview ==
Sokoban is a single-player puzzle game. The player pushes boxes onto
target positions inside a grid-based warehouse. The puzzle is solved
when every box is on a target.

== Symbols ==
  #   wall (impassable)
  .   target position
  $   box
  *   box on a target
  @   player
  +   player standing on a target
  (space)  empty floor

== Rules ==
1. The player can move one step at a time: UP, DOWN, LEFT, or RIGHT.
2. The player can push a box by walking into it, but only if the cell
   on the far side of the box is empty floor or a target (not a wall
   and not another box).
3. The player cannot pull boxes — only push.
4. The puzzle is solved when ALL boxes are on target positions.
5. A game is lost (unsolvable) when a box is stuck in a position from
   which it can never reach any remaining target (deadlock). The
   simplest deadlock is a box pushed into a corner that is not a target.

== Actions ==
The action space is {0, 1, 2, 3} corresponding to:
  0 = UP    (row - 1)
  1 = DOWN  (row + 1)
  2 = LEFT  (col - 1)
  3 = RIGHT (col + 1)

== State Representation ==
Each state is described by:
  - The player's (row, col) position.
  - The set of (row, col) positions of all boxes.
  - A step counter (game terminates if max_steps is reached).
  - Derived metrics: boxes_on_targets count and total_box_distance
    (sum of Manhattan distance from each box to its nearest target).

== Reward ==
The state.returns() method returns a SHAPED continuous reward in [0, 1]:
  - Solved (all boxes on targets): 1.0
  - Deadlocked (box stuck in non-target corner): 0.0
  - Otherwise: a weighted combination of:
       70% — fraction of boxes on target  (boxes_on_targets / num_targets)
       30% — distance score  (1 − total_box_distance / max_dist)
    This gives a fine-grained gradient even when the puzzle is not yet solved.

== GameState API ==
Public attributes (not methods):
  walls     : frozenset[tuple[int,int]]   – wall positions
  targets   : frozenset[tuple[int,int]]   – target positions
  boxes     : set[tuple[int,int]]         – current box positions
  player    : tuple[int,int]              – current player position
  height, width : int                     – grid dimensions
  num_targets   : int                     – number of targets
  steps, max_steps : int                  – current / maximum step count

Public methods (call with parentheses):
  clone()            → new independent copy of the state
  legal_actions()    → list[int]  (subset of {0,1,2,3})
  apply_action(a)    → None (mutates the state in-place)
  is_terminal()      → bool
  returns()          → list[float]  (shaped reward, see above)
  current_player()   → int (always 0, single-player game)
  state_key()        → str (hashable key for transposition)
  boxes_on_targets() → int
  total_box_distance() → int  (sum of min Manhattan dist per box)

== Key Strategic Concepts ==
  - Avoid pushing boxes into corners (unless the corner IS a target).
  - Avoid pushing boxes against walls in directions where no target
    lies along that wall — the box becomes permanently stuck.
  - Plan pushes so that earlier pushes do not block corridors needed
    for later pushes.
  - Minimize total push count; fewer pushes usually means fewer
    opportunities for deadlock.
  - The order in which boxes are placed on targets matters:
    placing one box may block the path needed for another.


------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes required; the draft implementation is correct and efficient.
Generated:   2026-03-08T17:00:46.644718
"""

import random
import math
import itertools
from typing import Set, Tuple, List

# ----------------------------------------------------------------------
# Helper geometry / dead‑lock functions
# ----------------------------------------------------------------------


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """Detect a box in a 2‑wall corner that is not already on a target."""
    if pos in targets:
        return False
    r, c = pos
    combos = [((r - 1, c), (r, c - 1)),   # up & left
              ((r - 1, c), (r, c + 1)),   # up & right
              ((r + 1, c), (r, c - 1)),   # down & left
              ((r + 1, c), (r, c + 1))]   # down & right
    for a, b in combos:
        if a in walls and b in walls:
            return True
    return False


def _is_wall_line_deadlock(box: Tuple[int, int],
                           walls: Set[Tuple[int, int]],
                           targets: Set[Tuple[int, int]],
                           height: int,
                           width: int) -> bool:
    """
    Simple wall‑line dead‑lock:
    If a box is directly adjacent to a wall in a direction where
    *no* target exists further along that line, the box can never reach a target.
    """
    r, c = box
    # left wall
    if (r, c - 1) in walls:
        if not any((r, tc) in targets for tc in range(c - 1)):
            return True
    # right wall
    if (r, c + 1) in walls:
        if not any((r, tc) in targets for tc in range(c + 2, width)):
            return True
    # up wall
    if (r - 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r - 1)):
            return True
    # down wall
    if (r + 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r + 2, height)):
            return True
    return False


def _matching_distance(state) -> int:
    """
    Minimum total Manhattan distance after optimally assigning each box
    to a distinct target (Hungarian‑style via brute‑force, suitable for
    the small numbers of boxes typical in Sokoban).
    """
    boxes = list(state.boxes)
    targets = list(state.targets)

    n_boxes = len(boxes)
    n_targets = len(targets)

    best = math.inf
    if n_boxes == n_targets:
        for perm in itertools.permutations(targets, n_boxes):
            total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
            if total < best:
                best = total
    else:
        for targ_subset in itertools.combinations(targets, n_boxes):
            for perm in itertools.permutations(targ_subset):
                total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
    return best if best != math.inf else 0


def _heuristic(state,
               push_count: int = 0,
               max_steps: int = 1,
               push_lambda: float = 0.05) -> float:
    """
    Refined heuristic used during roll‑outs.
    • 70 % – fraction of boxes already on targets.
    • 30 % – normalized *matching* distance (more informative than
      independent nearest‑target distances).
    • Small penalty for the number of pushes already taken.
    """
    num_targets = state.num_targets
    if num_targets == 0:
        return 0.0

    # 1) fraction on target
    on_target = state.boxes_on_targets()
    frac = on_target / num_targets

    # 2) matching distance (total Manhattan after optimal assignment)
    matched_dist = _matching_distance(state)
    max_dist = (state.height + state.width) * num_targets
    dist_score = 1.0 - (matched_dist / max_dist) if max_dist else 0.0

    # 3) push penalty – encourages shorter solutions
    push_penalty = push_lambda * (push_count / max_steps)

    return max(0.0, 0.7 * frac + 0.3 * dist_score - push_penalty)


# ----------------------------------------------------------------------
# Main simulation function
# ----------------------------------------------------------------------


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided rollout for Sokoban with improved dead‑lock detection,
    a matching‑distance heuristic, a push‑count penalty, and an adaptive
    ε‑greedy schedule.

    Args:
        state:               GameState to roll out from (will be cloned).
        perspective_player:  Player index whose reward is returned.
        max_depth:           Upper bound on rollout length; 0 → automatic.
        epsilon:             Base exploration probability (scaled with depth).

    Returns:
        Float reward from the perspective of `perspective_player`.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    sim_state = state.clone()

    if max_depth <= 0:
        max_depth = max(30, state.num_targets * 20)

    # keep original step count to compute pushes taken during the rollout
    base_steps = state.steps
    max_steps = getattr(state, "max_steps", max_depth)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        best_h = -math.inf
        best_actions: List[Tuple[int, object]] = []
        safe_actions: List[Tuple[int, object]] = []

        # ----------------------------------------------------------------
        # Evaluate each legal action
        # ----------------------------------------------------------------
        for a in legal:
            child = sim_state.clone()
            child.apply_action(a)

            # ----- dead‑lock pruning ------------------------------------------------
            dead = False
            for bpos in child.boxes:
                if (_is_corner_deadlock(bpos, child.walls, child.targets) or
                        _is_wall_line_deadlock(bpos, child.walls, child.targets,
                                               child.height, child.width)):
                    dead = True
                    break

            if dead:
                continue   # discard this action

            safe_actions.append((a, child))

            # ----- heuristic evaluation -------------------------------------------
            pushes = child.steps - base_steps
            h = _heuristic(child,
                           push_count=pushes,
                           max_steps=max_steps,
                           push_lambda=0.05)

            if h > best_h:
                best_h = h
                best_actions = [(a, child)]
            elif h == best_h:
                best_actions.append((a, child))

        # ----------------------------------------------------------------
        # Choose action
        # ----------------------------------------------------------------
        if not safe_actions:
            # no safe move – fall back to random among all legal actions
            chosen_action, next_state = random.choice([
                (a, sim_state.clone()) for a in legal
            ])
            next_state.apply_action(chosen_action)
        else:
            # depth‑dependent ε
            depth_eps = max(epsilon * (1 - depth / max_depth), 0.05)

            if random.random() < depth_eps:
                # pure exploration among safe actions
                chosen_action, next_state = random.choice(safe_actions)
            else:
                # exploit best heuristic actions (break ties randomly)
                chosen_action, next_state = random.choice(best_actions)

        # advance simulation
        sim_state = next_state
        depth += 1

    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        # exact return for solved or dead‑locked states
        return sim_state.returns()[perspective_player]

    # otherwise use the refined heuristic for the final non‑terminal state
    pushes = sim_state.steps - base_steps
    return _heuristic(sim_state,
                      push_count=pushes,
                      max_steps=max_steps,
                      push_lambda=0.05)
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (from trace) | Evidence / Effect |
|------|----------------------|-------------------|
| 1️⃣ | **Roll‑outs never push a box** – the tree drifts forever left/right (actions 3/2) while the “total distance” stays at **7** for 200 steps. All child nodes have `avg=0.0`, indicating that every simulation ends with a dead‑locked/unsolved reward. |
| 2️⃣ | **Excessive dead‑lock pruning** – the simulation’s wall‑line dead‑lock test discards *all* actions that would push a box next to a wall, even when the push is perfectly safe. Consequently the only surviving actions are pure player moves, which cannot change the box configuration. |
| 3️⃣ | **Heuristic does not favour pushes** – `_heuristic` rewards *fraction on target* and *matching distance* only. A push that leaves the matching distance unchanged receives the same score as a pure move, while the `push_penalty` (λ = 0.05) actually **discourages** pushes early in the rollout. |
| 4️⃣ | **UCB‑driven selection amplifies the bias** – because every rollout returns the same low value, the tree quickly saturates the “move‑right/left” branch (200 visits each) and never explores a push branch. |

The combination of (2) and (3) explains why the search never discovers a useful push and ends in a dead‑end loop.

---

**2. ROOT CAUSE**  

*Dead‑lock test* (simulation version)  

```python
if (r, c - 1) in walls:
    if not any((r, tc) in targets for tc in range(c - 1)):   # ← wrong
        return True
...
if (r, c + 1) in walls:
    if not any((r, tc) in targets for tc in range(c + 2, width)):
        return True
```

`range(c‑1)` stops **before** the column immediately left of the box, and `range(c+2, width)` skips the column right next to the box. For a box adjacent to a wall, the function falsely reports a wall‑line dead‑lock unless a target lies **strictly farther**, eliminating virtually all pushes that touch a wall (the most common situation in Sokoban).  

*Heuristic bias*  

The push‑penalty term (`push_lambda * (push_count / max_steps)`) is applied from the first push onward, giving a *negative* bias to any rollout that pushes, while the reward component for pushes (distance improvement) is either zero or negligible because `matching_distance` often does not change after a single push. Hence the rollout prefers *no‑push* actions.

---

**3. PROPOSED APPROACH – Strategy A (incremental)**  

The problems are localized and can be fixed without redesigning the whole MCTS pipeline. The following targeted changes will eliminate the false dead‑locks and give pushes a positive signal, while preserving the already‑working matching‑distance heuristic and the adaptive ε‑greedy schedule.

1. **Fix the wall‑line dead‑lock check** – copy the corrected version from `default_expansion` (the one that uses `range(c)` for left, `range(c+1, width)` for right, and analogous vertical ranges). This will keep legitimate pushes that touch a wall.

2. **Make pushes attractive**:  
   * a. Compute a *push‑bonus* inside the rollout loop: after `child.apply_action(a)`, detect whether the action moved a box (`len(child.boxes - sim_state.boxes) > 0`). If true, add a small constant (e.g., `+0.05`) to the heuristic value `h`.  
   * b. Reduce the push‑penalty magnitude for early rollout steps (e.g., `push_lambda = 0.01`) or apply it only after a configurable depth (e.g., start penalising after 10 pushes). This removes the perverse early‑push discouragement.

3. **Prefer pushes when all heuristic scores tie** – after the loop that builds `best_actions`, if `best_actions` contains only non‑push moves, fall back to a “push‑first” tie‑breaker: collect all safe push actions (same dead‑lock filter) and, if any exist, select randomly among them. This guarantees that the tree will eventually explore a push branch.

4. **Clamp the matching‑distance normalization** – the current `max_dist = (height + width) * num_targets` can be overly generous, leaving `dist_score` close to 1 for many states and flattening the gradient. Replace it with a tighter bound such as `max_dist = max(height, width) * num_targets` or compute the *worst‑case* Manhattan distance between the farthest possible box‑target pair once per level and reuse it. This yields a more discriminative distance term, helping pushes that truly reduce distance to stand out.

5. **Safety‑check the ε‑schedule** – keep the depth‑scaled ε but enforce a minimum exploration of `0.1` when *any* push action is available, ensuring the rollout does not get stuck in a deterministic move‑only loop.

These five lightweight edits directly address the two dominant failure modes (false dead‑lock pruning and push‑averse heuristic) while leaving the rest of the MCTS machinery untouched. After the changes we expect roll‑outs to start exhibiting pushes, the tree to receive non‑zero Q‑values for push branches, and the solver to break out of the endless left/right oscillation observed in the trace.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
import random
import math
import itertools
from typing import Set, Tuple, List

# ----------------------------------------------------------------------
# Geometry / dead‑lock helpers
# ----------------------------------------------------------------------


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Manhattan distance between two grid cells."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """Detect a box in a 2‑wall corner that is not already on a target."""
    if pos in targets:
        return False
    r, c = pos
    combos = [((r - 1, c), (r, c - 1)),   # up & left
              ((r - 1, c), (r, c + 1)),   # up & right
              ((r + 1, c), (r, c - 1)),   # down & left
              ((r + 1, c), (r, c + 1))]   # down & right
    for a, b in combos:
        if a in walls and b in walls:
            return True
    return False


def _is_wall_line_deadlock(box: Tuple[int, int],
                           walls: Set[Tuple[int, int]],
                           targets: Set[Tuple[int, int]],
                           height: int,
                           width: int) -> bool:
    """
    Corrected wall‑line dead‑lock detection.
    If a box is adjacent to a wall and there is *no* target further
    along that line, the box can never reach a target.
    """
    r, c = box
    # left wall – any target left of the box (including column 0)
    if (r, c - 1) in walls:
        if not any((r, tc) in targets for tc in range(c)):
            return True
    # right wall – any target right of the box
    if (r, c + 1) in walls:
        if not any((r, tc) in targets for tc in range(c + 1, width)):
            return True
    # up wall – any target above the box
    if (r - 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r)):
            return True
    # down wall – any target below the box
    if (r + 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r + 1, height)):
            return True
    return False


# ----------------------------------------------------------------------
# Matching distance (optimal box‑to‑target assignment)
# ----------------------------------------------------------------------


def _matching_distance(state) -> int:
    """
    Minimum total Manhattan distance after optimally assigning each box
    to a distinct target (brute‑force, acceptable for typical Sokoban sizes).
    """
    boxes = list(state.boxes)
    targets = list(state.targets)

    n_boxes = len(boxes)
    n_targets = len(targets)

    best = math.inf
    if n_boxes == n_targets:
        for perm in itertools.permutations(targets, n_boxes):
            total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
            if total < best:
                best = total
    else:
        for targ_subset in itertools.combinations(targets, n_boxes):
            for perm in itertools.permutations(targ_subset):
                total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
    return best if best != math.inf else 0


# ----------------------------------------------------------------------
# Heuristic used during roll‑outs
# ----------------------------------------------------------------------


def _heuristic(state,
               push_count: int = 0,
               max_steps: int = 1,
               push_lambda: float = 0.01,
               push_bonus: float = 0.05) -> float:
    """
    Roll‑out heuristic.
    Components:
      • 70 % – fraction of boxes already on targets.
      • 30 % – normalised matching distance (tighter normalisation).
      • Small *bonus* for having performed a push in the current rollout.
      • Small (reduced) penalty for many pushes.
    """
    num_targets = state.num_targets
    if num_targets == 0:
        return 0.0

    # 1) fraction of boxes already placed
    on_target = state.boxes_on_targets()
    frac = on_target / num_targets

    # 2) matching distance – normalise with the *maximum possible* Manhattan
    #    distance for one box in this board: (height‑1)+(width‑1)
    max_one = (state.height - 1) + (state.width - 1)
    max_dist = max_one * num_targets
    matched_dist = _matching_distance(state)
    dist_score = 1.0 - (matched_dist / max_dist) if max_dist else 0.0

    # 3) push penalty (scaled down)
    push_penalty = push_lambda * (push_count / max_steps)

    # 4) push bonus – encourages roll‑outs that actually move a box
    bonus = push_bonus if push_count > 0 else 0.0

    return max(0.0, 0.7 * frac + 0.3 * dist_score + bonus - push_penalty)


# ----------------------------------------------------------------------
# Main simulation (roll‑out) function
# ----------------------------------------------------------------------


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided rollout for Sokoban.

    Fixed dead‑lock detection, added a push bonus, reduced push penalty,
    tightened distance normalisation and ensured pushes are chosen when
    heuristic scores tie.
    """
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    sim_state = state.clone()

    if max_depth <= 0:
        max_depth = max(30, state.num_targets * 20)

    base_steps = state.steps                     # to compute pushes taken in rollout
    max_steps = getattr(state, "max_steps", max_depth)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        best_h = -math.inf
        best_actions: List[Tuple[int, object, bool]] = []   # (action, child_state, is_push)
        safe_actions: List[Tuple[int, object, bool]] = []   # same structure

        # ----------------------------------------------------------------
        # Evaluate each legal action
        # ----------------------------------------------------------------
        for a in legal:
            child = sim_state.clone()
            child.apply_action(a)

            # ----- dead‑lock pruning ------------------------------------------------
            dead = False
            for bpos in child.boxes:
                if (_is_corner_deadlock(bpos, child.walls, child.targets) or
                        _is_wall_line_deadlock(bpos, child.walls, child.targets,
                                               child.height, child.width)):
                    dead = True
                    break
            if dead:
                continue   # discard this action

            # ----- push detection ----------------------------------------------------
            is_push = len(child.boxes - sim_state.boxes) > 0

            safe_actions.append((a, child, is_push))

            # ----- heuristic evaluation ---------------------------------------------
            pushes = child.steps - base_steps
            h = _heuristic(child,
                           push_count=pushes,
                           max_steps=max_steps,
                           push_lambda=0.01,
                           push_bonus=0.05)

            if h > best_h:
                best_h = h
                best_actions = [(a, child, is_push)]
            elif h == best_h:
                best_actions.append((a, child, is_push))

        # ----------------------------------------------------------------
        # Action selection
        # ----------------------------------------------------------------
        if not safe_actions:
            # no safe move – fallback to a random legal action
            chosen_action, next_state = random.choice([
                (a, sim_state.clone()) for a in legal
            ])
            next_state.apply_action(chosen_action)
        else:
            # Determine if any safe push actions exist
            any_push = any(is_push for _, _, is_push in safe_actions)

            # Depth‑dependent ε, but guarantee at least 0.1 exploration
            # when pushes are available (helps escape move‑only loops)
            depth_eps = max(epsilon * (1 - depth / max_depth), 0.1 if any_push else 0.05)

            if random.random() < depth_eps:
                # pure exploration among safe actions
                chosen_action, next_state, _ = random.choice(safe_actions)
            else:
                # exploitation: pick among best heuristic actions
                # If best actions contain no pushes but pushes exist, prefer a push
                push_in_best = any(is_push for _, _, is_push in best_actions)
                if not push_in_best and any_push:
                    # Gather all safe pushes and pick one at random
                    push_choices = [(a, s) for a, s, isp in safe_actions if isp]
                    chosen_action, next_state = random.choice(push_choices)
                else:
                    # normal best‑action choice (break ties randomly)
                    chosen_action, next_state, _ = random.choice(best_actions)

        # advance simulation
        sim_state = next_state
        depth += 1

    # ------------------------------------------------------------------
    # Return the reward for the final state
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        # exact reward for solved / dead‑locked states
        return sim_state.returns()[perspective_player]

    # non‑terminal: use the refined heuristic
    pushes = sim_state.steps - base_steps
    return _heuristic(sim_state,
                      push_count=pushes,
                      max_steps=max_steps,
                      push_lambda=0.01,
                      push_bonus=0.05)
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
- No issues found; the draft correctly fixes dead‑lock detection, adds a push bonus, reduces the push penalty, tightens distance normalisation and ensures pushes are considered during tie‑breaking.  
- The code respects the GameState API, avoids unnecessary work, and should produce diverse rollout rewards.

ACTION: modify
FILE_NAME: sokoban_simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: No functional changes required; code is already correct and efficient.
```python
import random
import math
import itertools
from typing import Set, Tuple, List

# ----------------------------------------------------------------------
# Geometry / dead‑lock helpers
# ----------------------------------------------------------------------


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Manhattan distance between two grid cells."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _is_corner_deadlock(pos: Tuple[int, int],
                        walls: Set[Tuple[int, int]],
                        targets: Set[Tuple[int, int]]) -> bool:
    """Detect a box in a 2‑wall corner that is not already on a target."""
    if pos in targets:
        return False
    r, c = pos
    combos = [((r - 1, c), (r, c - 1)),   # up & left
              ((r - 1, c), (r, c + 1)),   # up & right
              ((r + 1, c), (r, c - 1)),   # down & left
              ((r + 1, c), (r, c + 1))]   # down & right
    for a, b in combos:
        if a in walls and b in walls:
            return True
    return False


def _is_wall_line_deadlock(box: Tuple[int, int],
                           walls: Set[Tuple[int, int]],
                           targets: Set[Tuple[int, int]],
                           height: int,
                           width: int) -> bool:
    """
    Corrected wall‑line dead‑lock detection.
    If a box is adjacent to a wall and there is *no* target further
    along that line, the box can never reach a target.
    """
    r, c = box
    # left wall – any target left of the box (including column 0)
    if (r, c - 1) in walls:
        if not any((r, tc) in targets for tc in range(c)):
            return True
    # right wall – any target right of the box
    if (r, c + 1) in walls:
        if not any((r, tc) in targets for tc in range(c + 1, width)):
            return True
    # up wall – any target above the box
    if (r - 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r)):
            return True
    # down wall – any target below the box
    if (r + 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r + 1, height)):
            return True
    return False


# ----------------------------------------------------------------------
# Matching distance (optimal box‑to‑target assignment)
# ----------------------------------------------------------------------


def _matching_distance(state) -> int:
    """
    Minimum total Manhattan distance after optimally assigning each box
    to a distinct target (brute‑force, acceptable for typical Sokoban sizes).
    """
    boxes = list(state.boxes)
    targets = list(state.targets)

    n_boxes = len(boxes)
    n_targets = len(targets)

    best = math.inf
    if n_boxes == n_targets:
        for perm in itertools.permutations(targets, n_boxes):
            total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
            if total < best:
                best = total
    else:
        for targ_subset in itertools.combinations(targets, n_boxes):
            for perm in itertools.permutations(targ_subset):
                total = sum(_manhattan(b, t) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
    return best if best != math.inf else 0


# ----------------------------------------------------------------------
# Heuristic used during roll‑outs
# ----------------------------------------------------------------------


def _heuristic(state,
               push_count: int = 0,
               max_steps: int = 1,
               push_lambda: float = 0.01,
               push_bonus: float = 0.05) -> float:
    """
    Roll‑out heuristic.
    Components:
      • 70 % – fraction of boxes already on targets.
      • 30 % – normalised matching distance (tighter normalisation).
      • Small *bonus* for having performed a push in the current rollout.
      • Small (reduced) penalty for many pushes.
    """
    num_targets = state.num_targets
    if num_targets == 0:
        return 0.0

    # 1) fraction of boxes already placed
    on_target = state.boxes_on_targets()
    frac = on_target / num_targets

    # 2) matching distance – normalise with the *maximum possible* Manhattan
    #    distance for one box in this board: (height‑1)+(width‑1)
    max_one = (state.height - 1) + (state.width - 1)
    max_dist = max_one * num_targets
    matched_dist = _matching_distance(state)
    dist_score = 1.0 - (matched_dist / max_dist) if max_dist else 0.0

    # 3) push penalty (scaled down)
    push_penalty = push_lambda * (push_count / max_steps)

    # 4) push bonus – encourages roll‑outs that actually move a box
    bonus = push_bonus if push_count > 0 else 0.0

    return max(0.0, 0.7 * frac + 0.3 * dist_score + bonus - push_penalty)


# ----------------------------------------------------------------------
# Main simulation (roll‑out) function
# ----------------------------------------------------------------------


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided rollout for Sokoban.

    Fixed dead‑lock detection, added a push bonus, reduced push penalty,
    tightened distance normalisation and ensured pushes are chosen when
    heuristic scores tie.
    """
    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    sim_state = state.clone()

    if max_depth <= 0:
        max_depth = max(30, state.num_targets * 20)

    base_steps = state.steps                     # to compute pushes taken in rollout
    max_steps = getattr(state, "max_steps", max_depth)

    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        legal = sim_state.legal_actions()
        if not legal:
            break

        best_h = -math.inf
        best_actions: List[Tuple[int, object, bool]] = []   # (action, child_state, is_push)
        safe_actions: List[Tuple[int, object, bool]] = []   # same structure

        # ----------------------------------------------------------------
        # Evaluate each legal action
        # ----------------------------------------------------------------
        for a in legal:
            child = sim_state.clone()
            child.apply_action(a)

            # ----- dead‑lock pruning ------------------------------------------------
            dead = False
            for bpos in child.boxes:
                if (_is_corner_deadlock(bpos, child.walls, child.targets) or
                        _is_wall_line_deadlock(bpos, child.walls, child.targets,
                                               child.height, child.width)):
                    dead = True
                    break
            if dead:
                continue   # discard this action

            # ----- push detection ----------------------------------------------------
            is_push = len(child.boxes - sim_state.boxes) > 0

            safe_actions.append((a, child, is_push))

            # ----- heuristic evaluation ---------------------------------------------
            pushes = child.steps - base_steps
            h = _heuristic(child,
                           push_count=pushes,
                           max_steps=max_steps,
                           push_lambda=0.01,
                           push_bonus=0.05)

            if h > best_h:
                best_h = h
                best_actions = [(a, child, is_push)]
            elif h == best_h:
                best_actions.append((a, child, is_push))

        # ----------------------------------------------------------------
        # Action selection
        # ----------------------------------------------------------------
        if not safe_actions:
            # no safe move – fallback to a random legal action
            chosen_action, next_state = random.choice([
                (a, sim_state.clone()) for a in legal
            ])
            next_state.apply_action(chosen_action)
        else:
            # Determine if any safe push actions exist
            any_push = any(is_push for _, _, is_push in safe_actions)

            # Depth‑dependent ε, but guarantee at least 0.1 exploration
            # when pushes are available (helps escape move‑only loops)
            depth_eps = max(epsilon * (1 - depth / max_depth), 0.1 if any_push else 0.05)

            if random.random() < depth_eps:
                # pure exploration among safe actions
                chosen_action, next_state, _ = random.choice(safe_actions)
            else:
                # exploitation: pick among best heuristic actions
                # If best actions contain no pushes but pushes exist, prefer a push
                push_in_best = any(is_push for _, _, is_push in best_actions)
                if not push_in_best and any_push:
                    # Gather all safe pushes and pick one at random
                    push_choices = [(a, s) for a, s, isp in safe_actions if isp]
                    chosen_action, next_state = random.choice(push_choices)
                else:
                    # normal best‑action choice (break ties randomly)
                    chosen_action, next_state, _ = random.choice(best_actions)

        # advance simulation
        sim_state = next_state
        depth += 1

    # ------------------------------------------------------------------
    # Return the reward for the final state
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        # exact reward for solved / dead‑locked states
        return sim_state.returns()[perspective_player]

    # non‑terminal: use the refined heuristic
    pushes = sim_state.steps - base_steps
    return _heuristic(sim_state,
                      push_count=pushes,
                      max_steps=max_steps,
                      push_lambda=0.01,
                      push_bonus=0.05)
```
