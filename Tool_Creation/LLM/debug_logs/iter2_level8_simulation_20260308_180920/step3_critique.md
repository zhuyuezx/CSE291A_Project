# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 18:11:31 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 54.99s |
| Status     | success |
| Tokens     | 11346 |
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
Description: No functional changes required; code is already correct and efficient.
Generated:   2026-03-08T17:18:56.250366
"""

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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (from trace) | Why it matters |
|------|----------------------|----------------|
| 1 | The agent simply **oscillates** between actions 0 (UP) and 1 (DOWN) for the whole 200‑step horizon, never executing a horizontal move or a push. | No progress ⇒ 0 boxes on targets, matching‑distance stays at 6, return stays 0. |
| 2 | The **roll‑out heuristic** gives the same (or higher) score to pure walking actions as to pushes. The best‑action list is often filled with non‑push moves, and the tie‑breaker selects a non‑push at random. | The search never discovers a state where a box is moved closer to a goal, so the tree never expands useful branches. |
| 3 | The **push‑bonus/penalty** is tiny ( +0.05  vs ‑0.01·push_ratio) and the distance term is heavily diluted by the normalisation (max_one·num_targets). With 0 targets on‑board the 70 % “fraction‑on‑target” term is always 0, so the heuristic relies on a very small 0.3 · dist_score change that rarely exceeds the random noise. | The algorithm is blind to the fact that a push is the only way to improve the solution. |
| 4 | The **epsilon schedule** guarantees at least 0.1 exploration only when *any* push exists in the *safe‑action* set. Because the safe‑action set is filtered by the dead‑lock test, many legal pushes that temporarily increase distance are discarded, so *any_push* is often **False**, freezing ε at 0.05. | The roll‑out stays in a low‑exploration regime, reinforcing the vertical loop. |

**2. ROOT CAUSE**  

The simulation function evaluates a child state with  

```
score = 0.7 * (boxes_on_targets/num_targets) + 
        0.3 * (1 – matching_dist / max_dist) +
        push_bonus – push_penalty
```

* When no boxes are on targets the first term is zero.  
* `max_dist` = (height‑1 + width‑1) × num_targets, which for level 8 is large (≈ 30 × 4 = 120). A one‑step improvement in matching distance therefore changes the second term by at most 0.3 · 1/120 ≈ 0.0025.  
* The push bonus (0.05) is only added **once per rollout**, regardless of how many pushes are made, and the penalty is minuscule. Consequently a move that simply walks the player (no push) often yields an equal or higher score than a push that temporarily moves a box away from its nearest target.  

Because the heuristic does not *explicitly* reward pushes that **reduce the Manhattan sum** (or at least do not increase it) the best‑action list is dominated by safe walks. The tie‑breaker prefers non‑push actions, leading to the observed up/down loop.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement** (recommended). The current framework (UCB1 selection, expansion with dead‑lock checks, back‑propagation) works; we only need a richer roll‑out evaluation that makes pushes positively attractive and penalises moves that worsen box positions.

Targeted modifications to `default_simulation` (no full rewrite):

1. **Add a “push‑value” term** proportional to the improvement in the *raw* Manhattan sum of boxes (`total_box_distance`).  
   ```python
   manh_improve = parent_state.total_box_distance() - child_state.total_box_distance()
   push_value   = push_lambda2 * max(manh_improve, 0)
   ```
   Use a larger coefficient (e.g., 0.15) so a single step that brings a box closer yields ~0.15 score, outweighing the tiny distance‑normalisation effect.

2. **Replace the static push‑bonus** with a *progressive* bonus that scales with the number of pushes performed in the rollout, e.g. `push_bonus * (pushes / max_steps)`. This encourages roll‑outs that actually move boxes rather than idle walking.

3. **Introduce a small penalty for moves that *increase* the Manhattan sum**, to actively discourage pushes that backtrack.  
   ```python
   push_penalty = push_lambda * (pushes / max_steps) + push_lambda3 * max(-manh_improve, 0)
   ```

4. **Re‑weight the distance component**: give the *raw* Manhattan improvement a higher share (e.g., 0.4) and lower the matching‑distance share (e.g., 0.1). The matching distance is still useful for global guidance, but the cheaper Manhattan sum reacts quickly to local pushes.

5. **Adjust ε** so that when *any* push exists ε is at least **0.2** (instead of 0.1). This gives more exploration of pushes early in the rollout, helping the tree discover useful push sequences.

These changes keep the overall structure (dead‑lock pruning, safe‑action set, tie‑breaker logic) intact, requiring only extra cheap calculations already available on the state object. By making pushes visibly beneficial in the roll‑out score, the simulation will select pushes more often, breaking the vertical oscillation and allowing the MCTS tree to expand toward real solutions.

*Why incremental suffices*: The core issue is an under‑incentivised push signal; the existing dead‑lock detection, expansion, and back‑propagation are already sound. Enhancing the roll‑out heuristic directly resolves the dominant weakness without needing a complete redesign.

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
               manh_improve: int = 0,
               push_lambda: float = 0.01,
               push_lambda_improve: float = 0.15,
               push_lambda_worsen: float = 0.08,
               push_bonus_base: float = 0.05,
               dist_weight: float = 0.3,
               target_weight: float = 0.7) -> float:
    """
    Roll‑out heuristic with explicit push incentives.

    *target_weight* – fraction of boxes already on targets.
    *dist_weight*   – weight for the normalised matching distance.
    *push_lambda_improve* – reward per unit Manhattan improvement.
    *push_lambda_worsen* – penalty per unit Manhattan worsening.
    *push_bonus_base*    – scaled bonus proportional to pushes taken.
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

    # 3) push improvement / worsening terms
    improve_term = push_lambda_improve * max(manh_improve, 0)
    worsen_term = push_lambda_worsen * max(-manh_improve, 0)

    # 4) push penalty (scaled by rollout length)
    push_penalty = push_lambda * (push_count / max_steps)

    # 5) scaled push bonus – encourages roll‑outs that actually push
    push_bonus = push_bonus_base * (push_count / max_steps)

    score = (
        target_weight * frac +
        dist_weight * dist_score +
        improve_term -
        worsen_term -
        push_penalty +
        push_bonus
    )
    return max(0.0, score)


# ----------------------------------------------------------------------
# Main simulation (roll‑out) function
# ----------------------------------------------------------------------


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided rollout for Sokoban with stronger push incentives.

    * Pushes that decrease the raw Manhattan sum receive a sizable bonus.
    * Pushes that increase the sum are penalised.
    * Push‑bonus now scales with the number of pushes performed.
    * ε is forced to ≥ 0.2 whenever any push is available, giving more
      exploration of push actions early in the rollout.
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
        best_actions: List[Tuple[int, object, bool, int]] = []   # (action, child_state, is_push, manh_improve)
        safe_actions: List[Tuple[int, object, bool, int]] = []   # same structure

        # ----------------------------------------------------------------
        # Evaluate each legal action
        # ----------------------------------------------------------------
        parent_manh = sim_state.total_box_distance()
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

            # Manhattan improvement relative to parent state
            child_manh = child.total_box_distance()
            manh_improve = parent_manh - child_manh

            safe_actions.append((a, child, is_push, manh_improve))

            # ----- heuristic evaluation ---------------------------------------------
            pushes = child.steps - base_steps
            h = _heuristic(child,
                           push_count=pushes,
                           max_steps=max_steps,
                           manh_improve=manh_improve,
                           push_lambda=0.01,
                           push_lambda_improve=0.15,
                           push_lambda_worsen=0.08,
                           push_bonus_base=0.05,
                           dist_weight=0.3,
                           target_weight=0.7)

            if h > best_h:
                best_h = h
                best_actions = [(a, child, is_push, manh_improve)]
            elif h == best_h:
                best_actions.append((a, child, is_push, manh_improve))

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
            any_push = any(is_push for _, _, is_push, _ in safe_actions)

            # Depth‑dependent ε, but guarantee at least 0.2 exploration
            # when pushes are present (helps escape movement‑only loops)
            depth_eps = max(epsilon * (1 - depth / max_depth), 0.2 if any_push else 0.05)

            if random.random() < depth_eps:
                # pure exploration among safe actions
                chosen_action, next_state, _, _ = random.choice(safe_actions)
            else:
                # exploitation: pick among best heuristic actions
                push_in_best = any(is_push for _, _, is_push, _ in best_actions)
                if not push_in_best and any_push:
                    # No pushes among top‑scoring actions → force a push
                    push_choices = [(a, s) for a, s, isp, _ in safe_actions if isp]
                    chosen_action, next_state = random.choice(push_choices)
                else:
                    # Normal best‑action choice (break ties randomly)
                    chosen_action, next_state, _, _ = random.choice(best_actions)

        # advance simulation
        sim_state = next_state
        depth += 1

    # ------------------------------------------------------------------
    # Return the reward for the final state
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        # exact reward for solved / dead‑locked states
        return sim_state.returns()[perspective_player]

    # non‑terminal: use the refined heuristic (no manh improvement at leaf)
    pushes = sim_state.steps - base_steps
    return _heuristic(sim_state,
                      push_count=pushes,
                      max_steps=max_steps,
                      manh_improve=0,
                      push_lambda=0.01,
                      push_lambda_improve=0.15,
                      push_lambda_worsen=0.08,
                      push_bonus_base=0.05,
                      dist_weight=0.3,
                      target_weight=0.7)
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
- No functional bugs detected; the code respects the GameState API and uses correct variable names.
- Performance is acceptable for typical Sokoban sizes; although `_matching_distance` is costly, it was already present in the original version and is necessary for the heuristic.
- The heuristic now provides a varied reward signal that reacts to pushes and Manhattan distance changes, improving reward spread.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: No changes needed; preserve the incremental improvements already applied.
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
               manh_improve: int = 0,
               push_lambda: float = 0.01,
               push_lambda_improve: float = 0.15,
               push_lambda_worsen: float = 0.08,
               push_bonus_base: float = 0.05,
               dist_weight: float = 0.3,
               target_weight: float = 0.7) -> float:
    """
    Roll‑out heuristic with explicit push incentives.

    *target_weight* – fraction of boxes already on targets.
    *dist_weight*   – weight for the normalised matching distance.
    *push_lambda_improve* – reward per unit Manhattan improvement.
    *push_lambda_worsen* – penalty per unit Manhattan worsening.
    *push_bonus_base*    – scaled bonus proportional to pushes taken.
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

    # 3) push improvement / worsening terms
    improve_term = push_lambda_improve * max(manh_improve, 0)
    worsen_term = push_lambda_worsen * max(-manh_improve, 0)

    # 4) push penalty (scaled by rollout length)
    push_penalty = push_lambda * (push_count / max_steps)

    # 5) scaled push bonus – encourages roll‑outs that actually push
    push_bonus = push_bonus_base * (push_count / max_steps)

    score = (
        target_weight * frac +
        dist_weight * dist_score +
        improve_term -
        worsen_term -
        push_penalty +
        push_bonus
    )
    return max(0.0, score)


# ----------------------------------------------------------------------
# Main simulation (roll‑out) function
# ----------------------------------------------------------------------


def default_simulation(state,
                       perspective_player: int,
                       max_depth: int = 0,
                       epsilon: float = 0.2) -> float:
    """
    Heuristic‑guided rollout for Sokoban with stronger push incentives.

    * Pushes that decrease the raw Manhattan sum receive a sizable bonus.
    * Pushes that increase the sum are penalised.
    * Push‑bonus now scales with the number of pushes performed.
    * ε is forced to ≥ 0.2 whenever any push is available, giving more
      exploration of push actions early in the rollout.
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
        best_actions: List[Tuple[int, object, bool, int]] = []   # (action, child_state, is_push, manh_improve)
        safe_actions: List[Tuple[int, object, bool, int]] = []   # same structure

        # ----------------------------------------------------------------
        # Evaluate each legal action
        # ----------------------------------------------------------------
        parent_manh = sim_state.total_box_distance()
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

            # Manhattan improvement relative to parent state
            child_manh = child.total_box_distance()
            manh_improve = parent_manh - child_manh

            safe_actions.append((a, child, is_push, manh_improve))

            # ----- heuristic evaluation ---------------------------------------------
            pushes = child.steps - base_steps
            h = _heuristic(child,
                           push_count=pushes,
                           max_steps=max_steps,
                           manh_improve=manh_improve,
                           push_lambda=0.01,
                           push_lambda_improve=0.15,
                           push_lambda_worsen=0.08,
                           push_bonus_base=0.05,
                           dist_weight=0.3,
                           target_weight=0.7)

            if h > best_h:
                best_h = h
                best_actions = [(a, child, is_push, manh_improve)]
            elif h == best_h:
                best_actions.append((a, child, is_push, manh_improve))

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
            any_push = any(is_push for _, _, is_push, _ in safe_actions)

            # Depth‑dependent ε, but guarantee at least 0.2 exploration
            # when pushes are present (helps escape movement‑only loops)
            depth_eps = max(epsilon * (1 - depth / max_depth), 0.2 if any_push else 0.05)

            if random.random() < depth_eps:
                # pure exploration among safe actions
                chosen_action, next_state, _, _ = random.choice(safe_actions)
            else:
                # exploitation: pick among best heuristic actions
                push_in_best = any(is_push for _, _, is_push, _ in best_actions)
                if not push_in_best and any_push:
                    # No pushes among top‑scoring actions → force a push
                    push_choices = [(a, s) for a, s, isp, _ in safe_actions if isp]
                    chosen_action, next_state = random.choice(push_choices)
                else:
                    # Normal best‑action choice (break ties randomly)
                    chosen_action, next_state, _, _ = random.choice(best_actions)

        # advance simulation
        sim_state = next_state
        depth += 1

    # ------------------------------------------------------------------
    # Return the reward for the final state
    # ------------------------------------------------------------------
    if sim_state.is_terminal():
        # exact reward for solved / dead‑locked states
        return sim_state.returns()[perspective_player]

    # non‑terminal: use the refined heuristic (no manh improvement at leaf)
    pushes = sim_state.steps - base_steps
    return _heuristic(sim_state,
                      push_count=pushes,
                      max_steps=max_steps,
                      manh_improve=0,
                      push_lambda=0.01,
                      push_lambda_improve=0.15,
                      push_lambda_worsen=0.08,
                      push_bonus_base=0.05,
                      dist_weight=0.3,
                      target_weight=0.7)
```
