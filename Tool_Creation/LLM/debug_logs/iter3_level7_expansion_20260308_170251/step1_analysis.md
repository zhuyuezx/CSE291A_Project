# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:03:15 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 23.97s |
| Status     | success |
| Tokens     | 11721 |
| Validation | invalid — No code block found in response. |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban' (phase: expansion).

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
MCTS TOOL FUNCTIONS (all 4 phases)
------------------------------------------------------------

--- selection ---
```python
"""
Default selection: UCB1 tree policy.

Walk down the tree choosing the child with the highest UCB1 score
until we reach a node that is either terminal or has untried actions
(then expand it).
"""

import math


def default_selection(node, exploration_weight: float = 1.41):
    """
    Pure UCB1 tree walk.

    Descends the tree choosing the child with the highest UCB1 score.
    Stops when reaching a node that has untried actions (needs expansion)
    or is terminal.

    Args:
        node:               Root MCTSNode to start selection from.
        exploration_weight:  UCB1 exploration constant C.

    Returns:
        An MCTSNode that is either terminal or has untried actions.
    """
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion phase
        # UCB1 selection among fully-expanded children
        log_parent = math.log(node.visits)
        best, best_score = None, -math.inf
        for child in node.children.values():
            exploit = child.value / child.visits
            explore = exploration_weight * math.sqrt(log_parent / child.visits)
            score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        node = best
    return node
```

--- expansion ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: expansion
Description: Fix type handling of untried actions, remove unsafe attribute assignment, and add a safe fallback when no promising actions are found.
Generated:   2026-03-08T17:02:19.165731
"""

"""
Improved expansion for Sokoban MCTS.

Key ideas:
  1. Prefer actions that push a box.
  2. Discard actions that immediately create a dead‑locked state.
  3. Avoid expanding duplicate board positions (transposition detection).
  4. Within the same priority level, favour actions that reduce the
     total Manhattan distance of boxes to targets.
"""

from typing import Set, Tuple, List

# ----------------------------------------------------------------------
# Local dead‑lock helpers (copied from the simulation module)
# ----------------------------------------------------------------------
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
    *no* target exists further along that line, the box can never
    reach a target.
    """
    r, c = box
    # left wall
    if (r, c - 1) in walls:
        if not any((r, tc) in targets for tc in range(c)):
            return True
    # right wall
    if (r, c + 1) in walls:
        if not any((r, tc) in targets for tc in range(c + 2, width)):
            return True
    # up wall
    if (r - 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r)):
            return True
    # down wall
    if (r + 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r + 2, height)):
            return True
    return False


# ----------------------------------------------------------------------
# Global transposition table for expansion phase
# ----------------------------------------------------------------------
_expansion_seen_keys: Set[str] = set()


def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Improvements over the naive version:
      • Push actions are tried before pure moves.
      • Actions that lead to obvious dead‑locks are discarded.
      • Duplicate board positions (transpositions) are not re‑expanded.
      • Within the same priority, actions that reduce the total
        Manhattan distance of boxes to targets are preferred.
      • Falls back to the original naïve expansion if no promising
        actions exist.
    """
    from mcts.node import MCTSNode  # local import to avoid circular deps

    # ------------------------------------------------------------------
    # Helper to evaluate a single action
    # ------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns (valid, is_push, dist_improve, child_state)
        where `valid` is False if the resulting state is a dead‑lock or
        a known duplicate.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ----- dead‑lock pruning ---------------------------------------
        for bpos in temp_state.boxes:
            if (_is_corner_deadlock(bpos, temp_state.walls, temp_state.targets) or
                _is_wall_line_deadlock(bpos, temp_state.walls, temp_state.targets,
                                       temp_state.height, temp_state.width)):
                return False, False, 0, None

        # ----- transposition detection ---------------------------------
        key = temp_state.state_key()
        if key in _expansion_seen_keys:
            return False, False, 0, None

        # ----- push detection ------------------------------------------
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ----- distance improvement ------------------------------------
        old_dist = node.state.total_box_distance()
        new_dist = temp_state.total_box_distance()
        dist_improve = old_dist - new_dist  # >0 means we got closer

        return True, is_push, dist_improve, temp_state

    # ------------------------------------------------------------------
    # Gather candidates from the untried‑action pool
    # ------------------------------------------------------------------
    original_actions = list(node._untried_actions)  # make a copy
    candidates: List[Tuple[int, bool, int, object]] = []  # (action, is_push, improve, state)

    for a in original_actions:
        valid, is_push, improve, child_state = evaluate_action(a)
        if not valid:
            # discard permanently (dead‑lock or duplicate)
            continue
        candidates.append((a, is_push, improve, child_state))

    # ------------------------------------------------------------------
    # If we have at least one promising candidate, choose the best one.
    # Otherwise fall back to the original naïve expansion.
    # ------------------------------------------------------------------
    if candidates:
        # Sort: pushes first (is_push=True), then larger distance improvement
        candidates.sort(key=lambda tup: (not tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = candidates[0]

        # Re‑populate the node's untried actions with the leftovers
        remaining = [a for a, *_ in candidates[1:]]

        # Preserve actions that were discarded because they were dead‑locks
        # or duplicates – they should not be tried again.
        # Hence we only keep `remaining`.
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining)
        else:
            node._untried_actions = remaining

        # ------------------------------------------------------------------
        # Create the child node
        # ------------------------------------------------------------------
        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child

        # Register the new board position globally to avoid future duplicates
        _expansion_seen_keys.add(chosen_state.state_key())

        return child

    # ------------------------------------------------------------------
    # Fallback: no promising actions; use the original behaviour.
    # ------------------------------------------------------------------
    # Restore the original action pool (so future expansions can still try)
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    # Pop an arbitrary action (works for list or set)
    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    return child
```

--- simulation ---
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

--- backpropagation ---
```python
"""
Default backpropagation: walk up the tree updating visits and value.

Value convention: each node's value is stored from the perspective of
the player who CHOSE the action leading to that node (= node.parent's
current_player). This lets UCB1 always maximize, which is correct for
both the searching player and the opponent.

For single-player games (e.g. Sokoban), all nodes share the same
perspective so the sign never flips.
"""


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a simulation result from leaf to root.

    Args:
        node:   The leaf MCTSNode where simulation started.
        reward: The simulation reward from the ROOT player's perspective.
    """
    # Find root's current player (= perspective of the reward)
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # Walk back up, flipping sign at opponent nodes
    while node is not None:
        node.visits += 1
        # Who chose the move that created this node?
        mover = node.parent.state.current_player() if node.parent else perspective
        node.value += reward if mover == perspective else -reward
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (expansion)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: expansion
Description: Fix type handling of untried actions, remove unsafe attribute assignment, and add a safe fallback when no promising actions are found.
Generated:   2026-03-08T17:02:19.165731
"""

"""
Improved expansion for Sokoban MCTS.

Key ideas:
  1. Prefer actions that push a box.
  2. Discard actions that immediately create a dead‑locked state.
  3. Avoid expanding duplicate board positions (transposition detection).
  4. Within the same priority level, favour actions that reduce the
     total Manhattan distance of boxes to targets.
"""

from typing import Set, Tuple, List

# ----------------------------------------------------------------------
# Local dead‑lock helpers (copied from the simulation module)
# ----------------------------------------------------------------------
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
    *no* target exists further along that line, the box can never
    reach a target.
    """
    r, c = box
    # left wall
    if (r, c - 1) in walls:
        if not any((r, tc) in targets for tc in range(c)):
            return True
    # right wall
    if (r, c + 1) in walls:
        if not any((r, tc) in targets for tc in range(c + 2, width)):
            return True
    # up wall
    if (r - 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r)):
            return True
    # down wall
    if (r + 1, c) in walls:
        if not any((tr, c) in targets for tr in range(r + 2, height)):
            return True
    return False


# ----------------------------------------------------------------------
# Global transposition table for expansion phase
# ----------------------------------------------------------------------
_expansion_seen_keys: Set[str] = set()


def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Improvements over the naive version:
      • Push actions are tried before pure moves.
      • Actions that lead to obvious dead‑locks are discarded.
      • Duplicate board positions (transpositions) are not re‑expanded.
      • Within the same priority, actions that reduce the total
        Manhattan distance of boxes to targets are preferred.
      • Falls back to the original naïve expansion if no promising
        actions exist.
    """
    from mcts.node import MCTSNode  # local import to avoid circular deps

    # ------------------------------------------------------------------
    # Helper to evaluate a single action
    # ------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns (valid, is_push, dist_improve, child_state)
        where `valid` is False if the resulting state is a dead‑lock or
        a known duplicate.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ----- dead‑lock pruning ---------------------------------------
        for bpos in temp_state.boxes:
            if (_is_corner_deadlock(bpos, temp_state.walls, temp_state.targets) or
                _is_wall_line_deadlock(bpos, temp_state.walls, temp_state.targets,
                                       temp_state.height, temp_state.width)):
                return False, False, 0, None

        # ----- transposition detection ---------------------------------
        key = temp_state.state_key()
        if key in _expansion_seen_keys:
            return False, False, 0, None

        # ----- push detection ------------------------------------------
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ----- distance improvement ------------------------------------
        old_dist = node.state.total_box_distance()
        new_dist = temp_state.total_box_distance()
        dist_improve = old_dist - new_dist  # >0 means we got closer

        return True, is_push, dist_improve, temp_state

    # ------------------------------------------------------------------
    # Gather candidates from the untried‑action pool
    # ------------------------------------------------------------------
    original_actions = list(node._untried_actions)  # make a copy
    candidates: List[Tuple[int, bool, int, object]] = []  # (action, is_push, improve, state)

    for a in original_actions:
        valid, is_push, improve, child_state = evaluate_action(a)
        if not valid:
            # discard permanently (dead‑lock or duplicate)
            continue
        candidates.append((a, is_push, improve, child_state))

    # ------------------------------------------------------------------
    # If we have at least one promising candidate, choose the best one.
    # Otherwise fall back to the original naïve expansion.
    # ------------------------------------------------------------------
    if candidates:
        # Sort: pushes first (is_push=True), then larger distance improvement
        candidates.sort(key=lambda tup: (not tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = candidates[0]

        # Re‑populate the node's untried actions with the leftovers
        remaining = [a for a, *_ in candidates[1:]]

        # Preserve actions that were discarded because they were dead‑locks
        # or duplicates – they should not be tried again.
        # Hence we only keep `remaining`.
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining)
        else:
            node._untried_actions = remaining

        # ------------------------------------------------------------------
        # Create the child node
        # ------------------------------------------------------------------
        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child

        # Register the new board position globally to avoid future duplicates
        _expansion_seen_keys.add(chosen_state.state_key())

        return child

    # ------------------------------------------------------------------
    # Fallback: no promising actions; use the original behaviour.
    # ------------------------------------------------------------------
    # Restore the original action pool (so future expansions can still try)
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    # Pop an arbitrary action (works for list or set)
    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    return child
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban (level7)
Timestamp:  2026-03-08T17:02:40.797891
Iterations: 200
Solved:     False
Steps:      200
Returns:    [0.0]

  Move 1: action=3, total_visits=200
    State: Step 0/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=100, avg=0.000), 2(v=100, avg=0.000)]
  Move 2: action=3, total_visits=200
    State: Step 1/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 3: action=2, total_visits=200
    State: Step 2/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 4: action=3, total_visits=200
    State: Step 3/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 5: action=2, total_visits=200
    State: Step 4/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 6: action=3, total_visits=200
    State: Step 5/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 7: action=2, total_visits=200
    State: Step 6/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 8: action=3, total_visits=200
    State: Step 7/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 9: action=2, total_visits=200
    State: Step 8/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 10: action=3, total_visits=200
    State: Step 9/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 11: action=2, total_visits=200
    State: Step 10/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 12: action=3, total_visits=200
    State: Step 11/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 13: action=2, total_visits=200
    State: Step 12/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 14: action=3, total_visits=200
    State: Step 13/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 15: action=2, total_visits=200
    State: Step 14/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 16: action=3, total_visits=200
    State: Step 15/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 17: action=2, total_visits=200
    State: Step 16/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 18: action=3, total_visits=200
    State: Step 17/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 19: action=2, total_visits=200
    State: Step 18/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 20: action=3, total_visits=200
    State: Step 19/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 21: action=2, total_visits=200
    State: Step 20/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 22: action=3, total_visits=200
    State: Step 21/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 23: action=2, total_visits=200
    State: Step 22/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 24: action=3, total_visits=200
    State: Step 23/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 25: action=2, total_visits=200
    State: Step 24/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 26: action=3, total_visits=200
    State: Step 25/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 27: action=2, total_visits=200
    State: Step 26/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 28: action=3, total_visits=200
    State: Step 27/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  Move 29: action=2, total_visits=200
    State: Step 28/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 30: action=3, total_visits=200
    State: Step 29/200 | Boxes on target: 0/3 | Total distance: 5
    Children: [3(v=67, avg=0.000), 2(v=67, avg=0.000), 0(v=66, avg=0.000)]
  ... (170 more moves omitted)

Final state:
Step 200/200 | Boxes on target: 0/3 | Total distance: 5
 ########
 # . . .#
 # $$#$ #
 #     @#
 ########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=200, max_rollout_depth=500, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 3 levels): 0.3333

Per-level best composites so far:
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level7: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level4', 'level5', 'level6', 'level7', 'level8']
Mastered levels: ['level4']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level4] [simulation]: composite=1.0000, solve_rate=100%, eval_time=2.8s, desc=No changes required; the draft implementation is correct and efficient. ← accepted
  Iter 2 [level6] [expansion]: composite=0.0000, solve_rate=0%, eval_time=13.7s, desc=Fix type handling of untried actions, remove unsafe attribute assignment, and add a safe fallback when no promising actions are found. ← accepted

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'expansion'
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
      modifications to the 'expansion' function
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

| Rank | Symptom (trace) | Evidence |
|------|-----------------|----------|
| 1️⃣ | **Never pushes a box** – total‑box‑distance stays 5 for all 200 moves. The move list is a strict 3‑2 alternating pattern; no action ever changes the box set. |
| 2️⃣ | **Expansion repeatedly re‑expands the same two player positions** (left/right swing). Visits for each child stay at ~100/200, showing the tree is stuck in a 2‑node loop. |
| 3️⃣ | **Zero value / zero reward** – every leaf returns 0.0, confirming that roll‑outs never reach a state with any box on a target. |
| 4️⃣ | **Heavy pruning** – the candidate list is almost always empty, forcing the fallback “pop an arbitrary action”. This happens because most actions are discarded as “dead‑locks” or “duplicates”. |

**2. ROOT CAUSE**  

* **Over‑aggressive dead‑lock pruning** – `default_expansion` calls `_is_wall_line_deadlock` for **every** resulting state. The wall‑line test only looks at *any* target along the line; for many level‑7 boxes there is no target directly left/right/up/down, so the function returns `True` even when the push is perfectly legal (the box can later be manoeuvred around the wall). Consequently *all* push actions are flagged invalid and removed from the candidate pool.  

* **Global transposition table** (`_expansion_seen_keys`) treats the whole search as a single hash set. After the first left‑right swing both states are stored; subsequent expansions see them as duplicates and discard them. The fallback then draws from the *remaining* untried actions, which are just the non‑push moves that survived the dead‑lock filter.  

* **Push‑detection is not decisive** – the sorting key prefers pushes, but if *no* push survives the dead‑lock/duplicate filters, the algorithm has nothing to choose, so it defaults to a random move. This explains the endless 3‑2 cycle.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvements** (recommended). The current framework (action‑level pruning, candidate sorting, transposition detection) is sound; we only need to make it less restrictive and smarter.

1. **Relax wall‑line dead‑lock pruning in expansion**  
   * Replace the call to `_is_wall_line_deadlock` with a *lighter* check that only rejects pushes that create an *immediate* corner dead‑lock (the existing `_is_corner_deadlock`).  
   * Keep the wall‑line test for roll‑outs (simulation) where a deeper look‑ahead is acceptable, but omit it from expansion.

2. **Scope the transposition table**  
   * Change `_expansion_seen_keys` from a global set to a *per‑root* dictionary (e.g., stored on the root node or passed through the search).  
   * Reset it when a new root is selected (each new MCTS iteration). This prevents the algorithm from permanently banning positions that are reachable via different push histories.

3. **Guarantee at least one push candidate**  
   * After building `candidates`, if the list is empty **but** there exists at least one *legal push* that does **not** create a corner dead‑lock, force‑include the best of those (e.g., the one with smallest distance increase).  
   * This ensures the expansion never falls back to pure walking moves when a viable push exists.

4. **Distance‑improvement fallback**  
   * When `dist_improve` is zero for all pushes (common when a push moves a box away from its nearest target), prefer pushes that **reduce the matching distance** (the Hungarian‑style distance used in simulation) instead of raw Manhattan sum. This adds a more informative metric without heavy computation.

These targeted tweaks eliminate the main blockage (pushes being filtered out) while keeping the rest of the heuristic—push‑first ordering, distance‑based tie‑breaking, and dead‑lock safety—intact. Once pushes are allowed, the tree will explore deeper configurations, the value estimates will start to rise above zero, and the agent will be able to make progress toward solving level 7.
