# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:17:31 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 27.20s |
| Status     | success |
| Tokens     | 12763 |
| Validation | invalid — No function definition found in code. |

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

--- expansion ---
```python
"""
LLM-generated MCTS tool: expansion
Description: Fixed weak‑reference error by using a regular dictionary for the per‑root transposition cache and added a small guard to keep the cache from growing unbounded; also corrected the wall‑line dead‑lock checks that mistakenly built illegal `range` objects.
Generated:   2026-03-08T17:10:17.527736
"""

def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock and wall‑line deadlock detection.
      • Transposition key now includes player position.
      • Guarantees a push candidate when possible (push‑bias).
      • Uses matching‑distance and *player‑to‑box* distance as secondary
        improvement metrics.
    """
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (simple dict – MCTSNode is not weak‑ref‑able)
    # ----------------------------------------------------------------------
    if not hasattr(default_expansion, "_root_transposition_cache"):
        # mapping: root_node -> set of (state_key, player_pos)
        default_expansion._root_transposition_cache = {}
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of (state_key, player_pos) seen so far for the search root."""
        root = current_node
        while getattr(root, "parent", None) is not None:
            root = root.parent
        # Use the root object itself as a dict key (it is hashable). If the root
        # is ever garbage‑collected the cache entry will simply linger – it is
        # tiny and does not affect correctness.
        if root not in _cache:
            _cache[root] = set()
        return _cache[root]

    trans_set = get_root_transposition_set(node)

    # ----------------------------------------------------------------------
    # Helper: cheap corner dead‑lock test
    # ----------------------------------------------------------------------
    def _is_corner_deadlock(pos, walls, targets):
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

    # ----------------------------------------------------------------------
    # Helper: wall‑line dead‑lock test (same logic as simulation)
    # ----------------------------------------------------------------------
    def _is_wall_line_deadlock(box, walls, targets, height, width):
        """
        Simple wall‑line dead‑lock:
        If a box is directly adjacent to a wall in a direction where
        no target exists further along that line, the box can never reach a target.
        """
        r, c = box
        # left wall – look for any target in the same row left of the box
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                return True
        # right wall – look for any target in the same row right of the box
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                return True
        # up wall – look for any target in the same column above the box
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                return True
        # down wall – look for any target in the same column below the box
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r + 1, height)):
                return True
        return False

    # ----------------------------------------------------------------------
    # Helper: optimal matching distance (same as in simulation)
    # ----------------------------------------------------------------------
    def _matching_distance(state) -> int:
        """Brute‑force optimal total Manhattan distance after assigning boxes to targets."""
        boxes = list(state.boxes)
        targets = list(state.targets)
        n_boxes = len(boxes)
        n_targets = len(targets)

        best = math.inf
        if n_boxes == n_targets:
            for perm in itertools.permutations(targets, n_boxes):
                total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
        else:
            for targ_subset in itertools.combinations(targets, n_boxes):
                for perm in itertools.permutations(targ_subset):
                    total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                    if total < best:
                        best = total
        return best if best != math.inf else 0

    # ----------------------------------------------------------------------
    # Helper: distance from player to nearest box
    # ----------------------------------------------------------------------
    def _player_to_nearest_box(state):
        if not state.boxes:
            return 0
        pr, pc = state.player
        return min(abs(pr - br) + abs(pc - bc) for br, bc in state.boxes)

    # ----------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ----------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns a tuple:
          (valid, is_push, manh_imp, match_imp, player_imp, child_state)

        `valid` is False if the resulting state has a corner or wall‑line
        deadlock or is a transposition.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- dead‑lock pruning ----
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None

        # ---- transposition detection (per‑root) ----
        full_key = (temp_state.state_key(), temp_state.player)
        if full_key in trans_set:
            return False, False, 0, 0, 0, None

        # ---- push detection ----
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ---- distance improvements ----
        old_manh = node.state.total_box_distance()
        new_manh = temp_state.total_box_distance()
        manhattan_improve = old_manh - new_manh          # positive → closer

        old_match = _matching_distance(node.state)
        new_match = _matching_distance(temp_state)
        matching_improve = old_match - new_match          # positive → closer

        # ---- player‑to‑box distance improvement ----
        old_player_dist = _player_to_nearest_box(node.state)
        new_player_dist = _player_to_nearest_box(temp_state)
        player_improve = old_player_dist - new_player_dist   # positive → closer

        return True, is_push, manhattan_improve, matching_improve, player_improve, temp_state

    # ----------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ----------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates = []   # (action, is_push, manh_imp, match_imp, player_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, player_imp, child_state = evaluate_action(a)
        if not valid:
            continue
        candidates.append((a, is_push, manh_imp, match_imp, player_imp, child_state))

    # ----------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ----------------------------------------------------------------------
    if candidates:
        # Sort: pushes first, then larger Manhattan improvement,
        # then larger matching improvement, then larger player‑distance improvement
        candidates.sort(
            key=lambda tup: (
                not tup[1],          # pushes first (False < True)
                -tup[2],             # larger Manhattan improvement
                -tup[3],             # larger matching improvement
                -tup[4]              # larger player‑to‑box improvement
            )
        )
        chosen_action, _, _, _, _, chosen_state = candidates[0]

        # keep the remaining actions for future expansions
        remaining_actions = [a for a, *_ in candidates[1:]]
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining_actions)
        else:
            node._untried_actions = remaining_actions

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add((chosen_state.state_key(), chosen_state.player))
        return child

    # ----------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ----------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, player_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter (corner + wall‑line)
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            continue
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            continue

        # push detection
        if len(temp_state.boxes - node.state.boxes) == 0:
            continue

        manh_imp = node.state.total_box_distance() - temp_state.total_box_distance()
        match_imp = _matching_distance(node.state) - _matching_distance(temp_state)
        player_imp = _player_to_nearest_box(node.state) - _player_to_nearest_box(temp_state)

        push_candidates.append((a, manh_imp, match_imp, player_imp, temp_state))

    if push_candidates:
        push_candidates.sort(key=lambda tup: (-tup[1], -tup[2], -tup[3]))
        chosen_action, _, _, _, chosen_state = push_candidates[0]

        # Update untried actions: remove the chosen push, keep the rest
        remaining = [a for a, *_ in push_candidates[1:]]
        non_push_rest = [a for a in original_actions if a not in (remaining + [chosen_action])]
        final_remaining = remaining + non_push_rest

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(final_remaining)
        else:
            node._untried_actions = final_remaining

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add((chosen_state.state_key(), chosen_state.player))
        return child

    # ----------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ----------------------------------------------------------------------
    # Restore the original untried‑action collection (it may have been mutated)
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        # pop works for both list (pop last) and set (pop arbitrary)
        action = node._untried_actions.pop()
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add((child_state.state_key(), child_state.player))
    return child
```

--- simulation ◀ TARGET ---
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
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban (level6)
Timestamp:  2026-03-08T17:16:59.408040
Iterations: 200
Solved:     False
Steps:      200
Returns:    [0.0]

  Move 1: action=3, total_visits=200
    State: Step 0/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 2: action=2, total_visits=200
    State: Step 1/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 3: action=3, total_visits=200
    State: Step 2/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 4: action=2, total_visits=200
    State: Step 3/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 5: action=3, total_visits=200
    State: Step 4/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 6: action=2, total_visits=200
    State: Step 5/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 7: action=3, total_visits=200
    State: Step 6/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 8: action=2, total_visits=200
    State: Step 7/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 9: action=3, total_visits=200
    State: Step 8/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 10: action=2, total_visits=200
    State: Step 9/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 11: action=3, total_visits=200
    State: Step 10/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 12: action=2, total_visits=200
    State: Step 11/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 13: action=3, total_visits=200
    State: Step 12/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 14: action=2, total_visits=200
    State: Step 13/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 15: action=3, total_visits=200
    State: Step 14/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 16: action=2, total_visits=200
    State: Step 15/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 17: action=3, total_visits=200
    State: Step 16/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 18: action=2, total_visits=200
    State: Step 17/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 19: action=3, total_visits=200
    State: Step 18/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 20: action=2, total_visits=200
    State: Step 19/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 21: action=3, total_visits=200
    State: Step 20/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 22: action=2, total_visits=200
    State: Step 21/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 23: action=3, total_visits=200
    State: Step 22/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 24: action=2, total_visits=200
    State: Step 23/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 25: action=3, total_visits=200
    State: Step 24/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 26: action=2, total_visits=200
    State: Step 25/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 27: action=3, total_visits=200
    State: Step 26/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 28: action=2, total_visits=200
    State: Step 27/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  Move 29: action=3, total_visits=200
    State: Step 28/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [3(v=200, avg=0.000)]
  Move 30: action=2, total_visits=200
    State: Step 29/200 | Boxes on target: 0/2 | Total distance: 7
    Children: [2(v=100, avg=0.000), 0(v=100, avg=0.000)]
  ... (170 more moves omitted)

Final state:
Step 200/200 | Boxes on target: 0/2 | Total distance: 7
  ####   
###  ####
#     $ #
# #  #$ #
# . .#@ #
#########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level6
Current hyperparams: iterations=200, max_rollout_depth=500, exploration_weight=1.410
Baseline for level6 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 4 levels): 0.2500

Per-level best composites so far:
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level7: best=0.0000 (baseline=0.0000)
  level8: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level4', 'level5', 'level6', 'level7', 'level8']
Mastered levels: ['level4']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 2 [level6] [expansion]: composite=0.0000, solve_rate=0%, eval_time=13.7s, desc=Fix type handling of untried actions, remove unsafe attribute assignment, and add a safe fallback when no promising actions are found. ← accepted
  Iter 3 [level7] [expansion]: composite=0.0000, solve_rate=0%, desc=Preserve the incremental improvements while cleaning up minor redundancies. ✗ rejected
  Iter 4 [level8] [expansion]: composite=0.0000, solve_rate=0%, eval_time=401.9s, desc=Fixed weak‑reference error by using a regular dictionary for the per‑root transposition cache and added a small guard to keep the cache from growing unbounded; also corrected the wall‑line dead‑lock checks that mistakenly built illegal `range` objects. ← accepted

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'simulation'
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
      modifications to the 'simulation' function
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
