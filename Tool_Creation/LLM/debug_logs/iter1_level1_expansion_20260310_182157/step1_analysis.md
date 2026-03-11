# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-10 18:22:13 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 15.40s |
| Status     | success |
| Tokens     | 14554 |
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
Description: Clean up redundant imports, minor cache‑eviction tweak, and remove unnecessary inner import.
Generated:   2026-03-08T18:02:48.551302
"""

def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements over the previous version:
      • Reject pushes that do not improve any distance metric.
      • Additional cheap dead‑lock patterns:
          – 2×2 box block without a target.
          – Straight‑corridor dead‑lock (box between two walls on a line).
          – Simple frozen box (box against a wall with no target in that line
            *and* blocked orthogonally).
      • Verify that after the action the player can still reach at
        least one box (BFS on floor cells, boxes act as obstacles).
      • Transposition key now includes the current step count so that
        identical layouts reached at different depths are distinguished.
      • Per‑root transposition cache size is limited (≈2000 entries) to
        keep memory usage bounded.
    """
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math
    from collections import deque

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (simple dict – MCTSNode is not weak‑ref‑able)
    # ----------------------------------------------------------------------
    if not hasattr(default_expansion, "_root_transposition_cache"):
        # mapping: root_node -> set of (state_key, player_pos, steps)
        default_expansion._root_transposition_cache = {}
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of (state_key, player_pos, steps) seen so far for the search root."""
        root = current_node
        while getattr(root, "parent", None) is not None:
            root = root.parent
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
        # left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                return True
        # right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                return True
        # up wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                return True
        # down wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r + 1, height)):
                return True
        return False

    # ----------------------------------------------------------------------
    # Additional cheap dead‑lock patterns
    # ----------------------------------------------------------------------
    def _is_2x2_deadlock(boxes, walls, targets):
        """Return True if there exists a 2×2 block of boxes with no target inside."""
        bset = set(boxes)
        for (r, c) in bset:
            if ((r, c + 1) in bset and
                (r + 1, c) in bset and
                (r + 1, c + 1) in bset):
                if not any(((r + dr, c + dc) in targets) for dr in (0, 1) for dc in (0, 1)):
                    return True
        return False

    def _is_corridor_deadlock(box, walls, targets):
        """Dead‑lock when a box sits between two opposite walls on a line and is not on a target."""
        if box in targets:
            return False
        r, c = box
        if (r - 1, c) in walls and (r + 1, c) in walls:
            return True
        if (r, c - 1) in walls and (r, c + 1) in walls:
            return True
        return False

    def _is_frozen_deadlock(box, walls, boxes, targets, height, width):
        """
        Cheap frozen check: box adjacent to a wall where no target exists along that line
        *and* the orthogonal neighbour is also a wall/box, making it immovable.
        """
        if box in targets:
            return False
        r, c = box
        # left wall + vertical block
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # right wall + vertical block
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # up wall + horizontal block
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                if (r, c - 1) in walls or (r, c + 1) in walls or (r, c - 1) in boxes or (r, c + 1) in boxes:
                    return True
        # down wall + horizontal block
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r + 1, height)):
                if (r, c - 1) in walls or (r, c + 1) in walls or (r, c - 1) in boxes or (r, c + 1) in boxes:
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
    # Helper: player reachability – can the player reach any box after the action?
    # ----------------------------------------------------------------------
    def _player_can_reach_box(state) -> bool:
        """BFS from player over free cells (walls excluded, boxes are obstacles)."""
        walls = state.walls
        boxes = state.boxes
        height, width = state.height, state.width

        start = state.player
        if start in boxes:  # unlikely, but safe
            return True

        visited = {start}
        q = deque([start])

        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in walls:
                    continue
                if (nr, nc) in boxes:
                    # reaching a box cell means the player can stand next to it
                    return True
                if 0 <= nr < height and 0 <= nc < width:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    # ----------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ----------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns a tuple:
          (valid, is_push, manh_imp, match_imp, player_imp, child_state)

        `valid` is False if the resulting state has a deadlock, is a transposition,
        or the player becomes trapped.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- dead‑lock pruning (basic + extra patterns) ----
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if _is_2x2_deadlock(temp_state.boxes, temp_state.walls, temp_state.targets):
            return False, False, 0, 0, 0, None
        if any(_is_corridor_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if any(_is_frozen_deadlock(b, temp_state.walls, temp_state.boxes,
                                   temp_state.targets, temp_state.height, temp_state.width)
               for b in temp_state.boxes):
            return False, False, 0, 0, 0, None

        # ---- player reachability pruning ----
        if not _player_can_reach_box(temp_state):
            return False, False, 0, 0, 0, None

        # ---- transposition detection (per‑root, includes step count) ----
        full_key = (temp_state.state_key(), temp_state.player, temp_state.steps)
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
        # push‑first bias only if the push yields *some* improvement
        if is_push and (manh_imp <= 0 and match_imp <= 0):
            is_push = False
        candidates.append((a, is_push, manh_imp, match_imp, player_imp, child_state))

    # ----------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ----------------------------------------------------------------------
    if candidates:
        # Sort: pushes first (only those with genuine improvement), then larger
        # Manhattan improvement, then larger matching improvement, then larger
        # player‑distance improvement
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
        trans_set.add((chosen_state.state_key(), chosen_state.player, chosen_state.steps))

        # Bounded cache: keep size ≤ 2000 entries
        if len(trans_set) > 2000:
            # remove arbitrary entries until within limit
            while len(trans_set) > 2000:
                trans_set.pop()
        return child

    # ----------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ----------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, player_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter (basic + extra)
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            continue
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            continue
        if _is_2x2_deadlock(temp_state.boxes, temp_state.walls, temp_state.targets):
            continue
        if any(_is_corridor_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            continue
        if any(_is_frozen_deadlock(b, temp_state.walls, temp_state.boxes,
                                   temp_state.targets, temp_state.height, temp_state.width)
               for b in temp_state.boxes):
            continue
        if not _player_can_reach_box(temp_state):
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
        trans_set.add((chosen_state.state_key(), chosen_state.player, chosen_state.steps))
        if len(trans_set) > 2000:
            while len(trans_set) > 2000:
                trans_set.pop()
        return child

    # ----------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ----------------------------------------------------------------------
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add((child_state.state_key(), child_state.player, child_state.steps))
    if len(trans_set) > 2000:
        while len(trans_set) > 2000:
            trans_set.pop()
    return child
```

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: Added player-to-closest-box proximity component (A* heuristic from
             sokoban_astar.ipynb): h = player_dist + sum(box_to_nearest_target).
             Weights redistributed from 70/30 (frac/dist) to 63/27/10
             (frac/dist/player_prox) so player position guides rollout action
             selection toward boxes that still need pushing.
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
      • 63 % – fraction of boxes already on targets.
      • 27 % – normalised matching distance (tighter normalisation).
      • 10 % – player proximity to nearest unplaced box (A* component:
                player_to_closest_box normalised by max board distance).
                Guides rollout action selection toward boxes that still need
                pushing, mirroring the h = player_dist + box_dist heuristic
                used in sokoban_astar.ipynb.
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

    # 3) player proximity to nearest unplaced box  (A* heuristic component)
    #    Unplaced boxes are those not yet on a target – they still need pushing.
    unplaced = [b for b in state.boxes if b not in state.targets]
    if unplaced and max_one > 0:
        pr, pc = state.player
        player_dist = min(_manhattan((pr, pc), b) for b in unplaced)
        player_prox = 1.0 - (player_dist / max_one)
    else:
        # All boxes placed (solved) or degenerate board – full proximity score.
        player_prox = 1.0

    # 4) push penalty (scaled down)
    push_penalty = push_lambda * (push_count / max_steps)

    # 5) push bonus – encourages roll‑outs that actually move a box
    bonus = push_bonus if push_count > 0 else 0.0

    return max(0.0, 0.63 * frac + 0.27 * dist_score + 0.10 * player_prox
               + bonus - push_penalty)


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
Description: Clean up redundant imports, minor cache‑eviction tweak, and remove unnecessary inner import.
Generated:   2026-03-08T18:02:48.551302
"""

def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements over the previous version:
      • Reject pushes that do not improve any distance metric.
      • Additional cheap dead‑lock patterns:
          – 2×2 box block without a target.
          – Straight‑corridor dead‑lock (box between two walls on a line).
          – Simple frozen box (box against a wall with no target in that line
            *and* blocked orthogonally).
      • Verify that after the action the player can still reach at
        least one box (BFS on floor cells, boxes act as obstacles).
      • Transposition key now includes the current step count so that
        identical layouts reached at different depths are distinguished.
      • Per‑root transposition cache size is limited (≈2000 entries) to
        keep memory usage bounded.
    """
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math
    from collections import deque

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (simple dict – MCTSNode is not weak‑ref‑able)
    # ----------------------------------------------------------------------
    if not hasattr(default_expansion, "_root_transposition_cache"):
        # mapping: root_node -> set of (state_key, player_pos, steps)
        default_expansion._root_transposition_cache = {}
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of (state_key, player_pos, steps) seen so far for the search root."""
        root = current_node
        while getattr(root, "parent", None) is not None:
            root = root.parent
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
        # left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                return True
        # right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                return True
        # up wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                return True
        # down wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r + 1, height)):
                return True
        return False

    # ----------------------------------------------------------------------
    # Additional cheap dead‑lock patterns
    # ----------------------------------------------------------------------
    def _is_2x2_deadlock(boxes, walls, targets):
        """Return True if there exists a 2×2 block of boxes with no target inside."""
        bset = set(boxes)
        for (r, c) in bset:
            if ((r, c + 1) in bset and
                (r + 1, c) in bset and
                (r + 1, c + 1) in bset):
                if not any(((r + dr, c + dc) in targets) for dr in (0, 1) for dc in (0, 1)):
                    return True
        return False

    def _is_corridor_deadlock(box, walls, targets):
        """Dead‑lock when a box sits between two opposite walls on a line and is not on a target."""
        if box in targets:
            return False
        r, c = box
        if (r - 1, c) in walls and (r + 1, c) in walls:
            return True
        if (r, c - 1) in walls and (r, c + 1) in walls:
            return True
        return False

    def _is_frozen_deadlock(box, walls, boxes, targets, height, width):
        """
        Cheap frozen check: box adjacent to a wall where no target exists along that line
        *and* the orthogonal neighbour is also a wall/box, making it immovable.
        """
        if box in targets:
            return False
        r, c = box
        # left wall + vertical block
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # right wall + vertical block
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # up wall + horizontal block
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                if (r, c - 1) in walls or (r, c + 1) in walls or (r, c - 1) in boxes or (r, c + 1) in boxes:
                    return True
        # down wall + horizontal block
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r + 1, height)):
                if (r, c - 1) in walls or (r, c + 1) in walls or (r, c - 1) in boxes or (r, c + 1) in boxes:
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
    # Helper: player reachability – can the player reach any box after the action?
    # ----------------------------------------------------------------------
    def _player_can_reach_box(state) -> bool:
        """BFS from player over free cells (walls excluded, boxes are obstacles)."""
        walls = state.walls
        boxes = state.boxes
        height, width = state.height, state.width

        start = state.player
        if start in boxes:  # unlikely, but safe
            return True

        visited = {start}
        q = deque([start])

        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in walls:
                    continue
                if (nr, nc) in boxes:
                    # reaching a box cell means the player can stand next to it
                    return True
                if 0 <= nr < height and 0 <= nc < width:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    # ----------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ----------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns a tuple:
          (valid, is_push, manh_imp, match_imp, player_imp, child_state)

        `valid` is False if the resulting state has a deadlock, is a transposition,
        or the player becomes trapped.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- dead‑lock pruning (basic + extra patterns) ----
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if _is_2x2_deadlock(temp_state.boxes, temp_state.walls, temp_state.targets):
            return False, False, 0, 0, 0, None
        if any(_is_corridor_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        if any(_is_frozen_deadlock(b, temp_state.walls, temp_state.boxes,
                                   temp_state.targets, temp_state.height, temp_state.width)
               for b in temp_state.boxes):
            return False, False, 0, 0, 0, None

        # ---- player reachability pruning ----
        if not _player_can_reach_box(temp_state):
            return False, False, 0, 0, 0, None

        # ---- transposition detection (per‑root, includes step count) ----
        full_key = (temp_state.state_key(), temp_state.player, temp_state.steps)
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
        # push‑first bias only if the push yields *some* improvement
        if is_push and (manh_imp <= 0 and match_imp <= 0):
            is_push = False
        candidates.append((a, is_push, manh_imp, match_imp, player_imp, child_state))

    # ----------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ----------------------------------------------------------------------
    if candidates:
        # Sort: pushes first (only those with genuine improvement), then larger
        # Manhattan improvement, then larger matching improvement, then larger
        # player‑distance improvement
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
        trans_set.add((chosen_state.state_key(), chosen_state.player, chosen_state.steps))

        # Bounded cache: keep size ≤ 2000 entries
        if len(trans_set) > 2000:
            # remove arbitrary entries until within limit
            while len(trans_set) > 2000:
                trans_set.pop()
        return child

    # ----------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ----------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, player_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter (basic + extra)
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            continue
        if any(_is_wall_line_deadlock(b, temp_state.walls, temp_state.targets,
                                      temp_state.height, temp_state.width) for b in temp_state.boxes):
            continue
        if _is_2x2_deadlock(temp_state.boxes, temp_state.walls, temp_state.targets):
            continue
        if any(_is_corridor_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            continue
        if any(_is_frozen_deadlock(b, temp_state.walls, temp_state.boxes,
                                   temp_state.targets, temp_state.height, temp_state.width)
               for b in temp_state.boxes):
            continue
        if not _player_can_reach_box(temp_state):
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
        trans_set.add((chosen_state.state_key(), chosen_state.player, chosen_state.steps))
        if len(trans_set) > 2000:
            while len(trans_set) > 2000:
                trans_set.pop()
        return child

    # ----------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ----------------------------------------------------------------------
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add((child_state.state_key(), child_state.player, child_state.steps))
    if len(trans_set) > 2000:
        while len(trans_set) > 2000:
            trans_set.pop()
    return child
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban (level1)
Timestamp:  2026-03-10T18:21:57.679450
Iterations: 500
Solved:     True
Steps:      1
Returns:    [1.0]

  Move 1: action=3, total_visits=500
    State: Step 0/200 | Boxes on target: 0/1 | Total distance: 1
    Children: [3(v=500, avg=1.000)]

Final state:
Step 1/200 | Boxes on target: 1/1 | Total distance: 0
#####
# @*#
#####

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

**1. KEY WEAKNESSES**  

| Rank | Symptom (observed in many runs) | Evidence / Reasoning |
|------|--------------------------------|----------------------|
| **1** | **Over‑pruning of pushes** – the tree expands almost exclusively with “move‑only” actions, delaying any necessary box movement.  In the trace the only successful expansion was a single push; in deeper levels the algorithm repeatedly selects actions that keep the Manhattan‑ and matching‑distance unchanged, then falls back to the “force‑push” block after **all** candidates are discarded.  This creates long, low‑information roll‑outs and often stalls before a solution is found. |
| **2** | **Transposition cache tied to step count** – identical board layouts reached at different depths are treated as distinct entries because the key includes `steps`.  Consequently the cache never re‑uses useful states, blowing up the effective branching factor and wasting visits on duplicated sub‑trees. |
| **3** | **Push‑first bias is unintentionally weakened** – after a push is evaluated the code does `if is_push and (manh_imp <= 0 and match_imp <= 0): is_push = False`.  This converts a genuine push into a “non‑push” when it does not improve either distance metric, pushing it to the end of the sorted list.  The subsequent sort (`not tup[1]`) then favours non‑pushes, so the algorithm actively avoids pushes that are temporarily “bad” (e.g. moving a box out of a corner to free a corridor). |
| **4** | **Reachability test is too permissive** – `_player_can_reach_box` returns *True* as soon as the player can step onto any box cell, not when the player can reach a *pushing position* for a future move.  This lets expansions that trap the player behind a newly‑pushed box slip through, increasing the chance of dead‑ends later in the search. |

**2. ROOT CAUSE**  

* The expansion routine tries to keep the search focused on pushes that *immediately* lower box‑target distance (`manh_imp` / `match_imp`).  Sokoban often requires a “detour” push that momentarily increases distance; the current filter (`is_push` → `False` when both improvements ≤ 0) discards exactly those moves.  

* By embedding `steps` into the transposition key the routine defeats one of the main benefits of transposition tables – sharing statistics across identical positions reached by different move orders.  The cache therefore grows needlessly and “prunes” only within a single depth slice, limiting its effectiveness.  

* The reachability test stops at the first encountered box, ignoring whether the player can stand *behind* the box to push it later.  This omission permits expansions that look safe but actually cut off the player’s ability to manipulate remaining boxes.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental optimisation (recommended).**  
The core architecture (dead‑lock filters, matching‑distance evaluation, BFS reachability) is sound; we only need to relax the overly strict push‑selection and improve state reuse.

1. **Allow pushes without immediate distance gain** – drop the `is_push and (manh_imp <= 0 and match_imp <= 0)` clause.  Instead keep a *soft* bias: add a small positive “push score” (e.g. `+0.1`) to the sorting key for any push that passes dead‑lock and reachability checks, regardless of distance improvement.  This preserves the push‑first intention while still permitting necessary detours.

2. **Decouple transposition from step count** – change the cache key to `(state.state_key(), state.player)`.  Keep the optional `steps` component only when the same layout can legitimately appear with different player‑to‑box distances (rare).  This dramatically raises cache hit‑rate, letting the algorithm recycle visit counts and avoid re‑expanding identical positions.

3. **Strengthen reachability** – modify `_player_can_reach_box` to require the player to reach a *free cell adjacent to a box* (i.e., a potential pushing position).  Perform a BFS that stops when a neighbour of a box (not the box cell itself) is reached.  This prevents expansions that trap the player behind a box while still being cheap.

4. **Tie‑break on pushes when heuristics are equal** – after selecting the best heuristic actions, if none are pushes but pushes exist among safe actions, preferentially pick a push (already done in simulation but not in expansion).  Implement this in the candidate‑selection block so that pushes are considered first when scores tie.

5. **Cache size management** – keep the 2000‑entry limit but now it will be populated with truly distinct positions, improving memory efficiency.

These targeted tweaks keep the existing dead‑lock detection, matching‑distance computation, and BFS reachability while removing the two biggest sources of myopic behaviour (push over‑pruning and step‑bound transpositions).  The changes are lightweight, maintainable, and should yield a noticeably higher solve rate without a full redesign.
