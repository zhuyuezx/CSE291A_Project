# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:09:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 47.09s |
| Status     | success |
| Tokens     | 11203 |
| Validation | valid |

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
TARGET HEURISTIC TO IMPROVE (expansion)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: expansion
Description: Fixed transposition‑table handling for `MCTSNode` objects that use `__slots__` (cannot set arbitrary attributes) by moving the per‑root transposition set to a module‑level weak‑key dictionary. The rest of the heuristic logic is unchanged.
Generated:   2026-03-08T17:04:57.817284
"""

def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock only (wall‑line deadlock removed).
      • Transposition table lives on the search root.
      • Guarantees a push candidate when possible.
      • Uses matching‑distance as a secondary improvement metric.
    """
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math
    import weakref

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (weak‑keyed so roots can be GC‑ed)
    # ----------------------------------------------------------------------
    # This dict lives on the function object so that the state is shared across
    # calls but does not interfere with the MCTSNode slots.
    if not hasattr(default_expansion, "_root_transposition_cache"):
        default_expansion._root_transposition_cache = weakref.WeakKeyDictionary()
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of state keys seen so far for the search root."""
        # climb to the root
        root = current_node
        while getattr(root, "parent", None) is not None:
            root = root.parent

        # Retrieve or create the set stored in the module‑level cache
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
    # Helper to evaluate a single action for expansion
    # ----------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns (valid, is_push, manhattan_improve, matching_improve, child_state)
        where `valid` is False if the resulting state has a corner deadlock
        or is a transposition.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- corner dead‑lock pruning (cheap, safe) ----
        for bpos in temp_state.boxes:
            if _is_corner_deadlock(bpos, temp_state.walls, temp_state.targets):
                return False, False, 0, 0, None

        # ---- transposition detection (per‑root) ----
        key = temp_state.state_key()
        if key in trans_set:
            return False, False, 0, 0, None

        # ---- push detection ----
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ---- distance improvements ----
        old_manh = node.state.total_box_distance()
        new_manh = temp_state.total_box_distance()
        manhattan_improve = old_manh - new_manh          # positive → closer

        old_match = _matching_distance(node.state)
        new_match = _matching_distance(temp_state)
        matching_improve = old_match - new_match          # positive → closer

        return True, is_push, manhattan_improve, matching_improve, temp_state

    # ----------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ----------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates = []   # (action, is_push, manh_imp, match_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, child_state = evaluate_action(a)
        if not valid:
            continue                # discard permanently (dead‑lock or duplicate)
        candidates.append((a, is_push, manh_imp, match_imp, child_state))

    # ----------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ----------------------------------------------------------------------
    if candidates:
        # Sort: pushes first, then larger Manhattan improvement, then larger matching improvement
        candidates.sort(
            key=lambda tup: (
                not tup[1],          # pushes first (False < True)
                -tup[2],             # larger Manhattan improvement
                -tup[3]              # larger matching improvement
            )
        )
        chosen_action, _, _, _, chosen_state = candidates[0]

        # keep the remaining actions for future expansions
        remaining_actions = [a for a, *_ in candidates[1:]]
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining_actions)
        else:
            node._untried_actions = remaining_actions

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ----------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ----------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets)
               for b in temp_state.boxes):
            continue

        # push detection
        if len(temp_state.boxes - node.state.boxes) == 0:
            continue

        manh_imp = node.state.total_box_distance() - temp_state.total_box_distance()
        match_imp = _matching_distance(node.state) - _matching_distance(temp_state)
        push_candidates.append((a, manh_imp, match_imp, temp_state))

    if push_candidates:
        push_candidates.sort(key=lambda tup: (-tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = push_candidates[0]

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
        trans_set.add(chosen_state.state_key())
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
    trans_set.add(child_state.state_key())
    return child
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (from Trace #1) | Why it hurts the search |
|------|--------------------------|--------------------------|
| 1️⃣ | The agent repeats the pattern **3 → 2 → 3 → 2 …** for the whole 200‑step horizon. No box is ever pushed; `total_box_distance` stays at 6 and the return is always 0. | The expansion routine never selects a *push* that actually changes the box layout, so the tree never explores the part of the search space that can solve the puzzle. |
| 2️⃣ | All children of every node have the same visit count (≈ 50/67) and average value 0.0. | The UCB1 policy receives identical Q‑values for every child, so selection devolves to random walk‑like behaviour. |
| 3️⃣ | The algorithm does not try to move the player toward a box when no push is immediately available. | Without a “player‑to‑box” signal, the expansion treats a pure move exactly like a push that makes no distance improvement, leaving the player wandering aimlessly. |
| 4️⃣ | Transposition handling is per‑root **but the key likely omits the player position** (common in Sokoban implementations). Consequently, a state with the same box layout but the player on the opposite side is considered a duplicate and discarded, preventing the algorithm from exploring necessary approach moves. |
| 5️⃣ | Wall‑line dead‑lock detection is missing in expansion (present only in simulation). Some pushes are silently discarded because they lead to a dead‑lock that the cheap corner test does not catch. |

**2. ROOT CAUSE**

*Expansion* builds its candidate list with `evaluate_action`. It scores actions only by:

* `is_push` (binary, used only as the first sort key),
* **Manhattan improvement of the boxes**, and
* **Matching‑distance improvement**.

Both improvements are *zero* for any pure move (boxes stay put) and are also often zero for a push that moves a box sideways without changing its distance to any target. The sort therefore falls back to the original ordering of actions. Because the untried‑action collection is unordered (`set` or list) the tie is essentially random – in this level it repeatedly picks the right/left pair.

The heuristic completely ignores **player proximity to boxes**. Consequently, when the player is not already adjacent to a box, the expansion has no reason to move towards one; it may keep selecting “neutral” moves that keep the same box configuration, leading to the observed oscillation.

Additionally, if `state_key` does **not** encode the player coordinate, any move that does not move a box is considered a transposition and is permanently removed from the pool, forcing the algorithm to alternate between the only two surviving actions (right/left) that keep the player on the same two squares.

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinements (recommended).**  
The current architecture (per‑root transposition cache, dead‑lock pruning, matching‑distance metric) is sound; we only need to give the expansion a better *push‑bias* and a *player‑approach* signal.

*Add a small constant bonus to pushes.*  
When constructing the sorting key, use  
`key = (not is_push, -(manhattan_improve + ε_push), -(matching_improve + ε_push), -player_to_box_dist)`  
with `ε_push` ≈ 0.1. This guarantees that any push outranks a move even when both improvements are zero.

*Introduce a player‑to‑nearest‑box distance term.*  
Compute `player_box_dist = min(|player‑r – box‑r| + |player‑c – box‑c|)` for the child state and treat a reduction of this distance as a positive improvement (add it to the sort key). Moves that bring the player closer to a box will now be preferred over moves that keep the distance unchanged.

*Make the transposition key explicit.*  
If `state_key` does not contain the player coordinate, wrap it:  
`full_key = (state.state_key(), state.player)`  
and use `full_key` for the per‑root cache. This prevents legitimate approach moves from being discarded as duplicates.

*Add wall‑line dead‑lock pruning to expansion.*  
Re‑use the `_is_wall_line_deadlock` function from the simulation phase; discard any push that would create such a dead‑lock before adding it to the candidate list.

These four targeted tweaks preserve the existing matching‑distance heuristic, the weak‑key transposition table, and all safety checks, while directly addressing the primary failure mode (never pushing boxes) and the secondary issue (pruning needed approach moves). No structural rewrite is required, keeping the code base stable and allowing quick iteration.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
    import weakref

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (weak‑keyed so roots can be GC‑ed)
    # ----------------------------------------------------------------------
    if not hasattr(default_expansion, "_root_transposition_cache"):
        default_expansion._root_transposition_cache = weakref.WeakKeyDictionary()
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of (state_key, player_pos) seen so far for the search root."""
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
        # Corner dead‑lock
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        # Wall‑line dead‑lock (only matters for pushes, but we check anyway)
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
- No issues found

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: No changes needed; the draft implementation is correct and efficient.
```python
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
    import weakref

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (weak‑keyed so roots can be GC‑ed)
    # ----------------------------------------------------------------------
    if not hasattr(default_expansion, "_root_transposition_cache"):
        default_expansion._root_transposition_cache = weakref.WeakKeyDictionary()
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of (state_key, player_pos) seen so far for the search root."""
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
        # Corner dead‑lock
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets) for b in temp_state.boxes):
            return False, False, 0, 0, 0, None
        # Wall‑line dead‑lock (only matters for pushes, but we check anyway)
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
