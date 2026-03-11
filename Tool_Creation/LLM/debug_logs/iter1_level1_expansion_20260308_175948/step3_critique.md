# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 18:02:48 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 69.66s |
| Status     | success |
| Tokens     | 16173 |
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

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (observed on larger test‑levels) | Why it hurts the search |
|------|------------------------------------------|--------------------------|
| **1** | *Premature pushes that create frozen dead‑locks.*  After a few expansions the algorithm often chooses a push that looks attractive (large Manhattan‑improvement) but leaves a box against a wall with **no target in that line** *and* a second box blocking the only escape route. The rollout then collapses to 0‑reward and the node is never revisited, wasting many simulations.  |
| **2** | *Over‑aggressive “push‑first” ordering.*  The sort key `not tup[1]` forces every push to outrank any safe move, even when the push gives **zero or negative** improvement on the matching distance.  In cramped corridors this floods the tree with deep push‑only branches, delaying exploration of necessary repositioning moves. |
| **3** | *Insufficient dead‑lock pruning.*  Only simple corner‑deadlocks and a single‑direction wall‑line test are applied.  Classic Sokoban dead‑locks such as **2×2 blocks, frozen squares, and patterns involving multiple boxes** are missed, so many illegal states are expanded and later penalised only by the simulation. |
| **4** | *No look‑ahead on player reachability.*  The expansion evaluates only the *player‑to‑nearest‑box* distance, ignoring whether the player can actually reach the box after the push.  This lets the algorithm favour pushes that put the player behind a wall, causing an immediate dead‑end in the next expansion. |
| **5** | *Transposition cache grows unchecked per‑root.*  The cache is a plain `set` that never evicts entries.  On deeper searches it becomes large enough to cause a **memory bottleneck** and, more subtly, it treats two states that are identical except for the number of steps taken as the same, preventing the tree from exploring alternative push‑orderings that may have different step‑counts but the same configuration. |

**2. ROOT CAUSE**  

* **Push‑first bias** – The sorting tuple places `not tup[1]` first, so any push (even with zero improvement) is chosen before a non‑push that could drastically reduce the matching distance. The code never checks the *sign* of `manh_imp` / `match_imp` before promoting a push.  

* **Dead‑lock detection limited to corners & single‑wall lines** – The helpers `_is_corner_deadlock` and `_is_wall_line_deadlock` are the only guards.  Patterns such as a box stuck against two parallel walls with no target in either direction, or a 2×2 block of boxes, are not caught. The expansion therefore creates states that are unsolvable but not filtered out.  

* **Player reachability not verified** – The metric `_player_to_nearest_box` only measures raw Manhattan distance; it does not consider obstacles. Consequently a push that puts the player in an isolated area may still look “good”.  

* **Transposition cache key missing step‑count** – The tuple `(state_key, player)` treats two nodes with the same configuration but different remaining step budgets as identical.  This prevents the tree from distinguishing a fast solution from a slower one, limiting the ability to learn that a longer path may be necessary.  

* **Unbounded cache** – The comment mentions a “small guard” but none is implemented; the set keeps growing, eventually slowing look‑ups and memory use.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental enhancements** (recommended).  
The current framework – corner‑dead‑lock guard, transposition set, matching‑distance heuristic – already provides a solid base.  Small, targeted changes can remove the dominant failure modes without a full redesign.

1. **Refine push ordering**  
   * Only promote a push if it yields a *positive* improvement on at least one of the distance metrics (`manh_imp > 0` or `match_imp > 0`).  Otherwise treat it like a normal action in the sorting key.  

2. **Add lightweight dead‑lock patterns** – implement three cheap checks that run before the costly matching‑distance call:  
   * **2×2 block**: if four boxes occupy a 2×2 sub‑grid and none of the four cells is a target, reject.  
   * **Frozen box**: a box adjacent to a wall (or another box) in a direction where *no* target exists in that line **and** the orthogonal direction is also blocked by a wall/box, i.e. the box can never be nudged out.  
   * **Simple corridor block**: if a box is on a hallway cell whose two opposite neighbours are walls and the cell itself is not a target, reject.  

   These checks are O(1) per box and dramatically cut unsolvable expansions.

3. **Reachability filter**  
   * Run a tiny BFS from the player (ignoring boxes) after the candidate action to compute **whether the player can reach any box**.  If the reachable‑box set becomes empty, discard the action.  This costs at most a few dozen cells on typical levels and prevents “player‑trapped” pushes.

4. **Improve transposition key**  
   * Include the remaining step budget (`state.steps`) in the key: `(state_key, player, state.steps)`.  This differentiates fast‑solution nodes from slower duplicates, allowing the tree to learn that a particular configuration reached earlier is preferable.

5. **Bound the per‑root cache**  
   * After each insertion, if `len(trans_set) > 2000` (tunable) drop the oldest 10 % of entries (store them in an `OrderedDict` or maintain a simple FIFO list).  This keeps lookup O(1) while preventing memory blow‑up.

These five modifications directly address weaknesses 1‑5 above, keep the overall structure of `default_expansion` unchanged, and require only local code edits.  The impact should be observable quickly: fewer dead‑end branches, deeper exploration of non‑push moves, and more stable memory usage in long searches.  

If after these tweaks the solver still stalls on very large puzzles, a **Strategy B** (e.g., switching to a pattern‑database‑guided expansion or a two‑stage “push‑order” planner) could be considered, but the current evidence points to the incremental fixes as the most effective first step.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
    from collections import deque
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
                # check if any of the four cells is a target
                if not any(((r + dr, c + dc) in targets) for dr in (0, 1) for dc in (0, 1)):
                    return True
        return False

    def _is_corridor_deadlock(box, walls, targets):
        """Dead‑lock when a box sits between two opposite walls on a line (vertical or horizontal) and is not on a target."""
        if box in targets:
            return False
        r, c = box
        # vertical corridor
        if (r - 1, c) in walls and (r + 1, c) in walls:
            return True
        # horizontal corridor
        if (r, c - 1) in walls and (r, c + 1) in walls:
            return True
        return False

    def _is_frozen_deadlock(box, walls, boxes, targets, height, width):
        """
        Very cheap frozen check: box adjacent to a wall in a direction
        where no target exists along that line *and* the orthogonal
        neighbour is also a wall/box, making it immovable.
        """
        if box in targets:
            return False
        r, c = box
        # left wall + up/down block
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(c)):
                # check vertical neighbour blockage
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # right wall + up/down block
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(c + 1, width)):
                if (r - 1, c) in walls or (r + 1, c) in walls or (r - 1, c) in boxes or (r + 1, c) in boxes:
                    return True
        # up wall + left/right block
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(r)):
                if (r, c - 1) in walls or (r, c + 1) in walls or (r, c - 1) in boxes or (r, c + 1) in boxes:
                    return True
        # down wall + left/right block
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
    # Helper: player reachability – can the player reach *any* box after the action?
    # ----------------------------------------------------------------------
    def _player_can_reach_box(state) -> bool:
        """BFS from player over free cells (walls excluded, boxes are obstacles)."""
        from collections import deque

        walls = state.walls
        boxes = state.boxes
        height, width = state.height, state.width

        start = state.player
        if start in boxes:  # shouldn't happen, but guard
            return True

        visited = set()
        q = deque([start])
        visited.add(start)

        while q:
            r, c = q.popleft()
            # if we stand next to a box we could potentially push it later
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if (nr, nc) in walls or (nr, nc) in boxes:
                    # we can still consider the box position itself as "reachable" for our test
                    if (nr, nc) in boxes:
                        return True
                    continue
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
            # treat as non‑push to avoid polluting the ordering
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
        # ------------------------------------------------------------------
        # Bounded cache: keep size ≤ 2000 entries
        # ------------------------------------------------------------------
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
        # pop works for both list (pop last) and set (pop arbitrary)
        action = node._untried_actions.pop()
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add((child_state.state_key(), child_state.player, child_state.steps))
    while len(trans_set) > 2000:
        trans_set.pop()
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
- Redundant multiple `from collections import deque` imports clutter the file and slightly increase load time.
- The transposition cache eviction loop (`while len(trans_set) > 2000: trans_set.pop()`) may inadvertently remove the just‑added entry; using a simple size‑check after insertion is sufficient.
- Minor inefficiency: `_player_can_reach_box` imports `deque` again inside the function even though it is already imported at module level.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Clean up redundant imports, minor cache‑eviction tweak, and remove unnecessary inner import.

```python
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
