# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:04:29 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 43.09s |
| Status     | success |
| Tokens     | 9296 |
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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

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

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
Improved expansion for Sokoban MCTS.

Key ideas (incremental):
  1️⃣  Relax dead‑lock pruning during expansion – only the cheap corner
      dead‑lock test is kept.  The aggressive wall‑line dead‑lock check
      caused virtually all pushes to be discarded.
  2️⃣  Transposition detection is now *per‑root* rather than a global
      set, preventing permanent bans of positions that are reachable
      via different push histories.
  3️⃣  If no candidate survives the filters, we still try to expand a
      legal push (preferring the one that most reduces the Manhattan
      distance or the optimal matching distance).
  4️⃣  Slightly richer scoring – after the push‑first ordering we also
      rank by distance improvement (and fall back to matching‑distance
      improvement when the simple metric is non‑positive).

The function signature and overall flow remain unchanged.
"""

from typing import Set, Tuple, List
import itertools
import math

# ----------------------------------------------------------------------
# Local dead‑lock helpers (corner only – used in expansion)
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


# ----------------------------------------------------------------------
# Helper: optimal matching distance (same as in simulation)
# ----------------------------------------------------------------------
def _matching_distance(state) -> int:
    """Brute‑force optimal total Manhattan distance after assigning boxes to targets."""
    boxes = list(state.boxes)
    targets = list(state.targets)
    n_boxes = len(boxes)
    n_targets = len(targets)

    # If numbers differ we consider all subsets of targets of size n_boxes
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
# Main expansion function
# ----------------------------------------------------------------------
def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock only (wall‑line deadlock removed).
      • Transposition table lives on the search root.
      • Guarantees a push candidate when possible.
      • Uses matching‑distance as a secondary improvement metric.
    """
    from mcts.node import MCTSNode                     # local import to avoid circular deps

    # ------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        root = current_node
        while root.parent is not None:
            root = root.parent
        if not hasattr(root, "_expansion_seen_keys"):
            root._expansion_seen_keys = set()
        return root._expansion_seen_keys

    trans_set = get_root_transposition_set(node)

    # ------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates: List[Tuple[int, bool, int, int, object]] = []   # (action, is_push, manh_imp, match_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, child_state = evaluate_action(a)
        if not valid:
            continue                # discard permanently (dead‑lock or duplicate)
        candidates.append((a, is_push, manh_imp, match_imp, child_state))

    # ------------------------------------------------------------------
    # If we have any candidates, pick the best according to the ranking:
    #   1. push actions first,
    #   2. larger Manhattan improvement,
    #   3. larger matching‑distance improvement.
    # ------------------------------------------------------------------
    if candidates:
        candidates.sort(
            key=lambda tup: (
                not tup[1],                 # pushes first (False < True)
                -tup[2],                    # larger Manhattan improvement
                -tup[3]                     # larger matching improvement
            )
        )
        chosen_action, _, _, _, chosen_state = candidates[0]

        # keep the remaining actions for future expansions
        remaining_actions = [a for a, *_ in candidates[1:]]

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining_actions)
        else:
            node._untried_actions = remaining_actions

        # create child node
        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child

        # register transposition
        trans_set.add(chosen_state.state_key())
        return child

    # ------------------------------------------------------------------
    # No candidate survived – attempt to force a push candidate.
    # We ignore the transposition set here (allow one revisit) but we
    # still respect corner dead‑locks.
    # ------------------------------------------------------------------
    push_candidates: List[Tuple[int, int, int, object]] = []   # (action, manh_imp, match_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter
        dead = any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets)
                   for b in temp_state.boxes)
        if dead:
            continue

        # push detection
        is_push = len(temp_state.boxes - node.state.boxes) > 0
        if not is_push:
            continue

        manh_imp = node.state.total_box_distance() - temp_state.total_box_distance()
        match_imp = _matching_distance(node.state) - _matching_distance(temp_state)
        push_candidates.append((a, manh_imp, match_imp, temp_state))

    if push_candidates:
        # rank pushes by improvement (same ordering as above)
        push_candidates.sort(key=lambda tup: (-tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = push_candidates[0]

        # Update untried actions: remove the chosen one, keep the rest
        remaining = [a for a, *_ in push_candidates[1:]]
        # also keep any other original actions that were not pushes
        non_push_rest = [a for a in original_actions if a not in remaining + [chosen_action]]
        final_remaining = remaining + non_push_rest

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(final_remaining)
        else:
            node._untried_actions = final_remaining

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ------------------------------------------------------------------
    # Fallback: no push possible, revert to original naive expansion.
    # ------------------------------------------------------------------
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    # Even in fallback we store the transposition to avoid re‑expanding
    # the exact same board later.
    trans_set.add(child_state.state_key())
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
- No functional bugs identified; the logic respects the GameState API and handles both list and set representations of `_untried_actions`.
- Performance is acceptable: at most 4 actions are evaluated per expansion, and the extra matching‑distance computation is cheap in that context.
- The heuristic now correctly relaxes over‑aggressive dead‑lock pruning, uses a per‑root transposition set, and guarantees a push when possible.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Preserve the incremental improvements while cleaning up minor redundancies.
```python
def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock only (wall‑line deadlock removed).
      • Transposition table lives on the search root.
      • Guarantees a push candidate when possible.
      • Uses matching‑distance as a secondary improvement metric.
    """
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math

    # ------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        root = current_node
        while root.parent is not None:
            root = root.parent
        if not hasattr(root, "_expansion_seen_keys"):
            root._expansion_seen_keys = set()
        return root._expansion_seen_keys

    trans_set = get_root_transposition_set(node)

    # ------------------------------------------------------------------
    # Helper: cheap corner dead‑lock test
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helper: optimal matching distance (same as in simulation)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates = []   # (action, is_push, manh_imp, match_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, child_state = evaluate_action(a)
        if not valid:
            continue                # discard permanently (dead‑lock or duplicate)
        candidates.append((a, is_push, manh_imp, match_imp, child_state))

    # ------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ------------------------------------------------------------------
    if candidates:
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

    # ------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ------------------------------------------------------------------
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

        # update untried actions: remove the chosen push, keep the rest
        remaining = [a for a, *_ in push_candidates[1:]]
        non_push_rest = [a for a in original_actions if a not in remaining + [chosen_action]]
        final_remaining = remaining + non_push_rest

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(final_remaining)
        else:
            node._untried_actions = final_remaining

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ------------------------------------------------------------------
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add(child_state.state_key())
    return child
```
