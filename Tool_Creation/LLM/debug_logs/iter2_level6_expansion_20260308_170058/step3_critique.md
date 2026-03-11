# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:02:19 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 33.05s |
| Status     | success |
| Tokens     | 7213 |
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
Default expansion: expand one untried action.

Pops an untried action from the node, creates a child node
with the resulting state, and returns it.
"""


def default_expansion(node):
    """
    Expand one untried action from the given node.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    action = node._untried_actions.pop()
    child_state = node.state.clone()
    child_state.apply_action(action)
    from mcts.node import MCTSNode
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

| Rank | Symptom (trace evidence) | Why it matters |
|------|--------------------------|----------------|
| **1** | The search walks a *single* corridor forever: actions alternate 3 (RIGHT) → 2 (LEFT) for every move, never a push. All 200 iterations end with `total_visits=200, avg=0.0`. | No progress → reward stays 0 → UCB‑1 never favours other branches, so the tree never discovers a box‑push or a solution. |
| **2** | Every child created during expansion receives *no* quality estimate; `default_expansion` simply `pop()`s an arbitrary untried action. | Expands move‑only actions with the same priority as pushes, drowning the few useful push actions in a huge sea of neutral moves. |
| **3** | Dead‑locked states are still added to the tree. The expansion creates a child, then later the simulation discards it, but the node already consumes a visit. | Wastes iterations on branches that can never lead to a solution; the value of those nodes stays 0, reinforcing the “no‑push” bias. |
| **4** | No caching of already‑seen states (transposition detection) at expansion time. | Identical board positions reached by different move sequences are expanded repeatedly, inflating the tree without adding new information. |

**2. ROOT CAUSE**

The expansion phase is deliberately *uninformed*:

```python
action = node._untried_actions.pop()
child_state = node.state.clone()
child_state.apply_action(action)
```

- **No push‑priority test** – it treats a simple walk and a box‑push identically.
- **No deadlock check** – it does not call `_is_corner_deadlock` / `_is_wall_line_deadlock` before creating the child.
- **No heuristic ordering** – the set of untried actions is unordered; the first popped action is essentially random.
- **No state‑duplicate guard** – it never checks `state_key()` against existing children of *any* ancestor.

Consequently the tree expands massive “move‑only” sub‑trees, each yielding zero value, which drives the UCB‑1 scores to favour further expansion of those same zero‑value branches. The rollout heuristic can only rescue an iteration *after* a child has already been created, which is too late.

**3. PROPOSED APPROACH – Strategy A (Incremental)**  

We can dramatically improve expansion with a few lightweight, targeted changes while keeping the rest of the MCTS pipeline untouched.

| Step | Change | Effect |
|------|--------|--------|
| **a. Detect pushes** | After cloning the state, call `state.apply_action(a)` on a *temporary* copy and inspect whether the player’s new position coincides with a former box position (i.e., a push occurred). Store a boolean `is_push`. | Enables us to rank actions: pushes are far more likely to change the reward. |
| **b. Order untried actions** | Replace the unordered `pop()` with a small priority queue: first all safe pushes, then safe moves, finally moves that lead to immediate dead‑locks. This can be done by converting `_untried_actions` to a list, sorting with a key `(not is_push, deadlock_flag)`, and popping the first element. | Guarantees that the first expansions from any node are the most promising, so early visits concentrate on meaningful branches. |
| **c. Early deadlock pruning** | After the temporary `apply_action`, run the same dead‑lock predicates used in the rollout (`_is_corner_deadlock`, `_is_wall_line_deadlock`). If the resulting state is dead‑locked, **discard** the action (do not add a child) and continue with the next candidate. | Prevents obvious losing nodes from ever consuming visits. |
| **d. Duplicate‑state check** | Before creating a child, compute `child_state.state_key()` and compare against a global (or node‑local) `seen_keys` dict that maps keys to existing nodes. If the key exists, skip creating a new node and remove the action from `_untried_actions`. | Avoids expanding identical board positions multiple times, keeping the tree compact. |
| **e. Light value hint (optional)** | When a child is created, initialise `child.value` with a cheap estimate such as `+1.0` for a push that reduces the matching distance, and `0.0` otherwise. This tiny bias helps UCB‑1 prefer branches that immediately improve the heuristic. | Gives the selection phase a better starting signal without altering the unbiased back‑propagation later. |

All five tweaks require only a few extra lines inside `default_expansion`; they reuse the existing dead‑lock helpers and the `state.clone()`/`apply_action` API, so no other component (selection, simulation, back‑propagation) needs alteration.

**Why incremental works:**  
The core problem is *uninformed node creation*. By adding a push‑first ordering, dead‑lock pruning, and duplicate detection we steer the limited 200‑iteration budget toward the few actions that actually change the puzzle’s state. This addresses the dominant weakness (rank 1) without discarding the already‑well‑tuned simulation and back‑propagation code. A full paradigm shift (Strategy B) would be unnecessary; the MCTS framework is sound, only the expansion heuristic is starving it of useful information.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
Improved expansion for Sokoban MCTS.

Key ideas:
  1. Prefer actions that *push* a box (they change the puzzle).
  2. Discard actions that immediately create a dead‑locked state.
  3. Do not expand states that have already been visited elsewhere
     (transposition detection via a global key set).
  4. Within the same priority level, favour actions that decrease the
     total box‑to‑target Manhattan distance.
  5. Initialise the child node with a small value hint based on the
     distance improvement (optional – helps UCB1 early on).

The function keeps the original signature and uses only the public
GameState API plus the dead‑lock helpers duplicated locally.
"""

import itertools
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
      • The newly created child receives a tiny value hint based on
        the distance improvement, giving UCB‑1 a better initial signal.

    Args:
        node: MCTSNode with at least one untried action.

    Returns:
        The newly created child MCTSNode.
    """
    from mcts.node import MCTSNode  # local import to avoid circular deps

    # ------------------------------------------------------------------
    # Helper to evaluate a single action
    # ------------------------------------------------------------------
    def evaluate_action(action):
        """Return a tuple (valid, is_push, dist_improve, child_state)."""
        # Clone and apply the action on a temporary state
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- dead‑lock pruning -------------------------------------------------
        for bpos in temp_state.boxes:
            if (_is_corner_deadlock(bpos, temp_state.walls, temp_state.targets) or
                    _is_wall_line_deadlock(bpos, temp_state.walls, temp_state.targets,
                                           temp_state.height, temp_state.width)):
                return (False, False, 0, None)   # discard this action

        # ---- transposition detection -----------------------------------------
        key = temp_state.state_key()
        if key in _expansion_seen_keys:
            return (False, False, 0, None)       # already expanded elsewhere

        # ---- push detection ---------------------------------------------------
        # A push occurs iff the set of box positions changed
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ---- distance improvement --------------------------------------------
        old_dist = node.state.total_box_distance()
        new_dist = temp_state.total_box_distance()
        dist_improve = old_dist - new_dist  # positive => got closer

        return (True, is_push, dist_improve, temp_state)

    # ------------------------------------------------------------------
    # Gather candidates from the untried‑action pool
    # ------------------------------------------------------------------
    candidates: List[Tuple[int, bool, int, object]] = []   # (action, is_push, improve, state)
    # The node's untried actions might be a set, list or other iterable.
    # We will pop them one‑by‑one while examining, but we must keep the ones
    # that are not selected for later attempts.
    remaining_actions = list(node._untried_actions)   # snapshot
    node._untried_actions.clear()                    # we'll refill the survivors later

    for a in remaining_actions:
        valid, is_push, improve, child_state = evaluate_action(a)
        if not valid:
            # discard permanently (dead‑lock or duplicate)
            continue
        candidates.append((a, is_push, improve, child_state))

    # If no promising candidates remain, fall back to any still‑legal action
    # (this can happen in very tight dead‑end situations).
    if not candidates:
        # Re‑populate the node's untried pool with the original actions
        # so future expansions can still try them after the global key set
        # changes (unlikely but safe).
        node._untried_actions.update(remaining_actions)
        raise RuntimeError("Expansion failed: no non‑dead‑lock, non‑duplicate actions available.")

    # ------------------------------------------------------------------
    # Choose the best candidate:
    #   1) pushes first
    #   2) larger distance improvement
    #   3) arbitrary tie‑breaker (the order in `candidates` after sorting)
    # ------------------------------------------------------------------
    candidates.sort(key=lambda tup: (not tup[1], -tup[2]))
    chosen_action, is_push, improve, chosen_state = candidates[0]

    # All other still‑legal actions (including those we examined but did not pick)
    # must be put back into the node's untried pool for future expansion.
    for a, _, _, _ in candidates[1:]:
        node._untried_actions.add(a)

    # ------------------------------------------------------------------
    # Create the child node
    # ------------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child

    # Register the new board position globally to avoid future duplicates
    _expansion_seen_keys.add(chosen_state.state_key())

    # Tiny heuristic hint: if we reduced the distance, give a small positive
    # value; otherwise leave at 0. This does NOT break unbiased back‑propagation.
    if improve > 0:
        # Scale the hint to be modest (max 0.5) to keep it advisory.
        child.value = min(0.5, 0.01 * improve)

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
- The code assumes `node._untried_actions` is a mutable set (`clear`, `add`, `update`), but the original implementation may use a list; this can cause AttributeError.
- Assigning `child.value` may break if `MCTSNode` does not expose a `value` attribute.
- Raising an exception when no non‑dead‑lock actions exist can crash the search; a graceful fallback to the original naïve expansion is safer.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Fix type handling of untried actions, remove unsafe attribute assignment, and add a safe fallback when no promising actions are found.
```python
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
