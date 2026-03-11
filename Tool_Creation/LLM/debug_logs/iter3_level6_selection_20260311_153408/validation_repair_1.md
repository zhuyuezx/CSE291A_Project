# validation_repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:35:11 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 5.30s |
| Status     | success |
| Tokens     | 2480 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS selection function, but it failed validation.

== BROKEN CODE ==
```python
"""
A*-guided MCTS selection with exploration and richer heuristic.

Improvements:
  • Uses the actual accumulated step cost (state.steps) as g instead of
    pure tree depth.
  • Adds a UCB‑style exploration term (weighted by exploration_weight)
    to avoid over‑exploitation of a single low‑f branch.
  • Enriches the heuristic h with a small dead‑lock penalty and a bonus
    for pushes that increase the number of boxes on targets.
  • Keeps the original A*‑style best‑first walk and root‑search reset.
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state (one instance shared across all phase files) ──
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag

# ── Heuristic tuning constants ────────────────────────────────────────
DEADLOCK_PENALTY = 5.0   # pushes a dead‑locked node higher in f
TARGET_BONUS = 0.5       # reward for each additional box placed on a target


def _is_corner_deadlocked(state) -> bool:
    """Simple corner dead‑lock detection (ignores targets)."""
    walls = state.walls
    targets = state.targets
    for bx, by in state.boxes:
        if (bx, by) in targets:
            continue
        horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
        vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
        if horiz and vert:
            return True
    return False


def _is_wall_line_deadlocked(state, box) -> bool:
    """
    Detect wall‑line dead‑locks: a box adjacent to a wall where, looking
    past that wall in the same direction, no target exists.
    """
    walls = state.walls
    targets = state.targets
    h, w = state.height, state.width
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in dirs:
        neighbor = (box[0] + dx, box[1] + dy)
        if neighbor not in walls:
            continue        # not a wall in this direction
        step = 2
        while True:
            check = (box[0] + dx * step, box[1] + dy * step)
            if not (0 <= check[0] < h and 0 <= check[1] < w):
                break
            if check in walls:
                break
            if check in targets:
                # a target beyond the wall – not a dead‑lock this way
                return False
            step += 1
        # No target found before hitting another wall/out‑of‑bounds → dead‑lock
        return True
    return False


def _is_deadlocked(state) -> bool:
    """Combine corner and wall‑line checks."""
    if _is_corner_deadlocked(state):
        return True
    for b in state.boxes:
        if b not in state.targets and _is_wall_line_deadlocked(state, b):
            return True
    return False


def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    At each level the child minimizing

        f = g + h - exploration

    is selected, where
        g = real accumulated step cost (state.steps)
        h = Manhattan distance + dead‑lock penalty – target bonus
        exploration = exploration_weight * sqrt(log(parent_visits) / child_visits)

    Ties are broken by preferring the child with the higher average value
    (more promising) and, finally, by fewer visits.

    The function also detects a new root (new search) and resets the shared
    A* globals.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Detect a new search and reset shared A* state.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # 2️⃣ Walk down the tree, always picking the child with minimal f.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion create a fresh child

        best_child = None
        best_f = math.inf
        best_q = -math.inf   # average value, used only when f ties

        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: base Manhattan distance + dead‑lock / target tweaks -----
            h_base = _ag.h_sokoban(child.state)
            dead_pen = DEADLOCK_PENALTY if _is_deadlocked(child.state) else 0.0

            # Target bonus: difference in boxes on targets compared to parent
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- exploration term (UCB) -----
            if child.visits == 0:
                # Strongly encourage unvisited nodes
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore

            # ----- tie‑break on average value -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        if best_child is None:
            break   # safety net; should not happen

        node = best_child

    return node
```

== VALIDATION ERRORS ==
  • Function 'default_selection' param 0 is 'node', expected 'root'. Full expected: ['root', 'exploration_weight']

== REQUIRED ==
Expected params: ['root', 'exploration_weight']
For selection, the first param MUST be named `root` (not `node`). Fix the function signature and any other validation issues. Return using the SAME structured format.

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**Description:** Fixed the function signature to use `root` as the first parameter (as required by the validator) and updated internal variable naming accordingly while preserving the original logic.

```python
def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    Parameters
    ----------
    root : Node
        The current root node of the MCTS tree (may be any node in the tree;
        the function will walk down from this node).
    exploration_weight : float, optional
        Weight applied to the UCB exploration term (default is √2 ≈ 1.41).

    Returns
    -------
    Node
        The selected node for expansion/simulation.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Detect a new search and reset shared A* state.
    # ------------------------------------------------------------------
    # Find the true tree root (node with no parent)
    true_root = root
    while true_root.parent is not None:
        true_root = true_root.parent

    root_key = true_root.state.state_key()
    if true_root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # 2️⃣ Walk down the tree, always picking the child with minimal f.
    # ------------------------------------------------------------------
    node = root
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion create a fresh child

        best_child = None
        best_f = math.inf
        best_q = -math.inf   # average value, used only when f ties

        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: base Manhattan distance + dead‑lock / target tweaks -----
            h_base = _ag.h_sokoban(child.state)
            dead_pen = DEADLOCK_PENALTY if _is_deadlocked(child.state) else 0.0

            # Target bonus: difference in boxes on targets compared to parent
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- exploration term (UCB) -----
            if child.visits == 0:
                # Strongly encourage unvisited nodes
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore

            # ----- tie‑break on average value -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        if best_child is None:
            break   # safety net; should not happen

        node = best_child

    return node
```
