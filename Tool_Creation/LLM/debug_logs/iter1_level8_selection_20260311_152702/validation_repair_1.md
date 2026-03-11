# validation_repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:28:02 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 11.19s |
| Status     | success |
| Tokens     | 2363 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS selection function, but it failed validation.

== BROKEN CODE ==
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight UCB exploration and Sokoban‑specific
    progress / dead‑lock signals.

    - g  : accumulated macro‑push cost (state.steps) – real cost so far.
    - h  : base heuristic from shared module (e.g., total box distance).
    - p  : progress bonus proportional to fraction of boxes already on targets.
    - d  : dead‑lock penalty (simple corner check).
    - u  : UCB‑style exploration term using parent and child visit counts.

    The child with the *lowest* (g + h - p + u) is selected.
    Tie‑break falls back to fewer visits.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper: cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def _is_deadlocked(state) -> bool:
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # Two orthogonal walls adjacent → corner
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Walk down the tree until we hit a node that needs expansion.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion handle it

        best_child = None
        best_score = math.inf
        best_visits = math.inf  # tie‑break on fewer visits

        # Use node.visits + 1 so log is always > 0, even at the very start.
        parent_visits = node.visits + 1

        for child in node.children.values():
            # ----- g : real macro‑push cost accumulated so far -----
            g = getattr(child.state, "steps", 0)

            # ----- h : base heuristic from shared module -----
            h = _ag.h_sokoban(child.state)

            # ----- dead‑lock penalty -----
            if _is_deadlocked(child.state):
                h += 5.0          # heavy penalty to push dead‑ends up

            # ----- progress bonus (more boxes on targets → lower score) -----
            num_targets = child.state.num_targets
            if num_targets:
                progress = child.state.boxes_on_targets() / num_targets
            else:
                progress = 0.0
            PROGRESS_WEIGHT = 0.5
            progress_bonus = PROGRESS_WEIGHT * progress

            # ----- UCB exploration term -----
            child_visits = child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(parent_visits) / (child_visits + 1e-9)
            )

            # ----- total selection score (lower is better) -----
            score = g + h - progress_bonus + exploration

            # Choose the child with the minimal score; break ties on fewer visits.
            if (score < best_score) or (
                abs(score - best_score) < 1e-9 and child_visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child_visits

        if best_child is None:
            # Fallback – shouldn't happen but keeps function safe.
            return node
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

**Fixed function signature and variable naming to meet validation requirements (first parameter renamed to `root`). Updated internal logic to use the new name while preserving original behavior.**

```python
def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight UCB exploration and Sokoban‑specific
    progress / dead‑lock signals.

    - g  : accumulated macro‑push cost (state.steps) – real cost so far.
    - h  : base heuristic from shared module (e.g., total box distance).
    - p  : progress bonus proportional to fraction of boxes already on targets.
    - d  : dead‑lock penalty (simple corner check).
    - u  : UCB‑style exploration term using parent and child visit counts.

    The child with the *lowest* (g + h - p + u) is selected.
    Tie‑break falls back to fewer visits.
    """
    import math  # ensure math is available if not imported elsewhere

    # ── Detect new MCTS search → reset shared A* state ───────────────
    # Find the true root of the tree (the node without a parent)
    true_root = root
    while true_root.parent is not None:
        true_root = true_root.parent
    root_key = true_root.state.state_key()
    if true_root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper: cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def _is_deadlocked(state) -> bool:
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # Two orthogonal walls adjacent → corner
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Walk down the tree until we hit a node that needs expansion.
    # ------------------------------------------------------------------
    node = root
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion handle it

        best_child = None
        best_score = math.inf
        best_visits = math.inf  # tie‑break on fewer visits

        # Use node.visits + 1 so log is always > 0, even at the very start.
        parent_visits = node.visits + 1

        for child in node.children.values():
            # ----- g : real macro‑push cost accumulated so far -----
            g = getattr(child.state, "steps", 0)

            # ----- h : base heuristic from shared module -----
            h = _ag.h_sokoban(child.state)

            # ----- dead‑lock penalty -----
            if _is_deadlocked(child.state):
                h += 5.0          # heavy penalty to push dead‑ends up

            # ----- progress bonus (more boxes on targets → lower score) -----
            num_targets = child.state.num_targets
            if num_targets:
                progress = child.state.boxes_on_targets() / num_targets
            else:
                progress = 0.0
            PROGRESS_WEIGHT = 0.5
            progress_bonus = PROGRESS_WEIGHT * progress

            # ----- UCB exploration term -----
            child_visits = child.visits
            exploration = exploration_weight * math.sqrt(
                math.log(parent_visits) / (child_visits + 1e-9)
            )

            # ----- total selection score (lower is better) -----
            score = g + h - progress_bonus + exploration

            # Choose the child with the minimal score; break ties on fewer visits.
            if (score < best_score) or (
                abs(score - best_score) < 1e-9 and child_visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child_visits

        if best_child is None:
            # Fallback – shouldn't happen but keeps function safe.
            return node
        node = best_child

    return node
```
