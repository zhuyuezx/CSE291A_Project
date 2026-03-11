# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:28:20 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.27s |
| Status     | success |
| Tokens     | 2773 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS selection function, but it raised a runtime error.

== BROKEN CODE ==
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

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 470, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/selection/selection.py", line 30, in default_selection
    _ag.reset(root_key)
    ^^^
NameError: name '_ag' is not defined


== ACTUAL GameState PUBLIC API (for node.state or state param) ==
  apply_action: method
  boxes: set
  boxes_on_targets: method
  clone: method
  current_player: method
  height: int
  is_terminal: method
  legal_actions: method
  max_steps: int
  num_targets: int
  player: tuple
  returns: method
  state_key: method
  steps: int
  targets: frozenset
  total_box_distance: method
  walls: frozenset
  width: int

== MCTSNode API (node parameter) ==
MCTSNode uses __slots__. You CANNOT add new attributes.
Use ONLY these: node.state, node.parent, node.parent_action,
node.children, node._untried_actions, node.visits, node.value.
  - node.state = GameState (use for legal_actions, clone, etc.)
  - node._untried_actions = list of actions to expand
  - node.children = dict[action, child_node]
  - node.visits, node.value = backpropagated stats
Do NOT assign node._mcts_root_key or any attribute not listed above.

CRITICAL: The function must be SELF-CONTAINED. Define ALL helper functions inline. Do NOT use undefined names (e.g. _ag, _bfs_distance). Fix ONLY the broken parts. Keep the heuristic strategy the same.
Return using the SAME structured format.

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**Description:** Fixed undefined `_ag` reference, replaced it with a self‑contained heuristic using the public `GameState.total_box_distance` method, and adapted the node‑state checks to the documented `MCTSNode` API.

```python
def default_selection(root, exploration_weight: float = 1.41):
    """
    A*‑guided tree walk with lightweight UCB exploration and Sokoban‑specific
    progress / dead‑lock signals.

    - g : accumulated macro‑push cost (state.steps) – real cost so far.
    - h : sum of Manhattan distances from each box to the nearest target
          (provided by GameState.total_box_distance()).
    - p : progress bonus proportional to fraction of boxes already on targets.
    - d : dead‑lock penalty (simple corner check).
    - u : UCB‑style exploration term using parent and child visit counts.

    The child with the *lowest* (g + h – p + u) is selected.
    Ties are broken by choosing the child with fewer visits.
    """
    import math

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
    while True:
        # Terminal check – use the GameState API.
        if node.state.is_terminal():
            return node

        # Fully‑expanded check – a node is fully expanded when it has no
        # untried actions left.
        fully_expanded = len(node._untried_actions) == 0
        if not fully_expanded:
            # Not ready for selection – let the expansion step handle it.
            return node

        # --------------------------------------------------------------
        # Choose the best child according to the composite score.
        # --------------------------------------------------------------
        best_child = None
        best_score = math.inf
        best_visits = math.inf  # tie‑break on fewer visits

        parent_visits = node.visits + 1  # avoid log(0)

        for child in node.children.values():
            # ----- g : real macro‑push cost accumulated so far -----
            g = getattr(child.state, "steps", 0)

            # ----- h : base heuristic – sum of box‑to‑target distances -----
            # GameState provides a method for this.
            h = child.state.total_box_distance()

            # ----- dead‑lock penalty -----
            if _is_deadlocked(child.state):
                h += 5.0  # heavy penalty to push dead‑ends up

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

        # If for any reason no child was selected, return the current node.
        if best_child is None:
            return node

        # Continue the walk down the tree.
        node = best_child
```
