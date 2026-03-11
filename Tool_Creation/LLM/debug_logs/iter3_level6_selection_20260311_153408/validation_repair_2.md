# validation_repair_2

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:35:20 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.30s |
| Status     | success |
| Tokens     | 1867 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS selection function, but it failed validation.

== BROKEN CODE ==
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

== VALIDATION ERRORS ==
  • Missing ACTION field.
  • Missing FILE_NAME field.
  • Missing FUNCTION_NAME field.

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

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Added required metadata fields and ensured the function signature uses `root` as the first parameter.

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
