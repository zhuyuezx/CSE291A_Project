# repair_2

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:35:45 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.44s |
| Status     | success |
| Tokens     | 3227 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS selection function, but it raised a runtime error.

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

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 470, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/selection/selection.py", line 76, in default_selection
    if not node.is_fully_expanded():
           ~~~~~~~~~~~~~~~~~~~~~~^^
TypeError: 'bool' object is not callable


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

**Description:** Fixed the runtime error caused by treating `is_fully_expanded` as a method, removed undefined external references, and added all required helper functions (heuristic, dead‑lock detection, and constants) inside the function while preserving the original selection strategy.

```python
def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    Parameters
    ----------
    root : MCTSNode
        The current node from which the walk starts (may be any node in the tree).
    exploration_weight : float, optional
        Weight applied to the UCB exploration term (default is √2 ≈ 1.41).

    Returns
    -------
    MCTSNode
        The selected node for expansion or simulation.
    """
    import math

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0   # large penalty to discourage dead‑locked states
    TARGET_BONUS    = 10.0       # reward for moving a box onto a target

    def heuristic(state):
        """
        Base heuristic: total Manhattan distance from every box to the
        nearest target (the GameState already provides this via
        `total_box_distance`).
        """
        # The method exists in the public API.
        return state.total_box_distance()

    def is_deadlocked(state):
        """
        Very cheap dead‑lock detection: a box that is not on a target
        and is stuck in a corner formed by two orthogonal walls.
        """
        walls = state.walls
        for box in state.boxes:
            if box in state.targets:
                continue
            x, y = box
            # check four possible corner configurations
            if ((x - 1, y) in walls and (x, y - 1) in walls) or \
               ((x - 1, y) in walls and (x, y + 1) in walls) or \
               ((x + 1, y) in walls and (x, y - 1) in walls) or \
               ((x + 1, y) in walls and (x, y + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # 2️⃣  Tree walk – always pick the child with minimal f‑value
    # ------------------------------------------------------------------
    node = root
    while not node.state.is_terminal():
        # A node is **fully expanded** when there are no untried actions left.
        fully_expanded = (len(node._untried_actions) == 0)

        if not fully_expanded:
            # Let the expansion phase create a fresh child.
            return node

        best_child = None
        best_f = math.inf
        best_q = -math.inf

        # Guard against division‑by‑zero when the parent has never been visited.
        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: heuristic + dead‑lock penalty – target bonus -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: boxes placed on targets relative to the parent.
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- Exploration term (UCB) -----
            if child.visits == 0:
                # Unvisited nodes are given the most attractive (lowest) f.
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore  # lower f is better

            # ----- Tie‑break on average value (q) -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        # Safety net – should never happen, but avoids an infinite loop.
        if best_child is None:
            break

        node = best_child

    return node
```
