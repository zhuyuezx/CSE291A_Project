# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:06:37 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 17.44s |
| Status     | success |
| Tokens     | 4907 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand the best untried action using richer scoring and safer pruning.

    Improvements over the original version:
      * Incorporates the player‑walk distance to the push position
        (penalised with WALK_WEIGHT) so long‑range pushes are deprioritised.
      * Adds a heavy penalty when a push would move a box off a target
        (TARGET_MOVE_PENALTY) – protects already‑solved boxes.
      * Performs a cheap corner‑dead‑lock test on the resulting state;
        pushes that create a box in a non‑target corner are pruned.
      * Visited‑g pruning no longer permanently removes the action from
        ``_untried_actions`` – actions are only skipped for this expansion,
        preserving breadth for later reconsideration.

    The rest of the A*‑style visited handling and deterministic ordering
    remains unchanged.
    """
    # ------------------------------------------------------------------
    # Reset shared A* visited table if we have entered a new MCTS search.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper utilities – everything defined here to keep the function
    # self‑contained.
    # ------------------------------------------------------------------
    from collections import deque

    # Directions: (dr, dc) matching the action encoding 0=UP,1=DOWN,2=LEFT,3=RIGHT
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    WALK_WEIGHT = 0.5          # importance of walk‑cost in the final score
    TARGET_MOVE_PENALTY = 5.0  # penalty for moving a box off a target
    # (no explicit corner penalty needed; we simply prune such actions)

    def bfs_distances(state):
        """Return a dict mapping each reachable cell to its shortest walk distance."""
        walls = state.walls
        boxes = state.boxes
        start = state.player
        visited = {start: 0}
        q = deque([start])
        while q:
            r, c = q.popleft()
            d = visited[(r, c)] + 1
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in walls or (nr, nc) in boxes:
                    continue
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = d
                    q.append((nr, nc))
        return visited

    def is_corner(pos, walls_set):
        """Detect a static corner (two orthogonal walls)."""
        r, c = pos
        return ((r + 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r + 1, c) in walls_set and (r, c - 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c - 1) in walls_set)

    # ------------------------------------------------------------------
    # Pre‑compute walk distances once for the current state (before any push).
    # ------------------------------------------------------------------
    walk_map = bfs_distances(node.state)

    g_new   = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    scored      = []  # list of tuples (score, action, next_state, key)
    dead_actions = []  # actions that are truly deadlocked and must be deleted

    for action in list(node._untried_actions):
        push_pos, direction = action
        dr, dc = DIRS[direction]

        # ------------------------------------------------------------------
        # Walk cost: distance from player to the required push position.
        # If unreachable (should not happen for legal actions) we skip.
        # ------------------------------------------------------------------
        walk_cost = walk_map.get(push_pos, None)
        if walk_cost is None:
            # unreachable – treat as deadlocked for safety
            dead_actions.append(action)
            continue

        # ------------------------------------------------------------------
        # Apply the action to obtain the successor state.
        # ------------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ------------------------------------------------------------------
        # 1️⃣  Dead‑lock pruning – original check (if any) plus cheap corner test.
        # ------------------------------------------------------------------
        dead = False
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            dead = True
        else:
            # cheap corner test on all boxes after the push
            for b in next_state.boxes:
                if b in next_state.targets:
                    continue          # target boxes are safe
                if is_corner(b, next_state.walls):
                    dead = True
                    break
        if dead:
            dead_actions.append(action)
            continue

        # ------------------------------------------------------------------
        # 2️⃣  Visited‑g check – only *skip* for this expansion, do NOT delete.
        # ------------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # skip this action for now; keep it for possible later use
            continue

        # ------------------------------------------------------------------
        # 3️⃣  Scoring – combine box‑only heuristic, walk cost and target‑move penalty.
        # ------------------------------------------------------------------
        h_box = _ag.h_sokoban_box_only(next_state)

        # Detect "moving a box off a target"
        box_before = (push_pos[0] + dr, push_pos[1] + dc)   # original box location
        box_after  = (box_before[0] + dr, box_before[1] + dc)  # where it ends up
        target_move_pen = TARGET_MOVE_PENALTY if (
            box_before in node.state.targets and box_after not in node.state.targets
        ) else 0.0

        score = h_box + WALK_WEIGHT * walk_cost + target_move_pen

        scored.append((score, action, next_state, key))

    # ------------------------------------------------------------------
    # Remove actions that are certainly deadlocked.
    # ------------------------------------------------------------------
    for a in dead_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # Choose the best scored action, if any.
    # ------------------------------------------------------------------
    if scored:
        # lower score → more promising
        scored.sort(key=lambda x: x[0])
        _, chosen_action, chosen_state, chosen_key = scored[0]
        # permanently remove the chosen action from the untried pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # No action survived scoring (all skipped by visited check or deadlocked).
        # Fallback: expand the first remaining untried action, if any.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # Nothing left – return an existing child if we have one,
            # otherwise return the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # Create the child node and attach it.
    # ------------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 32, in default_expansion
    if getattr(_ag, "_root_key", None) != root_key:
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
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**Fixed issue:** Added the missing shared A*‑style visited table (`_visited`) and root‑key handling, replaced the undefined `_ag` helper with local equivalents, and used the public `total_box_distance` heuristic. All helper utilities are now defined inside the function and no external names are referenced.

```python
def default_expansion(node):
    """
    Expand the best untried action using richer scoring and safer pruning.

    The implementation is self‑contained:
      * A module‑level visited‑g table (`_visited`) is created lazily and
        cleared whenever we start a new MCTS search (detected by a change
        of the root state's key).
      * Node depth is computed locally.
      * The box‑only heuristic uses the public `total_box_distance` method.
      * All helpers (BFS distance, corner test, etc.) are defined inside the
        function, so no external undefined names are used.
    """
    # --------------------------------------------------------------
    # Global visited‑g table – created on first call and reset on a new
    # root state.  This mimics the original _ag behaviour without relying
    # on an undefined object.
    # --------------------------------------------------------------
    global _visited, _root_key
    try:
        _visited
    except NameError:          # first invocation
        _visited = {}
    try:
        _root_key
    except NameError:
        _root_key = None

    # ------------------------------------------------------------------
    # Determine the root of the current search tree.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if _root_key != root_key:          # a fresh search – clear the table
        _visited.clear()
        _root_key = root_key

    # ------------------------------------------------------------------
    # Helper utilities – everything defined here to keep the function
    # self‑contained.
    # ------------------------------------------------------------------
    from collections import deque

    # Directions: (dr, dc) matching the action encoding 0=UP,1=DOWN,2=LEFT,3=RIGHT
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    WALK_WEIGHT = 0.5          # importance of walk‑cost in the final score
    TARGET_MOVE_PENALTY = 5.0  # penalty for moving a box off a target

    def bfs_distances(state):
        """Return a dict mapping each reachable cell to its shortest walk distance."""
        walls = state.walls
        boxes = state.boxes
        start = state.player
        visited = {start: 0}
        q = deque([start])
        while q:
            r, c = q.popleft()
            d = visited[(r, c)] + 1
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in walls or (nr, nc) in boxes:
                    continue
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = d
                    q.append((nr, nc))
        return visited

    def is_corner(pos, walls_set):
        """Detect a static corner (two orthogonal walls)."""
        r, c = pos
        return ((r + 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r + 1, c) in walls_set and (r, c - 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c + 1) in walls_set) or \
               ((r - 1, c) in walls_set and (r, c - 1) in walls_set)

    def node_depth(n):
        """Number of edges from the root to *n*."""
        d = 0
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ------------------------------------------------------------------
    # Pre‑compute walk distances once for the current state (before any push).
    # ------------------------------------------------------------------
    walk_map = bfs_distances(node.state)

    g_new = node_depth(node) + 1          # g‑cost of the successor
    scored = []        # (score, action, next_state, key)
    dead_actions = []  # actions that are definitely dead‑locked

    for action in list(node._untried_actions):
        # Expected action format: (push_position, direction)
        push_pos, direction = action
        dr, dc = DIRS[direction]

        # --------------------------------------------------------------
        # Walk cost: distance from player to the required push position.
        # --------------------------------------------------------------
        walk_cost = walk_map.get(push_pos)
        if walk_cost is None:                 # unreachable – treat as dead
            dead_actions.append(action)
            continue

        # --------------------------------------------------------------
        # Generate the successor state.
        # --------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # --------------------------------------------------------------
        # 1️⃣  Dead‑lock pruning – cheap corner test (no external _is_deadlocked).
        # --------------------------------------------------------------
        dead = False
        for b in next_state.boxes:
            if b in next_state.targets:
                continue          # safe: already on a target
            if is_corner(b, next_state.walls):
                dead = True
                break
        if dead:
            dead_actions.append(action)
            continue

        # --------------------------------------------------------------
        # 2️⃣  Visited‑g check – skip for this expansion only.
        # --------------------------------------------------------------
        key = next_state.state_key()
        if key in _visited and _visited[key] <= g_new:
            # skip this action for now; keep it for later reconsideration
            continue

        # --------------------------------------------------------------
        # 3️⃣  Scoring – box heuristic + walk cost + target‑move penalty.
        # --------------------------------------------------------------
        # Box‑only heuristic: use the public total_box_distance method.
        h_box = next_state.total_box_distance()

        # Detect moving a box off a target.
        box_before = (push_pos[0] + dr, push_pos[1] + dc)          # original box location
        box_after = (box_before[0] + dr, box_before[1] + dc)       # location after push
        target_move_pen = (TARGET_MOVE_PENALTY
                           if (box_before in node.state.targets and
                               box_after not in node.state.targets)
                           else 0.0)

        score = h_box + WALK_WEIGHT * walk_cost + target_move_pen
        scored.append((score, action, next_state, key))

    # --------------------------------------------------------------
    # Remove actions that are certainly dead‑locked.
    # --------------------------------------------------------------
    for a in dead_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # Choose the best‑scored action, if any.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: x[0])      # lower score is better
        _, chosen_action, chosen_state, chosen_key = scored[0]
        # permanently remove the chosen action from the untried pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        _visited[chosen_key] = g_new
    else:
        # No viable actions after scoring – fall back to any remaining action.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            _visited[chosen_key] = g_new
        else:
            # No actions left; return an existing child if possible, else the node.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # Create the child node and attach it.
    # --------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
