# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:09:23 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.26s |
| Status     | success |
| Tokens     | 3974 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements:
      * Uses true macro‑push cost (walk steps + 1) as g‑score for visited
        pruning.
      * Orders actions by a richer heuristic:
          h = box_distance + α * walk_distance   (α = 0.5)
      * Performs lightweight static dead‑lock detection and permanently
        removes those actions.
      * Does **not** permanently discard actions that fail the visited check;
        they stay in `_untried_actions` for future attempts.
      * In the fallback case, the visited map is only updated if the new
        g‑score is better than any previously recorded one.
    """
    # --------------------------------------------------------------
    # 0. Reset shared A* visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # --------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # --------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)
    visited = _ag.get_visited()

    # --------------------------------------------------------------
    # 2. Scan untried actions.
    # --------------------------------------------------------------
    alpha = 0.5  # weight for walk distance in the heuristic
    scored = []  # (h, action, next_state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action

        # walk distance from current player location to the required position
        walk_len = _bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- a) static dead‑lock detection (permanent) -----
        if (hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked()) \
                or _is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ----- b) visited pruning (temporary) -----
        g_new = cur_steps + walk_len + 1
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for possible later expansion
            continue

        # ----- c) heuristic scoring -----
        box_h = _ag.h_sokoban_box_only(next_state)   # box‑only Manhattan distance
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h).  If none, fall back.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        h_best, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No scored actions – fall back to any remaining untried action.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            player_pos, _ = chosen_action
            walk_len = _bfs_walk_len(node.state, node.state.player, player_pos)
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            chosen_g = cur_steps + walk_len + 1
            # update visited only if this g is an improvement
            if (chosen_key not in visited) or (visited[chosen_key] > chosen_g):
                visited[chosen_key] = chosen_g
        else:
            # No actions left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # 4. Create and register the child node.
    # --------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    actions = prev.legal_actions()
                 ^^^^^^^^^
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 30, in default_expansion
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

**Fixed issue** – the original implementation referenced undefined globals (`_ag`, `_bfs_walk_len`, `_is_simple_deadlocked`).  
**Solution** – all required helpers are now defined locally inside `default_expansion`. A per‑root visited map is stored as function attributes, BFS computes walk distances, a tiny static dead‑lock detector checks corner boxes, and the original heuristic strategy is preserved.

```python
def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements (preserved):
      * Uses true macro‑push cost (walk steps + 1) as g‑score for visited
        pruning.
      * Orders actions by a richer heuristic:
          h = box_distance + α * walk_distance   (α = 0.5)
      * Performs lightweight static dead‑lock detection and permanently
        removes those actions.
      * Does **not** permanently discard actions that fail the visited check;
        they stay in `_untried_actions` for future attempts.
      * In the fallback case, the visited map is only updated if the new
        g‑score is better than any previously recorded one.
    """
    import sys
    from collections import deque
    from mcts.node import MCTSNode

    # ------------------------------------------------------------------
    # Helper 1: BFS walk length (player moves only, boxes are static)
    # ------------------------------------------------------------------
    def bfs_walk_len(state, start, goal):
        """Return shortest walk length from start to goal avoiding walls & boxes.
        If unreachable, return sys.maxsize."""
        if start == goal:
            return 0
        walls = state.walls
        boxes = state.boxes
        width, height = state.width, state.height

        # simple 4‑direction moves
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in walls or (nx, ny) in boxes:
                    continue
                if (nx, ny) == goal:
                    return nd
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, nd))
        return sys.maxsize

    # ------------------------------------------------------------------
    # Helper 2: Very cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def is_simple_deadlocked(state):
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        # pre‑compute wall adjacency for quick checks
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # check four corner patterns
            if ((bx + 1, by) in walls or (bx - 1, by) in walls) and \
               ((bx, by + 1) in walls or (bx, by - 1) in walls):
                # Two perpendicular walls -> corner
                # Need both a horizontal and a vertical wall adjacent
                horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
                vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
                if horiz and vert:
                    return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Per‑root visited dictionary (persistent across calls)
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # --------------------------------------------------------------
    # 0. Reset shared visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._root_key = root_key

    visited = default_expansion._visited

    # --------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # --------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # --------------------------------------------------------------
    # 2. Scan untried actions.
    # --------------------------------------------------------------
    alpha = 0.5                     # weight for walk distance in the heuristic
    scored = []                     # (h, action, next_state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action

        # a) walk distance from current player location to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- static dead‑lock detection (permanent) -----
        if is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ----- visited pruning (temporary) -----
        g_new = cur_steps + walk_len + 1   # real cost to reach this macro‑state
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for potential later expansion
            continue

        # ----- heuristic scoring -----
        # box‑only Manhattan distance (provided by GameState)
        box_h = next_state.total_box_distance()
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h). If none, fall back.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No scored actions – fall back to any remaining untried action.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            player_pos, _ = chosen_action
            walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            chosen_g = cur_steps + walk_len + 1
            if (chosen_key not in visited) or (visited[chosen_key] > chosen_g):
                visited[chosen_key] = chosen_g
        else:
            # No actions left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # 4. Create and register the child node.
    # --------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

**Description of fix:**  
Replaced the undefined globals with self‑contained helpers (`bfs_walk_len`, `is_simple_deadlocked`) and introduced a per‑root visited dictionary stored as attributes of `default_expansion`. The function now runs without external dependencies while preserving the original heuristic and pruning logic.
