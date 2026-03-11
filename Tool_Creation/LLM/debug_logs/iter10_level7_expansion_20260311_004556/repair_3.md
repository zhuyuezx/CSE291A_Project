# repair_3

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:47:41 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 18.49s |
| Status     | success |
| Tokens     | 4405 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand the best untried action using A* visited + richer heuristic
    filtering (real walk cost, progress and wall‑deadlocks).
    """
    # -----------------------------------------------------------------
    # 0. Reset visited table if we are at a new root (new MCTS search).
    # -----------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    visited = _ag.get_visited()

    # -----------------------------------------------------------------
    # 1. Pre‑compute walk distances from current player position.
    # -----------------------------------------------------------------
    walk_map = _bfs_walk(node.state.player,
                         node.state.walls,
                         node.state.boxes)

    # -----------------------------------------------------------------
    # 2. Evaluate each currently untried macro‑push.
    # -----------------------------------------------------------------
    scored = []      # (h, -progress_gain, action, next_state, key, g_new)
    deadlocked = []  # actions to remove permanently (hard deadlocks)

    cur_steps = node.state.steps
    cur_progress = node.state.boxes_on_targets()

    for action in list(node._untried_actions):
        # -------------------------------------------------------------
        # a) Apply action to obtain the successor state.
        # -------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------
        # b) Hard dead‑lock detection (corner + wall).
        # -------------------------------------------------------------
        is_dead = False
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            is_dead = True
        else:
            # locate the box that just moved (the one present in next_state
            # but not in the parent state)
            moved_box = None
            diff = next_state.boxes - node.state.boxes
            if diff:
                moved_box = next(iter(diff))
                if moved_box not in next_state.targets:
                    if _simple_wall_deadlock(next_state, moved_box):
                        is_dead = True

        if is_dead:
            deadlocked.append(action)
            continue

        # -------------------------------------------------------------
        # c) Compute real g‑cost (total steps so far after this push).
        # -------------------------------------------------------------
        push_pos = action[0]                     # where the player must stand
        walk_dist = walk_map.get(push_pos, 10 ** 6)   # large if somehow unreachable
        g_new = cur_steps + walk_dist + 1

        # -------------------------------------------------------------
        # d) Visited check – *soft*: skip this expansion if we have already
        #    reached the same state with an equal or cheaper g.
        # -------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # Do NOT remove the action; a cheaper path might appear later.
            continue

        # -------------------------------------------------------------
        # e) Heuristic score.
        #    h = α·box_distance + β·walk_dist – γ·progress_gain
        # -------------------------------------------------------------
        α, β, γ = 1.0, 0.5, 3.0
        box_dist = next_state.total_box_distance()
        progress_gain = next_state.boxes_on_targets() - cur_progress
        h = α * box_dist + β * walk_dist - γ * progress_gain

        scored.append((h, -progress_gain, action, next_state, key, g_new))

    # -----------------------------------------------------------------
    # 3. Permanently discard actions that were hard dead‑locked.
    # -----------------------------------------------------------------
    for a in deadlocked:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # -----------------------------------------------------------------
    # 4. Choose the most promising action.
    # -----------------------------------------------------------------
    if scored:
        # sort by (h, -progress_gain) → lower h, higher progress first
        scored.sort(key=lambda x: (x[0], x[1]))
        _, _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        # remove the chosen action from the pool
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # -----------------------------------------------------------------
        # Fallback: expand any remaining action (should be non‑deadlocked).
        # -----------------------------------------------------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            # compute its real g for the visited table
            push_pos = chosen_action[0]
            walk_dist = walk_map.get(push_pos, 10 ** 6)
            chosen_g = cur_steps + walk_dist + 1
            visited[chosen_key] = chosen_g
        else:
            # Nothing left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # -----------------------------------------------------------------
    # 5. Create the child node and attach it.
    # -----------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 22, in default_expansion
    from utils import bfs_walk, simple_wall_deadlock   # noqa: E402
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'utils'


== ACTUAL GameState PUBLIC API ==
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

Fix ONLY the broken parts. Keep the heuristic strategy the same.
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

**ACTION:** modify  
**FILE_NAME:** expansion.py  
**FUNCTION_NAME:** default_expansion  
**DESCRIPTION:** Fixed missing imports, corrected API usage, added local BFS walk and simple wall‑deadlock detection, and made the visited‑table handling robust.

```python
def default_expansion(node):
    """
    Expand the best untried macro‑push using a richer heuristic
    (real walk cost, progress and wall‑deadlocks).

    The function now works with the public ``GameState`` API
    (no external ``utils`` module) and safely handles the
    untried‑action container as well as the visited‑state table.
    """
    # -----------------------------------------------------------------
    # 0. Visited‑state table – reset when a new root is encountered.
    # -----------------------------------------------------------------
    # Attach a simple visited dict to the function object itself.
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # Find the root of the current search tree.
    root = node
    while getattr(root, "parent", None) is not None:
        root = root.parent

    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        # New search → clear the visited table.
        default_expansion._visited.clear()
        default_expansion._root_key = root_key

    visited = default_expansion._visited

    # -----------------------------------------------------------------
    # 1. Helper: BFS walk distances from the current player position.
    # -----------------------------------------------------------------
    from collections import deque

    def bfs_walk(start, walls, boxes):
        """
        Return a dict ``pos -> distance`` representing the shortest
        walk (ignoring pushes) from ``start`` to all reachable squares.
        """
        obstacles = walls | boxes
        dist = {start: 0}
        q = deque([start])

        while q:
            x, y = q.popleft()
            d = dist[(x, y)] + 1
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in obstacles:
                    continue
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = d
                    q.append((nx, ny))
        return dist

    # -----------------------------------------------------------------
    # 2. Helper: simple wall‑deadlock test for a single box.
    # -----------------------------------------------------------------
    def simple_wall_deadlock(state, box):
        """
        Detect a classic corner dead‑lock: a box that is not on a target
        and is adjacent to walls on two orthogonal sides.
        """
        if box in state.targets:
            return False

        w = state.walls
        x, y = box
        # check the four corner configurations
        if ((x - 1, y) in w and (x, y - 1) in w) or \
           ((x - 1, y) in w and (x, y + 1) in w) or \
           ((x + 1, y) in w and (x, y - 1) in w) or \
           ((x + 1, y) in w and (x, y + 1) in w):
            return True
        return False

    # -----------------------------------------------------------------
    # 3. Pre‑compute walk distances from the current player position.
    # -----------------------------------------------------------------
    walk_map = bfs_walk(
        node.state.player,
        node.state.walls,
        node.state.boxes,
    )

    # -----------------------------------------------------------------
    # 4. Gather untried actions (handle both possible attribute names).
    # -----------------------------------------------------------------
    untried = getattr(node, "_untried_actions", None)
    if untried is None:
        untried = getattr(node, "untried_actions", [])
    # Work on a copy because we may mutate the original list.
    untried = list(untried)

    scored = []      # (h, -progress_gain, action, next_state, key, g_new)
    deadlocked = []  # actions that are permanently invalid

    cur_steps = node.state.steps
    cur_progress = node.state.boxes_on_targets()

    for action in untried:
        # -------------------------------------------------------------
        # a) Apply the macro‑push to obtain the successor state.
        # -------------------------------------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------
        # b) Hard dead‑lock detection (corner + wall).
        # -------------------------------------------------------------
        is_dead = False
        # If the GameState already implements a dead‑lock detector use it.
        if hasattr(next_state, "_is_deadlocked") and callable(next_state._is_deadlocked):
            if next_state._is_deadlocked():
                is_dead = True
        else:
            # Identify the box that moved (present in child but not in parent).
            moved_box = None
            diff = next_state.boxes - node.state.boxes
            if diff:
                moved_box = next(iter(diff))
                if moved_box not in next_state.targets and simple_wall_deadlock(next_state, moved_box):
                    is_dead = True

        if is_dead:
            deadlocked.append(action)
            continue

        # -------------------------------------------------------------
        # c) Compute the real g‑cost (total steps after this push).
        # -------------------------------------------------------------
        push_pos = action[0]                     # player position required for the push
        walk_dist = walk_map.get(push_pos, 10 ** 6)   # large penalty if unreachable
        g_new = cur_steps + walk_dist + 1               # +1 for the push itself

        # -------------------------------------------------------------
        # d) Visited check – soft: skip if we already reached this state
        #    with an equal or cheaper g.
        # -------------------------------------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            continue

        # -------------------------------------------------------------
        # e) Heuristic score.
        #    h = α·box_distance + β·walk_dist – γ·progress_gain
        # -------------------------------------------------------------
        α, β, γ = 1.0, 0.5, 3.0
        box_dist = next_state.total_box_distance()
        progress_gain = next_state.boxes_on_targets() - cur_progress
        h = α * box_dist + β * walk_dist - γ * progress_gain

        scored.append((h, -progress_gain, action, next_state, key, g_new))

    # -----------------------------------------------------------------
    # 5. Remove permanently dead‑locked actions from the node's pool.
    # -----------------------------------------------------------------
    for a in deadlocked:
        # Try both attribute names.
        if hasattr(node, "_untried_actions"):
            if a in node._untried_actions:
                node._untried_actions.remove(a)
        elif hasattr(node, "untried_actions"):
            if a in node.untried_actions:
                node.untried_actions.remove(a)

    # -----------------------------------------------------------------
    # 6. Choose the most promising action.
    # -----------------------------------------------------------------
    if scored:
        # lower h is better; for equal h we prefer higher progress_gain
        scored.sort(key=lambda x: (x[0], x[1]))
        _, _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]

        # Remove the chosen action from the node's untried list.
        if hasattr(node, "_untried_actions"):
            if chosen_action in node._untried_actions:
                node._untried_actions.remove(chosen_action)
        elif hasattr(node, "untried_actions"):
            if chosen_action in node.untried_actions:
                node.untried_actions.remove(chosen_action)

        visited[chosen_key] = chosen_g
    else:
        # -----------------------------------------------------------------
        # Fallback: expand any remaining (non‑deadlocked) action.
        # -----------------------------------------------------------------
        if untried:
            chosen_action = untried.pop()
            # sync removal with the node's original container
            if hasattr(node, "_untried_actions"):
                if chosen_action in node._untried_actions:
                    node._untried_actions.remove(chosen_action)
            elif hasattr(node, "untried_actions"):
                if chosen_action in node.untried_actions:
                    node.untried_actions.remove(chosen_action)

            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            push_pos = chosen_action[0]
            walk_dist = walk_map.get(push_pos, 10 ** 6)
            chosen_g = cur_steps + walk_dist + 1
            visited[chosen_key] = chosen_g
        else:
            # No actions left – return an existing child or the node itself.
            if getattr(node, "children", None):
                return next(iter(node.children.values()))
            return node

    # -----------------------------------------------------------------
    # 7. Create the child node and attach it.
    # -----------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    # Ensure the children dictionary exists.
    if not hasattr(node, "children"):
        node.children = {}
    node.children[chosen_action] = child
    return child
```

**What was fixed**

1. Removed the nonexistent `utils` import and provided local `bfs_walk` and `simple_wall_deadlock` helpers.  
2. Adapted the code to the actual `GameState` public API (`player`, `walls`, `boxes`, `state_key`, etc.).  
3. Made the visited‑state table robust by storing it as an attribute of the function and resetting it when a new root is detected.  
4. Handled both possible attribute names for untried actions (`_untried_actions` and `untried_actions`).  
5. Guarded against missing `children` attribute and kept the original heuristic logic intact.
