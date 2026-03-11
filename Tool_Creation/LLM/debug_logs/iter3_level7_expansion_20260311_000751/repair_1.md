# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:09:03 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 20.66s |
| Status     | success |
| Tokens     | 4762 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand the best untried action using A* visited + heuristic filtering,
    now taking the actual walk cost into account and discarding actions that
    increase the heuristic or create simple corner deadlocks.

    Steps:
      0. Reset shared A* visited if this is a new root.
      1. Compute parent depth and parent heuristic.
      2. For every untried action:
           • generate next_state
           • prune if a simple corner deadlock is detected
           • prune if next_state's heuristic > parent heuristic (non‑monotonic)
           • compute walk cost from current player pos to the required push cell
           • total_g = parent_depth + walk_cost + 1   (push step)
           • prune if visited[key] <= total_g
           • score by f = total_g + h(next_state)
      3. Choose the action with minimal f (tie‑break on smaller walk cost).
      4. Fallback to first remaining untried action if everything was pruned.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset A* visited if we have a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    from collections import deque

    def bfs_distance(start, goal, walls, boxes):
        """Return shortest distance avoiding walls & boxes (Manhattan BFS)."""
        if start == goal:
            return 0
        q = deque([start])
        visited = {start}
        dist = {start: 0}
        while q:
            cur = q.popleft()
            d = dist[cur]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in walls or nxt in boxes or nxt in visited:
                    continue
                if nxt == goal:
                    return d + 1
                visited.add(nxt)
                dist[nxt] = d + 1
                q.append(nxt)
        # Should not happen for legal actions; return a large penalty
        return 10**6

    def simple_corner_deadlock(state):
        """Detect obvious corner deadlocks (box in a wall corner not on target)."""
        walls = state.walls
        targets = state.targets
        for b in state.boxes:
            if b in targets:
                continue
            r, c = b
            # four possible corner configurations
            if ((r - 1, c) in walls and (r, c - 1) in walls) or \
               ((r - 1, c) in walls and (r, c + 1) in walls) or \
               ((r + 1, c) in walls and (r, c - 1) in walls) or \
               ((r + 1, c) in walls and (r, c + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------ #
    # 1. Parent information
    # ------------------------------------------------------------------ #
    parent_depth = _ag.node_depth(node)          # number of pushes so far
    # safe heuristic lookup (full h if available, else box‑only h)
    if hasattr(_ag, "h_sokoban"):
        parent_h = _ag.h_sokoban(node.state)
    else:
        parent_h = _ag.h_sokoban_box_only(node.state)

    visited = _ag.get_visited()

    scored = []   # (f, walk, action, next_state, key, total_g)
    pruned = []   # actions to permanently discard

    for action in list(node._untried_actions):
        # -------------------------------------------------------------- #
        # Generate successor state
        # -------------------------------------------------------------- #
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------- #
        # a) Simple corner deadlock pruning
        # -------------------------------------------------------------- #
        if simple_corner_deadlock(next_state):
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # b) Existing deadlock method, if present
        # -------------------------------------------------------------- #
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # c) Monotonicity: discard actions that raise the heuristic
        # -------------------------------------------------------------- #
        if hasattr(_ag, "h_sokoban"):
            next_h = _ag.h_sokoban(next_state)
        else:
            next_h = _ag.h_sokoban_box_only(next_state)

        if next_h > parent_h:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # d) Compute walk distance from current player to the push position
        # -------------------------------------------------------------- #
        player_start = node.state.player
        push_pos = action[0]                     # cell where player must stand
        walk = bfs_distance(player_start, push_pos,
                            node.state.walls, node.state.boxes)

        # -------------------------------------------------------------- #
        # e) Total g‑score for visited table (push count + walk + 1)
        # -------------------------------------------------------------- #
        total_g = parent_depth + walk + 1

        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # f) Score by full A* f = g + h
        # -------------------------------------------------------------- #
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 2. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 3. Choose best action (minimal f, then minimal walk, then action order)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ------------------------------------------------------------------
        # Fallback: expand the first remaining untried action (if any)
        # ------------------------------------------------------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = bfs_distance(node.state.player, chosen_action[0],
                                node.state.walls, node.state.boxes)
            total_g = parent_depth + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 4. Create child node and register it
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 392, in _smoke_test
    result = fn(*test_args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 34, in default_expansion
    if getattr(_ag, "_root_key", None) != root_key:
               ^^^
NameError: name '_ag' is not defined


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
**DESCRIPTION:** Fixed the undefined `_ag` reference by introducing module‑level globals (`_EXPANSION_STATE`) that store the current root key and visited table, added a local `node_depth` helper, and replaced heuristic calls with the public `total_box_distance` method. All other logic (corner dead‑lock pruning, monotonicity, walk‑cost computation, A*‑style scoring) remains unchanged.

```python
# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine.
# This replaces the previous undefined `_ag` global that was used for:
#   * root‑key tracking & visited‑table reset
#   * node depth lookup
#   * heuristic functions
# --------------------------------------------------------------------------- #
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}

def default_expansion(node):
    """
    Expand the best untried action using A*‑style scoring while respecting
    actual walk costs and simple dead‑lock / monotonicity pruning.

    The implementation mirrors the original strategy but no longer relies on
    the undefined `_ag` object.  All required information is obtained from the
    public `GameState` API or from the module‑level `_EXPANSION_STATE`.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    # Find the root of the current tree
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        # New search – start with a clean visited dict
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    from collections import deque

    def bfs_distance(start, goal, walls, boxes):
        """Shortest walk distance (ignoring pushes) using Manhattan BFS."""
        if start == goal:
            return 0
        q = deque([start])
        visited = {start}
        dist = {start: 0}
        while q:
            cur = q.popleft()
            d = dist[cur]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in walls or nxt in boxes or nxt in visited:
                    continue
                if nxt == goal:
                    return d + 1
                visited.add(nxt)
                dist[nxt] = d + 1
                q.append(nxt)
        # Unreachable – assign a large penalty
        return 10 ** 6

    def simple_corner_deadlock(state):
        """Detect obvious corner deadlocks (box stuck in a wall corner)."""
        walls = state.walls
        targets = state.targets
        for b in state.boxes:
            if b in targets:
                continue
            r, c = b
            if ((r - 1, c) in walls and (r, c - 1) in walls) or \
               ((r - 1, c) in walls and (r, c + 1) in walls) or \
               ((r + 1, c) in walls and (r, c - 1) in walls) or \
               ((r + 1, c) in walls and (r, c + 1) in walls):
                return True
        return False

    def node_depth(n):
        """Number of pushes from the root to this node."""
        d = 0
        while n.parent is not None:
            d += 1          # each edge corresponds to one push
            n = n.parent
        return d

    # ------------------------------------------------------------------ #
    # 1. Parent information (depth & heuristic)
    # ------------------------------------------------------------------ #
    parent_depth = node_depth(node)

    # Heuristic: use the public `total_box_distance` method (sum of Manhattan
    # distances from boxes to the nearest targets).  It is available on every
    # GameState object according to the provided API.
    try:
        parent_h = node.state.total_box_distance()
    except Exception:
        # Fallback – if for some reason the method is missing, use 0.
        parent_h = 0

    visited = _EXPANSION_STATE["visited"]

    scored = []   # (f, walk, action, next_state, key, total_g)
    pruned = []   # actions we permanently discard from the node's pool

    # ------------------------------------------------------------------ #
    # 2. Examine every untried action
    # ------------------------------------------------------------------ #
    for action in list(node._untried_actions):
        # -------------------------------------------------------------- #
        # a) Generate successor state
        # -------------------------------------------------------------- #
        next_state = node.state.clone()
        next_state.apply_action(action)

        # -------------------------------------------------------------- #
        # b) Simple corner deadlock pruning
        # -------------------------------------------------------------- #
        if simple_corner_deadlock(next_state):
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # c) Existing deadlock check (if the state implements it)
        # -------------------------------------------------------------- #
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # d) Monotonicity: discard actions that increase the heuristic
        # -------------------------------------------------------------- #
        try:
            next_h = next_state.total_box_distance()
        except Exception:
            next_h = 0
        if next_h > parent_h:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # e) Compute walk distance from current player to the required push cell
        # -------------------------------------------------------------- #
        player_start = node.state.player
        push_pos = action[0]                     # cell where player must stand
        walk = bfs_distance(player_start, push_pos,
                            node.state.walls, node.state.boxes)

        # -------------------------------------------------------------- #
        # f) Total g‑score for visited table (push count + walk + 1)
        # -------------------------------------------------------------- #
        total_g = parent_depth + walk + 1

        # -------------------------------------------------------------- #
        # g) Visited‑table pruning
        # -------------------------------------------------------------- #
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # -------------------------------------------------------------- #
        # h) Score by full A* f = g + h
        # -------------------------------------------------------------- #
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose the best remaining action (minimal f, then minimal walk)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        # Remove the selected action from the untried list
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ------------------------------------------------------------------
        # Fallback: expand the first remaining untried action (if any)
        # ------------------------------------------------------------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = bfs_distance(node.state.player, chosen_action[0],
                                node.state.walls, node.state.boxes)
            total_g = parent_depth + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create the child node, register it and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
