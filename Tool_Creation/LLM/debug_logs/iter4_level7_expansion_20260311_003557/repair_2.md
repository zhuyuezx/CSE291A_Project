# repair_2

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:37:31 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 22.41s |
| Status     | success |
| Tokens     | 5709 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
"""
Improved expansion for Sokoban‑Macro MCTS.

Key upgrades:
  • **Relaxed monotonicity** – allow a small increase (ε) in the
    total‑box‑distance heuristic instead of discarding every non‑monotone
    move. This keeps necessary back‑track pushes.
  • **Accurate g‑score** – use the state's cumulative step count
    (`state.steps`) plus the current walk cost, so the visited‑pruning
    table respects the true cost of long walks.
  • **Weighted heuristic** – combine box‑distance with a penalty for
    boxes that are currently unreachable by the player, encouraging
    pushes that keep boxes reachable.
  • **Extra dead‑lock check** – simple wall‑line dead‑locks (box on a wall
    with no target on that line) are filtered out early.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple, Set

# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine (unchanged interface).
# --------------------------------------------------------------------------- #
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}


def _bfs_distance(start: Tuple[int, int],
                  goal: Tuple[int, int],
                  walls: Set[Tuple[int, int]],
                  boxes: Set[Tuple[int, int]]) -> int:
    """Shortest walk distance ignoring pushes (Manhattan BFS)."""
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


def _simple_corner_deadlock(state) -> bool:
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


def _wall_line_deadlock(state) -> bool:
    """
    Detect simple wall‑line deadlocks:
      * box on a wall (left/right/top/bottom)
      * no target exists on the same line beyond the wall direction
    This catches cases like a box pressed against a wall where it can
    never be moved onto a target.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in state.targets:
            continue
        r, c = b
        # Left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Top wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
        # Bottom wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
    return False


def _reachable_boxes(state) -> int:
    """Number of boxes not reachable by the player (ignoring pushes)."""
    walls = state.walls
    boxes = state.boxes
    start = state.player
    q = deque([start])
    visited = {start}
    while q:
        cur = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)
    # boxes outside the reachable set are “unreachable”
    return sum(1 for b in boxes if b not in visited)


def _heuristic(state,
               alpha: float = 0.7,
               beta: float = 0.3) -> float:
    """
    Weighted heuristic:
        h = α * total_box_distance  +  β * unreachable_boxes
    Both components are integer counts; the weights keep them on a comparable scale.
    """
    box_dist = state.total_box_distance()
    unreachable = _reachable_boxes(state)
    return alpha * box_dist + beta * unreachable


def default_expansion(node):
    """
    Expand the best untried action using an A*‑style f = g + h scoring,
    with improved pruning and cost estimation.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    visited = _EXPANSION_STATE["visited"]

    # ------------------------------------------------------------------ #
    # 1. Gather parent information
    # ------------------------------------------------------------------ #
    parent_steps = getattr(node.state, "steps", 0)          # cumulative steps so far
    parent_h = _heuristic(node.state)                      # weighted heuristic

    # ------------------------------------------------------------------ #
    # 2. Scan untried actions
    # ------------------------------------------------------------------ #
    scored = []          # tuples of (f, walk, action, next_state, key, total_g)
    pruned = []          # actions permanently removed

    EPSILON = 2          # tolerance for non‑monotone moves

    for action in list(node._untried_actions):
        # ---- a) generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- b) dead‑lock checks ----------------------------------------
        if _simple_corner_deadlock(next_state) or _wall_line_deadlock(next_state):
            pruned.append(action)
            continue

        # ---- c) heuristic monotonicity (relaxed) -----------------------
        next_h = _heuristic(next_state)
        if next_h > parent_h + EPSILON:
            pruned.append(action)
            continue

        # ---- d) walk distance to push position -------------------------
        player_start = node.state.player
        push_pos = action[0]                 # cell where player must stand
        walk = _bfs_distance(player_start, push_pos,
                             node.state.walls, node.state.boxes)

        # ---- e) g‑score: accumulate real steps (push count + walk) -----
        total_g = parent_steps + walk + 1    # +1 for the push itself

        # ---- f) visited‑table pruning ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # ---- g) compute f = g + h ---------------------------------------
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose best action (minimal f, then minimal walk, then deterministic)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ---------- fallback when everything was pruned -----------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            walk = _bfs_distance(node.state.player, chosen_action[0],
                                 node.state.walls, node.state.boxes)
            total_g = parent_steps + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create child node, register, and return
    # ------------------------------------------------------------------ #
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
    if _EXPANSION_STATE["root_key"] != root_key:
       ^^^^^^^^^^^^^^^^
NameError: name '_EXPANSION_STATE' is not defined


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
**DESCRIPTION:** Added missing global `_EXPANSION_STATE` definition and made the function robust to the exact `GameState` public API (uses `player` attribute or `current_player()` method, and guards against missing `steps` attribute).

```python
"""
Improved expansion for Sokoban‑Macro MCTS.

Key upgrades:
  • **Relaxed monotonicity** – allow a small increase (ε) in the
    total‑box‑distance heuristic instead of discarding every non‑monotone
    move. This keeps necessary back‑track pushes.
  • **Accurate g‑score** – use the state's cumulative step count
    (`state.steps`) plus the current walk cost, so the visited‑pruning
    table respects the true cost of long walks.
  • **Weighted heuristic** – combine box‑distance with a penalty for
    boxes that are currently unreachable by the player, encouraging
    pushes that keep boxes reachable.
  • **Extra dead‑lock check** – simple wall‑line dead‑locks (box on a wall
    with no target on that line) are filtered out early.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple, Set

# --------------------------------------------------------------------------- #
# Module‑level state for the expansion routine (unchanged interface).
# --------------------------------------------------------------------------- #
# NOTE: In the original snippet this variable was defined, but the
# importing structure of the project caused it to be missing at runtime.
# We (re)define it here to guarantee availability.
_EXPANSION_STATE = {
    "root_key": None,   # state_key of the current search root
    "visited": {}       # maps state_key -> best g‑score seen so far
}


def _bfs_distance(start: Tuple[int, int],
                  goal: Tuple[int, int],
                  walls: Set[Tuple[int, int]],
                  boxes: Set[Tuple[int, int]]) -> int:
    """Shortest walk distance ignoring pushes (Manhattan BFS)."""
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


def _simple_corner_deadlock(state) -> bool:
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


def _wall_line_deadlock(state) -> bool:
    """
    Detect simple wall‑line deadlocks:
      * box on a wall (left/right/top/bottom)
      * no target exists on the same line beyond the wall direction
    This catches cases like a box pressed against a wall where it can
    never be moved onto a target.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in state.targets:
            continue
        r, c = b
        # Left wall
        if (r, c - 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Right wall
        if (r, c + 1) in walls:
            if not any((r, tc) in targets for tc in range(state.width)):
                return True
        # Top wall
        if (r - 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
        # Bottom wall
        if (r + 1, c) in walls:
            if not any((tr, c) in targets for tr in range(state.height)):
                return True
    return False


def _reachable_boxes(state) -> int:
    """Number of boxes not reachable by the player (ignoring pushes)."""
    walls = state.walls
    boxes = state.boxes
    # The GameState may expose the player position as an attribute or via a method.
    start = getattr(state, "player", None)
    if start is None and callable(getattr(state, "current_player", None)):
        start = state.current_player()
    if start is None:
        raise ValueError("Unable to obtain player position from state.")
    q = deque([start])
    visited = {start}
    while q:
        cur = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)
    # boxes outside the reachable set are “unreachable”
    return sum(1 for b in boxes if b not in visited)


def _heuristic(state,
               alpha: float = 0.7,
               beta: float = 0.3) -> float:
    """
    Weighted heuristic:
        h = α * total_box_distance  +  β * unreachable_boxes
    Both components are integer counts; the weights keep them on a comparable scale.
    """
    box_dist = state.total_box_distance()
    unreachable = _reachable_boxes(state)
    return alpha * box_dist + beta * unreachable


def default_expansion(node):
    """
    Expand the best untried action using an A*‑style f = g + h scoring,
    with improved pruning and cost estimation.
    """
    # ------------------------------------------------------------------ #
    # 0. Reset visited table if this is a fresh root (new MCTS search)
    # ------------------------------------------------------------------ #
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()

    global _EXPANSION_STATE
    if _EXPANSION_STATE["root_key"] != root_key:
        _EXPANSION_STATE["visited"] = {}
        _EXPANSION_STATE["root_key"] = root_key

    visited = _EXPANSION_STATE["visited"]

    # ------------------------------------------------------------------ #
    # 1. Gather parent information
    # ------------------------------------------------------------------ #
    # The GameState may expose steps as an attribute; default to 0 if missing.
    parent_steps = getattr(node.state, "steps", 0)
    parent_h = _heuristic(node.state)                      # weighted heuristic

    # ------------------------------------------------------------------ #
    # 2. Scan untried actions
    # ------------------------------------------------------------------ #
    scored = []          # tuples of (f, walk, action, next_state, key, total_g)
    pruned = []          # actions permanently removed

    EPSILON = 2          # tolerance for non‑monotone moves

    for action in list(node._untried_actions):
        # ---- a) generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- b) dead‑lock checks ----------------------------------------
        if _simple_corner_deadlock(next_state) or _wall_line_deadlock(next_state):
            pruned.append(action)
            continue

        # ---- c) heuristic monotonicity (relaxed) -----------------------
        next_h = _heuristic(next_state)
        if next_h > parent_h + EPSILON:
            pruned.append(action)
            continue

        # ---- d) walk distance to push position -------------------------
        # player start position (attribute or method)
        player_start = getattr(node.state, "player", None)
        if player_start is None and callable(getattr(node.state, "current_player", None)):
            player_start = node.state.current_player()
        if player_start is None:
            raise ValueError("Unable to obtain player position from state.")

        push_pos = action[0]                 # cell where player must stand
        walk = _bfs_distance(player_start, push_pos,
                             node.state.walls, node.state.boxes)

        # ---- e) g‑score: accumulate real steps (push count + walk) -----
        total_g = parent_steps + walk + 1    # +1 for the push itself

        # ---- f) visited‑table pruning ----------------------------------
        key = next_state.state_key()
        if key in visited and visited[key] <= total_g:
            pruned.append(action)
            continue

        # ---- g) compute f = g + h ---------------------------------------
        f_score = total_g + next_h
        scored.append((f_score, walk, action, next_state, key, total_g))

    # ------------------------------------------------------------------ #
    # 3. Remove permanently pruned actions from the node's pool
    # ------------------------------------------------------------------ #
    for a in pruned:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------ #
    # 4. Choose best action (minimal f, then minimal walk, then deterministic)
    # ------------------------------------------------------------------ #
    if scored:
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        f_score, walk, chosen_action, chosen_state, chosen_key, total_g = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = total_g
    else:
        # ---------- fallback when everything was pruned -----------------
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            # Re‑compute walk distance for fallback
            player_start = getattr(node.state, "player", None)
            if player_start is None and callable(getattr(node.state, "current_player", None)):
                player_start = node.state.current_player()
            walk = _bfs_distance(player_start, chosen_action[0],
                                 node.state.walls, node.state.boxes)
            total_g = parent_steps + walk + 1
            visited[chosen_key] = total_g
        else:
            # No actions left – return an existing child or the node itself
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------ #
    # 5. Create child node, register, and return
    # ------------------------------------------------------------------ #
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
