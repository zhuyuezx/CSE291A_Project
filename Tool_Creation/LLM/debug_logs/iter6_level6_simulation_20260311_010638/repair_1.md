# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:07:22 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.46s |
| Status     | success |
| Tokens     | 2367 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS simulation function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    h = _ag.h_sokoban_box_only(state)   # sum of distances box→nearest target

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = _bfs_distances(state.player, walls, boxes, height, width)

    min_walk = None
    for (push_pos, _dir) in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    f = state.boxes_on_targets() / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # penalty for long walks
    GAMMA = 0.4   # stronger bonus for having boxes placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    if reward > 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    # Special case: all boxes on targets → perfect reward.
    if h == 0:
        reward = 1.0

    return reward
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/simulation/simulation.py", line 26, in default_simulation
    h = _ag.h_sokoban_box_only(state)   # sum of distances box→nearest target
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

CRITICAL: The function must be SELF-CONTAINED. Define ALL helper functions inline. Do NOT use undefined names (e.g. _ag, _bfs_distance). Fix ONLY the broken parts. Keep the heuristic strategy the same.
Return using the SAME structured format.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**Fixed**: Imported missing helper functions (`_ag.h_sokoban_box_only` and `_bfs_distances`) by implementing them inline, using only the public `GameState` API. The function now computes the box‑only Manhattan distance via `state.total_box_distance()` and performs a BFS to obtain player walk costs to all reachable squares.

```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """
        Compute the shortest‑path distance from `start` to every reachable
        cell on the board, treating both walls and boxes as obstacles.
        Returns a dict {(r, c): distance}.
        """
        frontier = deque([start])
        distances = {start: 0}
        # 4‑neighbour moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while frontier:
            cur = frontier.popleft()
            cur_dist = distances[cur]

            for dr, dc in moves:
                nr, nc = cur[0] + dr, cur[1] + dc
                # stay inside the board
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                # cannot step onto a wall or a box
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in distances:
                    distances[nxt] = cur_dist + 1
                    frontier.append(nxt)
        return distances

    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # `returns()` returns a sequence indexed by player id.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    # The public API provides `total_box_distance()` which returns the
    # sum of Manhattan distances from each box to its nearest target.
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    # Compute distances from the player's current location.
    distances = bfs_distances(state.player, walls, boxes, height, width)

    # Determine the minimal distance to any push position that is legal.
    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    # If there is no reachable push position, treat the walk cost as 0.
    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    # `boxes_on_targets()` returns the number of boxes that sit on targets.
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # penalty for long walks
    GAMMA = 0.4   # stronger bonus for having boxes placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    reward = max(0.0, min(1.0, reward))

    # Special case: all boxes on targets → perfect reward.
    if h == 0:
        reward = 1.0

    return reward
```

**Explanation of fixes**

* Replaced the undefined `_ag.h_sokoban_box_only` with the public `state.total_box_distance()`.
* Implemented an inline `bfs_distances` function that mirrors the missing `_bfs_distances`.
* Adjusted imports to be local (only `deque` from `collections`).
* Preserved the original heuristic logic while ensuring all referenced names exist within this self‑contained function.
