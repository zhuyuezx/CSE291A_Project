# repair_5

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:01:01 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.15s |
| Status     | success |
| Tokens     | 2957 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand the most promising untried action.

    Improvements over the baseline:
      * g‑score = real steps taken (walk + push) instead of tree depth.
      * Corner dead‑lock pruning (now checks the box’s *new* location).
      * Scoring = walk_cost + h_box_only + penalty for Δh > 0.
      * Actions that fail the visited‑check are kept for possible later use.
    """
    # ---- Reset visited info for a new MCTS search (once per root) ----
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    visited = _ag.get_visited()

    # Pre‑compute current heuristic for Δh calculations
    h_curr = _ag.h_sokoban_box_only(node.state)

    # Containers
    scored = []   # (score, action, next_state, key, g_new)
    deadlocked_actions = []  # permanently removed

    # -----------------------------------------------------------------
    # Examine each untried action
    # -----------------------------------------------------------------
    for action in list(node._untried_actions):
        push_pos, direction = action          # player location to push from

        # ----- walk cost ------------------------------------------------
        walk_cost = _bfs_distance(
            start=node.state.player,
            goal=push_pos,
            walls=node.state.walls,
            boxes=node.state.boxes,
        )
        if walk_cost == sys.maxsize:
            # unreachable – treat as dead‑lock for this branch
            deadlocked_actions.append(action)
            continue

        # ----- generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- corner dead‑lock test ------------------------------------
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        # new box location after the push = player_pos + 2*delta
        new_box = (push_pos[0] + 2 * dr, push_pos[1] + 2 * dc)
        if _is_corner_deadlock(new_box, next_state.walls, next_state.targets):
            deadlocked_actions.append(action)
            continue

        # ----- A* visited check (using real g) -------------------------
        g_new = node.state.steps + walk_cost + 1   # +1 for the push itself
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for possible reconsideration later
            continue

        # ----- heuristic scoring ----------------------------------------
        h_next = _ag.h_sokoban_box_only(next_state)
        delta_h = h_next - h_curr            # >0 means we moved away from goal
        penalty = 5.0 * delta_h if delta_h > 0 else 0.0
        score = walk_cost + h_next + penalty

        scored.append((score, action, next_state, key, g_new))

    # ---- permanently prune corner‑deadlocked actions -------------------
    for a in deadlocked_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ---- Choose best scored action ------------------------------------
    if scored:
        # lowest composite score first; tie‑break by smaller Δh (implicit via score)
        scored.sort(key=lambda x: x[0])
        best_score, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]

        # Remove the chosen action from the pool and record its cost
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No promising actions left – fallback to any remaining untried action
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)

            walk_cost = _bfs_distance(
                start=node.state.player,
                goal=chosen_action[0],
                walls=node.state.walls,
                boxes=node.state.boxes,
            )
            chosen_g = node.state.steps + walk_cost + 1
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = chosen_g
        else:
            # No actions at all – return an existing child if any, else node
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ---- Create child node --------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    result = fn(*args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 28, in default_expansion
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
**DESCRIPTION:** Fixed undefined `_ag` reference by using a per‑root visited dictionary stored on the root node, and replaced the unavailable heuristic call with the public `total_box_distance()` method of `GameState`. Updated comments accordingly while preserving the original scoring logic.

```python
def default_expansion(node):
    """
    Expand the most promising untried action.

    Improvements over the baseline:
      * g‑score = real steps taken (walk + push) instead of tree depth.
      * Corner dead‑lock pruning (now checks the box’s *new* location).
      * Scoring = walk_cost + h_box_only + penalty for Δh > 0.
      * Actions that fail the visited‑check are kept for possible later use.
    """
    # -----------------------------------------------------------------
    # Reset / obtain visited information for a new MCTS search (once per root)
    # -----------------------------------------------------------------
    # Find the root of the current tree.
    root = node
    while root.parent is not None:
        root = root.parent

    # Attach a visited‑dictionary to the root node the first time we see it.
    # The dictionary maps state keys to the best g‑score (steps) seen so far.
    if not hasattr(root, "_visited"):
        root._visited = {}
    visited = root._visited

    # -----------------------------------------------------------------
    # Pre‑compute current heuristic for Δh calculations
    # -----------------------------------------------------------------
    # The public API provides `total_box_distance()` which is the classic
    # “box‑only” heuristic (sum of manhattan distances from each box to its
    # closest target).  This replaces the previously used private
    # `_ag.h_sokoban_box_only`.
    h_curr = node.state.total_box_distance()

    # Containers for scoring
    scored = []               # (score, action, next_state, key, g_new)
    deadlocked_actions = []   # permanently removed

    # -----------------------------------------------------------------
    # Examine each untried action
    # -----------------------------------------------------------------
    for action in list(node._untried_actions):
        push_pos, direction = action          # player location to push from

        # ----- walk cost ------------------------------------------------
        walk_cost = _bfs_distance(
            start=node.state.player,
            goal=push_pos,
            walls=node.state.walls,
            boxes=node.state.boxes,
        )
        if walk_cost == sys.maxsize:
            # unreachable – treat as dead‑lock for this branch
            deadlocked_actions.append(action)
            continue

        # ----- generate successor state ---------------------------------
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- corner dead‑lock test ------------------------------------
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        # new box location after the push = player_pos + 2*delta
        new_box = (push_pos[0] + 2 * dr, push_pos[1] + 2 * dc)
        if _is_corner_deadlock(new_box, next_state.walls, next_state.targets):
            deadlocked_actions.append(action)
            continue

        # ----- visited check (using real g) -----------------------------
        g_new = node.state.steps + walk_cost + 1   # +1 for the push itself
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # we have already reached this state more cheaply – skip for now
            continue

        # ----- heuristic scoring ----------------------------------------
        h_next = next_state.total_box_distance()
        delta_h = h_next - h_curr            # >0 means we moved away from goal
        penalty = 5.0 * delta_h if delta_h > 0 else 0.0
        score = walk_cost + h_next + penalty

        scored.append((score, action, next_state, key, g_new))

    # ---- permanently prune corner‑deadlocked actions -------------------
    for a in deadlocked_actions:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ---- Choose best scored action ------------------------------------
    if scored:
        # lowest composite score first; tie‑break by smaller Δh (implicit via score)
        scored.sort(key=lambda x: x[0])
        best_score, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]

        # Remove the chosen action from the pool and record its cost
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No promising actions left – fallback to any remaining untried action
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)

            walk_cost = _bfs_distance(
                start=node.state.player,
                goal=chosen_action[0],
                walls=node.state.walls,
                boxes=node.state.boxes,
            )
            chosen_g = node.state.steps + walk_cost + 1
            chosen_key = chosen_state.state_key()
            visited[chosen_key] = chosen_g
        else:
            # No actions at all – return an existing child if any, else node
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ---- Create child node --------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```
