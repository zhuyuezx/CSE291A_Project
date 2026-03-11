# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 17:04:57 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 28.51s |
| Status     | success |
| Tokens     | 4577 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban MCTS expansion function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock only (wall‑line deadlock removed).
      • Transposition table lives on the search root.
      • Guarantees a push candidate when possible.
      • Uses matching‑distance as a secondary improvement metric.
    """
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math

    # ------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        root = current_node
        while root.parent is not None:
            root = root.parent
        if not hasattr(root, "_expansion_seen_keys"):
            root._expansion_seen_keys = set()
        return root._expansion_seen_keys

    trans_set = get_root_transposition_set(node)

    # ------------------------------------------------------------------
    # Helper: cheap corner dead‑lock test
    # ------------------------------------------------------------------
    def _is_corner_deadlock(pos, walls, targets):
        """Detect a box in a 2‑wall corner that is not already on a target."""
        if pos in targets:
            return False
        r, c = pos
        combos = [((r - 1, c), (r, c - 1)),   # up & left
                  ((r - 1, c), (r, c + 1)),   # up & right
                  ((r + 1, c), (r, c - 1)),   # down & left
                  ((r + 1, c), (r, c + 1))]   # down & right
        for a, b in combos:
            if a in walls and b in walls:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: optimal matching distance (same as in simulation)
    # ------------------------------------------------------------------
    def _matching_distance(state) -> int:
        """Brute‑force optimal total Manhattan distance after assigning boxes to targets."""
        boxes = list(state.boxes)
        targets = list(state.targets)
        n_boxes = len(boxes)
        n_targets = len(targets)

        best = math.inf
        if n_boxes == n_targets:
            for perm in itertools.permutations(targets, n_boxes):
                total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
        else:
            for targ_subset in itertools.combinations(targets, n_boxes):
                for perm in itertools.permutations(targ_subset):
                    total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                    if total < best:
                        best = total
        return best if best != math.inf else 0

    # ------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns (valid, is_push, manhattan_improve, matching_improve, child_state)
        where `valid` is False if the resulting state has a corner deadlock
        or is a transposition.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- corner dead‑lock pruning (cheap, safe) ----
        for bpos in temp_state.boxes:
            if _is_corner_deadlock(bpos, temp_state.walls, temp_state.targets):
                return False, False, 0, 0, None

        # ---- transposition detection (per‑root) ----
        key = temp_state.state_key()
        if key in trans_set:
            return False, False, 0, 0, None

        # ---- push detection ----
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ---- distance improvements ----
        old_manh = node.state.total_box_distance()
        new_manh = temp_state.total_box_distance()
        manhattan_improve = old_manh - new_manh          # positive → closer

        old_match = _matching_distance(node.state)
        new_match = _matching_distance(temp_state)
        matching_improve = old_match - new_match          # positive → closer

        return True, is_push, manhattan_improve, matching_improve, temp_state

    # ------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates = []   # (action, is_push, manh_imp, match_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, child_state = evaluate_action(a)
        if not valid:
            continue                # discard permanently (dead‑lock or duplicate)
        candidates.append((a, is_push, manh_imp, match_imp, child_state))

    # ------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ------------------------------------------------------------------
    if candidates:
        candidates.sort(
            key=lambda tup: (
                not tup[1],          # pushes first (False < True)
                -tup[2],             # larger Manhattan improvement
                -tup[3]              # larger matching improvement
            )
        )
        chosen_action, _, _, _, chosen_state = candidates[0]

        # keep the remaining actions for future expansions
        remaining_actions = [a for a, *_ in candidates[1:]]
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining_actions)
        else:
            node._untried_actions = remaining_actions

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets)
               for b in temp_state.boxes):
            continue

        # push detection
        if len(temp_state.boxes - node.state.boxes) == 0:
            continue

        manh_imp = node.state.total_box_distance() - temp_state.total_box_distance()
        match_imp = _matching_distance(node.state) - _matching_distance(temp_state)
        push_candidates.append((a, manh_imp, match_imp, temp_state))

    if push_candidates:
        push_candidates.sort(key=lambda tup: (-tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = push_candidates[0]

        # update untried actions: remove the chosen push, keep the rest
        remaining = [a for a, *_ in push_candidates[1:]]
        non_push_rest = [a for a in original_actions if a not in remaining + [chosen_action]]
        final_remaining = remaining + non_push_rest

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(final_remaining)
        else:
            node._untried_actions = final_remaining

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ------------------------------------------------------------------
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        action = node._untried_actions.pop()
    except (KeyError, IndexError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add(child_state.state_key())
    return child
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 392, in _smoke_test
    result = fn(*test_args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 32, in default_expansion
    trans_set = get_root_transposition_set(node)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/expansion/expansion.py", line 29, in get_root_transposition_set
    root._expansion_seen_keys = set()
    ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'MCTSNode' object has no attribute '_expansion_seen_keys' and no __dict__ for setting new attributes


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
**DESCRIPTION:** Fixed transposition‑table handling for `MCTSNode` objects that use `__slots__` (cannot set arbitrary attributes) by moving the per‑root transposition set to a module‑level weak‑key dictionary. The rest of the heuristic logic is unchanged.

```python
def default_expansion(node):
    """
    Expand one *promising* untried action from the given node.

    Incremental improvements:
      • Corner‑deadlock only (wall‑line deadlock removed).
      • Transposition table lives on the search root.
      • Guarantees a push candidate when possible.
      • Uses matching‑distance as a secondary improvement metric.
    """
    # ----------------------------------------------------------------------
    # Imports
    # ----------------------------------------------------------------------
    from mcts.node import MCTSNode                     # local import to avoid circular deps
    import itertools
    import math
    import weakref

    # ----------------------------------------------------------------------
    # Module‑level transposition cache (weak‑keyed so roots can be GC‑ed)
    # ----------------------------------------------------------------------
    # This dict lives on the function object so that the state is shared across
    # calls but does not interfere with the MCTSNode slots.
    if not hasattr(default_expansion, "_root_transposition_cache"):
        default_expansion._root_transposition_cache = weakref.WeakKeyDictionary()
    _cache = default_expansion._root_transposition_cache

    # ----------------------------------------------------------------------
    # Helper: obtain (or create) a per‑root transposition set
    # ----------------------------------------------------------------------
    def get_root_transposition_set(current_node):
        """Return the set of state keys seen so far for the search root."""
        # climb to the root
        root = current_node
        while getattr(root, "parent", None) is not None:
            root = root.parent

        # Retrieve or create the set stored in the module‑level cache
        if root not in _cache:
            _cache[root] = set()
        return _cache[root]

    trans_set = get_root_transposition_set(node)

    # ----------------------------------------------------------------------
    # Helper: cheap corner dead‑lock test
    # ----------------------------------------------------------------------
    def _is_corner_deadlock(pos, walls, targets):
        """Detect a box in a 2‑wall corner that is not already on a target."""
        if pos in targets:
            return False
        r, c = pos
        combos = [((r - 1, c), (r, c - 1)),   # up & left
                  ((r - 1, c), (r, c + 1)),   # up & right
                  ((r + 1, c), (r, c - 1)),   # down & left
                  ((r + 1, c), (r, c + 1))]   # down & right
        for a, b in combos:
            if a in walls and b in walls:
                return True
        return False

    # ----------------------------------------------------------------------
    # Helper: optimal matching distance (same as in simulation)
    # ----------------------------------------------------------------------
    def _matching_distance(state) -> int:
        """Brute‑force optimal total Manhattan distance after assigning boxes to targets."""
        boxes = list(state.boxes)
        targets = list(state.targets)
        n_boxes = len(boxes)
        n_targets = len(targets)

        best = math.inf
        if n_boxes == n_targets:
            for perm in itertools.permutations(targets, n_boxes):
                total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                if total < best:
                    best = total
        else:
            for targ_subset in itertools.combinations(targets, n_boxes):
                for perm in itertools.permutations(targ_subset):
                    total = sum(abs(b[0] - t[0]) + abs(b[1] - t[1]) for b, t in zip(boxes, perm))
                    if total < best:
                        best = total
        return best if best != math.inf else 0

    # ----------------------------------------------------------------------
    # Helper to evaluate a single action for expansion
    # ----------------------------------------------------------------------
    def evaluate_action(action):
        """
        Returns (valid, is_push, manhattan_improve, matching_improve, child_state)
        where `valid` is False if the resulting state has a corner deadlock
        or is a transposition.
        """
        temp_state = node.state.clone()
        temp_state.apply_action(action)

        # ---- corner dead‑lock pruning (cheap, safe) ----
        for bpos in temp_state.boxes:
            if _is_corner_deadlock(bpos, temp_state.walls, temp_state.targets):
                return False, False, 0, 0, None

        # ---- transposition detection (per‑root) ----
        key = temp_state.state_key()
        if key in trans_set:
            return False, False, 0, 0, None

        # ---- push detection ----
        is_push = len(temp_state.boxes - node.state.boxes) > 0

        # ---- distance improvements ----
        old_manh = node.state.total_box_distance()
        new_manh = temp_state.total_box_distance()
        manhattan_improve = old_manh - new_manh          # positive → closer

        old_match = _matching_distance(node.state)
        new_match = _matching_distance(temp_state)
        matching_improve = old_match - new_match          # positive → closer

        return True, is_push, manhattan_improve, matching_improve, temp_state

    # ----------------------------------------------------------------------
    # Gather candidates from the node's untried‑action collection
    # ----------------------------------------------------------------------
    original_actions = list(node._untried_actions)      # copy (list or set)
    candidates = []   # (action, is_push, manh_imp, match_imp, state)

    for a in original_actions:
        valid, is_push, manh_imp, match_imp, child_state = evaluate_action(a)
        if not valid:
            continue                # discard permanently (dead‑lock or duplicate)
        candidates.append((a, is_push, manh_imp, match_imp, child_state))

    # ----------------------------------------------------------------------
    # Choose the best candidate if any survive the filters
    # ----------------------------------------------------------------------
    if candidates:
        # Sort: pushes first, then larger Manhattan improvement, then larger matching improvement
        candidates.sort(
            key=lambda tup: (
                not tup[1],          # pushes first (False < True)
                -tup[2],             # larger Manhattan improvement
                -tup[3]              # larger matching improvement
            )
        )
        chosen_action, _, _, _, chosen_state = candidates[0]

        # keep the remaining actions for future expansions
        remaining_actions = [a for a, *_ in candidates[1:]]
        if isinstance(node._untried_actions, set):
            node._untried_actions = set(remaining_actions)
        else:
            node._untried_actions = remaining_actions

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ----------------------------------------------------------------------
    # No candidate survived – force a push if possible (ignore transposition)
    # ----------------------------------------------------------------------
    push_candidates = []   # (action, manh_imp, match_imp, state)
    for a in original_actions:
        temp_state = node.state.clone()
        temp_state.apply_action(a)

        # dead‑lock filter
        if any(_is_corner_deadlock(b, temp_state.walls, temp_state.targets)
               for b in temp_state.boxes):
            continue

        # push detection
        if len(temp_state.boxes - node.state.boxes) == 0:
            continue

        manh_imp = node.state.total_box_distance() - temp_state.total_box_distance()
        match_imp = _matching_distance(node.state) - _matching_distance(temp_state)
        push_candidates.append((a, manh_imp, match_imp, temp_state))

    if push_candidates:
        push_candidates.sort(key=lambda tup: (-tup[1], -tup[2]))
        chosen_action, _, _, chosen_state = push_candidates[0]

        # Update untried actions: remove the chosen push, keep the rest
        remaining = [a for a, *_ in push_candidates[1:]]
        non_push_rest = [a for a in original_actions if a not in (remaining + [chosen_action])]
        final_remaining = remaining + non_push_rest

        if isinstance(node._untried_actions, set):
            node._untried_actions = set(final_remaining)
        else:
            node._untried_actions = final_remaining

        child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = child
        trans_set.add(chosen_state.state_key())
        return child

    # ----------------------------------------------------------------------
    # Fallback: no push possible, revert to naive expansion.
    # ----------------------------------------------------------------------
    # Restore the original untried‑action collection (it may have been mutated)
    if isinstance(node._untried_actions, set):
        node._untried_actions = set(original_actions)
    else:
        node._untried_actions = original_actions

    try:
        # pop works for both list (pop last) and set (pop arbitrary)
        action = node._untried_actions.pop()
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError("Expansion failed: node has no untried actions.") from None

    child_state = node.state.clone()
    child_state.apply_action(action)
    child = MCTSNode(child_state, parent=node, parent_action=action)
    node.children[action] = child
    trans_set.add(child_state.state_key())
    return child
```
