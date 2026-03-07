"""
LLM-generated MCTS tool: expansion
Description: Fixed incorrect access to the Sokoban state API (used a non‑existent `player` attribute and assumed a missing `state` attribute). Updated to use `current_player()` method and streamlined attribute accesses while preserving the original heuristic logic.
Generated:   2026-03-06T23:31:36.067457
"""

def default_expansion(node):
    """
    Expand one untried action from the given node.

    Enhancements over the vanilla version:
      • Detect and prioritise *push* actions (moves that actually move a box).
      • Sort pushes by estimated Manhattan‑distance improvement for the moved box.
      • Skip actions that simply undo the parent's move (prevents left/right ping‑pong).
      • Perform a cheap dead‑lock check (corner box not on a target) and avoid
        adding such children to the tree.
      • Keep the original ``_untried_actions`` bookkeeping for compatibility.
    """
    import random

    # ----- 1. Gather legal actions and classify them -----
    # Action → (drow, dcol) mapping (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)
    _deltas = {
        0: (-1, 0),  # UP
        1: ( 1, 0),  # DOWN
        2: ( 0,-1),  # LEFT
        3: ( 0, 1),  # RIGHT
    }

    # Use the public API correctly
    legal = node.state.legal_actions()
    push_actions = []
    move_actions = []

    pr, pc = node.state.current_player()   # ← corrected access
    walls = node.state.walls
    boxes = node.state.boxes
    targets = node.state.targets

    for a in legal:
        dr, dc = _deltas[a]
        nxt = (pr + dr, pc + dc)          # square the player would step onto
        if nxt in boxes:                  # potential push
            beyond = (nxt[0] + dr, nxt[1] + dc)
            if beyond not in walls and beyond not in boxes:
                push_actions.append(a)
        else:
            move_actions.append(a)

    # ----- 2. Order push actions by distance improvement (best first) -----
    if push_actions:
        def dist_improvement(action):
            dr, dc = _deltas[action]
            box = (pr + dr, pc + dc)                # box before push
            new_box = (box[0] + dr, box[1] + dc)    # box after push
            cur = min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets)
            nxt = min(abs(new_box[0] - t[0]) + abs(new_box[1] - t[1]) for t in targets)
            return cur - nxt                      # positive ⇒ improvement
        push_actions.sort(key=dist_improvement, reverse=True)

    # ----- 3. Avoid immediate back‑tracking (undo of parent's move) -----
    parent_player = None
    if node.parent is not None:
        parent_player = node.parent.state.current_player()   # ← corrected access

    def is_undo(action):
        """True if applying *action* would move the player back to the parent's position."""
        if parent_player is None:
            return False
        dr, dc = _deltas[action]
        new_pos = (pr + dr, pc + dc)
        return new_pos == parent_player

    # Choose the first non‑undo push; if none, the first non‑undo move.
    chosen_action = None
    for a in push_actions:
        if not is_undo(a):
            chosen_action = a
            break
    if chosen_action is None:
        for a in move_actions:
            if not is_undo(a):
                chosen_action = a
                break

    # Fallback: take any remaining untried action or a random legal one.
    if chosen_action is None:
        if getattr(node, "_untried_actions", None):
            # pop from the original untried‑action set if it still has items
            chosen_action = node._untried_actions.pop()
        else:
            chosen_action = random.choice(legal)

    # Keep the bookkeeping consistent – remove from the stored set if present.
    if hasattr(node, "_untried_actions") and chosen_action in node._untried_actions:
        node._untried_actions.remove(chosen_action)

    # ----- 4. Create the child state -----
    child_state = node.state.clone()
    child_state.apply_action(chosen_action)

    # ----- 5. Early dead‑lock detection (simple corner check) -----
    def is_deadlocked(state):
        """Return True if any non‑target box is stuck in a corner formed by walls."""
        for b in state.boxes:
            if b in state.targets:
                continue
            r, c = b
            # Four possible corner patterns (two orthogonal walls)
            if ((r - 1, c) in state.walls and (r, c - 1) in state.walls) or \
               ((r - 1, c) in state.walls and (r, c + 1) in state.walls) or \
               ((r + 1, c) in state.walls and (r, c - 1) in state.walls) or \
               ((r + 1, c) in state.walls and (r, c + 1) in state.walls):
                return True
        return False

    if is_deadlocked(child_state):
        # Treat dead‑locked nodes as terminal leaves; they will get 0 reward from rollout.
        from mcts.node import MCTSNode
        dead_child = MCTSNode(child_state, parent=node, parent_action=chosen_action)
        node.children[chosen_action] = dead_child
        return dead_child

    # ----- 6. Normal child creation -----
    from mcts.node import MCTSNode
    child = MCTSNode(child_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
