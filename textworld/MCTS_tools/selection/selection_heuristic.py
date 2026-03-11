"""
LLM-generated MCTS tool: selection
Description: 
Generated:   2026-03-10T00:44:02.355415
"""

import math

def default_selection(node, exploration_weight: float = 1.41):
    """
    Enhanced UCB‑1 tree policy with lightweight domain knowledge.

    The function prefers actions that make progress toward the goal,
    penalises actions that leave the observable state unchanged,
    and gives a small bonus when the map is read for the first time.
    After a child has been visited many times its exploration term is
    zeroed to prevent endless expansion of low‑value loops.
    """
    # Helper: determine if an action caused no observable change.
    def is_no_op(parent_state, child_state):
        # Compare description texts; if both unchanged, treat as no‑op.
        try:
            return (parent_state.look_text() == child_state.look_text() and
                    parent_state.inventory_text() == child_state.inventory_text())
        except Exception:  # Defensive: if method missing, assume not a no‑op
            return False

    # Helper: distance to goal if available; otherwise 0.
    def get_distance(state):
        try:
            return state.distance_to_goal()
        except Exception:
            return 0.0

    # Main selection loop.
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # need expansion

        # Use at least 1 visit for the log to avoid math domain errors.
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        best_child = None
        best_score = -math.inf

        for child in node.children.values():
            # ---- Exploitation term ----
            exploit = child.value / child.visits if child.visits > 0 else 0.0

            # ---- Exploration term (capped after many visits) ----
            if child.visits == 0:
                explore = float('inf')   # force exploration of unseen child
            elif child.visits < 50:
                explore = exploration_weight * math.sqrt(log_parent / child.visits)
            else:
                explore = 0.0  # enough samples, stop exploring this branch

            # ---- Domain‑specific bonuses / penalties ----
            bonus = 0.0

            # 1. Penalty for actions that do not change observable state.
            if is_no_op(node.state, child.state):
                bonus -= 0.10  # discourage look/inventory/redundant open‑close loops

            # 2. Bonus when the map is read for the first time (mapreader task).
            if getattr(child.state, "map_read", 0) and not getattr(node.state, "map_read", 0):
                bonus += 0.20

            # 3. Distance‑to‑goal bias for movement actions.
            #    Prefer actions that bring the agent closer to the goal.
            parent_room = getattr(node.state, "room", None)
            child_room = getattr(child.state, "room", None)
            if parent_room is not None and child_room is not None and parent_room != child_room:
                dist = get_distance(child.state)
                bonus -= 0.05 * dist          # smaller distance → larger score
                bonus += 1e-6                 # tiny tie‑breaker favouring movement

            # ---- Total UCB‑1 score with heuristics ----
            score = exploit + explore + bonus

            if score > best_score:
                best_child = child
                best_score = score

        # Safety fallback (should not happen).
        if best_child is None:
            best_child = next(iter(node.children.values()))
        node = best_child

    return node
