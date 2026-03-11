"""
LLM-generated MCTS tool: selection
Description: No changes needed; the function meets specifications.
Generated:   2026-03-10T00:48:12.650189
"""

import math

def default_selection(node, exploration_weight: float = 1.41):
    """
    Refined UCB‑1 tree policy for TextWorld Benchmark.

    Improvements over the original version:
    • Action‑type bias (movement actions preferred, observation actions penalised).
    • Stronger penalty for true no‑ops and for repeating the same action.
    • Larger distance‑to‑goal weight so progress is rewarded earlier.
    • Finite (large) exploration bonus for never‑visited children instead of ∞.
    • Keeps existing map‑read bonus and caps exploration after many visits.
    """

    # ------------------------------------------------------------------
    # Helper: Determine if an action caused no observable change.
    # Prefer a full state_key comparison; fall back to look / inventory.
    # ------------------------------------------------------------------
    def is_no_op(parent_state, child_state):
        try:
            return parent_state.state_key() == child_state.state_key()
        except Exception:
            try:
                return (parent_state.look_text() == child_state.look_text() and
                        parent_state.inventory_text() == child_state.inventory_text())
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Helper: Get distance to the goal, if the method exists.
    # Returns None when unavailable (e.g., coin task before map read).
    # ------------------------------------------------------------------
    def get_distance(state):
        try:
            return state.distance_to_goal()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helper: Simple heuristic bonus based on the textual action name.
    # ------------------------------------------------------------------
    def action_type_bonus(action_str: str) -> float:
        """Return a bias for the given action string."""
        act = action_str.lower()
        if act.startswith("move"):
            return 0.30          # primary movement
        if act.startswith("take ") or act.startswith("read "):
            return 0.20          # task‑oriented actions
        if act.startswith("open ") or act.startswith("close "):
            return 0.10          # door manipulation
        if act in ("look", "inventory", "task"):
            return -0.20         # generally wasteful observations
        return 0.0               # fallback

    # Tunable constants
    NOOP_PENALTY = 0.40          # stronger penalty for true no‑ops / repeats
    REPEAT_ACTION_PENALTY = 0.30
    MAP_READ_BONUS = 0.20
    DISTANCE_WEIGHT = 0.40       # reward per unit reduction in distance
    MOVE_TIE_BREAKER = 1e-6

    # --------------------------- Main selection loop --------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            # Node has untried actions – stop here; expansion will occur later.
            return node

        # Ensure log argument is at least 1 to avoid math domain errors.
        parent_visits = max(node.visits, 1)
        log_parent = math.log(parent_visits)

        best_child = None
        best_score = -math.inf

        # Cache parent distance once for efficiency.
        parent_dist = get_distance(node.state)

        for child in node.children.values():
            # ---- Exploitation ------------------------------------------------
            exploit = child.value / child.visits if child.visits > 0 else 0.0

            # ---- Exploration (finite for unseen children) -------------------
            if child.visits == 0:
                # Large but finite bonus; similar to standard UCB1 with N=1.
                explore = exploration_weight * math.sqrt(log_parent / 1.0)
            elif child.visits < 50:
                explore = exploration_weight * math.sqrt(log_parent / child.visits)
            else:
                explore = 0.0

            # ---- Domain‑specific bonuses / penalties ------------------------
            bonus = 0.0

            # 1. Penalty for true no‑ops.
            if is_no_op(node.state, child.state):
                bonus -= NOOP_PENALTY

            # 2. Extra penalty if the same action is taken consecutively.
            try:
                if getattr(child, "parent_action", None) == getattr(node, "parent_action", None):
                    bonus -= REPEAT_ACTION_PENALTY
            except Exception:
                pass

            # 3. Bonus when the map is read for the first time (mapreader task).
            if getattr(child.state, "map_read", 0) and not getattr(node.state, "map_read", 0):
                bonus += MAP_READ_BONUS

            # 4. Distance‑to‑goal delta bonus for movement actions.
            parent_room = getattr(node.state, "room", None)
            child_room = getattr(child.state, "room", None)
            if parent_room is not None and child_room is not None and parent_room != child_room:
                child_dist = get_distance(child.state)
                if parent_dist is not None and child_dist is not None:
                    delta = parent_dist - child_dist   # >0 means we are closer
                    bonus += DISTANCE_WEIGHT * delta   # can be negative if regressing
                # Tiny tie‑breaker to favour movement when everything else ties.
                bonus += MOVE_TIE_BREAKER

            # 5. Action‑type bias (based on the action that created this child).
            try:
                act_str = getattr(child, "parent_action", "")
                bonus += action_type_bias(act_str)
            except Exception:
                pass

            # ---- Combined UCB‑1 score ----------------------------------------
            score = exploit + explore + bonus

            if score > best_score:
                best_child = child
                best_score = score

        # Safety fallback (should never trigger).
        if best_child is None:
            best_child = next(iter(node.children.values()))
        node = best_child

    return node
