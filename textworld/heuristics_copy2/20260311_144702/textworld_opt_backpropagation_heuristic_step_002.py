"""
LLM-generated MCTS tool: backpropagation
Description: Fixed runtime error by removing misplaced __future__ import and ensured the function is self‑contained with proper handling of game‑state attributes.
Generated:   2026-03-11T15:21:41.931047
"""

"""
Enhanced backpropagation for single‑player TextWorld games.

Changes compared to the original version:
* Penalises actions that leave the game state unchanged (e.g. look/inventory
  that produce no new description) to discourage stagnation, especially
  in the punishment variant.
* Normalises distance‑to‑goal improvement, giving a larger bonus when a
  step makes a significant move toward the goal.
* Increases the importance of map‑reading and goal‑reveal events.
* Keeps the depth discount and final clipping, but with tuned constant
  values for the benchmark.
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagation with shaping and stagnation penalties.

    Args:
        node:   The leaf MCTSNode where the simulation started.
        reward: The raw simulation reward from the root player's perspective.
    """
    # ---------- Hyper‑parameters ----------
    DISCOUNT = 0.99        # depth discount factor
    DIST_WEIGHT = 0.3      # max bonus for a perfect distance improvement (scaled)
    MAP_BONUS = 0.7        # bonus for acquiring/reading the map
    GOAL_BONUS = 0.7       # bonus for learning the goal room
    NOOP_PENALTY = 0.2     # penalty when child state is identical to parent

    depth = 0  # distance from leaf (depth = 0) upward in the tree
    while node is not None:
        # Increment visit count
        if hasattr(node, "visits"):
            node.visits += 1
        else:
            # Gracefully initialise if the attribute is missing
            node.visits = 1

        # ---------- 1️⃣ Discounted raw reward ----------
        discounted_reward = reward * (DISCOUNT ** depth)

        # ---------- 2️⃣ Shaping bonuses / penalties ----------
        bonus = 0.0
        if node.parent is not None:
            parent_state = node.parent.state
            child_state = node.state

            # ---- a) Penalty for unchanged state (no‑op) ----
            try:
                if parent_state.state_key() == child_state.state_key():
                    bonus -= NOOP_PENALTY
            except Exception:
                # state_key may be absent or raise; ignore in that case
                pass

            # ---- b) Normalised distance improvement (coin games) ----
            try:
                parent_dist = parent_state.distance_to_goal()
                child_dist = child_state.distance_to_goal()
                delta_dist = parent_dist - child_dist  # >0 -> got closer
                if delta_dist > 0:
                    # Normalise by the original distance to keep the bonus in [0, DIST_WEIGHT]
                    norm_improvement = delta_dist / max(parent_dist, 1)
                    bonus += DIST_WEIGHT * norm_improvement
            except Exception:
                # distance_to_goal may be irrelevant in some variants
                pass

            # ---- c) Map read transition (mapreader) ----
            try:
                if (not getattr(parent_state, "map_read", False) and
                        getattr(child_state, "map_read", False)):
                    bonus += MAP_BONUS
            except Exception:
                pass

            # ---- d) Goal room revelation transition (mapreader) ----
            try:
                parent_known = getattr(parent_state, "known_goal_room", None)
                child_known = getattr(child_state, "known_goal_room", None)
                if parent_known != child_known and child_known is not None:
                    bonus += GOAL_BONUS
            except Exception:
                pass

        # ---------- 3️⃣ Combine and clip ----------
        shaped_value = discounted_reward + bonus
        shaped_value = max(min(shaped_value, 1.0), -1.0)

        # ---------- 4️⃣ Accumulate ----------
        if hasattr(node, "value"):
            node.value += shaped_value
        else:
            # Initialise if missing
            node.value = shaped_value

        # Move up the tree
        node = node.parent
        depth += 1
