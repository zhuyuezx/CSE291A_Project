"""
LLM-generated MCTS tool: backpropagation
Description: Clean up unused imports while keeping the improved backpropagation logic.
Generated:   2026-03-11T15:20:22.523706
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Improved backpropagation for single‑player TextWorld games.

    * Applies a depth discount so long failing rollouts are penalised more.
    * Adds small shaping bonuses:
        - reduction in ``distance_to_goal()`` (coin games)
        - acquiring the map (mapreader)
        - learning the goal room (mapreader)
    * Removes opponent sign‑flip logic which is irrelevant for single‑player games.
    * Clips the final shaped value to the original reward bounds [-1, 1].

    Args:
        node:   The leaf MCTSNode where the simulation started.
        reward: The raw simulation reward from the ROOT player's perspective.
    """
    # Hyper‑parameters (tuned for the benchmark)
    DISCOUNT = 0.99          # depth discount factor
    DIST_WEIGHT = 0.1        # weight per unit reduction in distance_to_goal
    MAP_BONUS = 0.5          # bonus for reading the map
    GOAL_BONUS = 0.5         # bonus for revealing the goal room

    depth = 0  # distance from leaf (depth = 0) upward
    while node is not None:
        node.visits += 1

        # 1️⃣ Discounted raw reward
        discounted_reward = reward * (DISCOUNT ** depth)

        # 2️⃣ Shaping bonuses based on state change relative to parent
        bonus = 0.0
        if node.parent is not None:
            parent_state = node.parent.state
            child_state = node.state

            # --- distance improvement (coin games) ---
            try:
                parent_dist = parent_state.distance_to_goal()
                child_dist = child_state.distance_to_goal()
                delta_dist = parent_dist - child_dist  # >0 means we got closer
                if delta_dist > 0:
                    bonus += DIST_WEIGHT * delta_dist
            except Exception:
                # distance_to_goal may not be meaningful in some variants
                pass

            # --- map read transition (mapreader) ---
            try:
                if (not getattr(parent_state, "map_read", False) and
                        getattr(child_state, "map_read", False)):
                    bonus += MAP_BONUS
            except Exception:
                pass

            # --- known goal room transition (mapreader) ---
            try:
                parent_known = getattr(parent_state, "known_goal_room", None)
                child_known = getattr(child_state, "known_goal_room", None)
                if parent_known != child_known and child_known is not None:
                    bonus += GOAL_BONUS
            except Exception:
                pass

        # 3️⃣ Combine and clip to keep values within [-1, 1]
        shaped_value = discounted_reward + bonus
        shaped_value = max(min(shaped_value, 1.0), -1.0)

        # 4️⃣ Update node's cumulative value
        node.value += shaped_value

        # Move up the tree
        node = node.parent
        depth += 1
