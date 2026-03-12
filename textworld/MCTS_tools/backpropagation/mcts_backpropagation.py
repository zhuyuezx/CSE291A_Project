"""
LLM-generated MCTS tool: backpropagation
Description: Add lightweight distance‑to‑goal shaping with depth discount, use `hasattr` for safe attribute access, and streamline sign handling.
Generated:   2026-03-11T03:12:30.593786
"""

import math
from typing import Any

def default_backpropagation(
    node: Any,
    reward: float,
    *,
    gamma: float = 0.99,
    distance_weight: float = 0.05,
    min_reward: float = -1.0,
) -> None:
    """
    Backpropagate a simulation result with simple shaping.

    Improvements over the vanilla implementation:
      * Apply a depth‑discount (γ^depth) to reduce the impact of long,
        heavily‑penalised roll‑outs (e.g., punishment variant).
      * Add a small bonus proportional to the reduction in
        ``distance_to_goal`` between a node and its parent, giving a
        gradient toward progress actions.
      * Clamp the shaped value to ``min_reward`` to avoid a single
        disastrous rollout dragging every node’s average down.
      * Keep the original sign‑flip logic for completeness, but it
        becomes a no‑op for single‑player games.

    Args:
        node:   Leaf ``MCTSNode`` where the simulation originated.
        reward: Final return from the simulation (from the root player's perspective).
        gamma:  Per‑step discount factor (default 0.99).
        distance_weight: Weight applied to the reduction in distance to goal.
        min_reward: Lower bound for the shaped value before it is added to a node.

    The function updates ``visits`` and ``value`` for every node on the
    path from ``node`` up to the root.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Determine the root perspective (the player whose reward we have)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # ------------------------------------------------------------------
    # 2️⃣ Compute depth from leaf to root for discounting
    # ------------------------------------------------------------------
    depth = 0
    tmp = node
    while tmp.parent is not None:
        depth += 1
        tmp = tmp.parent

    base_discounted = reward * (gamma ** depth)  # same base for all ancestors

    # ------------------------------------------------------------------
    # 3️⃣ Walk up the tree, applying visits, shaping and value updates
    # ------------------------------------------------------------------
    current = node
    while current is not None:
        # Increment visit count
        current.visits += 1

        # Start shaping with the discounted reward
        shaping = base_discounted

        # ---- distance‑to‑goal shaping (if the attribute exists) ----
        if current.parent is not None:
            parent_state = current.parent.state
            cur_state = current.state
            if hasattr(parent_state, "distance_to_goal") and hasattr(cur_state, "distance_to_goal"):
                parent_dist = parent_state.distance_to_goal()
                cur_dist = cur_state.distance_to_goal()
                # Positive delta means we moved closer to the goal.
                delta = parent_dist - cur_dist
                shaping += distance_weight * delta

        # Clamp to avoid extreme negatives
        if shaping < min_reward:
            shaping = min_reward

        # ---- sign handling (single‑player games effectively keep the same sign) ----
        if current.parent is not None:
            mover = current.parent.state.current_player()
        else:
            mover = perspective   # root node case

        if mover == perspective:
            current.value += shaping
        else:
            current.value -= shaping

        # Move upwards
        current = current.parent
