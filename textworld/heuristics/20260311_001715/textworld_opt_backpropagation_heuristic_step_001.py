"""
LLM-generated MCTS tool: backpropagation
Description: Clean up imports, add documentation, and keep the incremental enhancements.
Generated:   2026-03-11T01:00:54.643423
"""

from typing import Any

def default_backpropagation(node, reward: float) -> None:
    """
    Enhanced backpropagation for single‑player TextWorld games.

    Improvements over the original version:
      • Punishment scaling – negative leaf rewards are down‑weighted so a
        single punished step does not dominate the node's value.
      • Depth‑aware discount (γ) – rewards are attenuated the further a node
        is from the leaf, reducing the impact of early penalties on the root.
      • Progress bonus – when a child state is strictly closer to the goal
        (according to ``distance_to_goal()``), a small positive term is added.
      • Retains the original sign‑flip logic for potential multi‑player use.

    Args:
        node:   The leaf MCTSNode where the simulation started.
        reward: The simulation reward from the ROOT player's perspective.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Locate the root to obtain the perspective player (needed for sign flips)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # ------------------------------------------------------------------
    # 2️⃣ Scale punished (negative) rewards so they do not drown out positives
    # ------------------------------------------------------------------
    PUNISH_SCALE = 0.3  # keep only 30 % of a pure penalty
    if reward < 0:
        reward *= PUNISH_SCALE

    # ------------------------------------------------------------------
    # 3️⃣ Set up depth discounting (γ) and progress‑bonus parameters
    # ------------------------------------------------------------------
    GAMMA = 0.95          # discount factor per level up the tree
    PROGRESS_WEIGHT = 0.2  # weight for each unit of distance improvement

    depth = 0  # distance from the leaf (0 for the leaf itself)

    # ------------------------------------------------------------------
    # 4️⃣ Walk up the tree, updating visits and value with the enriched signal
    # ------------------------------------------------------------------
    while node is not None:
        node.visits += 1

        # Discounted reward contribution
        discounted_reward = reward * (GAMMA ** depth)

        # Progress bonus: reward if we moved closer to the goal
        progress_bonus = 0.0
        if node.parent is not None:
            try:
                parent_dist = node.parent.state.distance_to_goal()
                child_dist = node.state.distance_to_goal()
                if (parent_dist is not None) and (child_dist is not None):
                    delta = parent_dist - child_dist
                    if delta > 0:
                        progress_bonus = PROGRESS_WEIGHT * delta
            except Exception:
                # Defensive: some state objects might not implement the method
                progress_bonus = 0.0

        total_increment = discounted_reward + progress_bonus

        # Determine which player "moved" to create this node
        mover = node.parent.state.current_player() if node.parent else perspective

        # Apply sign flip only when mover differs from root perspective
        if mover == perspective:
            node.value += total_increment
        else:
            node.value -= total_increment

        # Move up the tree
        node = node.parent
        depth += 1
