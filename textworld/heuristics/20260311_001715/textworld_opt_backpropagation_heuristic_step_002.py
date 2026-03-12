"""
LLM-generated MCTS tool: backpropagation
Description: Fix AttributeError and simplify no‑op penalty detection while preserving all heuristic enhancements.
Generated:   2026-03-11T01:02:05.006033
"""

"""
Enhanced backpropagation for TextWorld Benchmark MCTS.

Improvements:
  • Mild punishment scaling (0.8) preserves most negative signals.
  • Stronger progress bonus (0.8 per unit distance reduction).
  • Bonus for opening new doors that do not increase distance.
  • Penalty for actions that leave the observation unchanged (no‑op).
  • Linear decay for negative increments so early penalties are not
    completely erased by later discounts.
  • Retains sign‑flip logic for potential multi‑player use.

Parameters can be tuned to balance exploration vs. exploitation.
"""

from typing import Any

# ----------------------------------------------------------------------
# Configuration constants – adjust to fine‑tune the search behavior.
# ----------------------------------------------------------------------
PUNISH_SCALE = 0.8          # retain most of a negative leaf reward
GAMMA = 0.95                # exponential discount per tree level
PROGRESS_WEIGHT = 0.8       # reward per unit reduction in distance_to_goal
DOOR_BONUS = 0.2            # extra reward when a new door becomes open
NOOP_PENALTY = 0.05         # penalty for actions that do not change observation
NEG_DECAY_PER_LEVEL = 0.02  # linear decay factor for negative increments


def default_backpropagation(node: Any, reward: float) -> None:
    """
    Back‑propagate a simulation reward up the MCTS tree with enriched signals.

    Args:
        node:   The leaf MCTSNode where the simulation began.
        reward: The simulation return from the root player's perspective.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Identify the root node to obtain the perspective player.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # ------------------------------------------------------------------
    # 2️⃣ Scale negative leaf rewards (punishment variant).
    # ------------------------------------------------------------------
    if reward < 0:
        reward *= PUNISH_SCALE

    depth = 0  # distance from leaf (leaf itself = 0)

    # ------------------------------------------------------------------
    # 3️⃣ Walk up the tree, updating visits and enriched value.
    # ------------------------------------------------------------------
    while node is not None:
        # Increment visit count.
        node.visits += 1

        # Discount the leaf reward according to depth.
        discounted_reward = reward * (GAMMA ** depth)

        # ------- progress bonus -------------------------------------------------
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
                progress_bonus = 0.0

        # ------- door opening bonus ---------------------------------------------
        door_bonus = 0.0
        if node.parent is not None:
            try:
                if node.parent.state.doors != node.state.doors:
                    # Apply bonus only if distance does not increase.
                    parent_dist = node.parent.state.distance_to_goal()
                    child_dist = node.state.distance_to_goal()
                    if (parent_dist is None) or (child_dist is None) or (child_dist <= parent_dist):
                        door_bonus = DOOR_BONUS
            except Exception:
                door_bonus = 0.0

        # ------- no‑op penalty ---------------------------------------------------
        noop_penalty = 0.0
        if node.parent is not None:
            try:
                parent_obs = node.parent.state.observation_text()
                child_obs = node.state.observation_text()
                if parent_obs == child_obs:
                    noop_penalty = NOOP_PENALTY
            except Exception:
                noop_penalty = 0.0

        # Total increment before sign handling.
        total_increment = discounted_reward + progress_bonus + door_bonus - noop_penalty

        # Attenuate negative increments with a linear decay based on depth.
        if total_increment < 0:
            decay_factor = max(0.0, 1.0 - depth * NEG_DECAY_PER_LEVEL)
            total_increment *= decay_factor

        # ------------------------------------------------------------------
        # Determine which player performed the action that led to this node.
        # ------------------------------------------------------------------
        mover = node.parent.state.current_player() if node.parent else perspective

        # Apply sign flip only when mover differs from root perspective.
        if mover == perspective:
            node.value += total_increment
        else:
            node.value -= total_increment

        # Move up the tree.
        node = node.parent
        depth += 1
