"""
LLM-generated MCTS tool: backpropagation
Description: 
Generated:   2026-03-11T01:03:16.820415
"""

"""
Enhanced backpropagation for the TextWorld Benchmark MCTS.

Key improvements over the previous version:
  • Refined “no‑op” detection – actions that actually change room,
    inventory or door states are no longer penalised.
  • Adaptive no‑op penalty scaling for the *punishment* variant.
  • Small bonus when the map is read for the map‑reader task.
  • Simplified sign handling – single‑player games always add the
    increment to the node value.
  • Retains proven components: exponential discount, progress bonus,
    door‑opening bonus, and linear decay for negative increments.
"""

from typing import Any

# ----------------------------------------------------------------------
# Configuration constants – feel free to tune them for different
# variants or difficulty levels.
# ----------------------------------------------------------------------
PUNISH_SCALE = 0.8          # scale for negative leaf rewards
GAMMA = 0.95                # exponential discount per tree level
PROGRESS_WEIGHT = 0.8       # reward per unit reduction in distance_to_goal
DOOR_BONUS = 0.2            # extra reward when a new door becomes open
NOOP_PENALTY_BASE = 0.05    # base penalty for true no‑op actions
MAP_READ_BONUS = 0.3        # bonus when the map is read for the first time
NEG_DECAY_PER_LEVEL = 0.02  # linear decay factor for negative increments


def _variant_factor(state: Any) -> float:
    """
    Return a scaling factor for the no‑op penalty depending on the
    game variant.  The benchmark stores variant information in
    `state.config['variant']` (defaults to 'deterministic').
    """
    variant_cfg = getattr(state, "config", {})
    if isinstance(variant_cfg, dict):
        variant = variant_cfg.get("variant", "deterministic")
    else:
        # Fallback for non‑dict config objects.
        variant = getattr(variant_cfg, "variant", "deterministic")
    return 0.5 if variant == "punishment" else 1.0


def _is_true_noop(parent_state: Any, child_state: Any) -> bool:
    """
    Detect a genuine no‑op: room, inventory, doors and observation
    are all unchanged.  This prevents penalising useful actions
    such as opening a door or taking the map, which often leave
    the textual observation identical.
    """
    try:
        if parent_state.room != child_state.room:
            return False
        if parent_state.inventory_items != child_state.inventory_items:
            return False
        if parent_state.doors != child_state.doors:
            return False
        # Observation check is kept as a final safety net.
        if parent_state.observation_text() != child_state.observation_text():
            return False
        return True
    except Exception:
        # If any attribute is missing, assume it is *not* a true no‑op.
        return False


def default_backpropagation(node: Any, reward: float) -> None:
    """
    Back‑propagate a simulation reward up the MCTS tree, enriching the
    update with domain‑specific signals.

    Args:
        node:   The leaf MCTSNode where the simulation began.
        reward: The simulation return from the root player's perspective.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Determine the root to obtain the perspective player (kept for
    #     possible future multi‑player extensions; currently unused).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    _ = root.state.current_player()   # perspective retained for compatibility

    # ------------------------------------------------------------------
    # 2️⃣ Scale negative leaf rewards (punishment variant).
    # ------------------------------------------------------------------
    if reward < 0:
        reward *= PUNISH_SCALE

    # ------------------------------------------------------------------
    # 3️⃣ Walk up the tree, updating visits and the enriched value.
    # ------------------------------------------------------------------
    depth = 0  # leaf depth = 0
    while node is not None:
        # Increment visit count.
        node.visits += 1

        # Discount the leaf reward according to depth.
        discounted_reward = reward * (GAMMA ** depth)

        # ------------------- progress bonus --------------------------------
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

        # ------------------- door opening bonus ----------------------------
        door_bonus = 0.0
        if node.parent is not None:
            try:
                if node.parent.state.doors != node.state.doors:
                    # Ensure the new door configuration does not increase distance.
                    parent_dist = node.parent.state.distance_to_goal()
                    child_dist = node.state.distance_to_goal()
                    if (parent_dist is None) or (child_dist is None) or (child_dist <= parent_dist):
                        door_bonus = DOOR_BONUS
            except Exception:
                door_bonus = 0.0

        # ------------------- map‑read bonus --------------------------------
        map_bonus = 0.0
        if node.parent is not None:
            try:
                if (not node.parent.state.map_read) and node.state.map_read:
                    map_bonus = MAP_READ_BONUS
            except Exception:
                map_bonus = 0.0

        # ------------------- no‑op penalty ---------------------------------
        noop_penalty = 0.0
        if node.parent is not None and _is_true_noop(node.parent.state, node.state):
            factor = _variant_factor(node.state)
            noop_penalty = NOOP_PENALTY_BASE * factor

        # Total increment before decay handling.
        total_increment = (
            discounted_reward
            + progress_bonus
            + door_bonus
            + map_bonus
            - noop_penalty
        )

        # Attenuate negative increments with a linear decay based on depth.
        if total_increment < 0:
            decay_factor = max(0.0, 1.0 - depth * NEG_DECAY_PER_LEVEL)
            total_increment *= decay_factor

        # ------------------------------------------------------------------
        # 4️⃣ Update node value (single‑player ⇒ always add).
        # ------------------------------------------------------------------
        node.value += total_increment

        # Move upward.
        node = node.parent
        depth += 1
