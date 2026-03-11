"""
LLM-generated MCTS tool: backpropagation
Description: No changes needed – code is correct and efficient.
Generated:   2026-03-10T01:23:59.655926
"""

"""
Enhanced backpropagation for TextWorld Benchmark.

Changes relative to the previous version:
  * Larger shaping bonuses (map read, coin pick‑up, distance improvement).
  * More robust map‑read detection (handles `map_read`, `map_readable`,
    and the presence of a "map" item in the inventory).
  * New generic state‑change bonus rewarding any change in room,
    door configuration, or newly acquired inventory items.
  * Stronger penalty for immediate action repetition (prevents open/close loops).
  * Much milder depth discount (γ = 0.99) so the terminal reward
    propagates almost unchanged.
"""

from typing import Any

# ----------------------------------------------------------------------
# Hyper‑parameters (tuned for the benchmark)
# ----------------------------------------------------------------------
MAP_READ_BONUS: float = 1.0       # reward for reading the map the first time
COIN_BONUS: float = 1.2           # reward for picking up the coin the first time
DIST_WEIGHT: float = 0.30         # weight for distance‑to‑goal improvement
STATE_ROOM_BONUS: float = 0.05    # reward for changing rooms
STATE_DOOR_BONUS: float = 0.03    # reward for any change in door layout
STATE_ITEM_BONUS: float = 0.04    # reward per newly acquired inventory item
REPEAT_ACTION_PENALTY: float = 0.20  # penalty for taking the same action twice in a row
DISCOUNT: float = 0.99            # γ used to discount reward with depth (almost no discount)


def _distance_improvement(parent_state: Any, cur_state: Any) -> float:
    """Return a positive bonus proportional to distance reduction, or 0."""
    # distance_to_goal may be a method or a plain attribute
    parent_dist = (
        parent_state.distance_to_goal()
        if callable(getattr(parent_state, "distance_to_goal", None))
        else getattr(parent_state, "distance_to_goal", None)
    )
    cur_dist = (
        cur_state.distance_to_goal()
        if callable(getattr(cur_state, "distance_to_goal", None))
        else getattr(cur_state, "distance_to_goal", None)
    )
    if parent_dist is None or cur_dist is None:
        return 0.0
    if cur_dist < parent_dist:
        return DIST_WEIGHT * (parent_dist - cur_dist)
    return 0.0


def _map_read_bonus(parent_state: Any, cur_state: Any) -> float:
    """
    Detect the first moment the map becomes readable.
    Handles three possible signals:
      * Boolean attribute `map_read`
      * Boolean attribute `map_readable`
      * Presence of a "map" item in the inventory *and* the flag becoming true.
    """
    def _read_flag(state: Any) -> bool:
        flag = getattr(state, "map_read", None)
        if flag is None:
            flag = getattr(state, "map_readable", False)
        return bool(flag)

    parent_read = _read_flag(parent_state)
    cur_read = _read_flag(cur_state)

    # If the flag just turned true, give the bonus.
    if cur_read and not parent_read:
        return MAP_READ_BONUS

    # Fallback: sometimes the flag never flips; reward the action that
    # acquires the map item *and* subsequently allows reading.
    if hasattr(cur_state, "inventory_items") and hasattr(parent_state, "inventory_items"):
        parent_inv = set(parent_state.inventory_items)
        cur_inv = set(cur_state.inventory_items)
        if "map" in cur_inv and "map" not in parent_inv:
            # taking the map is a prerequisite, give a smaller bonus
            return MAP_READ_BONUS * 0.5
    return 0.0


def _coin_bonus(parent_state: Any, cur_state: Any) -> float:
    """Detect first acquisition of the coin."""
    if not (hasattr(cur_state, "inventory_items") and hasattr(parent_state, "inventory_items")):
        return 0.0
    parent_inv = set(parent_state.inventory_items)
    cur_inv = set(cur_state.inventory_items)
    if "coin" in cur_inv and "coin" not in parent_inv:
        return COIN_BONUS
    return 0.0


def _state_change_bonus(parent_state: Any, cur_state: Any) -> float:
    """
    General purpose shaping that rewards any observable progress:
      * Room change
      * Door layout change
      * New inventory items
    """
    bonus = 0.0

    # Room change
    if getattr(parent_state, "room", None) != getattr(cur_state, "room", None):
        bonus += STATE_ROOM_BONUS

    # Door configuration change (simple structural equality check)
    if getattr(parent_state, "doors", None) != getattr(cur_state, "doors", None):
        bonus += STATE_DOOR_BONUS

    # New inventory items (e.g., map, key, coin)
    if hasattr(parent_state, "inventory_items") and hasattr(cur_state, "inventory_items"):
        parent_items = set(parent_state.inventory_items)
        cur_items = set(cur_state.inventory_items)
        new_items = cur_items - parent_items
        bonus += STATE_ITEM_BONUS * len(new_items)

    return bonus


def _repeat_action_penalty(node: Any) -> float:
    """
    Penalise immediate repetition of the same action.
    This is a strong deterrent for open/close loops or redundant looks.
    """
    if node.parent is None:
        return 0.0
    cur_action = getattr(node, "parent_action", None)
    prev_action = getattr(node.parent, "parent_action", None)
    if cur_action is not None and cur_action == prev_action:
        return REPEAT_ACTION_PENALTY
    return 0.0


def default_backpropagation(node: Any, reward: float) -> None:
    """
    Backpropagate the simulation result, enriching it with intermediate
    progress signals and a mild depth discount.

    Args:
        node:   The leaf MCTSNode where the simulation originated.
        reward: The raw terminal reward (from the root player's perspective).
    """
    # Determine the root node once – retained for potential multi‑player extensions.
    root = node
    while root.parent is not None:
        root = root.parent
    # No‑op use of root.state.current_player() kept for compatibility.
    _ = root.state.current_player()  # type: ignore

    depth = 0  # distance from leaf (depth 0) upwards
    while node is not None:
        node.visits += 1

        # Apply a very mild discount to the terminal reward.
        discounted_reward = reward * (DISCOUNT ** depth)

        # Start with the discounted terminal reward.
        aug_reward: float = discounted_reward

        if node.parent is not None:
            parent_state = node.parent.state
            cur_state = node.state

            # Shaping bonuses
            aug_reward += _map_read_bonus(parent_state, cur_state)
            aug_reward += _coin_bonus(parent_state, cur_state)
            aug_reward += _distance_improvement(parent_state, cur_state)
            aug_reward += _state_change_bonus(parent_state, cur_state)

            # Penalties
            aug_reward -= _repeat_action_penalty(node)

        # Single‑player game: accumulate the (augmented) reward.
        node.value += aug_reward

        # Ascend the tree.
        node = node.parent
        depth += 1
