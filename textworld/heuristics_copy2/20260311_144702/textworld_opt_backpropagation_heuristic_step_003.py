"""
LLM-generated MCTS tool: backpropagation
Description: Fixed __future__ import placement, ensured `child_state` is always defined, corrected usage of API attributes, and added missing imports/constants for a self‑contained implementation.
Generated:   2026-03-11T15:23:03.848190
"""

# -*- coding: utf-8 -*-
"""Back‑propagation utilities for TextWorld MCTS.

This module provides a single function ``default_backpropagation`` that
updates the statistics of a leaf MCTS node and all its ancestors using
richer shaping signals (map‑take/read bonuses, no‑op penalties, distance
improvement, etc.).  All hyper‑parameters are defined at module level
so they can be tuned easily.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Hyper‑parameters (tuned for the TextWorld benchmark)
# --------------------------------------------------------------------------- #
DISCOUNT: float = 0.99          # depth discount factor
DIST_WEIGHT: float = 0.6        # weight for normalised distance improvement
MAP_TAKE_BONUS: float = 0.9     # bonus when acquiring the map item
MAP_READ_BONUS: float = 0.7     # bonus when reading the map
GOAL_BONUS: float = 0.7         # bonus when the goal room becomes known
NOOP_PENALTY: float = 0.5       # penalty for true no‑ops
STEP_COST: float = 0.03         # tiny per‑step cost to encourage brevity
DONE_BONUS: float = 0.2         # extra reward for reaching a terminal state

def default_backpropagation(node: Any, reward: float) -> None:
    """
    Back‑propagation with richer shaping signals for a single‑player TextWorld
    MCTS search.

    Parameters
    ----------
    node : Any
        The leaf ``MCTSNode`` from which the simulation originated. The node
        must expose the attributes ``parent``, ``state``, ``visits`` and
        ``value`` (the latter two may be missing on new nodes).
    reward : float
        Raw simulation reward from the root player's perspective.
    """
    depth = 0  # distance from leaf (depth = 0) upward in the tree

    while node is not None:
        # --------------------------------------------------------------- #
        # 1️⃣  Visit count
        # --------------------------------------------------------------- #
        node.visits = getattr(node, "visits", 0) + 1

        # --------------------------------------------------------------- #
        # 2️⃣  Discounted raw reward
        # --------------------------------------------------------------- #
        discounted_reward = reward * (DISCOUNT ** depth)

        # --------------------------------------------------------------- #
        # 3️⃣  Shaping bonuses / penalties
        # --------------------------------------------------------------- #
        bonus = 0.0

        # ``child_state`` is always the state stored in the current node.
        child_state = getattr(node, "state", None)

        if node.parent is not None and child_state is not None:
            parent_state = getattr(node.parent, "state", None)

            # ---- a) Penalty for true no‑ops ---------------------------------
            #   * identical low‑level state_key
            #   * identical observation text (covers look/inventory)
            try:
                if parent_state.state_key() == child_state.state_key():
                    bonus -= NOOP_PENALTY
            except Exception:
                pass

            try:
                if (hasattr(parent_state, "observation_text")
                        and hasattr(child_state, "observation_text")
                        and parent_state.observation_text() == child_state.observation_text()):
                    bonus -= NOOP_PENALTY
            except Exception:
                pass

            # ---- b) Bonus for taking the map (pre‑read) --------------------
            try:
                # ``inventory_items`` is a tuple according to the public API
                parent_inv = getattr(parent_state, "inventory_items", ())
                child_inv = getattr(child_state, "inventory_items", ())
                if "map" not in parent_inv and "map" in child_inv:
                    bonus += MAP_TAKE_BONUS
            except Exception:
                pass

            # ---- c) Bonus for reading the map --------------------------------
            try:
                if not getattr(parent_state, "map_read", False) and getattr(child_state, "map_read", False):
                    bonus += MAP_READ_BONUS
            except Exception:
                pass

            # ---- d) Bonus for revealing the goal room -----------------------
            try:
                parent_known = getattr(parent_state, "known_goal_room", None)
                child_known = getattr(child_state, "known_goal_room", None)
                if parent_known != child_known and child_known is not None:
                    bonus += GOAL_BONUS
            except Exception:
                pass

            # ---- e) Normalised distance improvement (after goal known) -----
            try:
                if getattr(child_state, "known_goal_room", None) is not None:
                    parent_dist = parent_state.distance_to_goal()
                    child_dist = child_state.distance_to_goal()
                    delta = parent_dist - child_dist  # >0 → got closer
                    if delta > 0:
                        norm_improvement = delta / max(parent_dist, 1)
                        bonus += DIST_WEIGHT * norm_improvement
            except Exception:
                pass

        # --------------------------------------------------------------- #
        # 4️⃣  Combine, add per‑step cost, optional terminal bonus and clip
        # --------------------------------------------------------------- #
        shaped_value = discounted_reward + bonus - STEP_COST

        try:
            if child_state is not None and child_state.is_terminal():
                shaped_value += DONE_BONUS
        except Exception:
            pass

        # Clip to the valid reward range used by the environment (typically [-1, 1])
        shaped_value = max(min(shaped_value, 1.0), -1.0)

        # --------------------------------------------------------------- #
        # 5️⃣  Accumulate back‑propagated value
        # --------------------------------------------------------------- #
        node.value = getattr(node, "value", 0.0) + shaped_value

        # --------------------------------------------------------------- #
        # 6️⃣  Move up the tree
        # --------------------------------------------------------------- #
        node = node.parent
        depth += 1
