__TOOL_META__ = {
    "name": "corner_max_tile_evaluator",
    "type": "state_evaluator",
    "description": "Assigns a higher evaluation when the largest tile (or highest‑valued feature) resides in one of the four corners of the board. Keeping the biggest tile in a corner encourages a monotonic “snake” layout, reduces fragmentation, and preserves merge opportunities, which directly addresses the observed dead‑ends caused by high tiles being scattered across the grid.",
}

import math
import re

def _safe_observation(state):
    """Return a flat list of numeric observations or an empty list on failure."""
    # For 2048, observation_tensor() fails on chance nodes (player -1).
    # We prioritize the string fallback for 2048 as it is more robust.
    try:
        s = str(state)
        # Extract all numbers from the board string
        nums = re.findall(r"\d+", s)
        if len(nums) >= 16:
            # Most likely the 4x4 grid
            return [float(x) for x in nums[:16]]
    except Exception:
        pass

    try:
        if hasattr(state, "is_chance_node") and state.is_chance_node():
            pass # Skip observation_tensor on chance nodes
        elif hasattr(state, "observation_tensor"):
            obs = state.observation_tensor()
            if obs is not None:
                if hasattr(obs, "tolist"):
                    obs = obs.tolist()
                return list(obs)
    except Exception:
        pass
    
    return []

def run(state) -> float:
    """
    Assigns a higher evaluation when the largest tile resides in a corner.
    """
    # Terminal states: use the actual game returns.
    try:
        if hasattr(state, "is_terminal") and state.is_terminal():
            returns = state.returns()
            # For 2048 single player, returns is [score]
            if returns:
                return 1.0 if returns[0] > 0 else -1.0
            return 0.0
    except Exception:
        pass

    obs = _safe_observation(state)

    if not obs:
        return 0.0

    # Find index and value of the maximum tile.
    max_idx = -1
    max_val = -math.inf
    for i, v in enumerate(obs):
        try:
            val = float(v)
        except Exception:
            continue
        if val > max_val:
            max_val = val
            max_idx = i

    if max_idx == -1 or max_val <= 0:
        return 0.0

    length = len(obs)
    size = int(round(math.sqrt(length)))
    if size * size == length:
        width = height = size
    else:
        width = int(math.floor(math.sqrt(length)))
        if width == 0:
            width = 1
        height = int(math.ceil(length / width))

    row = max_idx // width
    col = max_idx % width

    is_corner = (
        (row == 0 and col == 0) or
        (row == 0 and col == width - 1) or
        (row == height - 1 and col == 0) or
        (row == height - 1 and col == width - 1)
    )

    if is_corner:
        return 1.0

    # Manhattan distance to the nearest corner (normalized).
    d_top = row + col
    d_bottom = (height - 1 - row) + (width - 1 - col)
    d_left = row + (width - 1 - col)
    d_right = (height - 1 - row) + col
    min_dist = min(d_top, d_bottom, d_left, d_right)
    max_dist = (height - 1) + (width - 1)
    if max_dist == 0:
        return 0.0
    score = - (min_dist / max_dist)
    # Clamp to the expected range.
    if score < -1.0:
        score = -1.0
    return score