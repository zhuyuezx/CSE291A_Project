"""Filter pathfinding actions to avoid immediately reversing direction."""
import random

__TOOL_META__ = {
    "name": "backtrack_pruning_filter",
    "type": "action_filter",
    "description": "Remove the action that would immediately reverse the last move, reducing backtracking.",
}

# Action encoding for pathfinding (up=0,down=1,left=2,right=3,stay=4)
_REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}

_last_action: dict = {}   # state_id → last action (best-effort)


def run(state, legal_actions: list[int]) -> list[int]:
    state_id = id(state)
    last = _last_action.get(state_id)
    if last is not None and last in _REVERSE:
        reverse = _REVERSE[last]
        filtered = [a for a in legal_actions if a != reverse]
        if filtered:
            return filtered
    return legal_actions
