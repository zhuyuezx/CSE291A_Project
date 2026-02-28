"""Score 2048 states by number of empty cells (more empty = more options = better)."""

__TOOL_META__ = {
    "name": "empty_cell_evaluator",
    "type": "state_evaluator",
    "description": "Score states higher when more cells are empty (proxy for game not being lost).",
}

_BOARD_SIZE = 16  # 4x4


def run(state) -> float:
    if state.is_terminal():
        return -1.0

    board_str = str(state)
    empty = sum(1 for token in board_str.split() if token == "0")
    # Normalize: 0 empty → -1.0, full board empty → +1.0
    return 2.0 * empty / _BOARD_SIZE - 1.0
