"""Score 2048 states by tile monotonicity: prefer boards where tiles decrease away from a corner."""

__TOOL_META__ = {
    "name": "monotonicity_evaluator",
    "type": "state_evaluator",
    "description": "Reward boards where tile values are monotonically ordered toward a corner (corner strategy).",
}


def _parse_board(state) -> list[list[int]]:
    board_str = str(state)
    rows = []
    for line in board_str.strip().split("\n"):
        nums = []
        for token in line.split():
            try:
                nums.append(int(token))
            except ValueError:
                pass
        if nums:
            rows.append(nums)
    return rows


def run(state) -> float:
    if state.is_terminal():
        return -1.0

    board = _parse_board(state)
    if not board:
        return 0.0

    score = 0.0
    total = 0
    for row in board:
        for i in range(len(row) - 1):
            total += 1
            if row[i] >= row[i + 1]:
                score += 1  # left-to-right decreasing
    for col_i in range(len(board[0])):
        col = [board[r][col_i] for r in range(len(board)) if col_i < len(board[r])]
        for i in range(len(col) - 1):
            total += 1
            if col[i] >= col[i + 1]:
                score += 1  # top-to-bottom decreasing

    if total == 0:
        return 0.0
    return 2.0 * (score / total) - 1.0  # maps [0,1] → [-1, 1]
