"""Evaluate Connect Four states by counting pieces in center columns."""

__TOOL_META__ = {
    "name": "center_column_bias",
    "type": "state_evaluator",
    "description": "Score states higher when the current player has more pieces in center columns. Game-agnostic: uses observation tensor to infer board shape and center.",
}


def run(state) -> float:
    board_str = str(state)
    lines = [l for l in board_str.strip().split("\n") if l.strip()]

    # Count pieces per column for current player
    player = state.current_player()
    if player < 0:
        return 0.0

    # Parse the board string to find piece positions
    # Connect Four board string has 'x' and 'o' characters
    my_char = "x" if player == 0 else "o"
    opp_char = "o" if player == 0 else "x"

    my_center = 0
    opp_center = 0
    total_cols = 0

    for line in lines:
        chars = [c for c in line if c in ("x", "o", ".")]
        if not chars:
            continue
        total_cols = max(total_cols, len(chars))
        center_start = len(chars) // 2 - 1
        center_end = len(chars) // 2 + 1
        for i, c in enumerate(chars):
            if center_start <= i <= center_end:
                if c == my_char:
                    my_center += 1
                elif c == opp_char:
                    opp_center += 1

    if my_center + opp_center == 0:
        return 0.0

    score = (my_center - opp_center) / max(my_center + opp_center, 1)
    return max(-1.0, min(1.0, score))
