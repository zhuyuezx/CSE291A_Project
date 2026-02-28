"""Evaluate morpion_solitaire states by piece density (more pieces = better)."""

__TOOL_META__ = {
    "name": "density_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when more pieces are on the board. Proxies for progress toward max pieces.",
}

_MAX_PIECES = 35.0  # theoretical max for morpion solitaire


def run(state) -> float:
    if state.is_terminal():
        # returns()[0] is pieces placed, normalize to [-1, 1]
        raw = state.returns()[0]
        return max(-1.0, min(1.0, 2.0 * raw / _MAX_PIECES - 1.0))

    board_str = str(state)
    piece_count = sum(
        1 for ch in board_str if ch not in (".", " ", "\n", "\t", "|", "-", "+")
    )
    # Normalize to [-1, 1] relative to max pieces
    return max(-1.0, min(1.0, 2.0 * piece_count / (_MAX_PIECES * 3) - 1.0))
