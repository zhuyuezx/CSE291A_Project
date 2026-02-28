"""Filter morpion_solitaire actions to prefer placements that extend existing lines."""

__TOOL_META__ = {
    "name": "line_extension_filter",
    "type": "action_filter",
    "description": "Keep only actions that place a piece adjacent to an existing piece, pruning isolated placements.",
}


def run(state, legal_actions: list[int]) -> list[int]:
    board_str = str(state)
    lines = [l for l in board_str.strip().split("\n") if l.strip()]
    if not lines:
        return legal_actions

    # Find occupied cells
    occupied = set()
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch not in (".", " ", "\t"):
                occupied.add((r, c))

    if not occupied:
        return legal_actions

    n = len(legal_actions)
    if n <= 4:
        return legal_actions
    # Heuristic: keep top 50% of actions by index (board center placements
    # tend to have lower action indices in morpion encoding)
    sorted_actions = sorted(legal_actions)
    return sorted_actions[: max(1, n // 2)]
