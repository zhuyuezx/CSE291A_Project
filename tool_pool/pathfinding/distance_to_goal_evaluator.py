"""Evaluate pathfinding states by Manhattan distance to goal (approximated via state string)."""

__TOOL_META__ = {
    "name": "distance_to_goal_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when agent is closer to goal. Parses state string for position markers.",
}


def run(state) -> float:
    if state.is_terminal():
        returns = state.returns()
        return float(returns[0]) if returns else 0.0

    board = str(state)
    lines = [l for l in board.strip().split("\n") if l.strip()]
    if not lines:
        return 0.0

    agent_pos = goal_pos = None
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch in ("A", "@", "P", "p"):   # agent markers vary by config
                agent_pos = (r, c)
            elif ch in ("G", "X", "*", "g"): # goal markers
                goal_pos = (r, c)

    if agent_pos is None or goal_pos is None:
        return 0.0

    max_dist = len(lines) + max(len(l) for l in lines)
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    # Closer = higher score; normalize to [-1, 1]
    return 1.0 - 2.0 * dist / max(max_dist, 1)
