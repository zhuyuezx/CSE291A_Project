"""Filter actions to prioritize winning moves and blocking opponent wins."""

__TOOL_META__ = {
    "name": "threat_detector",
    "type": "action_filter",
    "description": "Prioritize actions that win immediately or block an opponent's immediate win. Falls back to all legal actions if no threats found.",
}


def run(state, legal_actions: list[int]) -> list[int]:
    player = state.current_player()
    if player < 0:
        return legal_actions

    # Check for immediate wins
    winning_moves = []
    for action in legal_actions:
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            returns = child.returns()
            if returns[player] > 0:
                winning_moves.append(action)

    if winning_moves:
        return winning_moves

    # Check for moves that block opponent's immediate win
    # Simulate opponent having the turn by checking each column
    blocking_moves = []
    for action in legal_actions:
        # Check if opponent could win by playing this action
        # We do this by seeing if the next state would give opponent a winning move
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            continue
        opp_legal = child.legal_actions()
        opponent_can_win = False
        for opp_action in opp_legal:
            grandchild = child.clone()
            grandchild.apply_action(opp_action)
            if grandchild.is_terminal() and grandchild.returns()[1 - player] > 0:
                opponent_can_win = True
                break
        if not opponent_can_win:
            blocking_moves.append(action)

    if blocking_moves:
        return blocking_moves

    return legal_actions
