__TOOL_META__ = {
    "name": "immediate_win_block_filter",
    "type": "action_filter",
    "description": "Filters legal actions to those that either win the game immediately for the current player or block an opponent's immediate win. By focusing MCTS on these critical tactical moves, the agent avoids losing simple threats and exploits winning chances it would otherwise miss.",
}


def run(state, legal_actions: list[int]) -> list[int]:
    """
    Filters legal actions to those that either win the game immediately for the current player
    or block an opponent's immediate win. By focusing MCTS on these critical tactical moves,
    the agent avoids losing simple threats and exploits winning chances it would otherwise miss.

    Args:
        state: An OpenSpiel game state object with methods:
            - state.legal_actions() -> list[int]
            - state.clone() -> State
            - state.apply_action(action: int) -> None (mutates in place)
            - state.is_terminal() -> bool
            - state.returns() -> list[float]
            - state.current_player() -> int
            - str(state) -> str (human‑readable board)
        legal_actions: List of legal action ids for the current player.

    Returns:
        Subset of `legal_actions` according to the heuristic.
    """
    # Safety checks
    if not legal_actions:
        return []
    if state.is_terminal():
        return []

    current_player = state.current_player()
    opponent = 1 - current_player  # works for two‑player zero‑sum games

    # 1. Immediate winning moves
    winning_actions = []
    for a in legal_actions:
        s1 = state.clone()
        s1.apply_action(a)
        if s1.is_terminal():
            # Verify that the terminal state is a win for the current player.
            # Some games may return 0 for draw; we treat any positive reward as a win.
            returns = s1.returns()
            if len(returns) > current_player and returns[current_player] > 0.0:
                winning_actions.append(a)
            else:
                # Even if reward is 0 (draw) we still consider it a terminal win‑avoidance,
                # but we keep searching for true winning actions.
                pass
    if winning_actions:
        return winning_actions

    # 2. Block opponent's immediate win (i.e., keep opponent from having a forced win next turn)
    safe_actions = []
    for a in legal_actions:
        s2 = state.clone()
        s2.apply_action(a)

        # If after our move the game is already terminal, we cannot block anything;
        # treat this as a safe move (already handled above for wins).
        if s2.is_terminal():
            safe_actions.append(a)
            continue

        opp_legal = s2.legal_actions()
        opponent_can_win = False

        for opp_a in opp_legal:
            s3 = s2.clone()
            s3.apply_action(opp_a)
            if s3.is_terminal():
                opp_returns = s3.returns()
                if len(opp_returns) > opponent and opp_returns[opponent] > 0.0:
                    opponent_can_win = True
                    break

        if not opponent_can_win:
            safe_actions.append(a)

    if safe_actions:
        return safe_actions

    # Fallback: no winning or blocking moves found; return all legal actions.
    return legal_actions