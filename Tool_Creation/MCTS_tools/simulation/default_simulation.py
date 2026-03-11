"""
Generic random-rollout simulation — game agnostic.

Works for any game that implements the GameState interface.
The LLM optimizer will replace this with Quoridor-specific heuristics.
"""
import random


def default_simulation(state, perspective_player: int, max_depth: int = 200) -> float:
    """
    Random rollout simulation.

    Plays random legal moves until a terminal state or max_depth is reached.
    Returns the reward from perspective_player's point of view.

    Args:
        state:              GameState (will be cloned internally).
        perspective_player: Index of the player whose reward we return.
        max_depth:          Maximum rollout steps.

    Returns:
        float: Reward for perspective_player (typically in [-1, 1]).
    """
    sim = state.clone()

    for _ in range(max_depth):
        if sim.is_terminal():
            break
        actions = sim.legal_actions()
        if not actions:
            break
        sim.apply_action(random.choice(actions))

    returns = sim.returns()
    if isinstance(returns, (list, tuple)) and len(returns) > perspective_player:
        return returns[perspective_player]
    return returns if isinstance(returns, (int, float)) else 0.0
