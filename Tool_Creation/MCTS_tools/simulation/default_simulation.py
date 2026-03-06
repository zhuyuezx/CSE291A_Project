"""
Default simulation: random rollout.

Play random actions from the given state until terminal or max depth,
then return the reward from the perspective player's viewpoint.
"""

import random


def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Random rollout simulation.

    Args:
        state:              GameState to simulate from (will be cloned).
        perspective_player: Player index whose reward we return.
        max_depth:          Maximum rollout steps.

    Returns:
        Float reward from perspective_player's viewpoint.
    """
    sim_state = state.clone()
    depth = 0
    while not sim_state.is_terminal() and depth < max_depth:
        action = random.choice(sim_state.legal_actions())
        sim_state.apply_action(action)
        depth += 1
    return sim_state.returns()[perspective_player]
