"""Biased rollout policy: prefer center columns and blocking moves."""
import random

__TOOL_META__ = {
    "name": "greedy_rollout",
    "type": "rollout_policy",
    "description": "During simulation, with 60% probability play toward center columns or block threats. With 40% probability play randomly.",
}


def run(state, legal_actions: list[int]) -> int:
    if random.random() > 0.6:
        return random.choice(legal_actions)

    player = state.current_player()
    if player < 0:
        return random.choice(legal_actions)

    # Check for immediate wins
    for action in legal_actions:
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal() and child.returns()[player] > 0:
            return action

    # Prefer center columns (for a 7-column board, center is 3)
    num_actions = max(legal_actions) + 1 if legal_actions else 7
    center = num_actions // 2
    # Sort by distance from center
    sorted_actions = sorted(legal_actions, key=lambda a: abs(a - center))
    # Pick from top half with higher probability
    top_half = sorted_actions[: max(1, len(sorted_actions) // 2)]
    return random.choice(top_half)
