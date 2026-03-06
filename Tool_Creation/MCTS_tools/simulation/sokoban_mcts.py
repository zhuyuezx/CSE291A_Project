"""
LLM-generated MCTS tool: simulation
Description: Fix corner deadlock detection logic
Generated:   2026-03-06T03:05:06.323260
"""

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Biased rollout simulation that prefers actions reducing total box distance
    and avoids immediate corner deadlocks.

    Args:
        state: GameState to simulate from (will be cloned).
        perspective_player: Player index whose reward we return.
        max_depth: Maximum rollout steps.

    Returns:
        Float reward from perspective_player's viewpoint.
    """
    import random, math

    sim_state = state.clone()
    depth = 0
    beta = 0.5  # strength of bias (penalises distance increase)

    # helper: detect simple corner deadlock (box not on target with two orthogonal walls)
    def has_deadlock(tmp_state):
        for b in tmp_state.boxes:
            if b in tmp_state.targets:
                continue
            r, c = b
            # check each orthogonal corner pair
            if ((r - 1, c) in tmp_state.walls and (r, c - 1) in tmp_state.walls) or \
               ((r - 1, c) in tmp_state.walls and (r, c + 1) in tmp_state.walls) or \
               ((r + 1, c) in tmp_state.walls and (r, c - 1) in tmp_state.walls) or \
               ((r + 1, c) in tmp_state.walls and (r, c + 1) in tmp_state.walls):
                return True
        return False

    while not sim_state.is_terminal() and depth < max_depth:
        cur_dist = sim_state.total_box_distance()
        legal = sim_state.legal_actions()
        weights = []
        for a in legal:
            tmp = sim_state.clone()
            tmp.apply_action(a)
            new_dist = tmp.total_box_distance()
            delta = new_dist - cur_dist                # >0 → worse, <0 → better
            w = math.exp(-beta * delta)                # >1 for improvements, <1 for regressions
            if has_deadlock(tmp):
                w = 0.0                                 # discard actions causing instant deadlock
            weights.append(w)

        # Fallback to uniform random if all actions are dead‑ended
        if all(w == 0 for w in weights):
            action = random.choice(legal)
        else:
            action = random.choices(legal, weights=weights, k=1)[0]

        sim_state.apply_action(action)
        depth += 1

    return sim_state.returns()[perspective_player]
