"""
LLM-generated MCTS tool: simulation
Description: Add no‑undo bias to break back‑and‑forth loops.
Generated:   2026-03-06T03:09:45.423830
"""

def default_simulation(state, perspective_player: int, max_depth: int = 1000) -> float:
    """
    Biased rollout simulation that prefers actions reducing total box distance,
    avoids immediate corner deadlocks, rewards actions that move a box, and
    discourages immediately undoing the previous move.
    """
    import random, math

    sim_state = state.clone()
    depth = 0
    beta = 0.5          # distance bias strength
    push_factor = 2.0   # extra weight for actions that move a box
    opposite = {0: 1, 1: 0, 2: 3, 3: 2}
    last_action = None

    # simple corner deadlock detection (non‑target box in a wall corner)
    def has_deadlock(tmp_state):
        for b in tmp_state.boxes:
            if b in tmp_state.targets:
                continue
            r, c = b
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
            delta = new_dist - cur_dist                     # >0 worse, <0 better
            w = math.exp(-beta * delta)                     # distance bias
            if has_deadlock(tmp):
                w = 0.0                                      # discard deadlocks
            if tmp.boxes != sim_state.boxes:
                w *= push_factor                             # boost pushes
            # discourage immediate reversal of the last move
            if last_action is not None and a == opposite[last_action]:
                w *= 0.1                                     # strong penalty
            weights.append(w)

        # fallback to uniform random if all weights are zero
        if all(w == 0 for w in weights):
            action = random.choice(legal)
        else:
            action = random.choices(legal, weights=weights, k=1)[0]

        sim_state.apply_action(action)
        last_action = action
        depth += 1

    return sim_state.returns()[perspective_player]
