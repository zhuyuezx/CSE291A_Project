"""
LLM-generated MCTS tool: hyperparams
Description: Return tuned MCTS hyperparameters for Sokoban.
Generated:   2026-03-06T23:44:58.680043
"""

def get_hyperparams():
    """
    Return MCTS hyperparameters as a dict.

    Parameters (returned keys)
    --------------------------
    iterations : int
        Number of MCTS tree-search iterations per move.
        Increased to allow deeper exploration on complex Sokoban levels.
    max_rollout_depth : int
        Maximum number of steps in a simulation rollout.
        Reduced to focus rollouts on realistic solution lengths and save time.
    exploration_weight : float
        UCB1 exploration constant C.
        Raised to promote exploring less‑visited push actions and avoid premature
        convergence on sub‑optimal branches.
    """
    return {
        "iterations": 1000,          # more iterations for stronger search
        "max_rollout_depth": 130,    # enough for typical solution depth, less waste
        "exploration_weight": 2.0,   # more exploration for deeper puzzles
    }
