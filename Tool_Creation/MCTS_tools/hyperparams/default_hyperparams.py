"""
Default MCTS hyperparameters and orchestrator configuration.

This file is the single source of truth for all tuning-related config.
The LLM optimizer can tune the values returned by ``get_hyperparams()``.
Game identity, training logic, and optimization orchestration params
are encoded as module-level constants.

The function signature must be:

    def get_hyperparams() -> dict

Returned dict keys:
    iterations         — int, MCTS iterations per move
    max_rollout_depth  — int, max simulation rollout depth
    exploration_weight — float, UCB1 exploration constant C
"""

# ── Game configuration ───────────────────────────────────────────────
GAME_NAME = "sokoban"
GAME_CLASS = "Sokoban"
GAME_MODULE = "mcts.games"
# Sokoban: max_steps = game step limit (unsolved games stop here). Must be >=
# longest solution you want to allow; otherwise eval reports 200 for all unsolved.
CONSTRUCTOR_KWARGS = {"max_steps": 1000}
TRAINING_LOGIC = "sokoban_training"

# ── Optimization configuration ───────────────────────────────────────
# PHASES: only phases that create MCTS tool Python files under MCTS_tools/<phase>/
# (selection, expansion, simulation, backpropagation). Omit "hyperparams" to avoid
# tuning get_hyperparams(); include it to also optimize engine parameters via LLM.
NUM_ITERS = 5
THREE_STEP = True
HISTORY_WINDOW = 3
PHASES = ["selection", "expansion", "simulation", "backpropagation"] # tool-creation phases only
LOGGING = True
# LLM smoke test: repair attempts when generated code fails
MAX_REPAIR_ATTEMPTS = 5


def get_hyperparams():
    """
    Return MCTS hyperparameters as a dict.

    Parameters (returned keys)
    --------------------------
    iterations : int
        Number of MCTS tree-search iterations per move.
        Higher values give stronger play but cost more time.
        Typical range for Sokoban: 100–2000.
    max_rollout_depth : int
        Maximum number of steps in a simulation rollout.
        Must be large enough to reach terminal or near-terminal states.
    exploration_weight : float
        UCB1 exploration constant C.  Controls the tradeoff between
        exploring new branches vs. exploiting known good ones.
        Default sqrt(2) ≈ 1.41.  Lower → more exploitation, higher → more exploration.
    """
    return {
        "iterations": 500,
        "max_rollout_depth": 1000,
        "exploration_weight": 1.41,
    }
