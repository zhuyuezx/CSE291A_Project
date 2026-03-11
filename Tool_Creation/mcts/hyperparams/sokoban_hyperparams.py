"""
Sokoban MCTS hyperparameters and orchestrator configuration.

This file is the single source of truth for all Sokoban tuning-related
config.  The LLM optimizer can tune the values returned by
``get_hyperparams()``.  Game identity, training logic, and optimization
orchestration params are encoded as module-level constants.

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
CONSTRUCTOR_KWARGS = {}
TRAINING_LOGIC = "sokoban_training"

# ── Optimization configuration ───────────────────────────────────────
# PHASES: only phases that create MCTS tool Python files under MCTS_tools/<phase>/
# (selection, expansion, simulation, backpropagation). Omit "hyperparams" to avoid
# tuning get_hyperparams(); include it to also optimize engine parameters via LLM.
NUM_ITERS = 5
THREE_STEP = True
HISTORY_WINDOW = 3
PHASES = ["simulation", "expansion"]  # tool-creation phases only
LOGGING = True

# ── Default tool overrides ───────────────────────────────────────────
# The default_simulation.py is now game-agnostic.  Sokoban uses its own
# Sokoban-tuned rollout as the starting point for LLM optimization.
DEFAULT_TOOL_FILES = {
    "simulation": "MCTS_tools/simulation/sokoban_default_simulation.py",
}

# ── Tool evolution configuration ─────────────────────────────────────
ENABLE_TOOL_REGISTRY = False
ENABLE_AGGREGATOR = False
ENABLE_CLUSTER_MERGE = False
CLUSTER_MERGE_INTERVAL = 5
REGISTRY_HISTORY_LEN = 10


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
        "iterations": 200,
        "max_rollout_depth": 500,
        "exploration_weight": 1.41,
    }
