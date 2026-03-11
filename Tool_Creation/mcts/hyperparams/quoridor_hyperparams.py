"""
Quoridor MCTS hyperparameters and orchestrator configuration.

This file is the single source of truth for all Quoridor tuning-related
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
GAME_NAME = "quoridor"
GAME_CLASS = "Quoridor"
GAME_MODULE = "mcts.games"
CONSTRUCTOR_KWARGS = {}
TRAINING_LOGIC = "quoridor_training"

# ── Optimization configuration ───────────────────────────────────────
# Key insight from the hand-tuned C++ Quoridor MCTS:
#   Selection     — standard UCT, no changes needed
#   Expansion     — heavily heuristic (probable wall pruning, path-disturbing walls)
#   Simulation    — heavily heuristic (70% shortest-path, 20% wall, 10% explore)
#   Backprop      — standard alternating-player, no changes needed
# Therefore we optimize ONLY expansion and simulation.
NUM_ITERS = 5
THREE_STEP = True
HISTORY_WINDOW = 3
PHASES = ["simulation", "expansion"]
LOGGING = True

# ── Default tool overrides ───────────────────────────────────────────
# The MCTS_tools/ defaults are Sokoban-tuned.  For Quoridor we start
# from a game-agnostic random-rollout simulation.
DEFAULT_TOOL_FILES = {
    "simulation": "MCTS_tools/simulation/generic_simulation.py",
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
        Typical range for Quoridor: 200–2000 (adversarial game, needs depth).
    max_rollout_depth : int
        Maximum number of steps in a simulation rollout.
        Quoridor games can run ≥ 100 moves; set high enough to reach terminal.
    exploration_weight : float
        UCB1 exploration constant C.  Controls the tradeoff between
        exploring new branches vs. exploiting known good ones.
        Default sqrt(2) ≈ 1.41.  Lower → more exploitation, higher → more exploration.
    """
    return {
        "iterations": 200,
        "max_rollout_depth": 200,
        "exploration_weight": 1.41,
    }
