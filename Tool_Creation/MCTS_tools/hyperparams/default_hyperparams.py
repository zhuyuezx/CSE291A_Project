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

# ── Tool evolution configuration (new — all optional) ────────────────
# Set to True to enable; False to disable. With all False, the pipeline
# behaves identically to the original (no registry, no aggregator, no merge).
ENABLE_TOOL_REGISTRY = False     # True: record every installed tool to LLM/registry/tool_registry.json
ENABLE_AGGREGATOR = False       # True: inject strategic summary from past tools into each optimizer run
ENABLE_CLUSTER_MERGE = False    # True: periodically cluster and merge tools per phase
CLUSTER_MERGE_INTERVAL = 5      # run cluster+merge every N iterations (when ENABLE_CLUSTER_MERGE)
REGISTRY_HISTORY_LEN = 10        # how many past tools per phase to show (when aggregator/registry on)


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
