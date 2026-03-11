"""
Phase-specific prompt content for LLM heuristic optimization.

Provides per-phase descriptions and mechanics blocks that are injected into
the system and task sections when building prompts. See docs/heuristictips.md
and docs/MCTS_phase_interaction.md for the source material.
"""

from __future__ import annotations

# Phase-specific guidance for the system section (injected before 70/30 rule)
PHASE_DESCRIPTIONS: dict[str, str] = {
    "selection": (
        "PHASE: selection\n"
        "  • What it does: Walks down the tree from root to a leaf. Chooses which "
        "existing branch to explore next. Must balance exploration (UCB) and exploitation.\n"
        "  • Optimization goal: Improve how we RANK existing nodes — favor promising "
        "branches, deprioritize dead ends. Your heuristic adjusts node scores used by UCB1.\n"
        "  • Constraints: Called very often. Keep it CHEAP — no multi-step rollouts, "
        "no deep deadlock simulation. Rank nodes, don't simulate.\n"
        "  • Good patterns: bonus for more boxes on targets, bonus for lower box "
        "distance, penalize obvious deadlocks, novelty bonus for under-visited nodes.\n"
        "  • Avoid: expensive rollout logic, final reward shaping (that belongs in simulation)."
    ),
    "expansion": (
        "PHASE: expansion\n"
        "  • What it does: Creates new child nodes from a frontier node. Decides which "
        "actions to materialize into the tree and in what order.\n"
        "  • Optimization goal: PRUNE bad actions and ORDER remaining actions so promising "
        "ones are tried first. Filter deadlocks before they enter the tree.\n"
        "  • Constraints: Best place for hard constraints. Order actions; optionally "
        "filter some entirely. No rollout policies or value aggregation.\n"
        "  • Good patterns: reject pushes into non-target corners, reject wall deadlocks, "
        "prefer pushes that reduce box distance, deprioritize no-op player movement.\n"
        "  • Avoid: long rollout policies, reward aggregation, node-value update rules."
    ),
    "simulation": (
        "PHASE: simulation\n"
        "  • What it does: Rolls forward from a leaf state to estimate how promising it is. "
        "Returns a reward (e.g. 0–1) that flows into backpropagation.\n"
        "  • Optimization goal: Produce REWARDS that reflect true state quality. Shaped "
        "partial progress helps MCTS distinguish good from bad actions.\n"
        "  • Constraints: Must return a FLOAT. Reward MUST vary across states — flat "
        "rewards ≈ random play. Called thousands of times per move — keep it fast.\n"
        "  • Good patterns: shaped score (boxes on targets, distance improvement), "
        "penalize deadlocks/loops/stagnation, prefer pushes over wandering, early termination when stuck.\n"
        "  • Avoid: tree-level visit balancing, acceptance criteria for tools — this phase only scores rollouts."
    ),
    "backpropagation": (
        "PHASE: backpropagation\n"
        "  • What it does: Sends the simulation result back up the visited path. Updates "
        "node statistics (visits, value) that selection's UCB1 uses.\n"
        "  • Optimization goal: Control HOW strongly rollout evidence affects node values. "
        "Calibrate depth discount, solved vs partial progress, path length.\n"
        "  • Constraints: Only aggregates evidence — no move generation, no deadlock "
        "pruning, no rollout policy. Must stay coherent with selection's expectations.\n"
        "  • Good patterns: depth discount so shorter plans dominate, weight solved "
        "outcomes above partial progress, reduce credit for noisy weak rollouts.\n"
        "  • Avoid: move generation, deadlock pruning, rollout action-choice policy."
    ),
}

# Phase-specific "How it works" mechanics for the task section
PHASE_MECHANICS: dict[str, str] = {
    "selection": (
        "  - Receives (root, exploration_weight). Returns the chosen child index.\n"
        "  - Adjusts UCB scores or selection policy to favor promising branches.\n"
        "  - Called at every level during tree descent — keep it FAST.\n"
        "  - Must produce a valid child index; selection drives which branch gets expanded next."
    ),
    "expansion": (
        "  - Receives (node). Returns the action to expand (or orders/filters untried actions).\n"
        "  - Orders actions so promising ones are tried first; optionally prunes bad ones.\n"
        "  - Produces children that selection will later choose among.\n"
        "  - Best place for hard constraints (e.g. deadlock pruning)."
    ),
    "simulation": (
        "  - Called from a LEAF node, receives (state, perspective_player, max_depth).\n"
        "  - Must return a FLOAT reward backpropagated up the tree.\n"
        "  - Reward MUST vary across states so MCTS can distinguish good from bad actions.\n"
        "  - Flat rewards ≈ random play. Called thousands of times per move — keep it FAST."
    ),
    "backpropagation": (
        "  - Receives (node, reward). Returns None; updates node.value and node.visits up the tree.\n"
        "  - Propagates the simulation result to ancestors; selection's UCB1 uses these values.\n"
        "  - Controls how strongly rollout evidence affects node values (depth discount, etc.).\n"
        "  - Only aggregates evidence — no move generation or rollout policy."
    ),
}

# Unique phrases per phase for test assertions (must appear in that phase's prompt only)
PHASE_UNIQUE_PHRASES: dict[str, list[str]] = {
    "selection": [
        "RANK existing nodes",
        "node scores used by UCB1",
        "Rank nodes, don't simulate",
        "chosen child index",
    ],
    "expansion": [
        "PRUNE bad actions",
        "Filter deadlocks before they enter the tree",
        "non-target corners",
        "action to expand",
    ],
    "simulation": [
        "Returns a reward",
        "Produce REWARDS",
        "Must return a FLOAT",
        "flat rewards",
    ],
    "backpropagation": [
        "Sends the simulation result back up",
        "depth discount",
        "weight solved outcomes above partial progress",
        "node.value and node.visits",
    ],
}
