"""
Default heuristic functions for MCTS.

These are the **"tools"** that the LLM agent can read, analyze, and rewrite
to improve MCTS play quality.  Each function has a clear docstring contract
so the LLM understands inputs/outputs.

──────────────────────────────────────────────────────────────
  MODIFIABLE BY LLM — this file is the optimisation target.
──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .game_interface import GameState


# ======================================================================
# 1. Rollout (simulation) policy
# ======================================================================

def random_rollout_policy(state: GameState) -> Any:
    """
    Choose an action during the simulation (rollout) phase.

    Args:
        state: Current game state (non-terminal, has legal actions).

    Returns:
        One of state.legal_actions().

    Default: uniform random.
    The LLM can replace this with a smarter policy (e.g., prefer centre
    columns in Connect Four, capture moves in Chess, etc.).
    """
    return random.choice(state.legal_actions())


# ======================================================================
# 2. Static evaluation (optional early-termination of rollouts)
# ======================================================================

def null_evaluation(state: GameState, perspective_player: int) -> float | None:
    """
    Optionally evaluate a non-terminal state and return a value estimate.

    Args:
        state:              Current (possibly non-terminal) game state.
        perspective_player: The player whose perspective the value is from.

    Returns:
        A float in [-1, 1] if you want to short-circuit the rollout,
        or None to let the rollout continue to a terminal state.

    Default: always returns None (pure Monte-Carlo rollout).
    The LLM can add a heuristic board evaluator here.
    """
    return None


# ======================================================================
# 3. Exploration weight (UCB1 constant)
# ======================================================================

def default_exploration_weight(root_visits: int) -> float:
    """
    Return the exploration constant C for UCB1.

    Args:
        root_visits: How many total simulations the root has run so far.

    Returns:
        A positive float (typically ~1.4).

    The LLM can implement an adaptive schedule, e.g. start high and decay.
    """
    return 1.4


# ======================================================================
# 4. Action priority / ordering (for expansion)
# ======================================================================

def default_action_priority(state: GameState, actions: list[Any]) -> list[Any]:
    """
    Return actions in the order they should be tried during expansion.

    Args:
        state:   Current game state.
        actions: Legal actions (will be popped from the end).

    Returns:
        A reordered copy of *actions*.  The LAST element is expanded first.

    Default: shuffled (so expansion order is random).
    The LLM can prioritise "good-looking" moves to expand first.
    """
    shuffled = list(actions)
    random.shuffle(shuffled)
    return shuffled
