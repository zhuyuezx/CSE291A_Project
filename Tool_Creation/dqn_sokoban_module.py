"""
Minimal Sokoban "DQN module" interface for PUCT integration.

This is a drop-in placeholder that matches the required exports:
    - q_model(model_input) -> list[float] of length 4
    - encode_state_fn(state) -> model_input
    - action_to_index_fn(action) -> int

It uses a handcrafted one-step heuristic to approximate Q-values.
Replace q_model with a trained network when available.
"""

from __future__ import annotations

from typing import Any, Callable

# Optional runtime-injected trained model.
_TRAINED_Q_MODEL: Callable[[Any], Any] | None = None


def encode_state_fn(state: Any) -> Any:
    """
    Encode GameState into model input.

    Placeholder behavior: pass through the state object directly.
    A trained model version can return a tensor/array here.
    """
    return state


def action_to_index_fn(action: Any) -> int:
    """Map Sokoban action (0/1/2/3) to Q-value index."""
    return int(action)


def set_trained_q_model(model_callable: Callable[[Any], Any]) -> None:
    """
    Register a trained Q model at runtime.

    model_callable should map encoded state -> Q values over actions.
    """
    global _TRAINED_Q_MODEL
    _TRAINED_Q_MODEL = model_callable


def clear_trained_q_model() -> None:
    """Remove runtime model and fall back to heuristic proxy."""
    global _TRAINED_Q_MODEL
    _TRAINED_Q_MODEL = None


def _transition_score(before_state: Any, after_state: Any) -> float:
    """
    Heuristic value for one-step transition.

    Positive if we move toward solution:
    - more boxes on targets
    - lower total box distance
    Negative if we create deadlock or waste steps.
    """
    score = 0.0

    # Encourage progress toward target coverage.
    score += 3.0 * (after_state.boxes_on_targets() - before_state.boxes_on_targets())

    # Encourage lower box-target Manhattan distance.
    score += 0.5 * (before_state.total_box_distance() - after_state.total_box_distance())

    # Penalize deadlocks heavily.
    if hasattr(after_state, "_is_deadlocked") and after_state._is_deadlocked():
        score -= 5.0

    # Small step penalty to reduce dithering.
    score -= 0.01
    return score


def q_model(model_input: Any) -> list[float]:
    """
    Return per-action Q-values for Sokoban actions [UP, DOWN, LEFT, RIGHT].

    This is not a trained DQN; it is a deterministic heuristic proxy that
    keeps the same callable contract expected by make_dqn_prior_fn.
    """
    # Prefer trained model when one is registered.
    if _TRAINED_Q_MODEL is not None:
        return _TRAINED_Q_MODEL(model_input)

    state = model_input
    q_values = [-1e9, -1e9, -1e9, -1e9]
    legal = set(state.legal_actions())

    if not legal:
        return [0.0, 0.0, 0.0, 0.0]

    for action in legal:
        next_state = state.clone()
        next_state.apply_action(action)
        q_values[int(action)] = _transition_score(state, next_state)

    return q_values
