# src/games/twenty48_adapter.py
"""
Custom 2048 adapter with a proper win/loss definition.

Rules:
  - WIN:  a tile of 2048 (or higher) appears on the board → returns +1.0
  - LOSS: no 2048 tile reached and board is stuck → returns -1.0

Auxiliary (shaping) reward during rollouts:
  - aux_reward(state) → normalized log score in [-1, 1]
    This guides MCTS toward high-score states even before a win/loss.

The adapter wraps the vanilla OpenSpiel "2048" game state and intercepts
is_terminal() / returns() to inject the custom semantics.
"""
from __future__ import annotations

import math

import pyspiel

from src.games.adapter import GameAdapter
from src.games.meta_registry import GAME_META, GameMeta

# Define the custom meta.  Final reward is binary ±1 (win/loss).
_2048_WIN_TILE = 2048
_2048_MAX_SCORE = 131072.0  # theoretical max score (all tiles merged to 131072)

_CUSTOM_META = GameMeta(
    name="2048",
    is_single_player=True,
    min_return=-1.0,
    max_return=1.0,
    metric_name="win_rate",
    max_sim_depth=1000,
)


def _max_tile(state) -> int:
    """Return the maximum tile value currently on the board."""
    try:
        import re
        s = str(state)
        max_val = 0
        for tok in re.findall(r'\d+', s):
            v = int(tok)
            # Only consider powers of 2 >= 2
            if v >= 2 and (v & (v - 1)) == 0:
                max_val = max(max_val, v)
        return max_val
    except Exception:
        return 0


def _score_from_returns(state) -> float:
    """The raw game score from OpenSpiel's returns()."""
    try:
        return state.returns()[0]
    except Exception:
        return 0.0


class TwentyFortyEightAdapter(GameAdapter):
    """
    Drop-in replacement for GameAdapter("2048") with custom terminal conditions.

    Terminal conditions (checked in order):
      1. Any tile >= WIN_TILE → WIN (+1.0)
      2. OpenSpiel says terminal (board stuck) → LOSS (-1.0)
    """

    def __init__(self, win_tile: int = _2048_WIN_TILE, max_steps: int = 8000):
        # Load via parent but override meta
        super().__init__("2048")
        self.win_tile = win_tile
        self.max_steps = max_steps
        self.meta = _CUSTOM_META
        self._win_tile_log = math.log2(win_tile)  # used for aux normalization
        self._step_counts: dict[int, int] = {}  # state id → step count

    # ------------------------------------------------------------------ #
    # Core overrides                                                       #
    # ------------------------------------------------------------------ #

    def is_terminal(self, state) -> bool:
        """Terminal if we've won (2048 tile), board stuck, or exceeded max_steps."""
        if _max_tile(state) >= self.win_tile:
            return True
        if state.is_terminal():
            return True
        # Check step count via state history length
        if self.max_steps > 0 and len(state.history()) >= self.max_steps:
            return True
        return False

    def returns(self, state) -> list[float]:
        """
        Binary return:
          +1.0 if max tile >= win_tile (WIN)
          -1.0 otherwise (game over without win, or max_steps exceeded)
        Returns a 1-element list to match the single-player convention.
        """
        if _max_tile(state) >= self.win_tile:
            return [1.0]
        return [-1.0]

    def aux_reward(self, state) -> float:
        """
        Score-based auxiliary reward in [-1, 1].
        Uses log-normalized score so MCTS rollouts are guided toward
        high-score intermediate states.
        """
        score = _score_from_returns(state)
        if score <= 0:
            return -1.0
        # Normalise: log2(score) / log2(max_theoretical_score)
        # log2(131072) = 17; we use 17 as ceiling
        _MAX_LOG = 17.0
        normalized = math.log2(score + 1) / _MAX_LOG
        return max(-1.0, min(1.0, 2.0 * normalized - 1.0))

    def normalize_return(self, raw: float) -> float:
        """Binary returns are already in {-1, +1}; pass through."""
        return max(-1.0, min(1.0, raw))

    # ------------------------------------------------------------------ #
    # Overriding apply_action / legal_actions to respect custom terminal  #
    # ------------------------------------------------------------------ #

    def legal_actions(self, state) -> list[int]:
        """Return empty list if custom terminal, otherwise delegate."""
        if self.is_terminal(state):
            return []
        return state.legal_actions()

    def apply_action(self, state, action: int):
        """Apply action to clone; return clone (same as base)."""
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state

    def game_description(self) -> str:
        return (
            f"Game: 2048 (win={self.win_tile}), "
            f"Players: 1, Actions: 4, "
            f"Reward: binary win/loss + aux score shaping"
        )
