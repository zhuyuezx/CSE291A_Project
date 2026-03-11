"""
Abstract game interface for the MCTS framework.

Any game that wants to use MCTS must implement GameState and Game.
This keeps the MCTS engine game-agnostic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class GameState(ABC):
    """
    Abstract state of a game.

    A GameState must be clonable, know its legal actions, and be able
    to apply an action to produce the next state.
    """

    @abstractmethod
    def clone(self) -> "GameState":
        """Return a deep copy of this state."""
        ...

    @abstractmethod
    def current_player(self) -> int:
        """Return the index of the player whose turn it is (0-indexed)."""
        ...

    @abstractmethod
    def legal_actions(self) -> list[Any]:
        """Return a list of legal actions from this state."""
        ...

    @abstractmethod
    def apply_action(self, action: Any) -> None:
        """Apply an action **in-place**, advancing the state."""
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        ...

    @abstractmethod
    def returns(self) -> list[float]:
        """
        Return a list of rewards, one per player.
        Only meaningful when is_terminal() is True.
        Convention: +1 win, -1 loss, 0 draw.
        """
        ...

    @abstractmethod
    def state_key(self) -> str:
        """
        A unique string key for this state, used for the transposition table.
        Equivalent positions should map to the same key.
        """
        ...

    def __str__(self) -> str:
        """Human-readable representation (used in trace logs)."""
        return self.state_key()


class Game(ABC):
    """Abstract game factory — creates initial states."""

    @abstractmethod
    def new_initial_state(self) -> GameState:
        """Return a fresh starting state."""
        ...

    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable game name."""
        ...

    def action_mapping(self) -> dict[str, str]:
        """
        Return a dict mapping str(action) → human-readable label.

        Override in concrete Game subclasses to provide meaningful action names.
        The default implementation returns an empty dict, meaning the raw
        str(action) will be used as the label in trace records.
        """
        return {}
