# src/games/adapter.py
from __future__ import annotations

import pyspiel


class GameAdapter:
    """Thin wrapper around OpenSpiel games with a consistent interface."""

    def __init__(self, game_name: str, **params):
        if params:
            param_str = ",".join(f"{k}={v}" for k, v in params.items())
            self._game = pyspiel.load_game(f"{game_name}({param_str})")
        else:
            self._game = pyspiel.load_game(game_name)
        self.game_name = game_name
        self.num_players = self._game.num_players()
        self.num_distinct_actions = self._game.num_distinct_actions()

    def new_game(self):
        return self._game.new_initial_state()

    def legal_actions(self, state) -> list[int]:
        return state.legal_actions()

    def apply_action(self, state, action: int):
        """Apply action to a clone of state (non-mutating)."""
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state

    def clone_state(self, state):
        return state.clone()

    def is_terminal(self, state) -> bool:
        return state.is_terminal()

    def current_player(self, state) -> int:
        return state.current_player()

    def returns(self, state) -> list[float]:
        return state.returns()

    def action_to_string(self, state, action: int) -> str:
        return state.action_to_string(state.current_player(), action)

    def game_description(self) -> str:
        return (
            f"Game: {self.game_name}, "
            f"Players: {self.num_players}, "
            f"Actions: {self.num_distinct_actions}, "
            f"Max length: {self._game.max_game_length()}"
        )

    @property
    def pyspiel_game(self):
        return self._game
