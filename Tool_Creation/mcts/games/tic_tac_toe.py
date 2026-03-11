"""
Tic-Tac-Toe for the MCTS framework.

A 2-player game on a 3x3 board. Players alternate placing marks
(X for player 0, O for player 1). First to get 3-in-a-row wins.

Board layout (positions 0-8):
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

from __future__ import annotations

from typing import Any

from .game_interface import Game, GameState


# All possible winning lines (indices into the flat board)
_WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


class TicTacToeState(GameState):
    """Mutable state for a Tic-Tac-Toe game."""

    def __init__(self):
        self.board: list[int | None] = [None] * 9  # None / 0 / 1
        self._current_player: int = 0
        self._terminal: bool = False
        self._winner: int | None = None  # None = draw or ongoing

    # ---------- GameState interface ----------

    def clone(self) -> "TicTacToeState":
        s = TicTacToeState.__new__(TicTacToeState)
        s.board = list(self.board)
        s._current_player = self._current_player
        s._terminal = self._terminal
        s._winner = self._winner
        return s

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[int]:
        if self._terminal:
            return []
        return [i for i in range(9) if self.board[i] is None]

    def apply_action(self, action: int) -> None:
        self.board[action] = self._current_player
        # Check win
        for a, b, c in _WIN_LINES:
            if (self.board[a] == self.board[b] == self.board[c]
                    == self._current_player):
                self._terminal = True
                self._winner = self._current_player
                self._current_player = 1 - self._current_player
                return
        # Check draw
        if all(cell is not None for cell in self.board):
            self._terminal = True
        self._current_player = 1 - self._current_player

    def is_terminal(self) -> bool:
        return self._terminal

    def returns(self) -> list[float]:
        if not self._terminal:
            return [0.0, 0.0]
        if self._winner is None:
            return [0.0, 0.0]  # draw
        r = [0.0, 0.0]
        r[self._winner] = 1.0
        r[1 - self._winner] = -1.0
        return r

    def state_key(self) -> str:
        return "".join("." if c is None else str(c) for c in self.board)

    def __str__(self) -> str:
        symbols = {None: ".", 0: "X", 1: "O"}
        lines = []
        for r in range(3):
            lines.append(" ".join(symbols[self.board[r * 3 + c]] for c in range(3)))
        return "\n".join(lines)


class TicTacToe(Game):
    """Tic-Tac-Toe game factory."""

    def new_initial_state(self) -> TicTacToeState:
        return TicTacToeState()

    def num_players(self) -> int:
        return 2

    def name(self) -> str:
        return "TicTacToe"

    def action_mapping(self) -> dict[str, str]:
        return {str(i): f"cell({i // 3},{i % 3})" for i in range(9)}
