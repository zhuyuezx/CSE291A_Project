"""
Pure-Python Connect Four implementation.

No external dependencies — the game logic is self-contained so the LLM
can easily read and reason about board states.

Board layout (6 rows × 7 columns):
    Row 0 is the TOP (visually), row 5 is BOTTOM.
    Columns are numbered 0-6.  A "drop" places a piece in the lowest
    available row of the chosen column.

Players: 0 (plays 'X') and 1 (plays 'O').
"""

from __future__ import annotations

import copy
from typing import Any

from .game_interface import Game, GameState

ROWS = 6
COLS = 7
WIN_LEN = 4


class ConnectFourState(GameState):
    """Mutable state for a Connect Four game."""

    def __init__(self):
        # board[r][c] = None | 0 | 1
        self.board: list[list[int | None]] = [
            [None] * COLS for _ in range(ROWS)
        ]
        self._current_player: int = 0
        self._terminal: bool = False
        self._winner: int | None = None  # None = draw or not finished
        self._move_count: int = 0

    # ---------- GameState interface ----------

    def clone(self) -> "ConnectFourState":
        s = ConnectFourState.__new__(ConnectFourState)
        s.board = [row[:] for row in self.board]
        s._current_player = self._current_player
        s._terminal = self._terminal
        s._winner = self._winner
        s._move_count = self._move_count
        return s

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[int]:
        if self._terminal:
            return []
        return [c for c in range(COLS) if self.board[0][c] is None]

    def apply_action(self, action: int) -> None:
        col = action
        # Find lowest empty row in this column
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] is None:
                self.board[row][col] = self._current_player
                self._move_count += 1
                if self._check_win(row, col):
                    self._terminal = True
                    self._winner = self._current_player
                elif self._move_count == ROWS * COLS:
                    self._terminal = True  # draw
                self._current_player = 1 - self._current_player
                return
        raise ValueError(f"Column {col} is full")

    def is_terminal(self) -> bool:
        return self._terminal

    def returns(self) -> list[float]:
        if not self._terminal:
            return [0.0, 0.0]
        if self._winner is None:
            return [0.0, 0.0]  # draw
        # winner gets +1, loser gets -1
        r = [0.0, 0.0]
        r[self._winner] = 1.0
        r[1 - self._winner] = -1.0
        return r

    def state_key(self) -> str:
        rows = []
        for r in range(ROWS):
            rows.append("".join(
                "." if self.board[r][c] is None else str(self.board[r][c])
                for c in range(COLS)
            ))
        return "|".join(rows) + f"|p{self._current_player}"

    def __str__(self) -> str:
        symbols = {None: ".", 0: "X", 1: "O"}
        header = " ".join(str(c) for c in range(COLS))
        lines = [header]
        for r in range(ROWS):
            lines.append(" ".join(symbols[self.board[r][c]] for c in range(COLS)))
        lines.append(header)
        return "\n".join(lines)

    # ---------- Internal helpers ----------

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move at (row, col) created a 4-in-a-row."""
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag
        for dr, dc in directions:
            count = 1
            # Forward
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc
            # Backward
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= WIN_LEN:
                return True
        return False

    # ---------- Extra helpers for heuristics ----------

    def get_piece(self, row: int, col: int) -> int | None:
        """Return the piece at (row, col): 0, 1, or None."""
        return self.board[row][col]

    def column_height(self, col: int) -> int:
        """Return how many pieces are in this column."""
        for r in range(ROWS):
            if self.board[r][col] is not None:
                return ROWS - r
        return 0

    @property
    def winner(self) -> int | None:
        return self._winner


class ConnectFour(Game):
    """Connect Four game factory."""

    def new_initial_state(self) -> ConnectFourState:
        return ConnectFourState()

    def num_players(self) -> int:
        return 2

    def name(self) -> str:
        return "Connect Four"
