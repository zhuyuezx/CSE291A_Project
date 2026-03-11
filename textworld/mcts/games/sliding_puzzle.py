"""
Sliding Puzzle (8-puzzle / 15-puzzle) for the MCTS framework.

A single-player puzzle where numbered tiles must be arranged in order
by sliding them into a blank space.

Board layout (3×3 goal state):
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │
    ├───┼───┼───┤
    │ 4 │ 5 │ 6 │
    ├───┼───┼───┤
    │ 7 │ 8 │   │
    └───┴───┴───┘

Actions move the **blank** in the given direction.
"""

from __future__ import annotations

import random
from typing import Any

from .game_interface import GameState, Game


# Actions — direction the BLANK moves
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SlidingPuzzleState(GameState):
    """
    State of an N×N sliding puzzle.

    Board is a flat list of size N². The value 0 represents the blank.
    Goal state: [1, 2, 3, ..., N²-1, 0] (blank at bottom-right).
    """

    def __init__(self, size: int = 3, max_steps: int = 100):
        self.size = size
        n = size * size
        # Goal state — tiles in order, blank at end
        self.goal = list(range(1, n)) + [0]
        self.board = list(self.goal)
        self.blank_pos = n - 1        # index of blank
        self.steps = 0
        self.max_steps = max_steps
        self._solved = True            # starts at goal

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "SlidingPuzzleState":
        new = SlidingPuzzleState.__new__(SlidingPuzzleState)
        new.size = self.size
        new.goal = self.goal           # immutable, share reference
        new.board = list(self.board)
        new.blank_pos = self.blank_pos
        new.steps = self.steps
        new.max_steps = self.max_steps
        new._solved = self._solved
        return new

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        row, col = divmod(self.blank_pos, self.size)
        actions = []
        if row > 0:
            actions.append(UP)
        if row < self.size - 1:
            actions.append(DOWN)
        if col > 0:
            actions.append(LEFT)
        if col < self.size - 1:
            actions.append(RIGHT)
        return actions

    def apply_action(self, action: Any) -> None:
        row, col = divmod(self.blank_pos, self.size)
        dr, dc = ACTION_DELTAS[action]
        new_row, new_col = row + dr, col + dc
        new_pos = new_row * self.size + new_col
        # Swap blank with target tile
        self.board[self.blank_pos], self.board[new_pos] = (
            self.board[new_pos], self.board[self.blank_pos]
        )
        self.blank_pos = new_pos
        self.steps += 1
        self._solved = (self.board == self.goal)

    def is_terminal(self) -> bool:
        return self._solved or self.steps >= self.max_steps

    def returns(self) -> list[float]:
        if self._solved:
            return [1.0]
        return [0.0]

    def state_key(self) -> str:
        return ",".join(map(str, self.board))

    # ------------------------------------------------------------------
    # Puzzle-specific helpers (available to heuristics via the state)
    # ------------------------------------------------------------------

    def misplaced_tiles(self) -> int:
        """Count tiles not in their goal position (excluding blank)."""
        return sum(
            1 for a, g in zip(self.board, self.goal) if a != g and a != 0
        )

    def manhattan_distance(self) -> int:
        """Sum of Manhattan distances of each tile from its goal position."""
        total = 0
        for i, tile in enumerate(self.board):
            if tile == 0:
                continue
            goal_idx = tile - 1          # tile k belongs at index k-1
            r1, c1 = divmod(i, self.size)
            r2, c2 = divmod(goal_idx, self.size)
            total += abs(r1 - r2) + abs(c1 - c2)
        return total

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        w = len(str(self.size * self.size - 1))  # digit width
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                val = self.board[r * self.size + c]
                if val == 0:
                    row.append(" " * w + " ")
                else:
                    row.append(f"{val:>{w}} ")
            lines.append("|" + "|".join(row) + "|")
        sep = "+" + "+".join(["-" * (w + 1)] * self.size) + "+"
        header = (f"Step {self.steps}/{self.max_steps} | "
                  f"Misplaced: {self.misplaced_tiles()} | "
                  f"Manhattan: {self.manhattan_distance()}")
        return header + "\n" + sep + "\n" + ("\n" + sep + "\n").join(lines) + "\n" + sep


# ======================================================================
# Game factory
# ======================================================================

class SlidingPuzzle(Game):
    """
    Factory for Sliding Puzzle games.

    Each call to new_initial_state() returns a freshly scrambled puzzle.
    Scrambling is done by applying random legal moves from the goal state,
    which guarantees solvability.
    """

    def __init__(
        self,
        size: int = 3,
        max_steps: int = 100,
        scramble_moves: int = 20,
    ):
        self.size = size
        self.max_steps = max_steps
        self.scramble_moves = scramble_moves

    def new_initial_state(self) -> SlidingPuzzleState:
        state = SlidingPuzzleState(self.size, self.max_steps)
        self._scramble(state)
        return state

    def _scramble(self, state: SlidingPuzzleState):
        """Apply random legal moves from the goal state."""
        prev_action = None
        for _ in range(self.scramble_moves):
            actions = state.legal_actions()
            # Don't immediately undo the previous move
            if prev_action is not None:
                opp = OPPOSITE[prev_action]
                if opp in actions and len(actions) > 1:
                    actions = [a for a in actions if a != opp]
            action = random.choice(actions)
            state.apply_action(action)
            prev_action = action
        # Reset step counter after scrambling
        state.steps = 0
        state._solved = (state.board == state.goal)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"Sliding Puzzle {self.size}x{self.size}"
