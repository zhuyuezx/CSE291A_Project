"""
Sokoban macro-push adapter for the MCTS framework.

Wraps a SokobanState and exposes macro-push actions: each action is a
(player_pos, direction) tuple representing a box push reachable from
the player's current connected region. Walking is handled internally
via BFS.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .game_interface import GameState, Game
from .sokoban import SokobanState, LEVELS, UP, DOWN, LEFT, RIGHT, ACTION_DELTAS


class SokobanMacroState(GameState):
    """
    Macro-push adapter over SokobanState.

    State is still (player_pos, boxes, walls). Actions are (player_pos, direction)
    tuples -- macro-pushes reachable from the player's BFS-connected region.
    """

    def __init__(self, inner: SokobanState):
        self._inner = inner

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------

    def _compute_reachable(self) -> set[tuple[int, int]]:
        """BFS flood-fill from player through non-wall, non-box cells."""
        start = self._inner.player
        walls = self._inner.walls
        boxes = self._inner.boxes
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and (nr, nc) not in walls and (nr, nc) not in boxes:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def _bfs_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[int]:
        """BFS shortest path from start to goal, returns list of direction ints.
        Assumes goal is reachable (in the same connected region)."""
        if start == goal:
            return []
        walls = self._inner.walls
        boxes = self._inner.boxes
        visited: set[tuple[int, int]] = {start}
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
        while queue:
            (r, c), path = queue.popleft()
            for direction, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) == goal:
                    return path + [direction]
                if (nr, nc) not in visited and (nr, nc) not in walls and (nr, nc) not in boxes:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [direction]))
        raise ValueError(f"No path from {start} to {goal}")

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "SokobanMacroState":
        return SokobanMacroState(self._inner.clone())

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        """Enumerate all macro-push actions from the current reachable region.

        Each action is (player_pos, direction) where:
        - player_pos is a cell in the reachable region adjacent to a box
        - direction is the push direction (box is at player_pos + delta)
        - the cell beyond the box (player_pos + 2*delta) must be free
        """
        reachable = self._compute_reachable()
        actions: list[tuple[tuple[int, int], int]] = []
        walls = self._inner.walls
        boxes = self._inner.boxes
        for cell in reachable:
            cr, cc = cell
            for direction, (dr, dc) in ACTION_DELTAS.items():
                box_pos = (cr + dr, cc + dc)
                if box_pos not in boxes:
                    continue
                beyond = (cr + 2 * dr, cc + 2 * dc)
                if beyond not in walls and beyond not in boxes:
                    actions.append((cell, direction))
        return actions

    def apply_action(self, action: Any) -> None:
        """Apply a macro-push: BFS walk to player_pos, then push."""
        player_pos, direction = action
        walk_path = self._bfs_path(self._inner.player, player_pos)
        for move in walk_path:
            self._inner.apply_action(move)
        self._inner.apply_action(direction)

    def is_terminal(self) -> bool:
        return self._inner.is_terminal()

    def returns(self) -> list[float]:
        return self._inner.returns()

    def state_key(self) -> str:
        return self._inner.state_key()

    def __str__(self) -> str:
        return str(self._inner)

    # ------------------------------------------------------------------
    # Expose inner state attributes for heuristic tools
    # ------------------------------------------------------------------

    @property
    def player(self):
        return self._inner.player

    @property
    def boxes(self):
        return self._inner.boxes

    @property
    def targets(self):
        return self._inner.targets

    @property
    def walls(self):
        return self._inner.walls

    @property
    def height(self):
        return self._inner.height

    @property
    def width(self):
        return self._inner.width

    @property
    def num_targets(self):
        return self._inner.num_targets

    @property
    def steps(self):
        return self._inner.steps

    @property
    def max_steps(self):
        return self._inner.max_steps

    def boxes_on_targets(self) -> int:
        return self._inner.boxes_on_targets()

    def total_box_distance(self) -> int:
        return self._inner.total_box_distance()

    def _is_solved(self) -> bool:
        return self._inner._is_solved()

    def _is_deadlocked(self) -> bool:
        return self._inner._is_deadlocked()


# =====================================================================
# Game factory
# =====================================================================

class SokobanMacro(Game):
    """Factory for macro-push Sokoban games."""

    def __init__(
        self,
        level_name: str = "level1",
        max_steps: int = 200,
        level_lines: list[str] | None = None,
    ):
        if level_lines is not None:
            self.level_lines = level_lines
            self.level_name = "custom"
        else:
            if level_name not in LEVELS:
                raise ValueError(
                    f"Unknown level '{level_name}'. "
                    f"Available: {list(LEVELS.keys())}"
                )
            self.level_name = level_name
            self.level_lines = LEVELS[level_name]
        self.max_steps = max_steps

    def new_initial_state(self) -> SokobanMacroState:
        inner = SokobanState(self.level_lines, self.max_steps)
        return SokobanMacroState(inner)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"Sokoban_Macro ({self.level_name})"
