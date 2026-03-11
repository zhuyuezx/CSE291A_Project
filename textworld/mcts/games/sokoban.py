"""
Sokoban puzzle for the MCTS framework.

A single-player puzzle where the player pushes boxes onto target positions.

Standard symbols:
    # = wall     (space) = floor
    $ = box      . = target
    * = box on target
    @ = player   + = player on target

Includes several built-in Microban levels (easy / small).
"""

from __future__ import annotations

from typing import Any

from .game_interface import GameState, Game


# ── Actions ──────────────────────────────────────────────────────────
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


# ── Built-in levels (10 levels, easy → hard) ────────────────────────
# Difficulty: boxes × room complexity × required push count
LEVELS = {
    "level1": [          # 1 box
        "#####",
        "#@$.#",
        "#####",
    ],
    "level2": [          # 1 box
        "######",
        "#    #",
        "# @$ #",
        "#  . #",
        "######",
    ],
    "level3": [          # 2 boxes
        "#######",
        "#     #",
        "# .$. #",
        "#  $  #",
        "#  @  #",
        "#######",
    ],
    "level4": [          # 2 boxes
        "#######",
        "#     #",
        "# $ $ #",
        "# .@. #",
        "#######",
    ],
    "level5": [          # 3 boxes
        "########",
        "#      #",
        "# .$$  #",
        "# . $@ #",
        "# .    #",
        "########",
    ],
    "level6": [          # 2 boxes
        "  ####",
        "###  ####",
        "#     $ #",
        "# #  #$ #",
        "# . .#@ #",
        "#########",
    ],
    "level7": [          # 4 boxes
        " ########",
        " # . . .#",
        " # $$#$ #",
        " #   @  #",
        " ########",
    ],
    "level8": [          # 4 boxes, the offset cluster
        "########",
        "#  ....#",
        "# $$   #",
        "#  $$  #",
        "#   @  #",
        "########",
    ],
    "level9": [          # 4 boxes
        "########",
        "#   .  #",
        "#  $$$ #",
        "#  $@  #",
        "# ...  #",
        "########",
    ],
    "level10": [         # 4 boxes
        "#######",
        "#.   .#",
        "# $@$ #",
        "# ### #",
        "# $ $ #",
        "#.   .#",
        "#######",
    ],
}


# =====================================================================
# State
# =====================================================================

class SokobanState(GameState):
    """
    State of a Sokoban level.

    Immutable structures (shared via reference on clone):
        walls  : frozenset of (row, col)
        targets: frozenset of (row, col)

    Mutable state:
        boxes  : set of (row, col)
        player : (row, col)
        steps  : int
    """

    def __init__(self, level_lines: list[str], max_steps: int = 200):
        self.max_steps = max_steps
        self.steps = 0
        self.height = len(level_lines)
        self.width = max(len(line) for line in level_lines)

        walls: set[tuple[int, int]] = set()
        boxes: set[tuple[int, int]] = set()
        targets: set[tuple[int, int]] = set()
        self.player: tuple[int, int] = (0, 0)

        for r, line in enumerate(level_lines):
            for c, ch in enumerate(line):
                if ch == '#':
                    walls.add((r, c))
                elif ch == '$':
                    boxes.add((r, c))
                elif ch == '.':
                    targets.add((r, c))
                elif ch == '*':          # box on target
                    boxes.add((r, c))
                    targets.add((r, c))
                elif ch == '@':
                    self.player = (r, c)
                elif ch == '+':          # player on target
                    self.player = (r, c)
                    targets.add((r, c))

        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.boxes = boxes
        self.num_targets = len(self.targets)

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "SokobanState":
        new = SokobanState.__new__(SokobanState)
        new.max_steps = self.max_steps
        new.steps = self.steps
        new.height = self.height
        new.width = self.width
        new.walls = self.walls         # frozenset — shared
        new.targets = self.targets     # frozenset — shared
        new.boxes = set(self.boxes)    # mutable copy
        new.player = self.player       # tuple — immutable
        new.num_targets = self.num_targets
        return new

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        actions: list[int] = []
        pr, pc = self.player
        for action in (UP, DOWN, LEFT, RIGHT):
            dr, dc = ACTION_DELTAS[action]
            nr, nc = pr + dr, pc + dc
            if (nr, nc) in self.walls:
                continue
            if (nr, nc) in self.boxes:
                # Can push only if the cell behind the box is free
                br, bc = nr + dr, nc + dc
                if (br, bc) in self.walls or (br, bc) in self.boxes:
                    continue
            actions.append(action)
        return actions

    def apply_action(self, action: Any) -> None:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.player[0] + dr, self.player[1] + dc

        if (nr, nc) in self.boxes:
            # Push box
            br, bc = nr + dr, nc + dc
            self.boxes.remove((nr, nc))
            self.boxes.add((br, bc))

        self.player = (nr, nc)
        self.steps += 1

    def is_terminal(self) -> bool:
        if self._is_solved():
            return True
        if self.steps >= self.max_steps:
            return True
        if self._is_deadlocked():
            return True
        return False

    def returns(self) -> list[float]:
        if self._is_solved():
            return [1.0]
        return [0.0]

    def state_key(self) -> str:
        return f"P{self.player}B{tuple(sorted(self.boxes))}"

    # ------------------------------------------------------------------
    # Puzzle-specific helpers
    # ------------------------------------------------------------------

    def _is_solved(self) -> bool:
        return self.boxes == set(self.targets)

    def _is_deadlocked(self) -> bool:
        """
        Simple deadlock detection: a box in a corner (not on a target)
        is permanently stuck.
        """
        for br, bc in self.boxes:
            if (br, bc) in self.targets:
                continue
            wall_up   = (br - 1, bc) in self.walls
            wall_down = (br + 1, bc) in self.walls
            wall_left = (br, bc - 1) in self.walls
            wall_right = (br, bc + 1) in self.walls
            if (wall_up or wall_down) and (wall_left or wall_right):
                return True
        return False

    def boxes_on_targets(self) -> int:
        """Count boxes currently on target positions."""
        return len(self.boxes & set(self.targets))

    def total_box_distance(self) -> int:
        """Sum of minimum Manhattan distance from each box to nearest target."""
        total = 0
        for br, bc in self.boxes:
            min_dist = min(
                abs(br - tr) + abs(bc - tc) for tr, tc in self.targets
            )
            total += min_dist
        return total

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines: list[str] = []
        for r in range(self.height):
            row: list[str] = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.player:
                    row.append('+' if pos in self.targets else '@')
                elif pos in self.boxes:
                    row.append('*' if pos in self.targets else '$')
                elif pos in self.targets:
                    row.append('.')
                elif pos in self.walls:
                    row.append('#')
                else:
                    row.append(' ')
            lines.append("".join(row))
        header = (
            f"Step {self.steps}/{self.max_steps} | "
            f"Boxes on target: {self.boxes_on_targets()}/{self.num_targets} | "
            f"Total distance: {self.total_box_distance()}"
        )
        return header + "\n" + "\n".join(lines)


# =====================================================================
# Game factory
# =====================================================================

class Sokoban(Game):
    """
    Factory for Sokoban games.

    Pass a level name from the built-in collection, or provide custom
    level lines.
    """

    def __init__(
        self,
        level_name: str = "micro1",
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

    def new_initial_state(self) -> SokobanState:
        return SokobanState(self.level_lines, self.max_steps)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"Sokoban ({self.level_name})"
