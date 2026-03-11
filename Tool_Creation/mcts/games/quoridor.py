"""
Quoridor – Python implementation for the MCTS framework.

Two-player adversarial board game on a 9×9 grid.
Players alternate moving their pawn or placing walls.
First to reach the opponent's starting row wins.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .game_interface import Game, GameState

# ── Constants ────────────────────────────────────────────────────────
ROWS = 9
COLS = 9
INITIAL_WALLS = 10
MAX_MOVES = 200

# Directions: (dr, dc)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


# ── Move types ───────────────────────────────────────────────────────

class PawnMove:
    __slots__ = ("row", "col")

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PawnMove) and self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash(("P", self.row, self.col))

    def __repr__(self) -> str:
        return f"P({self.row},{self.col})"


class HWall:
    """Horizontal wall at slot (row, col).  Blocks passage between
    rows row and row+1, spanning columns col and col+1."""
    __slots__ = ("row", "col")

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HWall) and self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash(("H", self.row, self.col))

    def __repr__(self) -> str:
        return f"H({self.row},{self.col})"


class VWall:
    """Vertical wall at slot (row, col).  Blocks passage between
    columns col and col+1, spanning rows row and row+1."""
    __slots__ = ("row", "col")

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VWall) and self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash(("V", self.row, self.col))

    def __repr__(self) -> str:
        return f"V({self.row},{self.col})"


# ── Game State ───────────────────────────────────────────────────────

class QuoridorState(GameState):
    """Full Quoridor game state with efficient wall/passage tracking."""

    def __init__(self) -> None:
        # Pawn positions: player 0 starts top-center, player 1 bottom-center
        self.pawn_pos: list[tuple[int, int]] = [(0, 4), (8, 4)]
        self.goal_row: list[int] = [8, 0]  # player 0 goes down, player 1 goes up
        self.walls_left: list[int] = [INITIAL_WALLS, INITIAL_WALLS]

        # Placed walls
        self.h_walls: set[tuple[int, int]] = set()
        self.v_walls: set[tuple[int, int]] = set()

        # Passage graph: open_way[(r1,c1,r2,c2)] = True if passage is open
        # We track blockages instead (faster): blocked passages
        self._blocked: set[tuple[int, int, int, int]] = set()

        self._current_player: int = 0
        self._move_count: int = 0
        self._terminal: bool = False
        self._winner: int | None = None

    # ── Passage queries ──────────────────────────────────────────────

    def _is_open(self, r: int, c: int, dr: int, dc: int) -> bool:
        """Check if passage from (r,c) in direction (dr,dc) is open."""
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            return False
        return (r, c, nr, nc) not in self._blocked

    def _block_passage(self, r1: int, c1: int, r2: int, c2: int) -> None:
        """Block passage in both directions."""
        self._blocked.add((r1, c1, r2, c2))
        self._blocked.add((r2, c2, r1, c1))

    # ── Wall placement ───────────────────────────────────────────────

    def _place_h_wall(self, r: int, c: int) -> None:
        """Place horizontal wall at slot (r, c)."""
        self.h_walls.add((r, c))
        # Blocks (r,c)↔(r+1,c) and (r,c+1)↔(r+1,c+1)
        self._block_passage(r, c, r + 1, c)
        self._block_passage(r, c + 1, r + 1, c + 1)

    def _place_v_wall(self, r: int, c: int) -> None:
        """Place vertical wall at slot (r, c)."""
        self.v_walls.add((r, c))
        # Blocks (r,c)↔(r,c+1) and (r+1,c)↔(r+1,c+1)
        self._block_passage(r, c, r, c + 1)
        self._block_passage(r + 1, c, r + 1, c + 1)

    def _is_valid_h_wall(self, r: int, c: int) -> bool:
        """Check if horizontal wall at (r,c) is a legal placement."""
        if not (0 <= r < 8 and 0 <= c < 7):
            return False
        if (r, c) in self.h_walls:
            return False
        # Can't overlap with adjacent horizontal wall
        if (r, c - 1) in self.h_walls or (r, c + 1) in self.h_walls:
            return False
        # Can't cross a vertical wall at the same slot
        if (r, c) in self.v_walls:
            return False
        return True

    def _is_valid_v_wall(self, r: int, c: int) -> bool:
        """Check if vertical wall at (r,c) is a legal placement."""
        if not (0 <= r < 7 and 0 <= c < 8):
            return False
        if (r, c) in self.v_walls:
            return False
        # Can't overlap with adjacent vertical wall
        if (r - 1, c) in self.v_walls or (r + 1, c) in self.v_walls:
            return False
        # Can't cross a horizontal wall at the same slot
        if (r, c) in self.h_walls:
            return False
        return True

    # ── Path existence (BFS) ─────────────────────────────────────────

    def _bfs_dist(self, start_r: int, start_c: int, goal_row: int) -> int:
        """BFS shortest distance from (start_r, start_c) to any cell on goal_row.
        Returns -1 if no path exists."""
        if start_r == goal_row:
            return 0
        visited = set()
        visited.add((start_r, start_c))
        queue = deque([(start_r, start_c, 0)])
        while queue:
            r, c, d = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self._is_open(r, c, dr, dc):
                    if nr == goal_row:
                        return d + 1
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
        return -1

    def _both_can_reach_goal(self) -> bool:
        """Verify both players can still reach their goal row."""
        for p in range(2):
            r, c = self.pawn_pos[p]
            if self._bfs_dist(r, c, self.goal_row[p]) < 0:
                return False
        return True

    # ── Pawn movement ────────────────────────────────────────────────

    def _valid_pawn_destinations(self) -> list[tuple[int, int]]:
        """All legal pawn destinations for the current player (with jumping)."""
        p = self._current_player
        opp = 1 - p
        r, c = self.pawn_pos[p]
        opp_r, opp_c = self.pawn_pos[opp]
        dests: list[tuple[int, int]] = []

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self._is_open(r, c, dr, dc):
                continue
            if (nr, nc) == (opp_r, opp_c):
                # Opponent is adjacent in this direction — try jumping
                jr, jc = nr + dr, nc + dc
                if self._is_open(nr, nc, dr, dc) and 0 <= jr < ROWS and 0 <= jc < COLS:
                    dests.append((jr, jc))
                else:
                    # Can't jump straight — try diagonal jumps
                    for sdr, sdc in DIRECTIONS:
                        if (sdr, sdc) == (-dr, -dc):
                            continue  # don't go backwards
                        if (sdr, sdc) == (dr, dc):
                            continue  # already tried straight
                        sr, sc = nr + sdr, nc + sdc
                        if self._is_open(nr, nc, sdr, sdc) and 0 <= sr < ROWS and 0 <= sc < COLS:
                            dests.append((sr, sc))
            else:
                dests.append((nr, nc))
        return dests

    # ── Probable walls (heuristic subset) ────────────────────────────

    def _probable_walls(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return subsets of probable horizontal and vertical wall slots.

        A wall is 'probable' if it is near an existing wall or near a pawn.
        This dramatically reduces the branching factor.
        """
        h_prob: set[tuple[int, int]] = set()
        v_prob: set[tuple[int, int]] = set()

        # Near existing horizontal walls
        for wr, wc in self.h_walls:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = wr + dr, wc + dc
                    if 0 <= nr < 8 and 0 <= nc < 7:
                        h_prob.add((nr, nc))
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        v_prob.add((nr, nc))

        # Near existing vertical walls
        for wr, wc in self.v_walls:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = wr + dr, wc + dc
                    if 0 <= nr < 8 and 0 <= nc < 7:
                        h_prob.add((nr, nc))
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        v_prob.add((nr, nc))

        # Near pawns (after a few moves)
        if self._move_count >= 3:
            opp_r, opp_c = self.pawn_pos[1 - self._current_player]
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = opp_r + dr, opp_c + dc
                    if 0 <= nr < 8 and 0 <= nc < 7:
                        h_prob.add((nr, nc))
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        v_prob.add((nr, nc))

        if self._move_count >= 6:
            my_r, my_c = self.pawn_pos[self._current_player]
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = my_r + dr, my_c + dc
                    if 0 <= nr < 8 and 0 <= nc < 7:
                        h_prob.add((nr, nc))
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        v_prob.add((nr, nc))

        return list(h_prob), list(v_prob)

    # ── GameState interface ──────────────────────────────────────────

    def clone(self) -> QuoridorState:
        s = QuoridorState.__new__(QuoridorState)
        s.pawn_pos = list(self.pawn_pos)
        s.goal_row = list(self.goal_row)
        s.walls_left = list(self.walls_left)
        s.h_walls = set(self.h_walls)
        s.v_walls = set(self.v_walls)
        s._blocked = set(self._blocked)
        s._current_player = self._current_player
        s._move_count = self._move_count
        s._terminal = self._terminal
        s._winner = self._winner
        return s

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[Any]:
        if self._terminal:
            return []

        actions: list[Any] = []
        p = self._current_player

        # 1. Pawn moves
        for r, c in self._valid_pawn_destinations():
            actions.append(PawnMove(r, c))

        # 2. Wall placements (only probable walls)
        if self.walls_left[p] > 0:
            h_prob, v_prob = self._probable_walls()
            for wr, wc in h_prob:
                if self._is_valid_h_wall(wr, wc):
                    # Temporarily place and check paths
                    self._place_h_wall(wr, wc)
                    ok = self._both_can_reach_goal()
                    # Undo
                    self.h_walls.discard((wr, wc))
                    self._blocked.discard((wr, wc, wr + 1, wc))
                    self._blocked.discard((wr + 1, wc, wr, wc))
                    self._blocked.discard((wr, wc + 1, wr + 1, wc + 1))
                    self._blocked.discard((wr + 1, wc + 1, wr, wc + 1))
                    if ok:
                        actions.append(HWall(wr, wc))

            for wr, wc in v_prob:
                if self._is_valid_v_wall(wr, wc):
                    self._place_v_wall(wr, wc)
                    ok = self._both_can_reach_goal()
                    self.v_walls.discard((wr, wc))
                    self._blocked.discard((wr, wc, wr, wc + 1))
                    self._blocked.discard((wr, wc + 1, wr, wc))
                    self._blocked.discard((wr + 1, wc, wr + 1, wc + 1))
                    self._blocked.discard((wr + 1, wc + 1, wr + 1, wc))
                    if ok:
                        actions.append(VWall(wr, wc))

        return actions

    def apply_action(self, action: Any) -> None:
        if isinstance(action, PawnMove):
            self.pawn_pos[self._current_player] = (action.row, action.col)
            # Check win
            if action.row == self.goal_row[self._current_player]:
                self._terminal = True
                self._winner = self._current_player
        elif isinstance(action, HWall):
            self._place_h_wall(action.row, action.col)
            self.walls_left[self._current_player] -= 1
        elif isinstance(action, VWall):
            self._place_v_wall(action.row, action.col)
            self.walls_left[self._current_player] -= 1

        self._move_count += 1

        # Max moves → draw (or distance-based winner)
        if not self._terminal and self._move_count >= MAX_MOVES:
            self._terminal = True
            d0 = self._bfs_dist(*self.pawn_pos[0], self.goal_row[0])
            d1 = self._bfs_dist(*self.pawn_pos[1], self.goal_row[1])
            if d0 < 0:
                d0 = 999
            if d1 < 0:
                d1 = 999
            if d0 < d1:
                self._winner = 0
            elif d1 < d0:
                self._winner = 1
            # else draw: _winner stays None

        self._current_player = 1 - self._current_player

    def is_terminal(self) -> bool:
        return self._terminal

    def returns(self) -> list[float]:
        if self._winner is None:
            return [0.0, 0.0]
        r = [0.0, 0.0]
        r[self._winner] = 1.0
        r[1 - self._winner] = -1.0
        return r

    def state_key(self) -> str:
        hw = tuple(sorted(self.h_walls))
        vw = tuple(sorted(self.v_walls))
        return (
            f"p{self.pawn_pos[0]}{self.pawn_pos[1]}"
            f"w{self.walls_left[0]},{self.walls_left[1]}"
            f"h{hw}v{vw}t{self._current_player}"
        )

    def shortest_dist(self, player: int) -> int:
        """BFS shortest distance for player to their goal row."""
        r, c = self.pawn_pos[player]
        return self._bfs_dist(r, c, self.goal_row[player])

    def __str__(self) -> str:
        # Compact board representation
        lines = []
        lines.append(f"Turn {self._move_count} | Player {self._current_player}")
        lines.append(f"P0@{self.pawn_pos[0]} walls={self.walls_left[0]}  "
                      f"P1@{self.pawn_pos[1]} walls={self.walls_left[1]}")

        # Draw the grid
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                if (r, c) == self.pawn_pos[0]:
                    row_str += "0 "
                elif (r, c) == self.pawn_pos[1]:
                    row_str += "1 "
                else:
                    row_str += ". "
            lines.append(row_str.rstrip())

        if self.h_walls:
            lines.append(f"H-walls: {sorted(self.h_walls)}")
        if self.v_walls:
            lines.append(f"V-walls: {sorted(self.v_walls)}")
        return "\n".join(lines)


# ── Game factory ─────────────────────────────────────────────────────

class Quoridor(Game):
    """Quoridor game factory."""

    def new_initial_state(self) -> QuoridorState:
        return QuoridorState()

    def num_players(self) -> int:
        return 2

    def name(self) -> str:
        return "quoridor"

    def action_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        # Pawn moves
        for r in range(ROWS):
            for c in range(COLS):
                m = PawnMove(r, c)
                mapping[str(m)] = f"move-to({r},{c})"
        # Horizontal walls
        for r in range(8):
            for c in range(7):
                m = HWall(r, c)
                mapping[str(m)] = f"h-wall({r},{c})"
        # Vertical walls
        for r in range(7):
            for c in range(8):
                m = VWall(r, c)
                mapping[str(m)] = f"v-wall({r},{c})"
        return mapping
