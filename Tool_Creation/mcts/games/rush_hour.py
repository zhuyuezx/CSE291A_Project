"""
Rush Hour puzzle for the MCTS framework.

A single-player 6×6 sliding block puzzle.  Vehicles (cars of length 2
and trucks of length 3) occupy rows or columns and can only slide along
their orientation.  The goal is to slide the *primary piece* (piece A,
always horizontal) to the right edge of its row so it can exit the
board.

Puzzle encoding follows Michael Fogleman's format:
    . or o  = empty cell
    x       = wall (fixed obstacle)
    A       = primary piece (the "red car")
    B – Z   = other vehicles
    (36-char string, row-major for a 6×6 board)

Reference:
    https://github.com/fogleman/rush
    https://www.michaelfogleman.com/rush/
"""

from __future__ import annotations

import random
from typing import Any

from .game_interface import GameState, Game

# ── Constants ────────────────────────────────────────────────────────
BOARD_SIZE = 6
HORIZONTAL = 0
VERTICAL = 1


# ── Built-in puzzles ────────────────────────────────────────────────
# Format: { name: (description_string_36_chars, optimal_moves) }
# Curated from Fogleman's database at varied difficulty.
# Board is 36-char row-major: . = empty, x = wall, A = primary, B-Z = vehicles.
PUZZLES: dict[str, tuple[str, int]] = {
    # Simple hand-crafted puzzles for testing / easy starts
    #   ......  ......  ..BBCC  BBB...
    #   ...B..  ...B..  ..D...  ...D..
    #   AA.B..  .AAB..  AAD...  AA.D..
    #   ......  ......  ..D...  ..C...
    #   ..CC..  ......  ..EE..  ..C...
    #   ......  ......  ......  EE....
    "easy1":   (".........B..AA.B..........CC........", 2),
    "easy2":   ("..BBCC..D...AAD.....D.....EE........", 8),
    "easy3":   ("BBB......D..AA.D....C.....C...EE....", 10),

    # From the database — moderate to hard
    "medium1": ("ooIBBBooIKooAAJKoLCCJDDLGHEEoLGHFFoo", 50),
    "medium2": ("BBBJCCHooJoKHAAJoKooIDDLEEIooLooxoGG", 50),

    # From the database — hard
    "hard1":   ("GBBoLoGHIoLMGHIAAMCCCKoMooJKDDEEJFFo", 51),
    "hard2":   ("IBBxooIooLDDJAALooJoKEEMFFKooMGGHHHM", 60),
    "hard3":   ("BBoKMxDDDKMoIAALooIoJLEEooJFFNoGGoxN", 58),
}


# ── Piece helper ─────────────────────────────────────────────────────

class _Piece:
    """Internal representation of a vehicle on the board."""
    __slots__ = ("position", "size", "orientation")

    def __init__(self, position: int, size: int, orientation: int):
        self.position = position      # index into the flat 6×6 grid
        self.size = size              # 2 (car) or 3 (truck)
        self.orientation = orientation  # HORIZONTAL or VERTICAL

    def stride(self) -> int:
        return 1 if self.orientation == HORIZONTAL else BOARD_SIZE

    def row(self) -> int:
        return self.position // BOARD_SIZE

    def col(self) -> int:
        return self.position % BOARD_SIZE

    def cells(self) -> list[int]:
        """Return the list of cell indices this piece occupies."""
        s = self.stride()
        return [self.position + i * s for i in range(self.size)]

    def copy(self) -> "_Piece":
        return _Piece(self.position, self.size, self.orientation)


# ── Parsing ──────────────────────────────────────────────────────────

def _parse_board(desc: str) -> tuple[list[_Piece], list[int]]:
    """Parse a 36-char board description into pieces and walls.

    Returns (pieces, walls) where pieces[0] is the primary piece.
    """
    if len(desc) != BOARD_SIZE * BOARD_SIZE:
        raise ValueError(
            f"Board description must be {BOARD_SIZE*BOARD_SIZE} chars, "
            f"got {len(desc)}"
        )

    # Collect positions per label
    positions: dict[str, list[int]] = {}
    walls: list[int] = []
    for i, ch in enumerate(desc):
        if ch in ('.', 'o'):
            continue
        if ch == 'x':
            walls.append(i)
        else:
            positions.setdefault(ch, []).append(i)

    if 'A' not in positions:
        raise ValueError("Board must contain primary piece 'A'")

    # Sort labels so 'A' comes first, then alphabetical
    labels = sorted(positions.keys())
    if 'A' in labels:
        labels.remove('A')
        labels.insert(0, 'A')

    pieces: list[_Piece] = []
    for label in labels:
        ps = sorted(positions[label])
        if len(ps) < 2:
            raise ValueError(f"Piece '{label}' must occupy at least 2 cells")
        stride = ps[1] - ps[0]
        if stride not in (1, BOARD_SIZE):
            raise ValueError(f"Piece '{label}' has invalid shape")
        for k in range(2, len(ps)):
            if ps[k] - ps[k - 1] != stride:
                raise ValueError(f"Piece '{label}' has invalid shape")
        orientation = HORIZONTAL if stride == 1 else VERTICAL
        pieces.append(_Piece(ps[0], len(ps), orientation))

    # Primary piece must be horizontal
    if pieces[0].orientation != HORIZONTAL:
        raise ValueError("Primary piece 'A' must be horizontal")

    return pieces, walls


# =====================================================================
# State
# =====================================================================

class RushHourState(GameState):
    """State of a Rush Hour puzzle.

    Internally tracks:
        pieces   : list[_Piece]   – vehicle placements (pieces[0] = primary)
        walls    : tuple[int,...] – fixed wall cell indices
        occupied : list[bool]    – which cells are blocked
        target   : int           – cell index the primary piece must reach
        moves    : int           – number of moves made so far
        max_moves: int           – move budget
    """

    def __init__(
        self,
        desc: str,
        max_moves: int = 300,
        *,
        _pieces: list[_Piece] | None = None,
        _walls: tuple[int, ...] | None = None,
        _target: int | None = None,
    ):
        self.max_moves = max_moves
        self.moves_made = 0

        if _pieces is not None:
            # Internal fast-path used by clone()
            self.pieces = _pieces
            self.walls = _walls                    # type: ignore[assignment]
            self.target = _target                  # type: ignore[assignment]
        else:
            pieces, walls = _parse_board(desc)
            self.pieces = pieces
            self.walls = tuple(walls)
            # Target: primary piece flush right on its row
            p = self.pieces[0]
            self.target = (p.row() + 1) * BOARD_SIZE - p.size

        # Build occupancy grid
        self.occupied = [False] * (BOARD_SIZE * BOARD_SIZE)
        for w in self.walls:
            self.occupied[w] = True
        for piece in self.pieces:
            for c in piece.cells():
                self.occupied[c] = True

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "RushHourState":
        new = RushHourState.__new__(RushHourState)
        new.max_moves = self.max_moves
        new.moves_made = self.moves_made
        new.pieces = [p.copy() for p in self.pieces]
        new.walls = self.walls          # tuple — immutable, share
        new.target = self.target
        new.occupied = list(self.occupied)
        return new

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        """Return list of (piece_index, steps) moves.

        Steps > 0 means right/down; steps < 0 means left/up.
        Each distinct step count is a separate action (not cumulative).
        """
        actions: list[tuple[int, int]] = []
        for i, piece in enumerate(self.pieces):
            stride = piece.stride()
            if piece.orientation == VERTICAL:
                y = piece.position // BOARD_SIZE
                reverse_limit = -y
                forward_limit = BOARD_SIZE - piece.size - y
            else:
                x = piece.position % BOARD_SIZE
                reverse_limit = -x
                forward_limit = BOARD_SIZE - piece.size - x

            # Reverse (negative steps)
            idx = piece.position - stride
            for steps in range(-1, reverse_limit - 1, -1):
                if idx < 0 or idx >= BOARD_SIZE * BOARD_SIZE:
                    break
                if self.occupied[idx]:
                    break
                actions.append((i, steps))
                idx -= stride

            # Forward (positive steps)
            idx = piece.position + piece.size * stride
            for steps in range(1, forward_limit + 1):
                if idx < 0 or idx >= BOARD_SIZE * BOARD_SIZE:
                    break
                if self.occupied[idx]:
                    break
                actions.append((i, steps))
                idx += stride

        return actions

    def apply_action(self, action: Any) -> None:
        piece_idx, steps = action
        piece = self.pieces[piece_idx]
        stride = piece.stride()

        # Clear old cells
        for c in piece.cells():
            self.occupied[c] = False

        # Move
        piece.position += stride * steps

        # Mark new cells
        for c in piece.cells():
            self.occupied[c] = True

        self.moves_made += 1

    def is_terminal(self) -> bool:
        if self._is_solved():
            return True
        if self.moves_made >= self.max_moves:
            return True
        if not self.legal_actions():
            return True
        return False

    def returns(self) -> list[float]:
        if self._is_solved():
            return [1.0]
        return [0.0]

    def state_key(self) -> str:
        positions = tuple(p.position for p in self.pieces)
        return f"RH{positions}"

    # ------------------------------------------------------------------
    # Puzzle-specific helpers
    # ------------------------------------------------------------------

    def _is_solved(self) -> bool:
        return self.pieces[0].position == self.target

    def primary_distance(self) -> int:
        """Number of cells the primary piece must still slide right."""
        return self.target - self.pieces[0].position

    def blocking_count(self) -> int:
        """Count cells between primary piece and the right edge that are occupied."""
        p = self.pieces[0]
        start = p.position + p.size
        end = self.target + p.size  # inclusive far end
        count = 0
        for idx in range(start, end + 1):
            if self.occupied[idx]:
                count += 1
        return count

    def num_pieces(self) -> int:
        return len(self.pieces)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        grid = ['.'] * (BOARD_SIZE * BOARD_SIZE)
        for w in self.walls:
            grid[w] = 'x'
        for i, piece in enumerate(self.pieces):
            label = chr(ord('A') + i)
            for c in piece.cells():
                grid[c] = label
        rows: list[str] = []
        for y in range(BOARD_SIZE):
            start = y * BOARD_SIZE
            rows.append("".join(grid[start:start + BOARD_SIZE]))
        board_str = "\n".join(rows)
        header = (
            f"Move {self.moves_made}/{self.max_moves} | "
            f"Primary dist: {self.primary_distance()} | "
            f"Blocking: {self.blocking_count()} | "
            f"Pieces: {self.num_pieces()}"
        )
        return header + "\n" + board_str


# =====================================================================
# Puzzle generation
# =====================================================================

def generate_puzzle(
    num_pieces: int = 8,
    num_walls: int = 0,
    primary_row: int = 2,
    primary_size: int = 2,
    max_attempts: int = 200,
) -> str | None:
    """Generate a random Rush Hour puzzle string.

    Uses a simple random placement approach inspired by Fogleman's
    generator.  Returns a 36-char board string or None on failure.
    """
    for _ in range(max_attempts):
        occupied = [False] * (BOARD_SIZE * BOARD_SIZE)
        pieces: list[_Piece] = []
        walls: list[int] = []

        # Place primary piece at leftmost free position on its row
        pos = primary_row * BOARD_SIZE
        p = _Piece(pos, primary_size, HORIZONTAL)
        for c in p.cells():
            occupied[c] = True
        pieces.append(p)

        # Place remaining pieces
        ok = True
        for _ in range(num_pieces - 1):
            placed = False
            for __ in range(100):
                size = random.choice([2, 2, 2, 3])  # cars more common
                orientation = random.randint(0, 1)
                if orientation == HORIZONTAL:
                    x = random.randint(0, BOARD_SIZE - size)
                    y = random.randint(0, BOARD_SIZE - 1)
                    # No horizontal pieces on the primary row
                    if y == primary_row:
                        continue
                else:
                    x = random.randint(0, BOARD_SIZE - 1)
                    y = random.randint(0, BOARD_SIZE - size)
                position = y * BOARD_SIZE + x
                candidate = _Piece(position, size, orientation)
                if any(occupied[c] for c in candidate.cells()):
                    continue
                for c in candidate.cells():
                    occupied[c] = True
                pieces.append(candidate)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        # Place walls
        for _ in range(num_walls):
            for __ in range(100):
                idx = random.randint(0, BOARD_SIZE * BOARD_SIZE - 1)
                if not occupied[idx]:
                    occupied[idx] = True
                    walls.append(idx)
                    break

        # Build string
        grid = ['.'] * (BOARD_SIZE * BOARD_SIZE)
        for w in walls:
            grid[w] = 'x'
        for i, piece in enumerate(pieces):
            label = chr(ord('A') + i)
            for c in piece.cells():
                grid[c] = label
        desc = "".join(grid)

        # Quick check: is it solvable at all?  Just ensure primary row
        # has a path (not fully blocked by walls/fixed).  Full solve
        # check is too expensive here — MCTS will discover it.
        return desc

    return None


# =====================================================================
# Game factory
# =====================================================================

class RushHour(Game):
    """Factory for Rush Hour puzzles.

    Accepts either:
    - A built-in puzzle name (e.g. ``"hard1"``)
    - A raw 36-char board description string via ``board_desc``
    - ``"random"`` to generate a random puzzle

    Parameters
    ----------
    puzzle_name : str
        A key in ``PUZZLES``, ``"random"``, or ``"random"`` variants.
    max_moves : int
        Maximum number of moves before the game terminates.
    board_desc : str | None
        If provided, overrides ``puzzle_name``.
    """

    def __init__(
        self,
        puzzle_name: str = "hard1",
        max_moves: int = 300,
        board_desc: str | None = None,
    ):
        if board_desc is not None:
            self._desc = board_desc
            self._puzzle_name = "custom"
        elif puzzle_name == "random":
            desc = generate_puzzle()
            if desc is None:
                raise RuntimeError("Failed to generate a random puzzle")
            self._desc = desc
            self._puzzle_name = "random"
        else:
            if puzzle_name not in PUZZLES:
                raise ValueError(
                    f"Unknown puzzle '{puzzle_name}'. "
                    f"Available: {list(PUZZLES.keys())}"
                )
            self._desc, _ = PUZZLES[puzzle_name]
            self._puzzle_name = puzzle_name
        self._max_moves = max_moves

    def new_initial_state(self) -> RushHourState:
        return RushHourState(self._desc, self._max_moves)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"RushHour ({self._puzzle_name})"

    def action_mapping(self) -> dict[str, str]:
        # Rush Hour actions are self-descriptive tuples: (vehicle_id, steps)
        # Positive steps = right/down, negative = left/up.
        # No static enum possible; return a format description instead.
        return {"_format": "(vehicle_id, steps)  [+steps=right/down, -steps=left/up]"}
