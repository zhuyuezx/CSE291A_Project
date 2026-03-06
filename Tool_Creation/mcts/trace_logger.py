"""
Trace logger for MCTS game play.

A passive recorder used internally by MCTSEngine when logging=True.
Collects per-move trace data during a game, then writes a structured
JSON file to the records directory.

Each record captures:
    - metadata (game name, timestamp, engine config, tool paths)
    - per-move trace (state before, action chosen, search stats)
    - outcome (solved/winner, total steps, final returns)

Usage (via engine, not directly)::

    engine = MCTSEngine(game, iterations=100, logging=True)
    result = engine.play_game()   # trace auto-written to mcts/records/
    # or with custom dir:
    engine = MCTSEngine(game, logging=True, records_dir="my_logs/")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# Default records directory: mcts/records/
_RECORDS_DIR = Path(__file__).resolve().parent / "records"


class TraceLogger:
    """
    Passive trace recorder for MCTS games.

    The engine creates and drives a TraceLogger instance internally.
    Call begin_game() before the game loop, record_move() after each
    MCTS search, and end_game() when the game terminates.
    """

    def __init__(self, records_dir: str | Path | None = None):
        self.records_dir = (Path(records_dir) if records_dir else _RECORDS_DIR).resolve()
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self._move_traces: list[dict] = []
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle API (called by MCTSEngine)
    # ------------------------------------------------------------------

    def begin_game(self, metadata: dict[str, Any]) -> None:
        """Start tracing a new game. Called once before the game loop."""
        self._metadata = metadata
        self._move_traces = []

    def record_move(self, move_trace: dict[str, Any]) -> None:
        """Record one move's trace data. Called after each MCTS search."""
        self._move_traces.append(move_trace)

    def end_game(self, outcome: dict[str, Any]) -> dict:
        """
        Finalize the trace, write to disk, and return the full trace dict.

        Called once after the game terminates.
        """
        trace: dict[str, Any] = {
            "metadata": self._metadata,
            "moves": self._move_traces,
            "outcome": outcome,
        }
        filepath = self._write_trace(trace)
        trace["log_file"] = str(filepath)
        return trace

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_trace(self, trace: dict) -> Path:
        """Write a trace dict to a JSON file and return the path."""
        game_name = trace["metadata"].get("game", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{game_name}_{ts}.json"
        filepath = self.records_dir / filename
        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2, default=str)
        return filepath

