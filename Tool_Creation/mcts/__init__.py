"""
MCTS Framework with pluggable heuristic tools.

Modules:
    games/game_interface — Abstract Game / GameState contracts
    games/               — Concrete game implementations
    node                 — MCTS tree node
    mcts_engine          — Core MCTS: select -> expand -> simulate -> backprop
    trace_logger         — Records game play traces (used internally by engine)
"""

from .games.game_interface import Game, GameState
from .node import MCTSNode
from .mcts_engine import MCTSEngine
