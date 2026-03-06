"""
MCTS Framework with pluggable heuristic tools.

Modules:
    games/game_interface — Abstract Game / GameState contracts
    games/               — Concrete game implementations
    node                 — MCTS tree node with UCB1
    mcts_engine          — Core MCTS: select → expand → simulate → backprop
"""

from .games.game_interface import Game, GameState
from .node import MCTSNode
from .mcts_engine import MCTSEngine
