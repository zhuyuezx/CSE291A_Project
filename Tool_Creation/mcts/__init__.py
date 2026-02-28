"""
MCTS Framework with LLM-optimizable heuristics.

Architecture:
    ┌────────────────────┐
    │     MCTS Engine    │  ◄── pluggable heuristic functions
    │  (mcts_engine.py)  │
    └────────┬───────────┘
             │ uses
    ┌────────▼───────────┐     ┌───────────────────┐
    │   Game Interface   │     │   Trace Logger     │
    │ (game_interface.py)│     │ (trace_logger.py)  │
    └────────────────────┘     └───────────────────┘
             ▲ implements
    ┌────────┴───────────┐
    │  Connect Four /    │
    │  other games       │
    │  (games/*.py)      │
    └────────────────────┘

The "tools" the LLM agent optimizes are the heuristic functions
in heuristics.py — evaluation, rollout policy, etc.
"""

from .game_interface import Game, GameState
from .node import MCTSNode
from .mcts_engine import MCTSEngine
from .trace_logger import TraceLogger
