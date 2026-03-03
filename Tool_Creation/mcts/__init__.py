"""
MCTS Framework with LLM-optimizable heuristics.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  OptimizationLoop                       │
    │  Orchestrates: play → LLM → load → validate → adopt    │
    └────┬──────────┬──────────────┬─────────────────────────┘
         │          │              │
    ┌────▼────┐ ┌───▼──────┐ ┌────▼────────────┐
    │  MCTS   │ │   LLM    │ │ HeuristicLoader  │
    │ Engine  │ │  Client   │ │  (extract/exec)  │
    └────┬────┘ └───┬──────┘ └─────────────────┘
         │          │
    ┌────▼────┐ ┌───▼──────────┐
    │  Game   │ │ PromptBuilder │
    │Interface│ │  + Templates  │
    └────┬────┘ └──────────────┘
         ▲
    ┌────┴───────────┐
    │  games/*.py    │  (Connect Four, Sliding Puzzle, Sokoban)
    └────────────────┘

Modules:
    game_interface   — Abstract Game / GameState contracts
    mcts_engine      — Core MCTS with pluggable heuristic slots
    heuristics       — Default heuristic implementations
    trace_logger     — Records gameplay decisions for LLM analysis
    llm_client       — Ollama REST API client
    prompt_builder   — Game-specific prompt templates + builder
    heuristic_loader — Safe code extraction, compilation & validation
    optimization_loop— Full play-analyse-improve cycle orchestrator
    node             — MCTS tree node with UCB1
    games/           — Concrete game implementations
"""

# Core MCTS
from .game_interface import Game, GameState
from .node import MCTSNode
from .mcts_engine import MCTSEngine
from .trace_logger import TraceLogger

# LLM integration
from .llm_client import LLMClient, LLMResponse
from .heuristic_loader import HeuristicLoader, HeuristicLoadError
from .prompt_builder import PromptBuilder, PromptTemplate
from .optimization_loop import OptimizationLoop, LoopResult, RoundResult
