"""
Orchestrator — iterative LLM-driven MCTS heuristic optimization.

Encapsulates the full train + eval loop into reusable, game-agnostic
Python modules. Configuration is loaded from ``config.json``.

Usage::

    from orchestrator import OptimizationRunner, Evaluator

    # From config file (recommended)
    runner = OptimizationRunner.from_config("orchestrator/config.json")
    summary = runner.run()

    # Access results
    best_fn = summary["best_fn"]
    evaluator = runner.evaluator
"""

from .evaluator import Evaluator
from .runner import OptimizationRunner

__all__ = ["Evaluator", "OptimizationRunner"]
