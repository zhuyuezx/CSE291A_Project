"""
Orchestrator — iterative LLM-driven MCTS heuristic optimization.

Encapsulates the full train + eval loop into reusable, game-agnostic
Python modules. Configuration is loaded from
``MCTS_tools/hyperparams/default_hyperparams.py``.

Usage::

    from orchestrator import OptimizationRunner, Evaluator

    runner = OptimizationRunner.from_config()
    summary = runner.run()

    # Access results
    best_fn = summary["best_fn"]
    evaluator = runner.evaluator
"""

from .evaluator import Evaluator
from .runner import OptimizationRunner

__all__ = ["Evaluator", "OptimizationRunner"]
