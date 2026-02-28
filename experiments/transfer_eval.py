# experiments/transfer_eval.py
from __future__ import annotations
from dataclasses import dataclass

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.training.evaluator import Evaluator, PerformanceResult


@dataclass
class TransferResult:
    source_game: str
    target_game: str
    cold_start: PerformanceResult
    transferred: PerformanceResult


def run_transfer_chain(
    source_game: str,
    target_game: str,
    source_tool_dir: str,
    target_tool_dir: str,
    n_eval_games: int = 50,
    sim_budget: int = 100,
) -> TransferResult:
    adapter = GameAdapter(target_game)
    evaluator = Evaluator(adapter)

    # Cold start: vanilla MCTS, no tools
    vanilla_registry = ToolRegistry()
    vanilla_engine = MCTSEngine(
        adapter, vanilla_registry,
        simulations=sim_budget,
        max_rollout_depth=adapter.meta.max_sim_depth
    )
    cold_start = evaluator.measure(vanilla_engine, n_games=n_eval_games)

    # Transferred: load source game tools into target registry
    transfer_registry = ToolRegistry()
    transfer_registry.load_from_directory(source_tool_dir)
    transfer_engine = MCTSEngine(
        adapter, transfer_registry,
        simulations=sim_budget,
        max_rollout_depth=adapter.meta.max_sim_depth
    )
    transferred = evaluator.measure(transfer_engine, n_games=n_eval_games)

    return TransferResult(
        source_game=source_game,
        target_game=target_game,
        cold_start=cold_start,
        transferred=transferred,
    )


TRANSFER_CHAINS = [
    ("quoridor",     "pathfinding",       "tool_pool/quoridor",     "tool_pool/pathfinding"),
    ("connect_four", "morpion_solitaire", "tool_pool/connect_four", "tool_pool/morpion_solitaire"),
    ("connect_four", "2048",              "tool_pool/connect_four", "tool_pool/2048"),
]


if __name__ == "__main__":
    for source, target, src_dir, tgt_dir in TRANSFER_CHAINS:
        print(f"\n=== Transfer: {source} → {target} ===")
        result = run_transfer_chain(source, target, src_dir, tgt_dir,
                                    n_eval_games=50, sim_budget=200)
        cs = result.cold_start
        tr = result.transferred
        print(f"  Cold start  ({cs.metric_name}): raw={cs.raw_value:.3f}  norm={cs.normalized_value:.3f}")
        print(f"  Transferred ({tr.metric_name}): raw={tr.raw_value:.3f}  norm={tr.normalized_value:.3f}")
        delta = tr.normalized_value - cs.normalized_value
        print(f"  Transfer delta: {delta:+.3f}")
