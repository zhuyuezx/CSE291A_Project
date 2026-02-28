# tests/training/test_evaluator.py
from src.training.evaluator import Evaluator
from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry


def test_evaluate_vs_random():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=50)

    evaluator = Evaluator(adapter)
    result = evaluator.evaluate_vs_random(engine, num_games=10, player=0)
    assert "wins" in result
    assert "losses" in result
    assert "draws" in result
    assert "win_rate" in result
    assert result["wins"] + result["losses"] + result["draws"] == 10


def test_evaluate_head_to_head():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine_a = MCTSEngine(adapter, registry, simulations=50)
    engine_b = MCTSEngine(adapter, registry, simulations=10)

    evaluator = Evaluator(adapter)
    result = evaluator.evaluate_head_to_head(engine_a, engine_b, num_games=10)
    assert "a_wins" in result
    assert "b_wins" in result
    assert result["a_wins"] + result["b_wins"] + result["draws"] == 10


def test_performance_result_fields():
    from src.training.evaluator import PerformanceResult
    pr = PerformanceResult(
        game="pathfinding", metric_name="success_rate",
        raw_value=0.6, normalized_value=0.2, n_games=50
    )
    assert pr.game == "pathfinding"
    assert pr.normalized_value == 0.2


def test_measure_two_player():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=20)
    evaluator = Evaluator(adapter)
    result = evaluator.measure(engine, n_games=10)
    assert result.metric_name == "win_rate"
    assert -1.0 <= result.normalized_value <= 1.0


def test_measure_single_player():
    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=20,
                        max_rollout_depth=adapter.meta.max_sim_depth)
    evaluator = Evaluator(adapter)
    result = evaluator.measure(engine, n_games=5)
    assert result.metric_name == "success_rate"
    assert -1.0 <= result.normalized_value <= 1.0
