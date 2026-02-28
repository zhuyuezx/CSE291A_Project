# tests/training/test_trainer.py
from src.training.trainer import PlateauDetector


def test_no_plateau_at_start():
    detector = PlateauDetector(window_size=5, improvement_threshold=0.02)
    detector.record(0.3)
    detector.record(0.35)
    detector.record(0.4)
    assert not detector.is_plateau()


def test_plateau_detected():
    detector = PlateauDetector(window_size=3, improvement_threshold=0.02)
    # Fill two windows with same performance
    for _ in range(3):
        detector.record(0.5)
    for _ in range(3):
        detector.record(0.51)  # <2% improvement
    assert detector.is_plateau()


def test_no_plateau_when_improving():
    detector = PlateauDetector(window_size=3, improvement_threshold=0.02)
    for _ in range(3):
        detector.record(0.3)
    for _ in range(3):
        detector.record(0.5)  # big improvement
    assert not detector.is_plateau()


def test_regression_detected():
    detector = PlateauDetector(
        window_size=3, improvement_threshold=0.02, regression_threshold=0.05
    )
    for _ in range(3):
        detector.record(0.6)
    for _ in range(3):
        detector.record(0.5)  # dropped by 0.1 > 0.05
    assert detector.is_plateau()


def test_trainer_single_player_game():
    """Trainer should run a single-player episode without errors."""
    from src.games.adapter import GameAdapter
    from src.mcts.tool_registry import ToolRegistry
    from src.training.trainer import Trainer

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    trainer = Trainer(adapter, registry, simulations=10)
    result = trainer.play_episode()
    assert isinstance(result, float)   # normalized return


def test_trainer_single_player_train():
    from src.games.adapter import GameAdapter
    from src.mcts.tool_registry import ToolRegistry
    from src.training.trainer import Trainer

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    trainer = Trainer(adapter, registry, simulations=10)
    stats = trainer.train(num_games=5)
    assert stats["games"] == 5
    assert "success_rate" in stats
