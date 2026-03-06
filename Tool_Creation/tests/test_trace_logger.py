"""
Tests for trace logging integrated into MCTSEngine.

Verifies:
    1. Engine with logging=True produces JSON files on disk
    2. Trace structure has all required fields (metadata, moves, outcome)
    3. Per-move trace contains state, action, children_stats
    4. play_many with logging produces one file per game
    5. logging=False produces no files (default)
    6. Works for both 1-player (Sokoban) and 2-player (TicTacToe)
"""

import json
import os

import pytest
from mcts import MCTSEngine
from mcts.trace_logger import TraceLogger
from mcts.games.tic_tac_toe import TicTacToe
from mcts.games.sokoban import Sokoban


@pytest.fixture
def tmp_records(tmp_path):
    """Provide a temporary directory for trace records."""
    d = tmp_path / "test_records"
    d.mkdir()
    return d


# =====================================================================
# Test 1: Engine with logging=True writes JSON files
# =====================================================================

class TestTraceFileCreation:
    """Engine with logging=True should produce JSON files on disk."""

    def test_creates_json_file(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()

        assert "log_file" in result
        assert os.path.isfile(result["log_file"])
        assert result["log_file"].endswith(".json")

    def test_file_is_valid_json(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()

        with open(result["log_file"]) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_filename_contains_game_name(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()

        filename = os.path.basename(result["log_file"])
        assert filename.startswith("TicTacToe_")

    def test_no_file_when_logging_disabled(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=False)

        result = engine.play_game()

        assert "log_file" not in result
        # tmp_records should still be empty
        assert len(list(tmp_records.glob("*.json"))) == 0

    def test_creates_records_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "nonexistent" / "subdir"
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=new_dir)

        assert new_dir.exists()


# =====================================================================
# Test 2: Trace structure has all required fields
# =====================================================================

class TestTraceStructure:
    """The trace dict must have metadata, moves, and outcome."""

    def test_has_metadata(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()
        trace = result["trace"]

        assert "metadata" in trace
        meta = trace["metadata"]
        assert meta["game"] == "TicTacToe"
        assert "timestamp" in meta
        assert meta["iterations"] == 10
        assert "exploration_weight" in meta
        assert "tools" in meta
        for phase in ("selection", "expansion", "simulation", "backpropagation"):
            assert phase in meta["tools"]

    def test_has_outcome(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()
        trace = result["trace"]

        assert "outcome" in trace
        outcome = trace["outcome"]
        assert "solved" in outcome
        assert "steps" in outcome
        assert "returns" in outcome
        assert "final_state" in outcome
        assert isinstance(outcome["returns"], list)
        assert outcome["steps"] > 0

    def test_has_moves(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()
        trace = result["trace"]

        assert "moves" in trace
        assert len(trace["moves"]) == trace["outcome"]["steps"]


# =====================================================================
# Test 3: Per-move trace details
# =====================================================================

class TestMoveTrace:
    """Each move entry should contain detailed search information."""

    def test_move_has_required_fields(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=20, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()
        move = result["trace"]["moves"][0]

        assert move["move_number"] == 1
        assert "player" in move
        assert "state_before" in move
        assert "state_key" in move
        assert "legal_actions" in move
        assert "action_chosen" in move
        assert "root_visits" in move
        assert "children_stats" in move

    def test_children_stats_have_visit_counts(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=30, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()
        move = result["trace"]["moves"][0]

        assert len(move["children_stats"]) > 0
        for action_key, stats in move["children_stats"].items():
            assert "visits" in stats
            assert "value" in stats
            assert "avg_value" in stats
            assert stats["visits"] > 0

    def test_action_chosen_is_in_legal_actions(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=20, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()

        for move in result["trace"]["moves"]:
            assert move["action_chosen"] in move["legal_actions"]

    def test_move_numbers_sequential(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        result = engine.play_game()

        for i, move in enumerate(result["trace"]["moves"]):
            assert move["move_number"] == i + 1


# =====================================================================
# Test 4: play_many with logging
# =====================================================================

class TestPlayManyWithLogging:
    """play_many with logging=True should produce one file per game."""

    def test_creates_multiple_files(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        stats = engine.play_many(num_games=3)

        files = list(tmp_records.glob("*.json"))
        assert len(files) == 3

    def test_each_result_has_log_file(self, tmp_records):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=10, logging=True,
                            records_dir=tmp_records)

        stats = engine.play_many(num_games=3)

        log_files = set()
        for r in stats["results"]:
            assert "log_file" in r
            log_files.add(r["log_file"])
        assert len(log_files) == 3  # all unique files


# =====================================================================
# Test 5: Sokoban trace (single player)
# =====================================================================

class TestSokobanTrace:
    """Logging should work for single-player games too."""

    def test_sokoban_trace_structure(self, tmp_records):
        game = Sokoban(level_name="level1", max_steps=20)
        engine = MCTSEngine(game, iterations=20, max_rollout_depth=10,
                            logging=True, records_dir=tmp_records)

        result = engine.play_game()
        trace = result["trace"]

        assert "Sokoban" in trace["metadata"]["game"]
        assert len(trace["moves"]) > 0
        for move in trace["moves"]:
            assert move["player"] == 0

    def test_sokoban_file_readable(self, tmp_records):
        game = Sokoban(level_name="level1", max_steps=15)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=10,
                            logging=True, records_dir=tmp_records)

        result = engine.play_game()

        with open(result["log_file"]) as f:
            data = json.load(f)
        assert "Sokoban" in data["metadata"]["game"]
        assert len(data["moves"]) == data["outcome"]["steps"]
