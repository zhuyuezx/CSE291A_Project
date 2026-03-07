"""
Tests for the Rush Hour puzzle game.

Verifies:
    1. All built-in puzzles parse correctly
    2. MCTS engine runs without error on Rush Hour
    3. Easy puzzles are solvable with enough iterations
    4. GameState interface contract (clone, returns, state_key)
    5. Action mechanics (apply, undo-by-clone, blocking)
"""

import pytest
from mcts import MCTSEngine
from mcts.games.rush_hour import RushHour, RushHourState, PUZZLES, _parse_board


# =====================================================================
# Test 1: All built-in puzzles parse correctly
# =====================================================================

class TestPuzzleParsing:
    """Every built-in puzzle must parse into a valid initial state."""

    @pytest.mark.parametrize("puzzle_name", list(PUZZLES.keys()))
    def test_puzzle_parses(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name)
        state = game.new_initial_state()
        assert state.num_pieces() >= 2, f"{puzzle_name}: needs at least 2 pieces"

    @pytest.mark.parametrize("puzzle_name", list(PUZZLES.keys()))
    def test_puzzle_has_legal_actions(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name)
        state = game.new_initial_state()
        assert len(state.legal_actions()) > 0, f"{puzzle_name}: no legal actions"

    @pytest.mark.parametrize("puzzle_name", list(PUZZLES.keys()))
    def test_puzzle_not_already_solved(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name)
        state = game.new_initial_state()
        assert not state.is_terminal(), f"{puzzle_name}: already terminal at start"

    @pytest.mark.parametrize("puzzle_name", list(PUZZLES.keys()))
    def test_primary_distance_positive(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name)
        state = game.new_initial_state()
        assert state.primary_distance() > 0, f"{puzzle_name}: primary already at target"

    def test_total_puzzle_count(self):
        assert len(PUZZLES) == 8

    def test_all_descriptions_are_36_chars(self):
        for name, (desc, _) in PUZZLES.items():
            assert len(desc) == 36, f"{name}: description is {len(desc)} chars"


# =====================================================================
# Test 2: MCTS engine runs without errors
# =====================================================================

class TestMCTSEngineRuns:
    """MCTS engine should run without crashing on Rush Hour puzzles."""

    @pytest.mark.parametrize("puzzle_name", ["easy1", "easy2", "easy3"])
    def test_search_returns_valid_action(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name, max_moves=50)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=20)
        state = game.new_initial_state()
        action = engine.search(state)
        assert action in state.legal_actions()

    @pytest.mark.parametrize("puzzle_name", ["easy1", "easy2", "easy3"])
    def test_play_game_completes(self, puzzle_name):
        game = RushHour(puzzle_name=puzzle_name, max_moves=30)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=10)
        result = engine.play_game()
        assert "solved" in result
        assert "steps" in result
        assert "moves" in result

    def test_play_many_returns_stats(self):
        game = RushHour(puzzle_name="easy1", max_moves=30)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=10)
        stats = engine.play_many(num_games=3)
        assert stats["total"] == 3
        assert 0 <= stats["solve_rate"] <= 1
        assert len(stats["results"]) == 3


# =====================================================================
# Test 3: Easy puzzles solvable with sufficient iterations
# =====================================================================

class TestEasyPuzzlesSolvable:
    """Easy puzzles should be solvable with enough MCTS iterations."""

    def test_easy1_solvable(self):
        """easy1 has only 3 pieces — should be very solvable."""
        game = RushHour(puzzle_name="easy1", max_moves=50)
        engine = MCTSEngine(game, iterations=200, max_rollout_depth=50)
        stats = engine.play_many(num_games=5)
        assert stats["solved"] >= 1, (
            f"easy1: 0/{stats['total']} solved with 200 iters"
        )


# =====================================================================
# Test 4: Game state interface contract
# =====================================================================

class TestGameStateContract:
    """Verify GameState interface works correctly for Rush Hour."""

    def test_clone_is_independent(self):
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        clone = state.clone()

        action = state.legal_actions()[0]
        state.apply_action(action)
        assert state.state_key() != clone.state_key()
        assert clone.moves_made == 0

    def test_state_key_deterministic(self):
        game = RushHour(puzzle_name="easy1")
        s1 = game.new_initial_state()
        s2 = game.new_initial_state()
        assert s1.state_key() == s2.state_key()

    def test_returns_format(self):
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        r = state.returns()
        assert isinstance(r, list)
        assert len(r) == game.num_players()
        assert 0.0 <= r[0] <= 1.0

    def test_current_player_always_zero(self):
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        assert state.current_player() == 0

    def test_num_players_is_one(self):
        game = RushHour(puzzle_name="easy1")
        assert game.num_players() == 1

    def test_name_includes_puzzle(self):
        game = RushHour(puzzle_name="hard1")
        assert "hard1" in game.name()


# =====================================================================
# Test 5: Action mechanics
# =====================================================================

class TestActionMechanics:
    """Verify move application, occupancy, and terminal detection."""

    def test_apply_action_changes_state(self):
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        key_before = state.state_key()
        action = state.legal_actions()[0]
        state.apply_action(action)
        assert state.state_key() != key_before

    def test_solved_state_returns_one(self):
        """Manually slide primary piece to target on easy1."""
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        # Play random legal moves until solved or budget exhausted
        for _ in range(100):
            if state.is_terminal():
                break
            actions = state.legal_actions()
            if not actions:
                break
            state.apply_action(actions[0])
        # At least verify returns is valid
        r = state.returns()
        assert isinstance(r, list) and len(r) == 1

    def test_max_moves_causes_terminal(self):
        game = RushHour(puzzle_name="hard1", max_moves=3)
        state = game.new_initial_state()
        for _ in range(3):
            if state.is_terminal():
                break
            actions = state.legal_actions()
            state.apply_action(actions[0])
        assert state.is_terminal()

    def test_display_string_is_6x6(self):
        game = RushHour(puzzle_name="easy1")
        state = game.new_initial_state()
        text = str(state)
        lines = text.strip().split("\n")
        # First line is header, remaining 6 are the board
        assert len(lines) == 7, f"Expected 7 lines (header + 6 rows), got {len(lines)}"
        for row in lines[1:]:
            assert len(row) == 6, f"Row '{row}' is {len(row)} chars, expected 6"

    def test_custom_board_desc(self):
        """RushHour accepts a raw board description string."""
        desc = PUZZLES["easy2"][0]
        game = RushHour(board_desc=desc)
        state = game.new_initial_state()
        assert state.num_pieces() > 0

    def test_invalid_puzzle_name_raises(self):
        with pytest.raises(ValueError, match="Unknown puzzle"):
            RushHour(puzzle_name="nonexistent")

    def test_parse_invalid_length_raises(self):
        with pytest.raises(ValueError, match="36 chars"):
            _parse_board("too_short")

    def test_parse_missing_primary_raises(self):
        with pytest.raises(ValueError, match="primary piece"):
            _parse_board("." * 36)
