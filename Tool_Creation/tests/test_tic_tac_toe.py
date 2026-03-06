"""
Tests for MCTS on Tic-Tac-Toe — the simplest 2-player base case.

Verifies:
    1. Game interface contract (clone, terminal, returns, state_key)
    2. MCTS produces legal actions and completes games
    3. MCTS with enough iterations plays near-optimally (high draw rate)
    4. All 4 tool phases are invoked during search
"""

import pytest
from mcts import MCTSEngine
from mcts.games.tic_tac_toe import TicTacToe, TicTacToeState


# =====================================================================
# Test 1: Game state contract
# =====================================================================

class TestTicTacToeState:
    """Verify TicTacToeState satisfies the GameState interface."""

    def test_initial_state_not_terminal(self):
        state = TicTacToeState()
        assert not state.is_terminal()

    def test_initial_legal_actions(self):
        state = TicTacToeState()
        assert state.legal_actions() == list(range(9))

    def test_current_player_starts_at_0(self):
        state = TicTacToeState()
        assert state.current_player() == 0

    def test_player_alternates(self):
        state = TicTacToeState()
        state.apply_action(0)
        assert state.current_player() == 1
        state.apply_action(1)
        assert state.current_player() == 0

    def test_clone_is_independent(self):
        state = TicTacToeState()
        state.apply_action(4)
        clone = state.clone()
        clone.apply_action(0)
        # Original should be unchanged
        assert state.board[0] is None
        assert clone.board[0] == 1

    def test_state_key_deterministic(self):
        s1 = TicTacToeState()
        s2 = TicTacToeState()
        assert s1.state_key() == s2.state_key()

    def test_state_key_changes_after_action(self):
        state = TicTacToeState()
        key_before = state.state_key()
        state.apply_action(4)
        assert state.state_key() != key_before

    def test_returns_format(self):
        game = TicTacToe()
        state = game.new_initial_state()
        r = state.returns()
        assert isinstance(r, list)
        assert len(r) == game.num_players()

    def test_win_detected(self):
        """Player 0 wins with top row."""
        state = TicTacToeState()
        # X O X O X . . . .
        for move in [0, 3, 1, 4, 2]:  # X takes 0,1,2 -> top row
            state.apply_action(move)
        assert state.is_terminal()
        assert state.returns() == [1.0, -1.0]

    def test_draw_detected(self):
        """Fill the board with no winner."""
        state = TicTacToeState()
        # X O X | X O O | O X X  -> draw
        for move in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
            state.apply_action(move)
        assert state.is_terminal()
        assert state.returns() == [0.0, 0.0]

    def test_no_legal_actions_when_terminal(self):
        state = TicTacToeState()
        for move in [0, 3, 1, 4, 2]:  # X wins
            state.apply_action(move)
        assert state.legal_actions() == []

    def test_game_factory(self):
        game = TicTacToe()
        assert game.name() == "TicTacToe"
        assert game.num_players() == 2
        state = game.new_initial_state()
        assert isinstance(state, TicTacToeState)


# =====================================================================
# Test 2: MCTS engine runs correctly on TicTacToe
# =====================================================================

class TestMCTSOnTicTacToe:
    """MCTS should produce valid moves and complete games."""

    def test_search_returns_legal_action(self):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=50)
        state = game.new_initial_state()
        action = engine.search(state)
        assert action in state.legal_actions()

    def test_search_mid_game(self):
        """Search from a partially played position."""
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=50)
        state = game.new_initial_state()
        state.apply_action(4)  # X plays center
        state.apply_action(0)  # O plays corner
        action = engine.search(state)
        assert action in state.legal_actions()

    def test_play_game_completes(self):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=50)
        result = engine.play_game()
        assert "solved" in result
        assert "steps" in result
        assert "moves" in result
        assert 5 <= result["steps"] <= 9  # min 5 moves to win, max 9

    def test_play_many_returns_stats(self):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=50)
        stats = engine.play_many(num_games=5)
        assert stats["total"] == 5
        assert len(stats["results"]) == 5
        assert 0 <= stats["solve_rate"] <= 1


# =====================================================================
# Test 3: MCTS plays near-optimally with enough iterations
# =====================================================================

class TestMCTSQuality:
    """With sufficient iterations, MCTS should play strong tic-tac-toe."""

    def test_mcts_finds_winning_move(self):
        """If player 0 can win in one move, MCTS should find it."""
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=200)
        state = game.new_initial_state()
        # Set up: X at 0,1 — winning move is 2 (top row)
        state.apply_action(0)  # X
        state.apply_action(3)  # O
        state.apply_action(1)  # X
        state.apply_action(4)  # O
        # X to move, should play 2 to win
        action = engine.search(state)
        assert action == 2

    def test_mcts_blocks_opponent_win(self):
        """If opponent can win next move, MCTS should block."""
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=200)
        state = game.new_initial_state()
        # X plays 4 (center), O plays 0, X plays 8, O plays 6
        # Board: O . . | . X . | O . X
        # O has 0,6 -> needs 3 to win. X must block at 3.
        state.apply_action(4)  # X center
        state.apply_action(0)  # O corner
        state.apply_action(8)  # X corner
        state.apply_action(6)  # O corner — threatens col 0 (0,3,6)
        action = engine.search(state)
        assert action == 3, f"MCTS should block at 3, got {action}"

    def test_high_draw_rate_self_play(self):
        """Two strong MCTS players should produce some draws."""
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=500)
        stats = engine.play_many(num_games=20)
        draw_count = sum(
            1 for r in stats["results"]
            if r["returns"] == [0.0, 0.0]
        )
        # With 500 iterations, expect at least 2/20 draws
        assert draw_count >= 2, (
            f"Only {draw_count}/20 draws — MCTS not playing strong enough"
        )


# =====================================================================
# Test 4: All 4 tool phases are called
# =====================================================================

class TestAllPhasesUsed:
    """Verify that selection, expansion, simulation, backpropagation are all invoked."""

    def test_all_four_phases_called(self):
        game = TicTacToe()
        engine = MCTSEngine(game, iterations=20)

        called = {phase: 0 for phase in engine.PHASES}

        original_tools = {phase: engine.get_tool(phase) for phase in engine.PHASES}

        def make_wrapper(phase, original_fn):
            def wrapper(*args, **kwargs):
                called[phase] += 1
                return original_fn(*args, **kwargs)
            return wrapper

        for phase in engine.PHASES:
            engine.set_tool(phase, make_wrapper(phase, original_tools[phase]))

        engine.search(game.new_initial_state())

        for phase in engine.PHASES:
            assert called[phase] > 0, f"Phase '{phase}' was never called"
