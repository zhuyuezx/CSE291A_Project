"""
Tests for the MCTS engine on Sokoban.

Verifies:
    1. All 10 levels parse correctly (matching box/target counts)
    2. MCTS engine runs without error on every level
    3. Easy levels (1-box) are solvable with enough iterations
    4. Default random rollouts produce valid game results
    5. Tool hot-swap works (set_tool / get_tool / reset_tool)
"""

import pytest
from mcts import MCTSEngine, GameState
from mcts.games.sokoban import Sokoban, SokobanState, LEVELS


# =====================================================================
# Test 1: All levels parse correctly
# =====================================================================

class TestLevelParsing:
    """Every level must have matching box and target counts."""

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_level_has_matching_boxes_and_targets(self, level_name):
        game = Sokoban(level_name=level_name)
        state = game.new_initial_state()
        assert len(state.boxes) == state.num_targets, (
            f"{level_name}: {len(state.boxes)} boxes != {state.num_targets} targets"
        )

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_level_has_legal_actions(self, level_name):
        game = Sokoban(level_name=level_name)
        state = game.new_initial_state()
        assert len(state.legal_actions()) > 0, f"{level_name}: no legal actions"

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_level_is_not_already_solved(self, level_name):
        game = Sokoban(level_name=level_name)
        state = game.new_initial_state()
        assert not state.is_terminal(), f"{level_name}: already terminal at start"

    def test_total_level_count(self):
        assert len(LEVELS) == 10


# =====================================================================
# Test 2: MCTS engine runs without errors
# =====================================================================

class TestMCTSEngineRuns:
    """MCTS engine should run without crashing on every level."""

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_search_returns_valid_action(self, level_name):
        """A single search call should return a legal action."""
        game = Sokoban(level_name=level_name, max_steps=50)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=20)
        state = game.new_initial_state()
        action = engine.search(state)
        assert action in state.legal_actions(), (
            f"search returned {action}, not in {state.legal_actions()}"
        )

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_play_game_completes(self, level_name):
        """play_game should run to completion (solved or max_steps)."""
        game = Sokoban(level_name=level_name, max_steps=30)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=10)
        result = engine.play_game()
        assert "solved" in result
        assert "steps" in result
        assert "moves" in result
        assert result["steps"] > 0

    def test_play_many_returns_stats(self):
        """play_many should return a well-formed stats dict."""
        game = Sokoban(level_name="level1", max_steps=30)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=10)
        stats = engine.play_many(num_games=3)
        assert stats["total"] == 3
        assert 0 <= stats["solve_rate"] <= 1
        assert stats["avg_steps"] > 0
        assert len(stats["results"]) == 3


# =====================================================================
# Test 3: Easy levels solvable with sufficient iterations
# =====================================================================

class TestEasyLevelsSolvable:
    """1-box levels should be solvable with enough MCTS iterations."""

    @pytest.mark.parametrize("level_name", ["level1", "level2", "level3"])
    def test_easy_level_solvable(self, level_name):
        """Run 5 attempts with 200 iters — at least 1 should solve."""
        game = Sokoban(level_name=level_name, max_steps=50)
        engine = MCTSEngine(game, iterations=200, max_rollout_depth=50)
        stats = engine.play_many(num_games=5)
        assert stats["solved"] >= 1, (
            f"{level_name}: 0/{stats['total']} solved with 200 iters "
            f"(avg steps: {stats['avg_steps']})"
        )


# =====================================================================
# Test 4: Game state interface contract
# =====================================================================

class TestGameStateContract:
    """Verify GameState interface works correctly for Sokoban."""

    def test_clone_is_independent(self):
        game = Sokoban(level_name="level1")
        state = game.new_initial_state()
        clone = state.clone()

        # Apply action to original — clone should be unchanged
        action = state.legal_actions()[0]
        state.apply_action(action)
        assert state.state_key() != clone.state_key()
        assert clone.steps == 0

    def test_state_key_deterministic(self):
        game = Sokoban(level_name="level1")
        s1 = game.new_initial_state()
        s2 = game.new_initial_state()
        assert s1.state_key() == s2.state_key()

    def test_returns_format(self):
        game = Sokoban(level_name="level1")
        state = game.new_initial_state()
        r = state.returns()
        assert isinstance(r, list)
        assert len(r) == game.num_players()


# =====================================================================
# Test 5: Tool hot-swap
# =====================================================================

class TestToolHotSwap:
    """set_tool / get_tool / reset_tool / get_tool_source."""

    def test_set_and_get_tool(self):
        game = Sokoban(level_name="level1")
        engine = MCTSEngine(game, iterations=10)

        def my_sim(state, player, max_depth=50):
            return 0.5

        engine.set_tool("simulation", my_sim)
        assert engine.get_tool("simulation") is my_sim

    def test_reset_tool(self):
        game = Sokoban(level_name="level1")
        engine = MCTSEngine(game, iterations=10)

        original_src = engine.get_tool_source()["simulation"]
        engine.set_tool("simulation", lambda s, p, d=50: 0.5)
        engine.reset_tool("simulation")
        # After reset the source should match the original default again
        reset_src = engine.get_tool_source()["simulation"]
        assert "default_simulation" in reset_src

    def test_invalid_tool_name_raises(self):
        game = Sokoban(level_name="level1")
        engine = MCTSEngine(game, iterations=10)
        with pytest.raises(KeyError):
            engine.set_tool("nonexistent_slot", lambda: None)

    def test_get_tool_source(self):
        game = Sokoban(level_name="level1")
        engine = MCTSEngine(game, iterations=10)
        sources = engine.get_tool_source()
        assert "selection" in sources
        assert "simulation" in sources
        assert "backpropagation" in sources
        assert "expansion" in sources
        assert "def " in sources["simulation"]

    def test_custom_simulation_is_used(self):
        """When a custom simulation is set, it should be called."""
        game = Sokoban(level_name="level2", max_steps=30)
        engine = MCTSEngine(game, iterations=20, max_rollout_depth=50)

        call_count = 0
        def counting_sim(state, player, max_depth=50):
            nonlocal call_count
            call_count += 1
            return 0.5

        engine.set_tool("simulation", counting_sim)
        engine.play_game()
        assert call_count > 0, "Custom simulation was never called"
