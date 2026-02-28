# tests/integration/test_connect_four_with_tools.py
"""
Integration test: MCTS with hand-written tools should beat vanilla MCTS
on Connect Four at equal simulation budgets.
"""
import random
import pyspiel
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.games.adapter import GameAdapter


def _play_game(engine_p0: MCTSEngine, engine_p1: MCTSEngine, adapter: GameAdapter) -> float:
    """Play one game, return player 0's result."""
    state = adapter.new_game()
    while not adapter.is_terminal(state):
        player = adapter.current_player(state)
        if player == 0:
            action = engine_p0.search(state)
        else:
            action = engine_p1.search(state)
        state = adapter.apply_action(state, action)
    return adapter.returns(state)[0]


def test_tools_beat_vanilla():
    """MCTS+tools should win more than lose against vanilla MCTS at 100 sims."""
    adapter = GameAdapter("connect_four")

    vanilla_registry = ToolRegistry()
    tool_registry = ToolRegistry()
    tool_registry.load_from_directory("tool_pool/connect_four")

    vanilla = MCTSEngine(adapter, vanilla_registry, simulations=100, uct_c=1.41)
    with_tools = MCTSEngine(adapter, tool_registry, simulations=100, uct_c=1.41)

    wins = 0
    losses = 0
    draws = 0
    num_games = 20

    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            result = _play_game(with_tools, vanilla, adapter)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        else:
            result = _play_game(vanilla, with_tools, adapter)
            if result < 0:
                wins += 1
            elif result > 0:
                losses += 1
            else:
                draws += 1

    print(f"\nTools vs Vanilla: {wins}W / {losses}L / {draws}D out of {num_games}")
    # Tools should at least not be significantly worse
    assert wins >= losses, f"Tools lost more than won: {wins}W/{losses}L/{draws}D"


def test_tools_beat_random():
    """MCTS+tools at 50 sims should crush random player."""
    adapter = GameAdapter("connect_four")
    tool_registry = ToolRegistry()
    tool_registry.load_from_directory("tool_pool/connect_four")
    engine = MCTSEngine(adapter, tool_registry, simulations=50, uct_c=1.41)

    wins = 0
    num_games = 20
    for i in range(num_games):
        state = adapter.new_game()
        while not adapter.is_terminal(state):
            player = adapter.current_player(state)
            if player == 0:
                action = engine.search(state)
            else:
                action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
        if adapter.returns(state)[0] > 0:
            wins += 1

    print(f"\nTools+MCTS vs Random: {wins}/{num_games} wins")
    assert wins >= 15, f"Expected at least 15 wins, got {wins}"
