# tests/mcts/test_engine.py
import pyspiel
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.games.adapter import GameAdapter


def test_vanilla_mcts_creates_root():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=10, uct_c=1.41)
    state = adapter.new_game()
    action = engine.search(state)
    assert action in adapter.legal_actions(state)


def test_vanilla_mcts_connect_four():
    adapter = GameAdapter("connect_four")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=100, uct_c=1.41)
    state = adapter.new_game()
    action = engine.search(state)
    assert action in adapter.legal_actions(state)


def test_vanilla_mcts_wins_vs_random_tic_tac_toe():
    """MCTS with 200 sims should beat random in tic-tac-toe most of the time."""
    import random

    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=200, uct_c=1.41)

    wins = 0
    num_games = 20
    for _ in range(num_games):
        state = adapter.new_game()
        while not adapter.is_terminal(state):
            if adapter.current_player(state) == 0:
                action = engine.search(state)
            else:
                action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
        if adapter.returns(state)[0] > 0:
            wins += 1

    # MCTS should win at least 80% against random
    assert wins >= 16, f"MCTS only won {wins}/{num_games} against random"


def test_search_returns_policy():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=50, uct_c=1.41)
    state = adapter.new_game()
    action, policy = engine.search_with_policy(state)
    assert action in adapter.legal_actions(state)
    assert isinstance(policy, dict)
    assert action in policy
    # Policy values should sum to ~1.0
    total = sum(policy.values())
    assert abs(total - 1.0) < 0.01
