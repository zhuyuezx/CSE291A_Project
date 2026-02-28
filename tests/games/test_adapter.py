# tests/games/test_adapter.py
import pyspiel
from src.games.adapter import GameAdapter


def test_create_connect_four():
    adapter = GameAdapter("connect_four")
    assert adapter.game_name == "connect_four"
    assert adapter.num_players == 2


def test_new_game_state():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    assert not state.is_terminal()
    assert state.current_player() == 0


def test_legal_actions():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    assert len(actions) == 7  # 7 columns


def test_apply_action():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    new_state = adapter.apply_action(state, actions[0])
    assert new_state.current_player() == 1


def test_clone_state():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    clone = adapter.clone_state(state)
    new_state = adapter.apply_action(state, 0)
    # Clone should not be affected by action applied to original
    assert str(new_state) != str(clone)


def test_play_random_game_to_terminal():
    import random

    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    while not adapter.is_terminal(state):
        actions = adapter.legal_actions(state)
        state = adapter.apply_action(state, random.choice(actions))
    returns = adapter.returns(state)
    assert len(returns) == 2
    assert all(isinstance(r, float) for r in returns)


def test_game_description():
    adapter = GameAdapter("connect_four")
    desc = adapter.game_description()
    assert "connect_four" in desc.lower() or len(desc) > 0
