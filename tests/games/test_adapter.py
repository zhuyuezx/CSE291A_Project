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


# --- Phase A: GameMeta + normalize_return tests ---

from src.games.meta_registry import GAME_META


def test_game_meta_fields():
    meta = GAME_META["connect_four"]
    assert meta.name == "connect_four"
    assert meta.is_single_player is False
    assert meta.min_return == -1.0
    assert meta.max_return == 1.0
    assert meta.metric_name == "win_rate"
    assert meta.max_sim_depth == 42


def test_game_meta_single_player_entries_exist():
    for name in ["pathfinding", "morpion_solitaire", "2048"]:
        assert name in GAME_META
        assert GAME_META[name].is_single_player is True


def test_normalize_return_two_player():
    adapter = GameAdapter("connect_four")
    assert adapter.normalize_return(1.0) == 1.0
    assert adapter.normalize_return(-1.0) == -1.0
    assert adapter.normalize_return(0.0) == 0.0


def test_normalize_return_single_player_pathfinding():
    adapter = GameAdapter("pathfinding")
    # raw 0.0 → -1.0, raw 1.0 → +1.0
    assert adapter.normalize_return(0.0) == -1.0
    assert adapter.normalize_return(1.0) == 1.0


def test_normalize_return_2048():
    adapter = GameAdapter("2048")
    # raw 0 → -1.0, raw 20000 → +1.0, raw 10000 → 0.0
    assert adapter.normalize_return(0.0) == -1.0
    assert abs(adapter.normalize_return(20000.0) - 1.0) < 1e-6
    assert abs(adapter.normalize_return(10000.0) - 0.0) < 1e-6


def test_normalize_return_clips():
    adapter = GameAdapter("2048")
    # scores above max_return should clip to 1.0
    assert adapter.normalize_return(99999.0) == 1.0
