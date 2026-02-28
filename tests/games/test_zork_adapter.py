import pytest
import shutil
from src.games.zork_adapter import ZorkAdapter

ZORK_PATH = "assets/zork/zork1.z3"


@pytest.fixture
def adapter():
    return ZorkAdapter(ZORK_PATH)


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_new_game_returns_state(adapter):
    state = adapter.new_game()
    assert state is not None
    assert not adapter.is_terminal(state)


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_legal_actions_nonempty(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    assert len(actions) > 0


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_action_to_string(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    cmd = adapter.action_to_string(state, actions[0])
    assert isinstance(cmd, str) and len(cmd) > 0


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_apply_action_changes_state(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    new_state = adapter.apply_action(state, actions[0])
    assert str(new_state) != str(state) or adapter.is_terminal(new_state)


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_returns_nonnegative(adapter):
    state = adapter.new_game()
    ret = adapter.returns(state)
    assert isinstance(ret, list)
    assert ret[0] >= 0.0


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_normalize_return_in_range(adapter):
    state = adapter.new_game()
    raw = adapter.returns(state)[0]
    norm = adapter.normalize_return(raw)
    assert -1.0 <= norm <= 1.0
