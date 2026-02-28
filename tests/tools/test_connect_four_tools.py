# tests/tools/test_connect_four_tools.py
import pyspiel
from src.tools.base import load_tool_from_file


def _make_state(moves: list[int] = None):
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    for m in (moves or []):
        state.apply_action(m)
    return state


def test_center_column_bias_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/center_column_bias.py"
    )
    assert meta.type.value == "state_evaluator"


def test_center_column_bias_prefers_center():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/center_column_bias.py"
    )
    state = _make_state()
    score = run_fn(state)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_threat_detector_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/threat_detector.py"
    )
    assert meta.type.value == "action_filter"


def test_threat_detector_returns_subset():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/threat_detector.py"
    )
    state = _make_state()
    legal = state.legal_actions()
    filtered = run_fn(state, legal)
    assert isinstance(filtered, list)
    assert len(filtered) > 0
    assert all(a in legal for a in filtered)


def test_greedy_rollout_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/greedy_rollout.py"
    )
    assert meta.type.value == "rollout_policy"


def test_greedy_rollout_returns_legal_action():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/greedy_rollout.py"
    )
    state = _make_state()
    legal = state.legal_actions()
    action = run_fn(state, legal)
    assert action in legal
