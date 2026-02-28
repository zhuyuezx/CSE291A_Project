# tests/training/test_trace_recorder.py
from src.training.trace_recorder import TraceRecorder, GameTrace
from src.games.adapter import GameAdapter
import random


def test_record_game():
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    state = adapter.new_game()
    recorder.start_game(state)

    while not adapter.is_terminal(state):
        action = random.choice(adapter.legal_actions(state))
        state = adapter.apply_action(state, action)
        recorder.record_step(state, action)

    recorder.end_game(adapter.returns(state))
    traces = recorder.get_traces()
    assert len(traces) == 1
    assert len(traces[0].actions) > 0
    assert traces[0].outcome is not None


def test_select_informative_traces():
    """Should select losses and close games preferentially."""
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    # Record several games
    for _ in range(10):
        state = adapter.new_game()
        recorder.start_game(state)
        while not adapter.is_terminal(state):
            action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
            recorder.record_step(state, action)
        recorder.end_game(adapter.returns(state))

    informative = recorder.select_informative_traces(player=0, n=3)
    assert len(informative) <= 3
    assert all(isinstance(t, GameTrace) for t in informative)


def test_trace_to_string():
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    state = adapter.new_game()
    recorder.start_game(state)
    action = adapter.legal_actions(state)[0]
    state = adapter.apply_action(state, action)
    recorder.record_step(state, action)
    recorder.end_game([1.0, -1.0])

    trace = recorder.get_traces()[0]
    text = trace.to_string()
    assert isinstance(text, str)
    assert len(text) > 0
