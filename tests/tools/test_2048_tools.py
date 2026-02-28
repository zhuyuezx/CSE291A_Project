import importlib.util, random
import pyspiel


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _2048_state():
    game = pyspiel.load_game("2048")
    state = game.new_initial_state()
    for _ in range(10):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_monotonicity_in_range():
    mod = _load_tool("tool_pool/2048/monotonicity_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _2048_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0


def test_empty_cell_in_range():
    mod = _load_tool("tool_pool/2048/empty_cell_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _2048_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0
