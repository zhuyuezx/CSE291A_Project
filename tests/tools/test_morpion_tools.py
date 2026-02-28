import importlib.util, random
import pyspiel


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _morpion_state():
    game = pyspiel.load_game("morpion_solitaire")
    state = game.new_initial_state()
    for _ in range(3):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_line_extension_filter_returns_subset():
    mod = _load_tool("tool_pool/morpion_solitaire/line_extension_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"
    state = _morpion_state()
    if not state.is_terminal():
        legal = state.legal_actions()
        filtered = mod.run(state, legal)
        assert isinstance(filtered, list)
        assert len(filtered) >= 1
        assert all(a in legal for a in filtered)


def test_density_evaluator_in_range():
    mod = _load_tool("tool_pool/morpion_solitaire/density_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _morpion_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0
