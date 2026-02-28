import importlib.util, pathlib, random
import pyspiel


def _load_tool(path: str):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_pathfinding_state():
    game = pyspiel.load_game("pathfinding")
    state = game.new_initial_state()
    for _ in range(5):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_distance_evaluator_returns_in_range():
    mod = _load_tool("tool_pool/pathfinding/distance_to_goal_evaluator.py")
    assert hasattr(mod, "__TOOL_META__")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _random_pathfinding_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0


def test_backtrack_filter_returns_subset():
    mod = _load_tool("tool_pool/pathfinding/backtrack_pruning_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"
    state = _random_pathfinding_state()
    if not state.is_terminal():
        legal = state.legal_actions()
        filtered = mod.run(state, legal)
        assert isinstance(filtered, list)
        assert len(filtered) >= 1
        assert all(a in legal for a in filtered)
