# tests/mcts/test_tool_registry.py
import os
import tempfile

from src.mcts.tool_registry import ToolRegistry
from src.tools.base import ToolType


def _write_tool_file(directory: str, name: str, tool_type: str, body: str) -> str:
    code = f'''
__TOOL_META__ = {{
    "name": "{name}",
    "type": "{tool_type}",
    "description": "test tool",
}}

{body}
'''
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def test_register_and_get_tools():
    registry = ToolRegistry()
    registry.register(
        name="eval1",
        tool_type=ToolType.STATE_EVALUATOR,
        run_fn=lambda state: 0.5,
    )
    tools = registry.get_tools(ToolType.STATE_EVALUATOR)
    assert len(tools) == 1
    assert tools[0].name == "eval1"
    assert tools[0].run_fn(None) == 0.5


def test_get_tools_empty():
    registry = ToolRegistry()
    tools = registry.get_tools(ToolType.ACTION_FILTER)
    assert tools == []


def test_load_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_tool_file(
            tmpdir,
            "eval_tool",
            "state_evaluator",
            "def run(state):\n    return 0.42",
        )
        _write_tool_file(
            tmpdir,
            "filter_tool",
            "action_filter",
            "def run(state, legal_actions):\n    return legal_actions[:2]",
        )

        registry = ToolRegistry()
        registry.load_from_directory(tmpdir)

        evals = registry.get_tools(ToolType.STATE_EVALUATOR)
        assert len(evals) == 1
        assert evals[0].run_fn(None) == 0.42

        filters = registry.get_tools(ToolType.ACTION_FILTER)
        assert len(filters) == 1
        assert filters[0].run_fn(None, [1, 2, 3]) == [1, 2]


def test_unregister():
    registry = ToolRegistry()
    registry.register("eval1", ToolType.STATE_EVALUATOR, lambda s: 0.5)
    registry.unregister("eval1")
    assert registry.get_tools(ToolType.STATE_EVALUATOR) == []


def test_list_all():
    registry = ToolRegistry()
    registry.register("eval1", ToolType.STATE_EVALUATOR, lambda s: 0.5)
    registry.register("filter1", ToolType.ACTION_FILTER, lambda s, a: a)
    names = registry.list_all()
    assert set(names) == {"eval1", "filter1"}
