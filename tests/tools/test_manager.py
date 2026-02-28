# tests/tools/test_manager.py
import json
import os
import tempfile
from src.tools.manager import ToolPoolManager


def _make_tool_file(directory, name, tool_type="state_evaluator"):
    code = f'''
__TOOL_META__ = {{
    "name": "{name}",
    "type": "{tool_type}",
    "description": "Test tool {name}",
}}

def run(state) -> float:
    return 0.0
'''
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def test_save_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ToolPoolManager(pool_dir=tmpdir)
        code = '''
__TOOL_META__ = {
    "name": "new_tool",
    "type": "state_evaluator",
    "description": "A new tool",
}

def run(state) -> float:
    return 0.0
'''
        manager.save_tool("connect_four", "new_tool", code)
        assert os.path.exists(os.path.join(tmpdir, "connect_four", "new_tool.py"))


def test_load_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"tool1": {"type": "state_evaluator", "origin_game": "ttt"}}, f)

        manager = ToolPoolManager(pool_dir=tmpdir)
        meta = manager.load_metadata()
        assert "tool1" in meta


def test_update_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({}, f)

        manager = ToolPoolManager(pool_dir=tmpdir)
        manager.update_metadata("new_tool", {
            "type": "state_evaluator",
            "origin_game": "connect_four",
            "games_tested": {"connect_four": 5.0},
        })
        meta = manager.load_metadata()
        assert "new_tool" in meta
        assert meta["new_tool"]["origin_game"] == "connect_four"


def test_list_tools_for_game():
    with tempfile.TemporaryDirectory() as tmpdir:
        game_dir = os.path.join(tmpdir, "connect_four")
        os.makedirs(game_dir)
        _make_tool_file(game_dir, "tool_a")
        _make_tool_file(game_dir, "tool_b")

        manager = ToolPoolManager(pool_dir=tmpdir)
        tools = manager.list_tools_for_game("connect_four")
        assert set(tools) == {"tool_a.py", "tool_b.py"}


def test_promote_to_global():
    with tempfile.TemporaryDirectory() as tmpdir:
        game_dir = os.path.join(tmpdir, "connect_four")
        global_dir = os.path.join(tmpdir, "global")
        os.makedirs(game_dir)
        os.makedirs(global_dir)
        _make_tool_file(game_dir, "good_tool")

        manager = ToolPoolManager(pool_dir=tmpdir)
        manager.promote_to_global("connect_four", "good_tool")
        assert os.path.exists(os.path.join(global_dir, "good_tool.py"))
