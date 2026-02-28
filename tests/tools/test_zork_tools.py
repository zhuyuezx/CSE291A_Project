import importlib.util, pytest, shutil


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_room_exit_filter_meta():
    mod = _load_tool("tool_pool/zork/room_exit_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_item_evaluator_meta():
    mod = _load_tool("tool_pool/zork/item_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"


def test_room_exit_filter_loads():
    """Tool file must be syntactically valid Python."""
    mod = _load_tool("tool_pool/zork/room_exit_filter.py")
    assert hasattr(mod, "run")
    assert hasattr(mod, "__TOOL_META__")


def test_item_evaluator_loads():
    """Tool file must be syntactically valid Python."""
    mod = _load_tool("tool_pool/zork/item_evaluator.py")
    assert hasattr(mod, "run")
    assert hasattr(mod, "__TOOL_META__")
