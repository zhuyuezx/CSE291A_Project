# tests/tools/test_base.py
from src.tools.base import (
    ToolType,
    ToolMeta,
    validate_tool_meta,
    load_tool_from_file,
)
import tempfile
import os


def test_tool_type_enum():
    assert ToolType.STATE_EVALUATOR == "state_evaluator"
    assert ToolType.ACTION_FILTER == "action_filter"
    assert ToolType.ROLLOUT_POLICY == "rollout_policy"
    assert ToolType.SELECTION_PRIOR == "selection_prior"
    assert ToolType.REWARD_SHAPER == "reward_shaper"
    assert ToolType.MACRO_ACTION == "macro_action"


def test_validate_tool_meta_valid():
    meta = {
        "name": "test_tool",
        "type": "state_evaluator",
        "description": "A test tool",
    }
    result = validate_tool_meta(meta)
    assert result.name == "test_tool"
    assert result.type == ToolType.STATE_EVALUATOR


def test_validate_tool_meta_invalid_type():
    meta = {
        "name": "test_tool",
        "type": "invalid_type",
        "description": "A test tool",
    }
    try:
        validate_tool_meta(meta)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_validate_tool_meta_missing_field():
    meta = {"name": "test_tool"}
    try:
        validate_tool_meta(meta)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_load_tool_from_file():
    tool_code = '''
__TOOL_META__ = {
    "name": "dummy_eval",
    "type": "state_evaluator",
    "description": "Returns 0.5 for any state",
}

def run(state) -> float:
    return 0.5
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(tool_code)
        f.flush()
        try:
            meta, run_fn = load_tool_from_file(f.name)
            assert meta.name == "dummy_eval"
            assert meta.type == ToolType.STATE_EVALUATOR
            assert run_fn(None) == 0.5
        finally:
            os.unlink(f.name)


def test_load_tool_from_file_no_meta():
    tool_code = '''
def run(state):
    return 0.5
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(tool_code)
        f.flush()
        try:
            load_tool_from_file(f.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "__TOOL_META__" in str(e)
        finally:
            os.unlink(f.name)
