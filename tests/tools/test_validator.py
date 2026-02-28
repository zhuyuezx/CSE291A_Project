# tests/tools/test_validator.py
import tempfile
import os
import pyspiel
from src.tools.validator import ToolValidator


def test_validate_valid_tool():
    code = '''
__TOOL_META__ = {
    "name": "test_eval",
    "type": "state_evaluator",
    "description": "Test evaluator",
}

def run(state) -> float:
    return 0.0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=10)
    result = validator.validate_code(code)
    assert result.valid
    assert result.error is None


def test_validate_missing_meta():
    code = '''
def run(state) -> float:
    return 0.0
'''
    validator = ToolValidator(game_name="tic_tac_toe")
    result = validator.validate_code(code)
    assert not result.valid
    assert "__TOOL_META__" in result.error


def test_validate_runtime_crash():
    code = '''
__TOOL_META__ = {
    "name": "crasher",
    "type": "state_evaluator",
    "description": "Crashes on purpose",
}

def run(state) -> float:
    return 1 / 0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert not result.valid
    assert "runtime" in result.error.lower() or "error" in result.error.lower()


def test_validate_bad_return_range():
    code = '''
__TOOL_META__ = {
    "name": "out_of_range",
    "type": "state_evaluator",
    "description": "Returns values outside [-1, 1]",
}

def run(state) -> float:
    return 999.0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert not result.valid


def test_validate_action_filter():
    code = '''
__TOOL_META__ = {
    "name": "pass_through",
    "type": "action_filter",
    "description": "Returns all actions",
}

def run(state, legal_actions):
    return legal_actions
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert result.valid
