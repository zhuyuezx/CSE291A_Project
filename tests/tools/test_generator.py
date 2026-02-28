# tests/tools/test_generator.py
from unittest.mock import MagicMock, patch
from src.tools.generator import ToolGenerator


def test_generator_builds_analysis_prompt():
    """Test that the generator constructs proper prompts from traces."""
    mock_client = MagicMock()
    mock_client.generate.return_value = '{"name": "test_tool", "type": "state_evaluator", "description": "test", "pseudocode": "return 0"}'

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    spec = generator.analyze_traces(
        traces_text="Step 0: Action=4\n...",
        game_description="Tic Tac Toe, 3x3 grid",
        current_tools_desc="No tools loaded",
    )

    assert mock_client.generate.called
    assert spec is not None
    assert "name" in spec


def test_generator_generates_code():
    mock_client = MagicMock()
    mock_client.generate.return_value = '''__TOOL_META__ = {
    "name": "test_tool",
    "type": "state_evaluator",
    "description": "test",
}

def run(state) -> float:
    return 0.0
'''

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    spec = {
        "name": "test_tool",
        "type": "state_evaluator",
        "description": "test",
        "pseudocode": "return 0",
    }
    code = generator.generate_code(spec)
    assert "__TOOL_META__" in code
    assert "def run" in code


def test_generator_full_pipeline_mock():
    """Test the full generate_tool pipeline with mocked LLM."""
    mock_client = MagicMock()

    # First call: trace analysis
    mock_client.generate.side_effect = [
        '{"name": "simple_eval", "type": "state_evaluator", "description": "Returns 0", "pseudocode": "return 0"}',
        # Second call: code generation
        '''__TOOL_META__ = {
    "name": "simple_eval",
    "type": "state_evaluator",
    "description": "Returns 0",
}

def run(state) -> float:
    return 0.0
''',
    ]

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    result = generator.generate_tool(
        traces_text="Game trace here",
        game_description="Tic Tac Toe",
        current_tools_desc="None",
    )

    assert result is not None
    assert result.valid
    assert "simple_eval" in result.code
