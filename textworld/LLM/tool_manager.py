"""
Tool Manager: parse, validate, and install LLM-generated MCTS tools.

Handles the full lifecycle of an LLM response → working tool file:
  1. Parse the structured response (ACTION / FILE_NAME / FUNCTION_NAME / code)
  2. Validate the extracted code (syntax, signature, function presence)
  3. Write the tool to the appropriate MCTS_tools/<phase>/ directory
  4. Optionally verify the tool is loadable at runtime

Expected LLM response format::

    ACTION: modify
    FILE_NAME: improved_simulation.py
    FUNCTION_NAME: improved_simulation
    DESCRIPTION: Added heuristic box-distance reward shaping
    ```python
    def improved_simulation(state, perspective_player, max_depth=500):
        ...
    ```

Usage::

    from LLM.tool_manager import ToolManager

    mgr = ToolManager()
    parsed = mgr.parse_response(llm_text)        # -> dict
    valid  = mgr.validate(parsed)                 # -> dict with 'valid'
    path   = mgr.install(parsed, phase="simulation")  # -> Path
"""

from __future__ import annotations

import ast
import importlib.util
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Defaults ─────────────────────────────────────────────────────────
_TOOL_CREATION_DIR = Path(__file__).resolve().parent.parent
_MCTS_TOOLS_DIR = _TOOL_CREATION_DIR / "MCTS_tools"

VALID_PHASES = ("selection", "expansion", "simulation", "backpropagation", "hyperparams")
VALID_ACTIONS = ("create", "modify")

# LLM-generated hyperparams are written here only (never overwrite hyperparams.py)
HYPERPARAMS_GENERATED_FILENAME = "generated_hyperparams.py"

# Expected function signatures (param names) per phase
EXPECTED_SIGNATURES: dict[str, list[str]] = {
    "selection": ["root", "exploration_weight"],
    "expansion": ["node"],
    "simulation": ["state", "perspective_player", "max_depth"],
    "backpropagation": ["node", "reward"],
    "hyperparams": [],  # get_hyperparams() takes no args
}

# Optional aliases for specific phase/parameter-position pairs.
_PARAM_ALIASES: dict[str, dict[int, set[str]]] = {
    # Accept either "root" or "node" for selection's first argument.
    "selection": {0: {"root", "node"}},
}


# ── Response parsing ─────────────────────────────────────────────────

def parse_response(response: str) -> dict[str, Any]:
    """
    Parse a structured LLM response into its components.

    Returns dict with keys:
        action        — 'create' or 'modify'
        file_name     — e.g. 'improved_simulation.py'
        function_name — e.g. 'improved_simulation'
        description   — one-line summary
        code          — extracted Python source code
        raw_response  — full original text
        parse_errors  — list of issues found (empty = clean parse)
    """
    result: dict[str, Any] = {
        "action": None,
        "file_name": None,
        "function_name": None,
        "description": None,
        "code": None,
        "raw_response": response,
        "parse_errors": [],
    }

    # Extract header fields
    _extract_field(response, r"ACTION:\s*(\S+)", "action", result)
    _extract_field(response, r"FILE_NAME:\s*(\S+)", "file_name", result)
    _extract_field(response, r"FUNCTION_NAME:\s*(\S+)", "function_name", result)

    # Description is optional — present but not required for validation
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", response)
    if desc_match:
        result["description"] = desc_match.group(1).strip().strip("*").strip()
    else:
        result["description"] = ""

    # Extract code block
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        result["code"] = code_match.group(1).strip()
    else:
        result["parse_errors"].append("No ```python code block found.")

    return result


def _extract_field(
    text: str, pattern: str, field: str, result: dict[str, Any]
) -> None:
    """Extract a single field from text via regex."""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        # Strip surrounding whitespace and any Markdown bold/italic markers (**)
        value = match.group(1).strip().strip("*").strip()
        result[field] = value.lower() if field == "action" else value
    else:
        result["parse_errors"].append(f"Missing {field.upper()} field.")


# ── Validation ───────────────────────────────────────────────────────

def validate(parsed: dict[str, Any], phase: str | None = None) -> dict[str, Any]:
    """
    Validate a parsed LLM response.

    Checks:
      1. All required fields present (no parse_errors)
      2. ACTION is 'create' or 'modify'
      3. FILE_NAME ends in .py, valid chars
      4. Code parses without SyntaxError
      5. Code defines the declared FUNCTION_NAME
      6. If phase given, function signature matches expected params

    Returns dict:
        valid   — bool
        errors  — list of error strings
    """
    errors: list[str] = list(parsed.get("parse_errors", []))

    # 1. Check action
    action = parsed.get("action")
    if action and action not in VALID_ACTIONS:
        errors.append(f"ACTION '{action}' invalid. Must be one of {VALID_ACTIONS}.")

    # 2. Check file name
    fname = parsed.get("file_name")
    if fname:
        if not fname.endswith(".py"):
            errors.append(f"FILE_NAME '{fname}' must end in .py.")
        stem = fname.removesuffix(".py")
        if not re.match(r"^[a-z0-9_]+$", stem):
            errors.append(f"FILE_NAME stem '{stem}' has invalid chars (use [a-z0-9_]).")
    
    # 3. Check code
    code = parsed.get("code")
    func_name = parsed.get("function_name")
    
    if not code:
        errors.append("No code to validate.")
        return {"valid": False, "errors": errors}

    # 4. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        errors.append(f"SyntaxError in code: {e}")
        return {"valid": False, "errors": errors}

    # 5. Function presence
    defined_funcs = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not defined_funcs:
        errors.append("Code defines no functions.")
    elif func_name and func_name not in defined_funcs:
        errors.append(
            f"FUNCTION_NAME '{func_name}' not found. "
            f"Defined functions: {defined_funcs}"
        )

    # 6. Signature check (if phase is known)
    if phase and func_name and func_name in defined_funcs:
        expected_params = EXPECTED_SIGNATURES.get(phase)
        if expected_params:
            _check_signature(tree, func_name, expected_params, errors, phase=phase)

    return {"valid": len(errors) == 0, "errors": errors}


def _check_signature(
    tree: ast.Module,
    func_name: str,
    expected_params: list[str],
    errors: list[str],
    phase: str | None = None,
) -> None:
    """Verify function has the expected parameter names."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            actual_params = [arg.arg for arg in node.args.args]
            # Check that all expected params appear (order matters for first N)
            for i, expected in enumerate(expected_params):
                if i >= len(actual_params):
                    errors.append(
                        f"Function '{func_name}' missing param '{expected}'. "
                        f"Expected params: {expected_params}, got: {actual_params}"
                    )
                    break
                actual = actual_params[i]
                allowed = {expected}
                if phase in _PARAM_ALIASES and i in _PARAM_ALIASES[phase]:
                    allowed = set(_PARAM_ALIASES[phase][i])
                if actual not in allowed:
                    errors.append(
                        f"Function '{func_name}' param {i} is '{actual}', "
                        f"expected one of {sorted(allowed)}. Full expected: {expected_params}"
                    )
                    break
            return


# ── Tool Manager class ───────────────────────────────────────────────

class ToolManager:
    """
    Manage LLM-generated MCTS tool files.

    Parameters
    ----------
    tools_dir : str | Path, optional
        Root directory for MCTS tools. Defaults to MCTS_tools/.
    """

    def __init__(self, tools_dir: str | Path | None = None):
        self.tools_dir = Path(tools_dir) if tools_dir else _MCTS_TOOLS_DIR

    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse an LLM response. Delegates to module-level parse_response."""
        return parse_response(response)

    def validate(
        self, parsed: dict[str, Any], phase: str | None = None
    ) -> dict[str, Any]:
        """Validate a parsed response. Delegates to module-level validate."""
        return validate(parsed, phase)

    def install(
        self,
        parsed: dict[str, Any],
        phase: str,
        overwrite: bool = False,
    ) -> Path:
        """
        Write the validated tool code to MCTS_tools/<phase>/<filename>.

        Parameters
        ----------
        parsed : dict
            Parsed and validated response from parse_response().
        phase : str
            MCTS phase directory to install into.
        overwrite : bool
            If False (default), refuse to overwrite existing files.

        Returns
        -------
        Path to the written file.

        Raises
        ------
        ValueError : if validation fails or phase is invalid.
        FileExistsError : if file exists and overwrite=False.
        """
        if phase not in VALID_PHASES:
            raise ValueError(f"Invalid phase '{phase}'. Must be one of {VALID_PHASES}.")

        # Validate before install
        validation = self.validate(parsed, phase=phase)
        if not validation["valid"]:
            raise ValueError(
                f"Validation failed:\n" +
                "\n".join(f"  - {e}" for e in validation["errors"])
            )

        code = parsed["code"]
        # Never write to hyperparams.py; use generated_hyperparams.py so user file is untouched
        if phase == "hyperparams":
            fname = HYPERPARAMS_GENERATED_FILENAME
        else:
            fname = parsed["file_name"]
        phase_dir = self.tools_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        target_path = phase_dir / fname
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Tool file already exists: {target_path}. "
                f"Use overwrite=True to replace."
            )

        # Add header comment
        desc = parsed.get("description", "LLM-generated tool")
        header = (
            f'"""\n'
            f"LLM-generated MCTS tool: {phase}\n"
            f"Description: {desc}\n"
            f"Generated:   {datetime.now().isoformat()}\n"
            f'"""\n\n'
        )
        target_path.write_text(header + code + "\n", encoding="utf-8")
        return target_path

    def verify_loadable(self, filepath: str | Path, function_name: str) -> dict[str, Any]:
        """
        Try to import the file and confirm the function is callable.

        Returns dict with 'loadable' bool and 'error' string.
        """
        filepath = Path(filepath)
        try:
            spec = importlib.util.spec_from_file_location(
                filepath.stem, str(filepath)
            )
            if spec is None or spec.loader is None:
                return {"loadable": False, "error": f"Cannot create module spec for {filepath}"}
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, function_name, None)
            if fn is None:
                return {"loadable": False, "error": f"Function '{function_name}' not found in module."}
            if not callable(fn):
                return {"loadable": False, "error": f"'{function_name}' is not callable."}
            return {"loadable": True, "error": None}
        except Exception as e:
            return {"loadable": False, "error": str(e)}

    def list_tools(self, phase: str) -> list[Path]:
        """List all tool files in a given phase directory."""
        phase_dir = self.tools_dir / phase
        if not phase_dir.exists():
            return []
        return sorted(phase_dir.glob("*.py"))
