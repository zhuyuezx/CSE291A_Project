# src/tools/validator.py
from __future__ import annotations

import ast
import importlib.util
import random
import tempfile
import os
import time
from dataclasses import dataclass

import pyspiel

from src.tools.base import ToolType, validate_tool_meta


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


class ToolValidator:
    def __init__(
        self,
        game_name: str = "tic_tac_toe",
        num_test_states: int = 50,
        timeout_ms: float = 100.0,
    ):
        self.game_name = game_name
        self.num_test_states = num_test_states
        self.timeout_ms = timeout_ms

    def validate_code(self, code: str) -> ValidationResult:
        # Step 1: Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, error=f"Syntax error: {e}")

        # Step 2: Check __TOOL_META__ exists in AST
        tree = ast.parse(code)
        has_meta = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "__TOOL_META__"
                for t in node.targets
            )
            for node in ast.walk(tree)
        )
        if not has_meta:
            return ValidationResult(
                valid=False, error="Missing __TOOL_META__ dict"
            )

        # Step 3: Load module
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmppath = f.name

        try:
            spec = importlib.util.spec_from_file_location("test_tool", tmppath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return ValidationResult(valid=False, error=f"Import error: {e}")
        finally:
            os.unlink(tmppath)

        # Step 4: Validate meta
        try:
            meta = validate_tool_meta(module.__TOOL_META__)
        except ValueError as e:
            return ValidationResult(valid=False, error=str(e))

        if not hasattr(module, "run"):
            return ValidationResult(valid=False, error="Missing run() function")

        # Step 5: Runtime check on random game states
        game = pyspiel.load_game(self.game_name)
        test_states = self._generate_random_states(game)

        for state in test_states:
            if state.is_terminal() or state.current_player() < 0:
                continue
            try:
                if meta.type == ToolType.STATE_EVALUATOR:
                    result = module.run(state)
                    if not isinstance(result, (int, float)):
                        return ValidationResult(
                            valid=False,
                            error=f"state_evaluator returned {type(result)}, expected float",
                        )
                    if not (-1.0 <= float(result) <= 1.0):
                        return ValidationResult(
                            valid=False,
                            error=f"state_evaluator returned {result}, must be in [-1, 1]",
                        )
                elif meta.type == ToolType.ACTION_FILTER:
                    legal = state.legal_actions()
                    result = module.run(state, legal)
                    if not isinstance(result, list):
                        return ValidationResult(
                            valid=False,
                            error=f"action_filter returned {type(result)}, expected list",
                        )
                elif meta.type == ToolType.ROLLOUT_POLICY:
                    legal = state.legal_actions()
                    result = module.run(state, legal)
                    if result not in legal:
                        return ValidationResult(
                            valid=False,
                            error=f"rollout_policy returned {result}, not in legal actions",
                        )
                elif meta.type == ToolType.REWARD_SHAPER:
                    result = module.run(state, 0.5)
                    if not isinstance(result, (int, float)):
                        return ValidationResult(
                            valid=False,
                            error=f"reward_shaper returned {type(result)}, expected float",
                        )
            except Exception as e:
                return ValidationResult(
                    valid=False, error=f"Runtime error on test state: {e}"
                )

        return ValidationResult(valid=True)

    def _generate_random_states(self, game) -> list:
        states = []
        for _ in range(self.num_test_states):
            state = game.new_initial_state()
            # Play random number of moves
            depth = random.randint(0, 20)
            for _ in range(depth):
                if state.is_terminal():
                    break
                if state.current_player() < 0:
                    break
                action = random.choice(state.legal_actions())
                state.apply_action(action)
            states.append(state)
        return states
