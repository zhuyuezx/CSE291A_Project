"""
Heuristic code loader for the MCTS optimization loop.

Handles:
- Extracting Python code from LLM responses (markdown code blocks)
- Safely compiling and executing the extracted code
- Resolving the target function from the exec namespace
- Sanity-testing the loaded function against a real game state
"""

from __future__ import annotations

import re
import logging
from typing import Any, Callable

from .game_interface import Game, GameState

logger = logging.getLogger(__name__)


class HeuristicLoadError(Exception):
    """Raised when heuristic code cannot be loaded or validated."""


class HeuristicLoader:
    """
    Extracts, compiles, and validates heuristic functions from LLM output.

    The loader provides a sandboxed namespace for ``exec()`` with only
    the types the heuristic is expected to use (GameState, math, etc.).

    Example::

        loader = HeuristicLoader(game_state_classes=[SlidingPuzzleState])
        fn, code = loader.load_from_response(llm_text, target_name="evaluation")
        # fn is a ready-to-use callable
    """

    def __init__(
        self,
        game_state_classes: list[type] | None = None,
        extra_imports: dict[str, Any] | None = None,
    ):
        """
        Args:
            game_state_classes: Concrete GameState subclasses to inject
                                into the exec namespace (so the LLM code
                                can reference them by name).
            extra_imports:      Additional name→object pairs to inject
                                (e.g. ``{"numpy": numpy}``).
        """
        self._state_classes = game_state_classes or []
        self._extra_imports = extra_imports or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_response(
        self,
        llm_text: str,
        target_name: str = "evaluation",
        test_state: GameState | None = None,
    ) -> tuple[Callable, str]:
        """
        End-to-end pipeline: extract → compile → resolve → validate.

        Args:
            llm_text:    Raw LLM response text (may contain markdown).
            target_name: Name of the function to find in the exec'd code.
            test_state:  If provided, do a sanity call to verify the
                         function runs without error.

        Returns:
            (function, source_code) tuple.

        Raises:
            HeuristicLoadError: If extraction, compilation, or
                                validation fails.
        """
        code = self.extract_code(llm_text)
        fn = self.compile_function(code, target_name)

        if test_state is not None:
            self.validate(fn, target_name, test_state)

        return fn, code

    # ------------------------------------------------------------------
    # Step 1: Extract Python code from markdown
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code(text: str) -> str:
        """
        Extract the first Python code block from markdown-formatted text.

        Tries ``python`` fenced blocks first, then generic fenced blocks,
        then falls back to the raw text.

        Returns:
            The extracted source code string.

        Raises:
            HeuristicLoadError: If no usable code is found.
        """
        # Try ```python ... ```
        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()

        # Try generic ``` ... ```
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()

        # Last resort: assume the whole text is code
        candidate = text.strip()
        if "def " in candidate:
            return candidate

        raise HeuristicLoadError(
            "No Python code block found in LLM response. "
            f"Response starts with: {text[:200]!r}"
        )

    # ------------------------------------------------------------------
    # Step 2: Compile the code in a sandboxed namespace
    # ------------------------------------------------------------------

    def compile_function(
        self, code: str, target_name: str = "evaluation"
    ) -> Callable:
        """
        Execute the code string and retrieve the target function.

        Args:
            code:        Python source code defining at least one function.
            target_name: The function name to look for.

        Returns:
            The resolved callable.

        Raises:
            HeuristicLoadError: On syntax errors or missing function.
        """
        namespace = self._build_namespace()

        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            raise HeuristicLoadError(
                f"Failed to compile heuristic code: {e}"
            ) from e

        fn = namespace.get(target_name)
        if fn is not None and callable(fn):
            return fn

        # Fallback: find any new callable that isn't a builtin/class
        skip = {"__builtins__", "GameState"} | {
            cls.__name__ for cls in self._state_classes
        }
        for name, obj in namespace.items():
            if callable(obj) and name not in skip and not name.startswith("_"):
                logger.info(
                    "Target '%s' not found; using '%s' instead",
                    target_name, name,
                )
                return obj

        raise HeuristicLoadError(
            f"No callable named '{target_name}' (or any other function) "
            f"found in the exec'd code."
        )

    # ------------------------------------------------------------------
    # Step 3: Validate the function
    # ------------------------------------------------------------------

    @staticmethod
    def validate(
        fn: Callable,
        target_name: str,
        test_state: GameState,
        perspective: int = 0,
    ) -> None:
        """
        Sanity-check the loaded function by calling it on a test state.

        Raises:
            HeuristicLoadError: If the call fails or returns bad types.
        """
        try:
            if target_name == "evaluation":
                val = fn(test_state, perspective)
                if val is not None and not isinstance(val, (int, float)):
                    raise HeuristicLoadError(
                        f"evaluation returned {type(val).__name__}, "
                        f"expected float | None"
                    )
            elif target_name == "rollout_policy":
                val = fn(test_state)
                if val not in test_state.legal_actions():
                    raise HeuristicLoadError(
                        f"rollout_policy returned {val!r}, "
                        f"not in legal_actions()"
                    )
            else:
                # Generic: just call it with the state
                fn(test_state)
        except HeuristicLoadError:
            raise
        except Exception as e:
            raise HeuristicLoadError(
                f"Sanity test of '{target_name}' failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_namespace(self) -> dict[str, Any]:
        """Build the exec namespace with allowed imports."""
        ns: dict[str, Any] = {
            "GameState": GameState,
            "math": __import__("math"),
            "random": __import__("random"),
        }
        for cls in self._state_classes:
            ns[cls.__name__] = cls
        ns.update(self._extra_imports)
        return ns
