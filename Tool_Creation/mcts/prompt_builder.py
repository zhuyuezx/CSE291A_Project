"""
Prompt templates for the LLM optimization loop.

Each game can register a PromptTemplate that knows how to describe
the game's rules, state API, and optimization constraints to the LLM.

The PromptBuilder combines a template with live engine data (traces,
heuristic source, stats) to produce the final prompt string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .mcts_engine import MCTSEngine


# =====================================================================
# Abstract template
# =====================================================================

class PromptTemplate(ABC):
    """
    Game-specific prompt template.

    Subclass this for each game to describe its rules, state API,
    and any game-specific optimization guidance.
    """

    @abstractmethod
    def game_description(self) -> str:
        """Return a multi-line description of the game rules and state API."""
        ...

    @abstractmethod
    def optimization_guidance(self) -> str:
        """Return game-specific tips for the LLM (what works, what doesn't)."""
        ...

    def target_heuristic(self) -> str:
        """Which heuristic slot to optimise. Default: 'evaluation'."""
        return "evaluation"

    def function_signature(self) -> str:
        """The expected function signature for the target heuristic."""
        target = self.target_heuristic()
        if target == "evaluation":
            return "def evaluation(state: GameState, perspective_player: int) -> float | None:"
        if target == "rollout_policy":
            return "def rollout_policy(state: GameState) -> Any:"
        return f"def {target}(...):"


# =====================================================================
# Concrete templates
# =====================================================================

class SlidingPuzzlePrompt(PromptTemplate):
    """Prompt template for the Sliding Puzzle (8-puzzle / 15-puzzle)."""

    def game_description(self) -> str:
        return """\
## Game: Sliding Puzzle (8-puzzle)
- 3x3 grid with tiles 1-8 and one blank space
- Actions: slide the blank UP(0), DOWN(1), LEFT(2), RIGHT(3)
- Goal: arrange tiles as [1,2,3,4,5,6,7,8,_] (blank at bottom-right)
- The puzzle state has helper methods:
  - state.misplaced_tiles() -> int (tiles not in goal position)
  - state.manhattan_distance() -> int (sum of Manhattan distances to goals)
  - state.board -> list[int] (flat board, 0=blank)
  - state.size -> int (3 for 3x3)
  - state.goal -> list[int] (the goal configuration)
  - state.blank_pos -> int (index of blank)
  - state.legal_actions() -> list[int]"""

    def optimization_guidance(self) -> str:
        return """\
CRITICAL DESIGN RULES:
1. Return a float in [0, 1] where 1.0 = solved and 0.0 = far from solution.
2. Values must have STRONG contrast: near-solved states ~0.9+, scrambled states ~0.05-0.1.
   If values are compressed (e.g. all in 0.5-0.9), MCTS cannot differentiate states.
3. Manhattan distance divided by (1 + md) works well. Avoid (max_md - md)/max_md shape.
4. Keep it simple -- complex functions often perform worse than simple ones."""


class SokobanPrompt(PromptTemplate):
    """Prompt template for Sokoban."""

    def game_description(self) -> str:
        return """\
## Game: Sokoban
- Grid-based puzzle: push boxes ($) onto target positions (.)
- Actions: move player UP(0), DOWN(1), LEFT(2), RIGHT(3); pushes box if adjacent
- Walls (#) block movement; boxes cannot be pulled, only pushed
- The puzzle state has helper methods:
  - state.boxes_on_targets() -> int (boxes currently on target positions)
  - state.total_box_distance() -> int (sum of Manhattan distances from each box to nearest target)
  - state.num_targets -> int (total number of targets)
  - state.player -> tuple[int, int] (row, col)
  - state.boxes -> set of tuple[int, int]
  - state.targets -> frozenset of tuple[int, int]
  - state.walls -> frozenset of tuple[int, int]
  - state.legal_actions() -> list[int]"""

    def optimization_guidance(self) -> str:
        return """\
CRITICAL DESIGN RULES:
1. Return a float in [0, 1] where 1.0 = solved and 0.0 = far from solution.
2. Consider both box-to-target distance AND deadlock detection.
3. Corner deadlocks (box pushed into corner with no target) should return 0.0.
4. boxes_on_targets() / num_targets gives a rough progress measure.
5. Keep it simple -- avoid expensive computations inside the evaluation."""


class ConnectFourPrompt(PromptTemplate):
    """Prompt template for Connect Four."""

    def game_description(self) -> str:
        return """\
## Game: Connect Four
- 6x7 board, two players drop discs into columns
- Win by connecting 4 discs horizontally, vertically, or diagonally
- Actions: column index 0-6 (must have space)
- The state has:
  - state.board -> 2D list (0=empty, 1=P1, 2=P2)
  - state.legal_actions() -> list[int] (non-full columns)
  - state.current_player() -> int (0 or 1)"""

    def optimization_guidance(self) -> str:
        return """\
CRITICAL DESIGN RULES:
1. Return a float in [-1, 1] where 1.0 = winning for perspective_player.
2. Consider threats (3-in-a-row with open end), centre control, connectivity.
3. Avoid expensive deep searches -- keep the evaluation O(board_size).
4. Return None to let the rollout continue if you're unsure."""


# =====================================================================
# Registry: game name → template
# =====================================================================

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "Sliding Puzzle 3x3": SlidingPuzzlePrompt(),
    "Sokoban": SokobanPrompt(),
    "Connect Four": ConnectFourPrompt(),
}


def register_prompt_template(game_name: str, template: PromptTemplate) -> None:
    """Register a prompt template for a game name."""
    PROMPT_TEMPLATES[game_name] = template


def get_prompt_template(game_name: str) -> PromptTemplate:
    """
    Look up the prompt template for a game name.

    Falls back to a partial match (prefix) if exact match fails.
    """
    if game_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[game_name]
    # Prefix match
    for key, tmpl in PROMPT_TEMPLATES.items():
        if game_name.startswith(key) or key.startswith(game_name):
            return tmpl
    raise KeyError(
        f"No prompt template registered for '{game_name}'. "
        f"Available: {list(PROMPT_TEMPLATES.keys())}"
    )


# =====================================================================
# Prompt builder — combines template + live data
# =====================================================================

class PromptBuilder:
    """
    Assembles the final LLM prompt from a template and live engine data.

    Example::

        builder = PromptBuilder(template=SlidingPuzzlePrompt())
        prompt = builder.build(engine, stats, prev_code="...")
    """

    def __init__(self, template: PromptTemplate | None = None):
        self.template = template

    def build(
        self,
        engine: MCTSEngine,
        stats: dict,
        *,
        prev_code: str | None = None,
        max_trace_games: int = 5,
        max_trace_chars: int = 3000,
    ) -> str:
        """
        Build the full prompt string.

        Args:
            engine:           MCTS engine (used for traces + heuristic source).
            stats:            Stats dict from engine.play_many().
            prev_code:        Source code from the previous LLM attempt
                              (to give it feedback for iteration).
            max_trace_games:  How many game traces to include.
            max_trace_chars:  Character limit for the trace section.

        Returns:
            The assembled prompt string.
        """
        template = self._resolve_template(engine)
        traces = engine.logger.format_for_llm(max_games=max_trace_games)
        sources = engine.get_heuristic_source()
        target = template.target_heuristic()
        win_pct = stats.get("win_rate", 0) * 100

        prev_section = ""
        if prev_code:
            prev_section = (
                f"\n## Previous Attempt (solve rate {win_pct:.0f}%):\n"
                f"```python\n{prev_code}\n```\n"
                f"This needs improvement. Try a different or better approach.\n"
            )

        prompt = f"""\
You are an expert game AI engineer optimizing MCTS heuristics for a puzzle/game.

{template.game_description()}

## Current Performance: {win_pct:.0f}% solve rate

### Current {target} function:
{sources.get(target, '# (not set)')}
{prev_section}
## Gameplay Traces (unsolved/lost games -- MCTS couldn't handle these):
{traces[:max_trace_chars]}

## Task
Write an improved `{target}` function.

{template.optimization_guidance()}

Function signature:
  {template.function_signature()}

Put your code in a ```python code block. Write ONLY the function.
"""
        return prompt

    def _resolve_template(self, engine: MCTSEngine) -> PromptTemplate:
        """Get the template, auto-detecting from the engine's game if needed."""
        if self.template is not None:
            return self.template
        game_name = engine.game.name()
        return get_prompt_template(game_name)
