"""
LLM Prompt Builder for MCTS heuristic improvement.

Assembles structured prompts that combine:
  1. Game rules / description  (from game_infos/<game>.txt)
  2. Current MCTS tool source  (the heuristic code being improved)
  3. Gameplay trace records     (from mcts/records/*.json)

The builder is parametrized by game name and target MCTS phase, and
can accept one or more record file paths. The output is a single
prompt string that can be saved to disk or sent to an LLM.

Usage::

    builder = PromptBuilder(game="sokoban", target_phase="simulation")
    prompt  = builder.build(record_files=["mcts/records/sokoban_xxx.json"])
    builder.save(prompt, "LLM/drafts/prompt_sokoban.txt")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Default directories ──────────────────────────────────────────────
_LLM_DIR = Path(__file__).resolve().parent
_GAME_INFOS_DIR = _LLM_DIR / "game_infos"
_RECORDS_DIR = _LLM_DIR.parent / "mcts" / "records"
_DRAFTS_DIR = _LLM_DIR / "drafts"


class PromptBuilder:
    """
    Build structured LLM prompts for MCTS heuristic improvement.

    Parameters
    ----------
    game : str
        Name of the game (must match a <game>.txt in game_infos/).
    target_phase : str
        MCTS phase whose tool we want the LLM to improve.
        One of: selection, expansion, simulation, backpropagation.
    game_infos_dir : str | Path | None
        Override default game_infos/ directory.
    records_dir : str | Path | None
        Override default mcts/records/ directory.
    """

    VALID_PHASES = ("selection", "expansion", "simulation", "backpropagation")

    def __init__(
        self,
        game: str,
        target_phase: str = "simulation",
        game_infos_dir: str | Path | None = None,
        records_dir: str | Path | None = None,
    ):
        if target_phase not in self.VALID_PHASES:
            raise ValueError(
                f"Invalid target_phase '{target_phase}'. "
                f"Must be one of {list(self.VALID_PHASES)}"
            )
        self.game = game.lower()
        self.target_phase = target_phase
        self.game_infos_dir = Path(game_infos_dir) if game_infos_dir else _GAME_INFOS_DIR
        self.records_dir = Path(records_dir) if records_dir else _RECORDS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        record_files: list[str | Path] | None = None,
        tool_source: str | None = None,
        all_tool_sources: dict[str, str] | None = None,
        max_moves_per_trace: int | None = None,
    ) -> str:
        """
        Assemble the full prompt string.

        Parameters
        ----------
        record_files : list of paths, optional
            Specific record JSON files to include. Paths can be absolute
            or relative to ``self.records_dir``.
        tool_source : str, optional
            Source code of the current heuristic tool (the target phase).
            If not provided, a placeholder note is inserted.
        all_tool_sources : dict[str, str], optional
            Source code of ALL 4 MCTS phase tools, keyed by phase name.
            Typically obtained via ``engine.get_tool_source()``.
            If provided, a full MCTS TOOL FUNCTIONS section is included.
        max_moves_per_trace : int, optional
            Limit the number of moves shown per trace to keep the prompt
            concise. None means include all moves.

        Returns
        -------
        str : the assembled prompt.
        """
        sections: list[str] = []

        # ── Section 1: System instruction ──
        sections.append(self._build_system_section())

        # ── Section 2: Game rules ──
        sections.append(self._build_game_rules_section())

        # ── Section 3: All MCTS tool function sources ──
        if all_tool_sources:
            sections.append(self._build_all_tools_section(all_tool_sources))

        # ── Section 4: Target heuristic code (highlighted) ──
        sections.append(self._build_tool_source_section(tool_source))

        # ── Section 5: Gameplay traces ──
        traces = self._load_traces(record_files)
        sections.append(self._build_traces_section(traces, max_moves_per_trace))

        # ── Section 6: Task instruction ──
        sections.append(self._build_task_section())

        return "\n\n".join(sections)

    def save(self, prompt: str, filepath: str | Path | None = None) -> Path:
        """
        Save the prompt string to a text file.

        Parameters
        ----------
        filepath : str | Path, optional
            Output path. Defaults to LLM/drafts/<game>_<phase>_prompt.txt.

        Returns
        -------
        Path : the path the file was written to.
        """
        if filepath is None:
            out_dir = _DRAFTS_DIR
        else:
            out_dir = None

        if filepath is None:
            _DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = _DRAFTS_DIR / f"{self.game}_{self.target_phase}_prompt.txt"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        filepath.write_text(prompt, encoding="utf-8")
        return filepath

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_system_section(self) -> str:
        sep = "=" * 60
        return (
            f"{sep}\n"
            f"SYSTEM: MCTS Heuristic Improvement\n"
            f"{sep}\n"
            f"You are an expert game-playing AI researcher.\n"
            f"Your task is to improve a specific MCTS heuristic function\n"
            f"for the game '{self.game}' (phase: {self.target_phase}).\n\n"
            f"Analyze the game rules, the current heuristic code, and the\n"
            f"gameplay traces below. Then produce an improved version of\n"
            f"the heuristic function that addresses the weaknesses you\n"
            f"identified."
        )

    def _build_game_rules_section(self) -> str:
        rules = self._load_game_info()
        sep = "-" * 60
        return f"{sep}\nGAME RULES\n{sep}\n{rules}"

    def _build_all_tools_section(self, all_tool_sources: dict[str, str]) -> str:
        sep = "-" * 60
        parts = [f"{sep}\nMCTS TOOL FUNCTIONS (all 4 phases)\n{sep}"]
        for phase in self.VALID_PHASES:
            src = all_tool_sources.get(phase)
            if src:
                marker = " ◀ TARGET" if phase == self.target_phase else ""
                parts.append(
                    f"--- {phase}{marker} ---\n"
                    f"```python\n{src.rstrip()}\n```"
                )
            else:
                parts.append(f"--- {phase} ---\n(source not available)")
        return "\n\n".join(parts)

    def _build_tool_source_section(self, tool_source: str | None) -> str:
        sep = "-" * 60
        header = f"{sep}\nTARGET HEURISTIC TO IMPROVE ({self.target_phase})\n{sep}"
        if tool_source:
            return header + "\n```python\n" + tool_source.rstrip() + "\n```"
        return header + "\n(No heuristic code provided — using MCTS default.)"

    def _build_traces_section(
        self,
        traces: list[dict[str, Any]],
        max_moves: int | None,
    ) -> str:
        sep = "-" * 60
        header = f"{sep}\nGAMEPLAY TRACES\n{sep}"
        if not traces:
            return header + "\n(No gameplay traces provided.)"

        parts = [header]
        for i, trace in enumerate(traces, 1):
            parts.append(self._format_trace(trace, index=i, max_moves=max_moves))
        return "\n\n".join(parts)

    def _build_task_section(self) -> str:
        sep = "-" * 60
        return (
            f"{sep}\n"
            f"TASK\n"
            f"{sep}\n"
            f"Write an improved Python function for the '{self.target_phase}'\n"
            f"phase of MCTS. Your function must:\n\n"
            f"1. Have the SAME function signature as the current code above.\n"
            f"2. Be a standalone function (no class needed).\n"
            f"3. Only use the Python standard library (no external packages).\n"
            f"4. Address specific weaknesses you observed in the gameplay\n"
            f"   traces (e.g., deadlocks, wasted moves, poor exploration).\n\n"
            f"Return ONLY the Python function code, enclosed in a\n"
            f"```python ... ``` code block."
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_game_info(self) -> str:
        """Load the game description text file."""
        path = self.game_infos_dir / f"{self.game}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Game info file not found: {path}\n"
                f"Available: {[f.name for f in self.game_infos_dir.glob('*.txt')]}"
            )
        return path.read_text(encoding="utf-8")

    def _load_traces(
        self,
        record_files: list[str | Path] | None,
    ) -> list[dict[str, Any]]:
        """Load trace dicts from JSON record files."""
        if not record_files:
            return []

        traces: list[dict[str, Any]] = []
        for f in record_files:
            p = Path(f)
            # Try as absolute path first, then relative to records_dir
            if not p.is_absolute():
                p = self.records_dir / p
            if not p.exists():
                raise FileNotFoundError(f"Record file not found: {p}")
            with open(p, encoding="utf-8") as fh:
                traces.append(json.load(fh))
        return traces

    # ------------------------------------------------------------------
    # Trace formatting
    # ------------------------------------------------------------------

    def _format_trace(
        self,
        trace: dict[str, Any],
        index: int = 1,
        max_moves: int | None = None,
    ) -> str:
        """Format a single trace dict into a human-readable string."""
        meta = trace.get("metadata", {})
        outcome = trace.get("outcome", {})
        moves = trace.get("moves", [])

        parts: list[str] = []

        # Header
        parts.append(f"--- Trace #{index} ---")
        parts.append(f"Game:       {meta.get('game', 'unknown')}")
        parts.append(f"Timestamp:  {meta.get('timestamp', 'unknown')}")
        parts.append(f"Iterations: {meta.get('iterations', '?')}")
        parts.append(f"Solved:     {outcome.get('solved', '?')}")
        parts.append(f"Steps:      {outcome.get('steps', '?')}")
        parts.append(f"Returns:    {outcome.get('returns', '?')}")
        parts.append("")

        # Moves
        display_moves = moves[:max_moves] if max_moves else moves
        for move in display_moves:
            parts.append(self._format_move(move))

        if max_moves and len(moves) > max_moves:
            parts.append(f"  ... ({len(moves) - max_moves} more moves omitted)")

        # Final state
        final = outcome.get("final_state")
        if final:
            parts.append(f"\nFinal state:\n{final}")

        return "\n".join(parts)

    @staticmethod
    def _format_move(move: dict[str, Any]) -> str:
        """Format a single move dict into a compact string."""
        num = move.get("move_number", "?")
        action = move.get("action_chosen", "?")
        visits = move.get("root_visits", "?")
        children = move.get("children_stats", {})

        # Build compact children summary
        child_parts = []
        for act, stats in children.items():
            v = stats.get("visits", 0)
            avg = stats.get("avg_value", 0.0)
            child_parts.append(f"{act}(v={v}, avg={avg:.3f})")
        children_str = ", ".join(child_parts)

        state_before = move.get("state_before", "")
        # Show just the first line of state (the metrics line)
        state_summary = state_before.split("\n")[0] if state_before else ""

        return (
            f"  Move {num}: action={action}, total_visits={visits}\n"
            f"    State: {state_summary}\n"
            f"    Children: [{children_str}]"
        )
