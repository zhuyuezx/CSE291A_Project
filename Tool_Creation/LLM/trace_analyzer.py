"""
Trace Analyzer — LLM-based interpretation of MCTS gameplay traces.

Loads recent trace JSON files produced by MCTSEngine (logging=True),
builds per-trace analysis prompts, queries the LLM in parallel via
``query_batch``, and returns a combined analysis string that is injected
into the main optimisation pipeline as ``additional_context``.

Flow in the runner
------------------
1. ``engine.play_game()`` writes a trace to ``mcts/records/<game>_<ts>.json``
2. ``TraceAnalyzer.analyze_single(trace_path)`` sends the trace to the LLM
   and returns a concise natural-language description of what happened and why.
3. The analysis text is prepended to ``history`` and forwarded as
   ``additional_context`` to ``Optimizer.run()``, giving the code-generation
   step richer context about the agent's actual behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm_querier import LLMQuerier


_LLM_DIR = Path(__file__).resolve().parent
_GAME_INFOS_DIR = _LLM_DIR / "game_infos"
_RECORDS_DIR = _LLM_DIR.parent / "mcts" / "records"


class TraceAnalyzer:
    """
    Analyse gameplay traces via LLM and produce text descriptions.

    Parameters
    ----------
    game : str
        Game name (must match a ``<game>.txt`` file in ``LLM/game_infos/``).
    querier : LLMQuerier, optional
        Re-use an existing querier.  A new one is created when ``None``.
    records_dir : str | Path | None
        Override the default records directory (``mcts/records/``).
    max_moves_per_trace : int | None
        Cap the number of moves displayed per trace in the prompt.
        ``None`` means no cap.
    """

    def __init__(
        self,
        game: str,
        querier: LLMQuerier | None = None,
        records_dir: str | Path | None = None,
        max_moves_per_trace: int | None = 40,
    ):
        self.game = game.lower()
        self._querier = querier
        self.records_dir = Path(records_dir) if records_dir else _RECORDS_DIR
        self.max_moves_per_trace = max_moves_per_trace

    # ------------------------------------------------------------------
    # Lazy querier
    # ------------------------------------------------------------------

    @property
    def querier(self) -> LLMQuerier:
        if self._querier is None:
            self._querier = LLMQuerier(session_tag=f"{self.game}_trace_analysis")
        return self._querier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_single(self, record_file: str | Path) -> str:
        """
        Analyse a single trace file and return a description string.

        Parameters
        ----------
        record_file : str | Path
            Path to a trace JSON file (absolute or relative to
            ``records_dir``).

        Returns
        -------
        str
            LLM-generated description of the trace.
        """
        path = Path(record_file)
        if not path.is_absolute():
            path = self.records_dir / path

        traces = self._load_traces([path])
        if not traces:
            return "(Trace file could not be loaded.)"

        game_rules = self._load_game_info()
        prompt = self._build_single_trace_prompt(game_rules, traces[0], index=1)

        result = self.querier.query(prompt, step_name="trace_analysis")
        if result.get("status") == "error":
            return f"(Trace analysis failed: {result.get('error', 'unknown error')})"
        return result.get("response", "(No response from LLM.)")

    def analyze(
        self,
        record_files: list[str | Path],
        concurrency: int | None = None,
    ) -> str:
        """
        Analyse multiple traces in parallel and return combined text.

        Parameters
        ----------
        record_files : list[str | Path]
            Paths to trace JSON files.
        concurrency : int, optional
            Max parallel LLM requests.

        Returns
        -------
        str
            Concatenated per-trace analysis, suitable for
            ``additional_context`` in ``Optimizer.run()``.
        """
        traces = self._load_traces(record_files)
        if not traces:
            return "(No traces provided for analysis.)"

        game_rules = self._load_game_info()
        prompts = [
            self._build_single_trace_prompt(game_rules, tr, index=i + 1)
            for i, tr in enumerate(traces)
        ]

        results = self.querier.query_batch(prompts, concurrency=concurrency)

        parts: list[str] = []
        for i, res in enumerate(results):
            header = f"=== Trace {i + 1} analysis ==="
            body = (
                res.get("response", "(No response.)")
                if res.get("status") != "error"
                else f"(Error: {res.get('error', 'unknown')})"
            )
            parts.append(f"{header}\n{body}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_game_info(self) -> str:
        path = _GAME_INFOS_DIR / f"{self.game}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return f"(No game info file found for '{self.game}'.)"

    def _load_traces(self, record_files: list[str | Path]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for rf in record_files:
            path = Path(rf)
            if not path.is_absolute():
                path = self.records_dir / path
            try:
                with open(path, encoding="utf-8") as f:
                    traces.append(json.load(f))
            except Exception:
                pass
        return traces

    def _build_single_trace_prompt(
        self, game_rules: str, trace: dict[str, Any], index: int
    ) -> str:
        sep = "-" * 60
        meta = trace.get("metadata", {})
        outcome = trace.get("outcome", {})
        solved = outcome.get("solved", False)
        steps = outcome.get("steps", "?")
        final_state = outcome.get("final_state", "")

        sections = [
            f"{sep}\nGAME RULES\n{sep}\n{game_rules}",
            f"{sep}\nGAMEPLAY TRACE #{index}\n{sep}\n"
            + self._format_trace(trace),
            (
                f"{sep}\n"
                f"TASK — TRACE INTERPRETATION (no code)\n"
                f"{sep}\n"
                f"Analyse the gameplay trace above for '{self.game}'.\n\n"
                f"Produce a concise description covering:\n\n"
                f"1. OUTCOME SUMMARY\n"
                f"   Result: {'SOLVED' if solved else 'UNSOLVED'} in {steps} steps.\n"
                f"   What was the final board state?\n\n"
                f"2. KEY MOMENTS\n"
                f"   Identify 2-3 critical moves where the agent made a\n"
                f"   particularly good or bad decision. Cite move numbers.\n\n"
                f"3. ROOT CAUSE\n"
                f"   What drove the outcome? Was it good positioning,\n"
                f"   poor exploration, a missed opportunity, or a strong\n"
                f"   tactical sequence?\n\n"
                f"4. SIMULATION HEURISTIC INSIGHT\n"
                f"   What change to the simulation (rollout) heuristic\n"
                f"   could help MCTS find better moves faster?\n\n"
                f"Keep the analysis under 350 words. Be specific and cite\n"
                f"move numbers. Do NOT write code."
            ),
        ]
        return "\n\n".join(sections)

    def _format_trace(self, trace: dict[str, Any]) -> str:
        meta = trace.get("metadata", {})
        outcome = trace.get("outcome", {})
        moves: list[dict[str, Any]] = trace.get("moves", [])

        lines: list[str] = [
            f"Game     : {meta.get('game', '?')}",
            f"Solved   : {outcome.get('solved', '?')}",
            f"Steps    : {outcome.get('steps', '?')}",
            f"Returns  : {outcome.get('returns', '?')}",
            "",
        ]

        display_moves = moves
        if self.max_moves_per_trace and len(moves) > self.max_moves_per_trace:
            display_moves = moves[: self.max_moves_per_trace]

        for move in display_moves:
            lines.append(self._format_move(move))

        if self.max_moves_per_trace and len(moves) > self.max_moves_per_trace:
            lines.append(
                f"... ({len(moves) - self.max_moves_per_trace} more moves omitted)"
            )

        final = outcome.get("final_state", "")
        if final:
            lines += ["", "Final state:", final]

        return "\n".join(lines)

    @staticmethod
    def _format_move(move: dict[str, Any]) -> str:
        move_num = move.get("move_number", "?")
        action = move.get("action_chosen", "?")
        visits = move.get("root_visits", "?")
        state = move.get("state_before", "")
        children = move.get("children_stats", {})

        parts = [f"  Move {move_num}: action={action}  root_visits={visits}"]

        if state:
            for line in str(state).split("\n")[:6]:
                parts.append(f"    {line}")

        if children:
            # children_stats is {action_key: {visits, value, avg_value}}
            if isinstance(children, dict):
                child_list = [
                    {
                        "action": k,
                        "visits": v.get("visits", 0),
                        "q": v.get("avg_value", v.get("value", 0.0)),
                    }
                    for k, v in children.items()
                ]
            else:
                child_list = [
                    {
                        "action": c.get("action", "?"),
                        "visits": c.get("visits", 0),
                        "q": c.get("q_value", c.get("avg_value", 0.0)),
                    }
                    for c in children
                ]
            top = sorted(child_list, key=lambda c: c["visits"], reverse=True)[:5]
            child_strs = [
                f"a={c['action']} v={c['visits']} "
                f"q={c['q']:.3f}" if isinstance(c["q"], (int, float))
                else f"a={c['action']} v={c['visits']}"
                for c in top
            ]
            parts.append(f"    top children: [{', '.join(child_strs)}]")

        return "\n".join(parts)
