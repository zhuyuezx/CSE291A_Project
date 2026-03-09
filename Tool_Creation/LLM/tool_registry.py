"""
Global Tool Registry — versioned history of MCTS heuristic tools.

Maintains a per-phase record of every tool that has been installed,
together with optional performance metrics.  The registry is
persisted as a single JSON file so that later runs (and the
aggregator / cluster / merge components) can reason over past tools.

Usage::

    from LLM.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        phase="simulation",
        path="MCTS_tools/simulation/improved_sim.py",
        function_name="default_simulation",
        description="Added Manhattan-distance reward shaping",
        iteration=3,
        metrics={"composite": 0.72, "solve_rate": 0.6},
    )
    history = registry.get_history("simulation", last_k=5)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


_LLM_DIR = Path(__file__).resolve().parent
_DEFAULT_REGISTRY_DIR = _LLM_DIR / "registry"
_DEFAULT_REGISTRY_FILE = _DEFAULT_REGISTRY_DIR / "tool_registry.json"


class ToolEntry:
    """Single entry in the tool history."""

    __slots__ = (
        "phase", "path", "function_name", "description",
        "iteration", "metrics", "timestamp", "source_snippet",
    )

    def __init__(
        self,
        phase: str,
        path: str,
        function_name: str,
        description: str = "",
        iteration: int = 0,
        metrics: dict[str, Any] | None = None,
        timestamp: str | None = None,
        source_snippet: str | None = None,
    ):
        self.phase = phase
        self.path = path
        self.function_name = function_name
        self.description = description
        self.iteration = iteration
        self.metrics = metrics or {}
        self.timestamp = timestamp or datetime.now().isoformat()
        self.source_snippet = source_snippet

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "path": self.path,
            "function_name": self.function_name,
            "description": self.description,
            "iteration": self.iteration,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "source_snippet": self.source_snippet,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolEntry":
        return cls(
            phase=d["phase"],
            path=d["path"],
            function_name=d.get("function_name", ""),
            description=d.get("description", ""),
            iteration=d.get("iteration", 0),
            metrics=d.get("metrics"),
            timestamp=d.get("timestamp"),
            source_snippet=d.get("source_snippet"),
        )


class ToolRegistry:
    """
    Persistent, per-phase history of installed MCTS tools.

    Parameters
    ----------
    registry_file : str | Path | None
        Path to the JSON persistence file.  Defaults to
        ``LLM/registry/tool_registry.json``.
    """

    def __init__(self, registry_file: str | Path | None = None):
        self._file = Path(registry_file) if registry_file else _DEFAULT_REGISTRY_FILE
        self._entries: list[ToolEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        phase: str,
        path: str | Path,
        function_name: str,
        description: str = "",
        iteration: int = 0,
        metrics: dict[str, Any] | None = None,
        source_snippet: str | None = None,
    ) -> ToolEntry:
        """Add a new tool entry and persist to disk."""
        entry = ToolEntry(
            phase=phase,
            path=str(path),
            function_name=function_name,
            description=description,
            iteration=iteration,
            metrics=metrics,
            source_snippet=source_snippet,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def get_history(
        self,
        phase: str,
        last_k: int | None = None,
    ) -> list[ToolEntry]:
        """Return tool entries for *phase*, newest-first."""
        entries = [e for e in self._entries if e.phase == phase]
        entries.sort(key=lambda e: e.iteration, reverse=True)
        if last_k is not None:
            entries = entries[:last_k]
        return entries

    def get_best(self, phase: str, metric_key: str = "composite") -> ToolEntry | None:
        """Return the entry with the highest *metric_key* for *phase*."""
        candidates = [
            e for e in self._entries
            if e.phase == phase and metric_key in e.metrics
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.metrics[metric_key])

    def get_all_phase_tools(self, phase: str) -> list[dict[str, Any]]:
        """
        Return metadata dicts for all tools in *phase* (for clustering).

        Each dict has keys: name, description, path, metrics, source_snippet.
        """
        return [
            {
                "name": e.function_name,
                "description": e.description,
                "path": e.path,
                "metrics": e.metrics,
                "source_snippet": e.source_snippet,
            }
            for e in self._entries if e.phase == phase
        ]

    def format_history_context(
        self,
        phase: str,
        last_k: int = 10,
    ) -> str:
        """
        Build a concise text summary of recent tool history for *phase*.

        Suitable for injection into ``additional_context``.
        """
        entries = self.get_history(phase, last_k=last_k)
        if not entries:
            return ""
        lines = [f"=== Tool History for '{phase}' (last {len(entries)}) ==="]
        for e in entries:
            m = e.metrics
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in m.items()) if m else "no metrics"
            lines.append(
                f"  Iter {e.iteration}: {e.description} [{metric_str}]"
            )
        best = self.get_best(phase)
        if best:
            lines.append(
                f"  >> BEST so far: iter {best.iteration} — "
                f"{best.description} (composite={best.metrics.get('composite', '?')})"
            )
        return "\n".join(lines)

    @property
    def entries(self) -> list[ToolEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text(encoding="utf-8"))
                self._entries = [ToolEntry.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._entries = []

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        self._file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
