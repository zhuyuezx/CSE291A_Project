# src/training/trace_recorder.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameTrace:
    states: list[str] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    outcome: list[float] | None = None

    def to_string(self) -> str:
        lines = []
        for i, (state_str, action) in enumerate(zip(self.states, self.actions)):
            lines.append(f"Step {i}: Action={action}")
            lines.append(state_str)
            lines.append("")
        if self.outcome:
            lines.append(f"Outcome: {self.outcome}")
        return "\n".join(lines)


class TraceRecorder:
    def __init__(self):
        self._traces: list[GameTrace] = []
        self._current: GameTrace | None = None

    def start_game(self, initial_state) -> None:
        self._current = GameTrace()
        self._current.states.append(str(initial_state))

    def record_step(self, state, action: int) -> None:
        if self._current is None:
            return
        self._current.states.append(str(state))
        self._current.actions.append(action)

    def end_game(self, returns: list[float]) -> None:
        if self._current is None:
            return
        self._current.outcome = returns
        self._traces.append(self._current)
        self._current = None

    def get_traces(self) -> list[GameTrace]:
        return list(self._traces)

    def clear(self) -> None:
        self._traces.clear()

    def select_informative_traces(self, player: int, n: int = 5) -> list[GameTrace]:
        """Select the most informative traces: losses first, then close games."""
        losses = [t for t in self._traces if t.outcome and t.outcome[player] < 0]
        draws = [t for t in self._traces if t.outcome and t.outcome[player] == 0]
        wins = [t for t in self._traces if t.outcome and t.outcome[player] > 0]

        # Prioritize losses, then draws, then wins
        selected = []
        for pool in [losses, draws, wins]:
            for trace in pool:
                if len(selected) >= n:
                    break
                selected.append(trace)

        return selected[:n]
