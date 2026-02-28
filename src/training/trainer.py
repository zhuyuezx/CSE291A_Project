# src/training/trainer.py
from __future__ import annotations

import random
from collections import deque
from typing import Callable

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.training.trace_recorder import TraceRecorder


class PlateauDetector:
    def __init__(
        self,
        window_size: int = 50,
        improvement_threshold: float = 0.02,
        regression_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold
        self._history: list[float] = []

    def record(self, win_rate: float) -> None:
        self._history.append(win_rate)

    def is_plateau(self) -> bool:
        if len(self._history) < 2 * self.window_size:
            return False

        prev_window = self._history[-(2 * self.window_size) : -self.window_size]
        curr_window = self._history[-self.window_size :]

        prev_avg = sum(prev_window) / len(prev_window)
        curr_avg = sum(curr_window) / len(curr_window)

        improvement = curr_avg - prev_avg

        # Regression
        if improvement < -self.regression_threshold:
            return True

        # Plateau (no significant improvement)
        if abs(improvement) < self.improvement_threshold:
            return True

        return False

    def current_win_rate(self) -> float | None:
        if not self._history:
            return None
        window = self._history[-self.window_size :]
        return sum(window) / len(window)


class Trainer:
    def __init__(
        self,
        adapter: GameAdapter,
        registry: ToolRegistry,
        simulations: int = 100,
        uct_c: float = 1.41,
        plateau_detector: PlateauDetector | None = None,
        on_plateau: Callable | None = None,
    ):
        self.adapter = adapter
        self.registry = registry
        self.engine = MCTSEngine(adapter, registry, simulations=simulations, uct_c=uct_c)
        self.recorder = TraceRecorder()
        self.plateau_detector = plateau_detector or PlateauDetector()
        self.on_plateau = on_plateau
        self.total_games = 0

    def play_game_vs_random(self, player: int = 0) -> float:
        """Play one game against a random opponent. Returns result for `player`."""
        state = self.adapter.new_game()
        self.recorder.start_game(state)

        while not self.adapter.is_terminal(state):
            current = self.adapter.current_player(state)
            if current == player:
                action = self.engine.search(state)
            else:
                action = random.choice(self.adapter.legal_actions(state))
            state = self.adapter.apply_action(state, action)
            self.recorder.record_step(state, action)

        returns = self.adapter.returns(state)
        self.recorder.end_game(returns)
        self.total_games += 1

        result = 1.0 if returns[player] > 0 else (0.0 if returns[player] == 0 else -1.0)
        self.plateau_detector.record(result)

        return returns[player]

    def play_episode(self) -> float:
        """Play one episode for single-player games. Returns normalized return."""
        state = self.adapter.new_game()
        self.recorder.start_game(state)

        while not self.adapter.is_terminal(state):
            action = self.engine.search(state)
            state = self.adapter.apply_action(state, action)
            self.recorder.record_step(state, action)

        raw_return = state.returns()[0]
        normalized = self.adapter.normalize_return(raw_return)
        self.recorder.end_game(state.returns())
        self.total_games += 1
        self.plateau_detector.record(normalized)
        return normalized

    def train(self, num_games: int, player: int = 0) -> dict:
        """Run training loop with plateau detection."""
        scores = []
        wins = 0
        for i in range(num_games):
            if self.adapter.meta.is_single_player:
                result = self.play_episode()
            else:
                result = self.play_game_vs_random(player)
                if result > 0:
                    wins += 1
            scores.append(result)

            if self.plateau_detector.is_plateau() and self.on_plateau:
                self.on_plateau(self)

        avg = sum(scores) / len(scores) if scores else 0.0
        metric = self.adapter.meta.metric_name
        stats = {"games": num_games, metric: avg, "scores": scores}
        if not self.adapter.meta.is_single_player:
            stats["wins"] = wins
            stats["win_rate"] = wins / num_games if num_games > 0 else 0
        return stats
