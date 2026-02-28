# src/training/evaluator.py
from __future__ import annotations

import random
from dataclasses import dataclass

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine


@dataclass
class PerformanceResult:
    game: str
    metric_name: str        # "win_rate" | "avg_score" | "success_rate"
    raw_value: float        # e.g. 14.3 avg pieces, 0.62 win rate
    normalized_value: float # always [-1, 1]
    n_games: int


class Evaluator:
    def __init__(self, adapter: GameAdapter):
        self.adapter = adapter

    def measure(self, engine: MCTSEngine, n_games: int = 100) -> PerformanceResult:
        """Unified evaluation for both single-player and two-player games."""
        scores_raw = []
        wins = 0

        for _ in range(n_games):
            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                cp = self.adapter.current_player(state)
                if self.adapter.meta.is_single_player or cp == 0:
                    action = engine.search(state)
                else:
                    action = random.choice(self.adapter.legal_actions(state))
                state = self.adapter.apply_action(state, action)

            raw = state.returns()[0]
            scores_raw.append(raw)
            if raw > 0:
                wins += 1

        avg_raw = sum(scores_raw) / n_games
        avg_norm = sum(self.adapter.normalize_return(r) for r in scores_raw) / n_games

        if self.adapter.meta.is_single_player:
            raw_value = avg_raw
        else:
            raw_value = wins / n_games  # win rate

        return PerformanceResult(
            game=self.adapter.game_name,
            metric_name=self.adapter.meta.metric_name,
            raw_value=raw_value,
            normalized_value=avg_norm,
            n_games=n_games,
        )

    def evaluate_vs_random(
        self, engine: MCTSEngine, num_games: int = 100, player: int = 0
    ) -> dict:
        wins, losses, draws = 0, 0, 0
        for i in range(num_games):
            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                current = self.adapter.current_player(state)
                if current == player:
                    action = engine.search(state)
                else:
                    action = random.choice(self.adapter.legal_actions(state))
                state = self.adapter.apply_action(state, action)

            result = self.adapter.returns(state)[player]
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0,
            "num_games": num_games,
        }

    def evaluate_head_to_head(
        self,
        engine_a: MCTSEngine,
        engine_b: MCTSEngine,
        num_games: int = 100,
    ) -> dict:
        a_wins, b_wins, draws = 0, 0, 0
        for i in range(num_games):
            # Alternate who plays first
            a_player = i % 2
            b_player = 1 - a_player

            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                current = self.adapter.current_player(state)
                if current == a_player:
                    action = engine_a.search(state)
                else:
                    action = engine_b.search(state)
                state = self.adapter.apply_action(state, action)

            result_a = self.adapter.returns(state)[a_player]
            if result_a > 0:
                a_wins += 1
            elif result_a < 0:
                b_wins += 1
            else:
                draws += 1

        return {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / num_games if num_games > 0 else 0,
            "b_win_rate": b_wins / num_games if num_games > 0 else 0,
            "num_games": num_games,
        }

    def sample_efficiency_curve(
        self,
        engine_factory,
        sim_budgets: list[int] = None,
        num_games_per_budget: int = 50,
        player: int = 0,
    ) -> dict[int, float]:
        """Evaluate win rate at different simulation budgets."""
        if sim_budgets is None:
            sim_budgets = [10, 50, 100, 500, 1000]

        curve = {}
        for budget in sim_budgets:
            engine = engine_factory(budget)
            result = self.evaluate_vs_random(engine, num_games_per_budget, player)
            curve[budget] = result["win_rate"]

        return curve
