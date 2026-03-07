"""
Evaluator — game-agnostic multi-run evaluation and mastery tracking.

Runs MCTS games with a given tool function, computes composite scores,
and tracks per-level baselines and mastery status.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from mcts import MCTSEngine, Game


class Evaluator:
    """
    Evaluate MCTS tool functions across game levels.

    Parameters
    ----------
    game_factory : callable(level) -> Game
        Given a level identifier, returns a Game instance.
    phase : str
        MCTS phase being optimized (e.g. "simulation").
    iterations : int
        MCTS iterations per move.
    max_rollout_depth : int
        Max rollout depth for MCTS.
    eval_runs : int
        Number of games per evaluation.
    solve_weight : float
        Weight for solve_rate in composite score.
    return_weight : float
        Weight for avg_returns in composite score.
    mastery_solve_rate : float
        Solve rate threshold to trigger mastery confirmation.
    mastery_confirm_runs : int
        Extra games to confirm mastery.
    mastery_max_steps : int | None
        Optional max avg_steps for mastery.
    """

    def __init__(
        self,
        game_factory: Callable[[Any], Game],
        phase: str,
        iterations: int = 200,
        max_rollout_depth: int = 500,
        eval_runs: int = 3,
        solve_weight: float = 0.6,
        return_weight: float = 0.4,
        mastery_solve_rate: float = 1.0,
        mastery_confirm_runs: int = 7,
        mastery_max_steps: int | None = None,
    ):
        self.game_factory = game_factory
        self.phase = phase
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth
        self.eval_runs = eval_runs
        self.solve_weight = solve_weight
        self.return_weight = return_weight
        self.mastery_solve_rate = mastery_solve_rate
        self.mastery_confirm_runs = mastery_confirm_runs
        self.mastery_max_steps = mastery_max_steps

        # Per-level tracking
        self.level_baselines: dict[str, dict] = {}
        self.level_best_scores: dict[str, float] = {}
        self.mastered_levels: set[str] = set()

    def composite_score(self, solve_rate: float, avg_returns: float) -> float:
        """Weighted score that prioritizes solving over raw returns."""
        return self.solve_weight * solve_rate + self.return_weight * avg_returns

    def multi_eval(
        self,
        fn: Callable | None,
        level: Any,
        n: int | None = None,
        logging: bool = True,
    ) -> tuple[float, float, float, list[dict], float]:
        """
        Evaluate a function over n games on a specific level.

        Returns (avg_returns, solve_rate, avg_steps, results, elapsed).
        """
        if n is None:
            n = self.eval_runs
        t0 = time.time()
        results = []
        for _ in range(n):
            g = self.game_factory(level)
            e = MCTSEngine(
                g,
                iterations=self.iterations,
                max_rollout_depth=self.max_rollout_depth,
                logging=logging,
            )
            if fn is not None:
                e.set_tool(self.phase, fn)
            r = e.play_game()
            results.append(r)
        elapsed = time.time() - t0
        avg_ret = sum(
            r["returns"][0] if isinstance(r["returns"], list)
            else r["returns"]
            for r in results
        ) / n
        solve_rate = sum(1 for r in results if r.get("solved")) / n
        avg_steps = sum(r.get("steps", 0) for r in results) / n
        return avg_ret, solve_rate, avg_steps, results, elapsed

    def check_mastery(
        self,
        level: Any,
        solve_rate: float,
        avg_steps: float,
        fn: Callable | None,
    ) -> bool:
        """
        Check if a level is mastered. Runs confirmation eval if initial
        eval hits 100% solve rate. Returns True if newly mastered.
        """
        if level in self.mastered_levels:
            return False
        if solve_rate < self.mastery_solve_rate:
            return False
        if self.mastery_max_steps is not None and avg_steps > self.mastery_max_steps:
            return False

        print(f"  ⏳ {level} hit {solve_rate:.0%} on {self.eval_runs} runs — "
              f"confirming with {self.mastery_confirm_runs} more games…")
        _, confirm_sr, confirm_steps, _, confirm_t = self.multi_eval(
            fn, level, n=self.mastery_confirm_runs
        )
        if confirm_sr < self.mastery_solve_rate:
            print(f"  ❌ Confirmation failed: {confirm_sr:.0%} on "
                  f"{self.mastery_confirm_runs} runs ({confirm_t:.1f}s)")
            return False
        if self.mastery_max_steps is not None and confirm_steps > self.mastery_max_steps:
            print(f"  ❌ Confirmation failed: avg_steps={confirm_steps:.0f} > "
                  f"{self.mastery_max_steps}")
            return False

        total_runs = self.eval_runs + self.mastery_confirm_runs
        self.mastered_levels.add(level)
        print(f"  🎓 {level} MASTERED — {total_runs}/{total_runs} solved "
              f"({confirm_t:.1f}s)")
        return True

    def get_baseline(self, level: Any) -> dict:
        """Return baseline info for a level, computing lazily if needed."""
        if level not in self.level_baselines:
            print(f"  Computing baseline for {level}…")
            avg, sr, steps, _, t = self.multi_eval(None, level)
            comp = self.composite_score(sr, avg)
            self.level_baselines[level] = {
                "avg_returns": avg,
                "solve_rate": sr,
                "avg_steps": steps,
                "eval_time": t,
                "composite": comp,
            }
            self.level_best_scores[level] = comp
            print(f"    {level}: composite={comp:.4f}, solve_rate={sr:.0%}, "
                  f"avg_returns={avg:.4f} ({t:.1f}s)")
            self.check_mastery(level, sr, steps, None)
        return self.level_baselines[level]
