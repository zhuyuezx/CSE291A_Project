"""
Automated MCTS heuristic optimization loop.

Orchestrates the two-component cycle:
    1. MCTS plays games → collects traces
    2. LLM analyses traces → proposes improved heuristic code
    3. Loader compiles & validates → hot-swaps into the engine
    4. Repeat, keeping only improvements ("keep best" safeguard)

This module ties together MCTSEngine, LLMClient, PromptBuilder,
and HeuristicLoader into a single high-level API.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .game_interface import Game, GameState
from .mcts_engine import MCTSEngine
from .llm_client import LLMClient
from .prompt_builder import PromptBuilder, PromptTemplate
from .heuristic_loader import HeuristicLoader, HeuristicLoadError

logger = logging.getLogger(__name__)


# =====================================================================
# Round result data
# =====================================================================

@dataclass
class RoundResult:
    """Result of one optimization round."""

    round_num: int
    play_rate: float          # solve/win rate during trace collection
    candidate_rate: float     # solve/win rate of the LLM candidate
    best_rate: float          # best rate after this round
    adopted: bool             # whether the candidate was adopted
    llm_time_sec: float       # LLM inference time
    candidate_code: str = ""  # source code of the candidate (if any)
    error: str | None = None  # error message if extraction/load failed


@dataclass
class LoopResult:
    """Aggregate result of the full optimization loop."""

    baseline_rate: float
    final_rate: float
    rounds: list[RoundResult] = field(default_factory=list)
    best_heuristic: Callable | None = None
    best_code: str | None = None

    @property
    def improvement_pp(self) -> float:
        """Total improvement in percentage points."""
        return (self.final_rate - self.baseline_rate) * 100

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = ["Solve rate progression:"]
        lines.append(f"  {'Baseline':>12s}: {self.baseline_rate*100:5.1f}%")
        for r in self.rounds:
            tag = "adopted" if r.adopted else ("error" if r.error else "rejected")
            lines.append(
                f"  {'R' + str(r.round_num):>12s}: "
                f"{r.best_rate*100:5.1f}% ({tag})"
            )
        lines.append(f"\nTotal improvement: {self.improvement_pp:+.1f} pp")
        return "\n".join(lines)


# =====================================================================
# Optimization loop
# =====================================================================

class OptimizationLoop:
    """
    Runs the full optimize-evaluate cycle.

    Example::

        from mcts.games.sliding_puzzle import SlidingPuzzle, SlidingPuzzleState

        loop = OptimizationLoop(
            game_factory=lambda: SlidingPuzzle(size=3, scramble_moves=10, max_steps=50),
            state_classes=[SlidingPuzzleState],
            mcts_iterations=30,
            games_per_round=30,
            validation_games=30,
        )
        result = loop.run(num_rounds=3, verbose=True)
        print(result.summary())
    """

    def __init__(
        self,
        game_factory: Callable[[], Game],
        state_classes: list[type] | None = None,
        *,
        mcts_iterations: int = 30,
        max_rollout_depth: int = 50,
        games_per_round: int = 30,
        validation_games: int = 30,
        target_heuristic: str = "evaluation",
        llm_client: LLMClient | None = None,
        prompt_template: PromptTemplate | None = None,
    ):
        """
        Args:
            game_factory:      Callable that returns a fresh Game instance.
            state_classes:     Concrete GameState subclasses for the
                               HeuristicLoader exec namespace.
            mcts_iterations:   MCTS iterations per move.
            max_rollout_depth: Max rollout depth.
            games_per_round:   Games for trace collection each round.
            validation_games:  Games to validate a candidate heuristic.
            target_heuristic:  Which heuristic slot to optimise.
            llm_client:        LLM client (default: Qwen 3.5 via Ollama).
            prompt_template:   Game-specific prompt template (auto-detected
                               from game name if omitted).
        """
        self.game_factory = game_factory
        self.mcts_iterations = mcts_iterations
        self.max_rollout_depth = max_rollout_depth
        self.games_per_round = games_per_round
        self.validation_games = validation_games
        self.target_heuristic = target_heuristic

        self.llm = llm_client or LLMClient()
        self.prompt_builder = PromptBuilder(template=prompt_template)
        self.loader = HeuristicLoader(game_state_classes=state_classes or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        num_rounds: int = 3,
        verbose: bool = True,
    ) -> LoopResult:
        """
        Execute the full optimization loop.

        Args:
            num_rounds: Number of LLM optimization rounds.
            verbose:    Print progress to stdout.

        Returns:
            LoopResult with baseline, final rate, round details,
            and the best heuristic function found.
        """
        # --- Baseline ---
        if verbose:
            print("Measuring baseline (default heuristics)...")
        baseline_rate = self._evaluate(heuristic_fn=None)
        if verbose:
            print(f"Baseline solve rate: {baseline_rate*100:.1f}%")

        best_fn: Callable | None = None
        best_rate = baseline_rate
        best_code: str | None = None
        prev_code: str | None = None
        rounds: list[RoundResult] = []

        for round_num in range(1, num_rounds + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_num}/{num_rounds}")
                print(f"{'='*60}")

            result = self._run_one_round(
                round_num=round_num,
                best_fn=best_fn,
                best_rate=best_rate,
                prev_code=prev_code,
                verbose=verbose,
            )
            rounds.append(result)

            if result.adopted:
                # Re-evaluate to get the new best_fn — it's stored in result
                best_rate = result.candidate_rate
                best_code = result.candidate_code
                # Recompile to get the function object
                best_fn, _ = self.loader.load_from_response(
                    f"```python\n{result.candidate_code}\n```",
                    target_name=self.target_heuristic,
                )

            prev_code = result.candidate_code or prev_code

        # --- Final ---
        final_rate = best_rate
        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS")
            print(f"{'='*60}")

        loop_result = LoopResult(
            baseline_rate=baseline_rate,
            final_rate=final_rate,
            rounds=rounds,
            best_heuristic=best_fn,
            best_code=best_code,
        )

        if verbose:
            print(loop_result.summary())

        return loop_result

    # ------------------------------------------------------------------
    # Single-round logic
    # ------------------------------------------------------------------

    def _run_one_round(
        self,
        round_num: int,
        best_fn: Callable | None,
        best_rate: float,
        prev_code: str | None,
        verbose: bool,
    ) -> RoundResult:
        """Execute one round of play → LLM → validate → adopt."""

        # 1. Play games with current best heuristic
        engine = self._make_engine(heuristic_fn=best_fn)
        t0 = time.time()
        stats = engine.play_many(
            num_games=self.games_per_round,
            clear_table_each_game=True,
        )
        play_time = time.time() - t0
        play_rate = stats["win_rate"]

        if verbose:
            print(f"  Play: {play_rate*100:.1f}% solved ({play_time:.1f}s)")

        # 2. Build prompt and call LLM
        prompt = self.prompt_builder.build(engine, stats, prev_code=prev_code)
        if verbose:
            print(f"  Calling LLM ({len(prompt)} char prompt)...")

        t1 = time.time()
        try:
            llm_resp = self.llm.chat(prompt)
        except Exception as e:
            if verbose:
                print(f"  LLM call failed: {e}")
            return RoundResult(
                round_num=round_num,
                play_rate=play_rate,
                candidate_rate=0.0,
                best_rate=best_rate,
                adopted=False,
                llm_time_sec=time.time() - t1,
                error=str(e),
            )
        llm_time = time.time() - t1

        if verbose:
            print(f"  LLM responded in {llm_time:.1f}s")

        # 3. Extract and load candidate
        try:
            test_state = self.game_factory().new_initial_state()
            candidate_fn, candidate_code = self.loader.load_from_response(
                llm_resp.full_text,
                target_name=self.target_heuristic,
                test_state=test_state,
            )
            if verbose:
                print(f"  Extracted code ({len(candidate_code)} chars)")
        except HeuristicLoadError as e:
            if verbose:
                print(f"  FAIL to load: {e}")
            return RoundResult(
                round_num=round_num,
                play_rate=play_rate,
                candidate_rate=0.0,
                best_rate=best_rate,
                adopted=False,
                llm_time_sec=llm_time,
                error=str(e),
            )

        # 4. Validate candidate
        candidate_rate = self._evaluate(heuristic_fn=candidate_fn)
        if verbose:
            print(
                f"  Candidate solve rate: {candidate_rate*100:.1f}% "
                f"(best so far: {best_rate*100:.1f}%)"
            )

        # 5. Adopt if better
        adopted = candidate_rate > best_rate
        new_best_rate = candidate_rate if adopted else best_rate

        if verbose:
            if adopted:
                print("  -> ADOPTED new heuristic (improvement!)")
            else:
                print("  -> Rejected -- not better than current best")

        return RoundResult(
            round_num=round_num,
            play_rate=play_rate,
            candidate_rate=candidate_rate,
            best_rate=new_best_rate,
            adopted=adopted,
            llm_time_sec=llm_time,
            candidate_code=candidate_code,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_engine(self, heuristic_fn: Callable | None = None) -> MCTSEngine:
        """Create a fresh MCTSEngine, optionally with a custom heuristic."""
        game = self.game_factory()
        engine = MCTSEngine(
            game,
            iterations=self.mcts_iterations,
            max_rollout_depth=self.max_rollout_depth,
        )
        if heuristic_fn is not None:
            engine.set_heuristic(self.target_heuristic, heuristic_fn)
        return engine

    def _evaluate(self, heuristic_fn: Callable | None = None) -> float:
        """Play validation games and return the win/solve rate."""
        engine = self._make_engine(heuristic_fn=heuristic_fn)
        stats = engine.play_many(
            num_games=self.validation_games,
            clear_table_each_game=True,
            verbose=False,
        )
        return stats["win_rate"]
