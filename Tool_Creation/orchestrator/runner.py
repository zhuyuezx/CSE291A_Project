"""
OptimizationRunner — game-agnostic iterative LLM optimization loop.

Encapsulates the full train loop from `test_llm_pipeline.ipynb`:
  1. Pick a random level from the active pool
  2. Play with the current tool → collect trace
  3. Build history context → run 3-step LLM optimizer
  4. Evaluate the new tool on the same level
  5. Accept/reject with per-level baselines
  6. Track mastery and remove solved levels

Usage::

    from orchestrator import OptimizationRunner

    runner = OptimizationRunner.from_config("orchestrator/config.json")
    summary = runner.run()
    # summary is a dict with all_results, best_fn, level_best_scores, etc.
"""

from __future__ import annotations

import importlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from mcts import MCTSEngine, Game
from LLM import Optimizer

from .evaluator import Evaluator


def _load_config(config_path: str | Path) -> dict:
    """Load and return the JSON configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_game_factory(
    game_class: type,
    levels: list[str],
    constructor_kwargs: dict,
) -> Callable[[Any], Game]:
    """
    Return a callable(level) -> Game.

    For games with a 'level_name' parameter (e.g. Sokoban), the level is
    passed as the first positional arg. For games without level support
    (e.g. TicTacToe), the level is ignored.
    """
    import inspect
    sig = inspect.signature(game_class.__init__)
    params = list(sig.parameters.keys())
    # Check if the constructor has a level-like first param
    has_level_param = len(params) > 1  # first is 'self'

    if has_level_param:
        def factory(level: Any) -> Game:
            return game_class(level, **constructor_kwargs)
    else:
        def factory(level: Any) -> Game:
            return game_class(**constructor_kwargs)

    return factory


class OptimizationRunner:
    """
    Iterative LLM optimization loop — game-agnostic.

    Parameters
    ----------
    game_name : str
        Game name for the LLM prompt builder (e.g. "sokoban").
    game_factory : callable(level) -> Game
        Creates a Game instance for a given level.
    levels : list[str]
        All available levels.
    start_level : str
        Level to start with.
    phase : str
        MCTS phase to optimize.
    mcts_iterations : int
        MCTS iterations per move.
    max_rollout_depth : int
        Max rollout depth.
    evaluator : Evaluator
        Pre-configured evaluator instance.
    num_iters : int
        Number of optimization iterations.
    three_step : bool
        Use 3-step LLM pipeline.
    history_window : int
        How many past iterations to include in LLM context.
    reject_threshold : float
        Reject if composite < baseline * this.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        game_name: str,
        game_factory: Callable[[Any], Game],
        levels: list[str],
        start_level: str,
        phase: str,
        mcts_iterations: int,
        max_rollout_depth: int,
        evaluator: Evaluator,
        num_iters: int = 5,
        three_step: bool = True,
        history_window: int = 3,
        reject_threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.game_name = game_name
        self.game_factory = game_factory
        self.levels = levels
        self.start_level = start_level
        self.phase = phase
        self.mcts_iterations = mcts_iterations
        self.max_rollout_depth = max_rollout_depth
        self.evaluator = evaluator
        self.num_iters = num_iters
        self.three_step = three_step
        self.history_window = history_window
        self.reject_threshold = reject_threshold
        self.verbose = verbose

        # State
        self.best_fn: Callable | None = None
        self.current_fn: Callable | None = None
        self.all_results: list[dict] = []
        self.active_levels: list[str] = list(levels)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.json",
        verbose: bool = True,
    ) -> "OptimizationRunner":
        """Create an OptimizationRunner from a JSON config file."""
        cfg = _load_config(config_path)

        game_cfg = cfg["game"]
        mcts_cfg = cfg["mcts"]
        opt_cfg = cfg["optimization"]

        # Dynamically import the game class
        module = importlib.import_module(game_cfg["module"])
        game_class = getattr(module, game_cfg["class"])
        ctor_kwargs = game_cfg.get("constructor_kwargs", {})
        levels = game_cfg["levels"]
        start_level = game_cfg.get("start_level", levels[0])

        game_factory = _make_game_factory(game_class, levels, ctor_kwargs)

        evaluator = Evaluator(
            game_factory=game_factory,
            phase=mcts_cfg["phase"],
            iterations=mcts_cfg["iterations"],
            max_rollout_depth=mcts_cfg["max_rollout_depth"],
            eval_runs=opt_cfg["eval_runs"],
            solve_weight=opt_cfg["solve_weight"],
            return_weight=opt_cfg["return_weight"],
            mastery_solve_rate=opt_cfg["mastery_solve_rate"],
            mastery_confirm_runs=opt_cfg["mastery_confirm_runs"],
            mastery_max_steps=opt_cfg.get("mastery_max_steps"),
        )

        return cls(
            game_name=game_cfg["name"],
            game_factory=game_factory,
            levels=levels,
            start_level=start_level,
            phase=mcts_cfg["phase"],
            mcts_iterations=mcts_cfg["iterations"],
            max_rollout_depth=mcts_cfg["max_rollout_depth"],
            evaluator=evaluator,
            num_iters=opt_cfg["num_iters"],
            three_step=opt_cfg["three_step"],
            history_window=opt_cfg["history_window"],
            reject_threshold=opt_cfg["reject_threshold"],
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # History builder
    # ------------------------------------------------------------------

    def _build_history(self, results: list[dict], current_level: str) -> str:
        """Build a concise history string for the LLM prompt."""
        ev = self.evaluator
        bl = ev.get_baseline(current_level)
        agg_best = (
            sum(ev.level_best_scores.values()) / len(ev.level_best_scores)
            if ev.level_best_scores else 0.0
        )

        lines = [
            f"Current level: {current_level}",
            f"Baseline for {current_level} (default MCTS): "
            f"composite={bl['composite']:.4f}, "
            f"solve_rate={bl['solve_rate']:.0%}, "
            f"avg_returns={bl['avg_returns']:.4f}",
            f"Aggregate best (avg across {len(ev.level_best_scores)} levels): "
            f"{agg_best:.4f}",
            "",
            "Per-level best composites so far:",
        ]
        for lv in sorted(ev.level_best_scores.keys()):
            bl_lv = ev.level_baselines.get(lv, {})
            tag = " [MASTERED]" if lv in ev.mastered_levels else ""
            lines.append(
                f"  {lv}: best={ev.level_best_scores[lv]:.4f} "
                f"(baseline={bl_lv.get('composite', 0):.4f}){tag}"
            )
        lines += [
            "",
            f"Active levels (not yet mastered): {sorted(self.active_levels)}",
            (f"Mastered levels: {sorted(ev.mastered_levels)}"
             if ev.mastered_levels else ""),
            "",
            "SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns",
            "  → SOLVING the puzzle is MORE important than heuristic accuracy.",
            "",
            "STRATEGY: Prefer gradual, incremental improvements. Build on the",
            "previous version rather than rewriting from scratch. However, if",
            "the current approach is fundamentally flawed, a larger restructure",
            "is acceptable.",
            "",
            "Recent iterations:",
        ]
        for r in results:
            tag = f"solve_rate={r['solve_rate']:.0%}"
            status = ""
            if r.get("is_best"):
                status = " ★ BEST-for-level"
            elif r.get("adopted"):
                status = " ← accepted"
            else:
                status = " ✗ rejected"
            time_info = ""
            if r.get("eval_time") is not None:
                time_info = f", eval_time={r['eval_time']:.1f}s"
            lines.append(
                f"  Iter {r['iteration']} [{r['level']}]: "
                f"composite={r['composite']:.4f}, "
                f"{tag}{time_info}, "
                f"desc={r.get('description', 'n/a')}{status}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute the iterative optimization loop.

        Returns a summary dict with:
            all_results, best_fn, level_best_scores, level_baselines,
            mastered_levels, active_levels
        """
        ev = self.evaluator
        current_level = self.start_level

        # Sync mastered levels into active_levels
        self.active_levels = [
            lv for lv in self.levels if lv not in ev.mastered_levels
        ]

        # Create the LLM optimizer
        opt = Optimizer(
            game=self.game_name,
            target_phase=self.phase,
            three_step=self.three_step,
            verbose=self.verbose,
        )

        # Compute initial baseline
        if self.verbose:
            print(f"Starting level: {current_level}")
        init_bl = ev.get_baseline(current_level)
        if self.verbose:
            print(f"  Reject floor for {current_level}: "
                  f"{init_bl['composite'] * self.reject_threshold:.4f}")
            print(f"  Active levels: {self.active_levels}")

        for iteration in range(1, self.num_iters + 1):
            if not self.active_levels:
                if self.verbose:
                    print(f"\n🎉 All levels mastered after "
                          f"{iteration - 1} iterations! Stopping early.")
                break

            cur_level = current_level
            bl = ev.get_baseline(cur_level)
            reject_floor = bl["composite"] * self.reject_threshold

            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"  ITERATION {iteration}/{self.num_iters}, "
                      f"LEVEL={cur_level}")
                print(f"  Baseline composite={bl['composite']:.4f}, "
                      f"reject_floor={reject_floor:.4f}")
                print(f"  Active levels: {len(self.active_levels)}/"
                      f"{len(self.levels)}, "
                      f"mastered: {sorted(ev.mastered_levels)}")
                print(f"{'#'*60}")

            # Capture level by value via default arg
            state_factory = (
                lambda _lv=cur_level: self.game_factory(_lv).new_initial_state()
            )

            # --- 1. Play with current tool ---
            t_play_start = time.time()
            g = self.game_factory(cur_level)
            eng = MCTSEngine(
                g,
                iterations=self.mcts_iterations,
                max_rollout_depth=self.max_rollout_depth,
                logging=True,
            )
            if self.current_fn is not None:
                eng.set_tool(self.phase, self.current_fn)

            play_result = eng.play_game()
            t_play = time.time() - t_play_start
            play_trace = play_result.get("log_file", "")
            tl = eng.get_tool_source()
            p_ret = play_result["returns"]
            p_ret_val = p_ret[0] if isinstance(p_ret, list) else p_ret
            ptag = "SOLVED" if play_result.get("solved") else "UNSOLVED"
            if self.verbose:
                print(f"  Play: {ptag} in {play_result.get('steps', '?')} "
                      f"steps  returns={p_ret_val:.4f}  ({t_play:.1f}s)")

            # --- 2. Build history context ---
            history = (
                self._build_history(
                    self.all_results[-self.history_window:], cur_level
                )
                if self.all_results else None
            )

            # --- 3. Optimize (3-step: analysis → draft → critique) ---
            t_opt_start = time.time()
            result = opt.run(
                record_files=[play_trace] if play_trace else [],
                tool_list=tl,
                state_factory=state_factory,
                additional_context=history,
            )
            t_opt = time.time() - t_opt_start
            if self.verbose:
                print(f"  Optimize: {t_opt:.1f}s")

            # --- 4. Multi-run evaluation ---
            iter_record = {
                "iteration": iteration,
                "level": cur_level,
                "smoke_test": result["smoke_test"],
                "avg_returns": bl["avg_returns"],
                "solve_rate": 0.0,
                "composite": 0.0,
                "avg_steps": 0,
                "description": (
                    (result.get("parsed") or {}).get("description", "")
                ),
                "error": result.get("error"),
                "adopted": False,
                "is_best": False,
                "play_time": t_play,
                "opt_time": t_opt,
                "eval_time": None,
            }

            fn = result.get("function")
            if fn is not None:
                avg_ret, solve_rate, avg_steps, _, eval_time = ev.multi_eval(
                    fn, cur_level
                )
                comp = ev.composite_score(solve_rate, avg_ret)
                iter_record["avg_returns"] = avg_ret
                iter_record["solve_rate"] = solve_rate
                iter_record["composite"] = comp
                iter_record["avg_steps"] = avg_steps
                iter_record["eval_time"] = eval_time

                if self.verbose:
                    print(
                        f"  Eval ({ev.eval_runs} runs, {cur_level}): "
                        f"avg_returns={avg_ret:.4f}, "
                        f"solve_rate={solve_rate:.0%}, "
                        f"composite={comp:.4f}, "
                        f"avg_steps={avg_steps:.0f}  ({eval_time:.1f}s)"
                    )

                # Check mastery with confirmation
                newly_mastered = ev.check_mastery(
                    cur_level, solve_rate, avg_steps, fn
                )
                if newly_mastered and cur_level in self.active_levels:
                    self.active_levels.remove(cur_level)
                    if self.verbose:
                        print(f"    {len(self.active_levels)} levels remain.")

                # --- 5. Accept-unless-terrible ---
                prev_level_best = ev.level_best_scores.get(
                    cur_level, bl["composite"]
                )

                if comp > prev_level_best:
                    if self.verbose:
                        print(f"  ★ NEW BEST for {cur_level} "
                              f"(prev={prev_level_best:.4f}) — adopting")
                    ev.level_best_scores[cur_level] = comp
                    self.best_fn = fn
                    self.current_fn = fn
                    iter_record["adopted"] = True
                    iter_record["is_best"] = True
                elif comp >= reject_floor:
                    if self.verbose:
                        print(
                            f"  → Accepted on {cur_level} "
                            f"(comp={comp:.4f} ≥ floor={reject_floor:.4f}, "
                            f"level_best={prev_level_best:.4f})"
                        )
                    self.current_fn = fn
                    iter_record["adopted"] = True
                else:
                    if self.verbose:
                        print(
                            f"  ✗ Rejected on {cur_level} "
                            f"(comp={comp:.4f} < floor={reject_floor:.4f}) "
                            f"— reverting to best"
                        )
                    self.current_fn = self.best_fn
            else:
                if self.verbose:
                    print("  Eval:  SKIPPED (smoke test failed or error)")
                    if result.get("error"):
                        print(f"         {result['error'][:120]}")

            total_time = t_play + t_opt + (iter_record["eval_time"] or 0)
            if self.verbose:
                print(f"  Iteration total: {total_time:.1f}s")
            self.all_results.append(iter_record)

            # Pick next level from non-mastered pool
            if self.active_levels:
                current_level = random.choice(self.active_levels)
            else:
                if self.verbose:
                    print("\n🎉 All levels mastered! Will stop next iteration.")
                current_level = random.choice(self.levels)

        # --- Summary ---
        summary = {
            "all_results": self.all_results,
            "best_fn": self.best_fn,
            "current_fn": self.current_fn,
            "level_best_scores": dict(ev.level_best_scores),
            "level_baselines": dict(ev.level_baselines),
            "mastered_levels": set(ev.mastered_levels),
            "active_levels": list(self.active_levels),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Iterative optimization complete — "
                  f"{len(self.all_results)} iterations done.")
            if ev.mastered_levels:
                print(f"Mastered {len(ev.mastered_levels)}/{len(self.levels)} "
                      f"levels: {sorted(ev.mastered_levels)}")
            if ev.level_best_scores:
                agg = (sum(ev.level_best_scores.values())
                       / len(ev.level_best_scores))
                print(f"Aggregate best composite "
                      f"(avg of {len(ev.level_best_scores)} levels): "
                      f"{agg:.4f}")
                for lv in sorted(ev.level_best_scores.keys()):
                    bl_lv = ev.level_baselines.get(lv, {})
                    delta = (ev.level_best_scores[lv]
                             - bl_lv.get("composite", 0))
                    tag = (" [MASTERED]"
                           if lv in ev.mastered_levels else "")
                    print(f"  {lv}: best={ev.level_best_scores[lv]:.4f} "
                          f"(baseline={bl_lv.get('composite', 0):.4f}, "
                          f"Δ={delta:+.4f}){tag}")
            print(f"{'='*60}")

        return summary
