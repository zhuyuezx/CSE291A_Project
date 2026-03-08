"""
OptimizationRunner — game-agnostic iterative LLM optimization loop.

Encapsulates the full train loop:
  1. Pick a level (via game-specific training logic)
  2. Pick which component to optimize (tool phase or hyperparams)
  3. Play with current tool + hyperparams → collect trace
  4. Build history context → run 3-step LLM optimizer
  5. Evaluate the new tool / hyperparams
  6. Accept/reject with per-level baselines
  7. Track mastery and remove solved levels

Configuration sources (all in ``MCTS_tools/``):
  - ``hyperparams/default_hyperparams.py`` — engine params, game identity,
    and optimization orchestration settings
  - ``training_logic/<game>_training.py`` — levels, mastery criteria, etc.

Usage::

    from orchestrator import OptimizationRunner

    runner = OptimizationRunner.from_config()
    summary = runner.run()
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import random
import time
from pathlib import Path
from typing import Any, Callable

from mcts import MCTSEngine, Game
from LLM import Optimizer

from .evaluator import Evaluator

# ── Paths ────────────────────────────────────────────────────────────
_TOOL_CREATION_DIR = Path(__file__).resolve().parent.parent
_MCTS_TOOLS_DIR = _TOOL_CREATION_DIR / "MCTS_tools"


def _load_training_logic(module_name: str):
    """Load a training logic module from MCTS_tools/training_logic/."""
    path = _MCTS_TOOLS_DIR / "training_logic" / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Training logic file not found: {path}\n"
            f"Create MCTS_tools/training_logic/{module_name}.py"
        )
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hyperparams_module(filename: str = "default_hyperparams.py"):
    """Load the full hyperparams module from MCTS_tools/hyperparams/."""
    path = _MCTS_TOOLS_DIR / "hyperparams" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Hyperparams file not found: {path}\n"
            f"Create MCTS_tools/hyperparams/{filename}"
        )
    spec = importlib.util.spec_from_file_location("hyperparams", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_hyperparams_source() -> str:
    """Return the source code of the current hyperparams file."""
    path = _MCTS_TOOLS_DIR / "hyperparams" / "default_hyperparams.py"
    return path.read_text(encoding="utf-8")


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
    sig = inspect.signature(game_class.__init__)
    params = list(sig.parameters.keys())
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

    Supports optimizing multiple components per loop: MCTS tool phases
    (e.g. simulation) and/or hyperparameters.  Training strategy
    (level selection, mastery criteria) is loaded from a game-specific
    training logic module.
    """

    def __init__(
        self,
        game_name: str,
        game_factory: Callable[[Any], Game],
        training,
        phases: list[str],
        evaluator: Evaluator,
        hyperparams_fn: Callable,
        num_iters: int = 5,
        three_step: bool = True,
        history_window: int = 3,
        logging: bool = True,
        verbose: bool = True,
    ):
        self.game_name = game_name
        self.game_factory = game_factory
        self.training = training
        self.phases = phases
        self.evaluator = evaluator
        self.num_iters = num_iters
        self.three_step = three_step
        self.history_window = history_window
        self.logging = logging
        self.verbose = verbose

        # Derived from training logic
        self.levels: list[str] = training.LEVELS
        self.start_level: str = training.START_LEVEL
        self.reject_threshold: float = training.REJECT_THRESHOLD

        # Separate tool phases (engine phases) from meta-phases
        self.tool_phases = [p for p in phases if p != "hyperparams"]
        self.primary_phase = self.tool_phases[0] if self.tool_phases else "simulation"

        # Hyperparams state
        self.hyperparams_fn = hyperparams_fn
        self.best_hyperparams_fn = hyperparams_fn
        self.current_hyperparams: dict = hyperparams_fn()

        # Tool function state — per-phase
        self.best_fns: dict[str, Callable | None] = {
            p: None for p in self.tool_phases
        }
        self.current_fns: dict[str, Callable | None] = {
            p: None for p in self.tool_phases
        }
        self.all_results: list[dict] = []
        self.active_levels: list[str] = list(self.levels)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        hyperparams_file: str = "default_hyperparams.py",
        verbose: bool = True,
    ) -> "OptimizationRunner":
        """Create an OptimizationRunner from the hyperparams module.

        All configuration (game identity, optimization settings, and
        engine parameters) is read from a single Python file in
        ``MCTS_tools/hyperparams/``.
        """
        hp_mod = _load_hyperparams_module(hyperparams_file)

        # Game identity from module-level constants
        game_module = importlib.import_module(
            getattr(hp_mod, "GAME_MODULE", "mcts.games")
        )
        game_class = getattr(
            game_module, getattr(hp_mod, "GAME_CLASS", "Sokoban")
        )
        ctor_kwargs = getattr(hp_mod, "CONSTRUCTOR_KWARGS", {})
        game_name = getattr(hp_mod, "GAME_NAME", "sokoban")

        # Training logic
        training_name = getattr(hp_mod, "TRAINING_LOGIC", "sokoban_training")
        training = _load_training_logic(training_name)

        # Engine hyperparams
        hyperparams_fn = hp_mod.get_hyperparams
        hp = hyperparams_fn()

        # Optimization orchestration
        phases = getattr(hp_mod, "PHASES", ["simulation"])
        primary_phase = next(
            (p for p in phases if p != "hyperparams"), "simulation"
        )

        game_factory = _make_game_factory(
            game_class, training.LEVELS, ctor_kwargs
        )

        evaluator = Evaluator(
            game_factory=game_factory,
            phase=primary_phase,
            iterations=hp["iterations"],
            max_rollout_depth=hp["max_rollout_depth"],
            exploration_weight=hp.get("exploration_weight", 1.41),
            eval_runs=training.EVAL_RUNS,
            solve_weight=training.SOLVE_WEIGHT,
            return_weight=training.RETURN_WEIGHT,
            mastery_solve_rate=training.MASTERY_SOLVE_RATE,
            mastery_confirm_runs=training.MASTERY_CONFIRM_RUNS,
            mastery_max_steps=getattr(training, "MASTERY_MAX_STEPS", None),
        )

        return cls(
            game_name=game_name,
            game_factory=game_factory,
            training=training,
            phases=phases,
            evaluator=evaluator,
            hyperparams_fn=hyperparams_fn,
            num_iters=getattr(hp_mod, "NUM_ITERS", 5),
            three_step=getattr(hp_mod, "THREE_STEP", True),
            history_window=getattr(hp_mod, "HISTORY_WINDOW", 3),
            logging=getattr(hp_mod, "LOGGING", True),
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_optimization_phase(self, iteration: int) -> str:
        """Pick which component to optimize this iteration."""
        return random.choice(self.phases)

    def _make_engine(self, level: str) -> MCTSEngine:
        """Create an MCTSEngine with current hyperparams and all tools."""
        hp = self.current_hyperparams
        g = self.game_factory(level)
        eng = MCTSEngine(
            g,
            iterations=hp["iterations"],
            max_rollout_depth=hp["max_rollout_depth"],
            exploration_weight=hp.get("exploration_weight", 1.41),
            logging=self.logging,
        )
        for phase, fn in self.current_fns.items():
            if fn is not None:
                eng.set_tool(phase, fn)
        return eng

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

        hp = self.current_hyperparams
        lines = [
            f"Current level: {current_level}",
            f"Current hyperparams: iterations={hp['iterations']}, "
            f"max_rollout_depth={hp['max_rollout_depth']}, "
            f"exploration_weight={hp.get('exploration_weight', 1.41):.3f}",
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
            f"SCORING: composite = {ev.solve_weight} × solve_rate "
            f"+ {ev.return_weight} × avg_returns",
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
            phase_info = ""
            if r.get("opt_phase"):
                phase_info = f" [{r['opt_phase']}]"
            lines.append(
                f"  Iter {r['iteration']} [{r['level']}]{phase_info}: "
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
            all_results, best_fn, current_fn, current_hyperparams,
            level_best_scores, level_baselines, mastered_levels, active_levels
        """
        ev = self.evaluator
        current_level = self.start_level

        # Sync mastered levels into active_levels
        self.active_levels = [
            lv for lv in self.levels if lv not in ev.mastered_levels
        ]

        # Create LLM optimizers — one per target phase
        optimizers: dict[str, Optimizer] = {}
        for phase in self.phases:
            optimizers[phase] = Optimizer(
                game=self.game_name,
                target_phase=phase,
                three_step=self.three_step,
                verbose=self.verbose,
            )

        # Compute initial baseline
        if self.verbose:
            hp = self.current_hyperparams
            print(f"Starting level: {current_level}")
            print(f"Hyperparams: iterations={hp['iterations']}, "
                  f"max_depth={hp['max_rollout_depth']}, "
                  f"C={hp.get('exploration_weight', 1.41):.3f}")
            print(f"Phases to optimize: {self.phases}")
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
            opt_phase = self._pick_optimization_phase(iteration)

            if self.verbose:
                print(f"\n{'#'*60}")
                print(f"  ITERATION {iteration}/{self.num_iters}, "
                      f"LEVEL={cur_level}, PHASE={opt_phase}")
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

            # --- 1. Play with current tool + hyperparams ---
            t_play_start = time.time()
            eng = self._make_engine(cur_level)
            play_result = eng.play_game()
            t_play = time.time() - t_play_start
            play_trace = play_result.get("log_file", "")
            tl = eng.get_tool_source()

            # Include hyperparams source in tool_list for LLM context
            if "hyperparams" in self.phases:
                tl["hyperparams"] = _get_hyperparams_source()

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

            # --- 3. Optimize (analysis → draft → critique) ---
            t_opt_start = time.time()
            opt = optimizers[opt_phase]
            result = opt.run(
                record_files=[play_trace] if play_trace else [],
                tool_list=tl,
                state_factory=state_factory,
                additional_context=history,
                session_tag=f"iter{iteration}_{cur_level}_{opt_phase}",
            )
            t_opt = time.time() - t_opt_start
            if self.verbose:
                print(f"  Optimize ({opt_phase}): {t_opt:.1f}s")

            # --- 4. Multi-run evaluation ---
            iter_record = {
                "iteration": iteration,
                "level": cur_level,
                "opt_phase": opt_phase,
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
                # Build extra_tools: all current tool fns except the phase being evaluated
                extra_tools = {
                    p: f for p, f in self.current_fns.items()
                    if f is not None and p != opt_phase
                } or None

                try:
                    if opt_phase == "hyperparams":
                        # fn is a get_hyperparams callable — evaluate new params
                        new_hp = fn()
                        old_hp = self.current_hyperparams.copy()
                        ev.update_hyperparams(new_hp)

                        # Eval with all current tool fns
                        all_tools = {
                            p: f for p, f in self.current_fns.items()
                            if f is not None
                        } or None
                        avg_ret, solve_rate, avg_steps, _, eval_time = ev.multi_eval(
                            None, cur_level, extra_tools=all_tools,
                        )
                        comp = ev.composite_score(solve_rate, avg_ret)
                        iter_record.update({
                            "avg_returns": avg_ret,
                            "solve_rate": solve_rate,
                            "composite": comp,
                            "avg_steps": avg_steps,
                            "eval_time": eval_time,
                        })
                        if self.verbose:
                            print(
                                f"  Eval ({ev.eval_runs} runs, {cur_level}): "
                                f"avg_returns={avg_ret:.4f}, "
                                f"solve_rate={solve_rate:.0%}, "
                                f"composite={comp:.4f}  ({eval_time:.1f}s)"
                            )

                        prev_level_best = ev.level_best_scores.get(
                            cur_level, bl["composite"]
                        )
                        if comp > prev_level_best:
                            if self.verbose:
                                print(f"  ★ NEW BEST (hyperparams) for {cur_level}")
                            ev.level_best_scores[cur_level] = comp
                            self.current_hyperparams = new_hp
                            self.hyperparams_fn = fn
                            self.best_hyperparams_fn = fn
                            iter_record["adopted"] = True
                            iter_record["is_best"] = True
                        elif comp >= reject_floor:
                            if self.verbose:
                                print(f"  → Accepted hyperparams "
                                      f"(comp={comp:.4f} ≥ floor={reject_floor:.4f})")
                            self.current_hyperparams = new_hp
                            self.hyperparams_fn = fn
                            iter_record["adopted"] = True
                        else:
                            if self.verbose:
                                print(f"  ✗ Rejected hyperparams "
                                      f"(comp={comp:.4f} < floor={reject_floor:.4f})")
                            ev.update_hyperparams(old_hp)

                        # Check mastery
                        newly_mastered = ev.check_mastery(
                            cur_level, solve_rate, avg_steps, None,
                            extra_tools=all_tools,
                        )
                        if newly_mastered and cur_level in self.active_levels:
                            self.active_levels.remove(cur_level)
                    else:
                        # Normal tool phase optimization
                        avg_ret, solve_rate, avg_steps, _, eval_time = ev.multi_eval(
                            fn, cur_level,
                            phase=opt_phase, extra_tools=extra_tools,
                        )
                        comp = ev.composite_score(solve_rate, avg_ret)
                        iter_record.update({
                            "avg_returns": avg_ret,
                            "solve_rate": solve_rate,
                            "composite": comp,
                            "avg_steps": avg_steps,
                            "eval_time": eval_time,
                        })

                        if self.verbose:
                            print(
                                f"  Eval ({ev.eval_runs} runs, {cur_level}): "
                                f"avg_returns={avg_ret:.4f}, "
                                f"solve_rate={solve_rate:.0%}, "
                                f"composite={comp:.4f}, "
                                f"avg_steps={avg_steps:.0f}  ({eval_time:.1f}s)"
                            )

                        newly_mastered = ev.check_mastery(
                            cur_level, solve_rate, avg_steps, fn,
                            phase=opt_phase, extra_tools=extra_tools,
                        )
                        if newly_mastered and cur_level in self.active_levels:
                            self.active_levels.remove(cur_level)
                            if self.verbose:
                                print(f"    {len(self.active_levels)} levels remain.")

                        prev_level_best = ev.level_best_scores.get(
                            cur_level, bl["composite"]
                        )
                        if comp > prev_level_best:
                            if self.verbose:
                                print(f"  ★ NEW BEST for {cur_level} "
                                      f"(prev={prev_level_best:.4f}) — adopting")
                            ev.level_best_scores[cur_level] = comp
                            self.best_fns[opt_phase] = fn
                            self.current_fns[opt_phase] = fn
                            iter_record["adopted"] = True
                            iter_record["is_best"] = True
                        elif comp >= reject_floor:
                            if self.verbose:
                                print(
                                    f"  → Accepted on {cur_level} "
                                    f"(comp={comp:.4f} ≥ floor={reject_floor:.4f})"
                                )
                            self.current_fns[opt_phase] = fn
                            iter_record["adopted"] = True
                        else:
                            if self.verbose:
                                print(
                                    f"  ✗ Rejected on {cur_level} "
                                    f"(comp={comp:.4f} < floor={reject_floor:.4f}) "
                                    f"— reverting to best"
                                )
                            self.current_fns[opt_phase] = self.best_fns[opt_phase]
                except Exception as exc:
                    if self.verbose:
                        print(f"  ✗ Eval crashed: {exc!r}")
                        print(f"    Rejecting {opt_phase} candidate")
                    iter_record["error"] = str(exc)
                    # Revert hyperparams if we changed them
                    if opt_phase == "hyperparams":
                        ev.update_hyperparams(old_hp)
            else:
                if self.verbose:
                    print("  Eval:  SKIPPED (smoke test failed or error)")
                    if result.get("error"):
                        print(f"         {result['error'][:120]}")

            total_time = t_play + t_opt + (iter_record["eval_time"] or 0)
            if self.verbose:
                print(f"  Iteration total: {total_time:.1f}s")
            self.all_results.append(iter_record)

            # Pick next level via training logic
            if self.active_levels:
                current_level = self.training.pick_next_level(
                    self.active_levels, self.levels, self.all_results
                )
            else:
                if self.verbose:
                    print("\n🎉 All levels mastered! Will stop next iteration.")
                current_level = random.choice(self.levels)

        # --- Summary ---
        summary = {
            "all_results": self.all_results,
            "best_fns": dict(self.best_fns),
            "current_fns": dict(self.current_fns),
            "current_hyperparams": dict(self.current_hyperparams),
            "level_best_scores": dict(ev.level_best_scores),
            "level_baselines": dict(ev.level_baselines),
            "mastered_levels": set(ev.mastered_levels),
            "active_levels": list(self.active_levels),
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Iterative optimization complete — "
                  f"{len(self.all_results)} iterations done.")
            hp = self.current_hyperparams
            print(f"Final hyperparams: iterations={hp['iterations']}, "
                  f"max_depth={hp['max_rollout_depth']}, "
                  f"C={hp.get('exploration_weight', 1.41):.3f}")
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
