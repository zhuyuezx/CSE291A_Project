"""
Fast heuristic evaluation — no LLM, no optimization.

Loads the current hyperparams and any installed non-default tool files from
MCTS_tools/<phase>/, then evaluates baseline vs optimized MCTS across a
configurable set of levels and prints a summary table.

Config (edit the CONFIG block below, or use CLI flags):
  --levels   level1,level2,...   (comma-separated, default: all from training logic)
  --runs     N                   games per level per variant (default: 5)
  --iters    N                   override MCTS iterations from hyperparams
  --baseline-only                skip optimized eval; just show baseline

Usage:
    python scripts/eval_heuristics.py
    python scripts/eval_heuristics.py --levels level1,level2,level3 --runs 10
    python scripts/eval_heuristics.py --baseline-only
    python scripts/eval_heuristics.py --iters 200 --runs 3
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts import MCTSEngine
from orchestrator.runner import _load_installed_tools


# ══════════════════════════════════════════════════════════════════════
# CONFIG — change these defaults without touching CLI args
# ══════════════════════════════════════════════════════════════════════
DEFAULT_RUNS = 5          # games per level per variant
DEFAULT_LEVELS = None     # None = use all levels from training logic
# ══════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Sokoban heuristics (no LLM).")
    p.add_argument(
        "--levels", type=str, default=None,
        help="Comma-separated level names, e.g. level1,level2,level3. "
             "Default: all levels from training logic.",
    )
    p.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Games per level per variant (default: {DEFAULT_RUNS}).",
    )
    p.add_argument(
        "--iters", type=int, default=None,
        help="Override MCTS iterations from hyperparams.",
    )
    p.add_argument(
        "--baseline-only", action="store_true",
        help="Only run baseline (default tools); skip optimized eval.",
    )
    p.add_argument(
        "--optimized-only", action="store_true",
        help="Only run optimized (installed tools); skip baseline eval.",
    )
    return p.parse_args()


def _load_config():
    """Load hyperparams module and training logic module."""
    hp_path = ROOT / "MCTS_tools" / "hyperparams" / "default_hyperparams.py"
    spec = importlib.util.spec_from_file_location("hp", str(hp_path))
    hp_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp_mod)

    tl_name = getattr(hp_mod, "TRAINING_LOGIC", "sokoban_training")
    tl_path = ROOT / "MCTS_tools" / "training_logic" / f"{tl_name}.py"
    spec2 = importlib.util.spec_from_file_location("tl", str(tl_path))
    tl_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tl_mod)

    return hp_mod, tl_mod


def _run_eval(
    game_class,
    ctor_kwargs: dict,
    level: str,
    hp: dict,
    n: int,
    installed_tools: dict,
    use_installed: bool,
) -> dict:
    """Play n games on a level; return aggregate stats."""
    returns_list, solved_list, steps_list = [], [], []
    t0 = time.time()
    for _ in range(n):
        g = game_class(level, **ctor_kwargs)
        eng = MCTSEngine(
            g,
            iterations=hp["iterations"],
            max_rollout_depth=hp["max_rollout_depth"],
            exploration_weight=hp.get("exploration_weight", 1.41),
            logging=False,
        )
        if use_installed:
            for phase, fn in installed_tools.items():
                if fn is not None:
                    eng.set_tool(phase, fn)
        result = eng.play_game()
        ret = result["returns"]
        returns_list.append(ret[0] if isinstance(ret, list) else ret)
        solved_list.append(bool(result.get("solved")))
        steps_list.append(result.get("steps", 0))
    elapsed = time.time() - t0
    n_solved = sum(solved_list)
    return {
        "avg_ret":    sum(returns_list) / n,
        "solve_rate": n_solved / n,
        "avg_steps":  sum(steps_list) / n,
        "solved":     n_solved,
        "total":      n,
        "elapsed":    elapsed,
    }


def _bar(solve_rate: float, width: int = 10) -> str:
    filled = round(solve_rate * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def main() -> None:
    args = _parse_args()
    hp_mod, tl_mod = _load_config()

    hp = hp_mod.get_hyperparams()
    if args.iters is not None:
        hp["iterations"] = args.iters

    # ── Game class ───────────────────────────────────────────────────
    game_module = importlib.import_module(getattr(hp_mod, "GAME_MODULE", "mcts.games"))
    game_class  = getattr(game_module, getattr(hp_mod, "GAME_CLASS", "Sokoban"))
    ctor_kwargs = getattr(hp_mod, "CONSTRUCTOR_KWARGS", {})

    # ── Levels ───────────────────────────────────────────────────────
    all_levels: list[str] = tl_mod.LEVELS
    if args.levels:
        levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    else:
        levels = list(all_levels)

    # ── Load installed (non-default) tools ───────────────────────────
    tool_phases: list[str] = [p for p in getattr(hp_mod, "PHASES", ["simulation"])
                               if p != "hyperparams"]
    state_factory = lambda: game_class(tl_mod.START_LEVEL, **ctor_kwargs).new_initial_state()
    installed = _load_installed_tools(tool_phases, state_factory=state_factory)
    has_installed = any(fn is not None for fn in installed.values())

    # ── Print config ─────────────────────────────────────────────────
    print("=" * 70)
    print("  SOKOBAN HEURISTIC EVAL")
    print("=" * 70)
    print(f"  Levels     : {levels}")
    print(f"  Runs/level : {args.runs}")
    print(f"  Iterations : {hp['iterations']}  "
          f"max_depth={hp['max_rollout_depth']}  "
          f"C={hp.get('exploration_weight', 1.41)}")
    print(f"  Tool phases: {tool_phases}")
    if has_installed:
        active = {p: fn.__module__ if fn else None for p, fn in installed.items()}
        print(f"  Installed  : { {p: ('✓' if fn else '–') for p, fn in installed.items()} }")
    else:
        print("  Installed  : none (using built-in defaults for all phases)")
    mode = "both"
    if args.baseline_only:
        mode = "baseline"
        print("  Mode       : baseline only")
    elif args.optimized_only:
        mode = "optimized"
        print("  Mode       : optimized only")
        if not has_installed:
            print("  WARNING    : no installed tools found — will use defaults anyway")
    print("=" * 70)

    # ── Evaluate ─────────────────────────────────────────────────────
    rows = []
    for lvl in levels:
        base = opt = None

        if mode in ("both", "baseline"):
            sys.stdout.write(f"  {lvl:<10} baseline ... ")
            sys.stdout.flush()
            base = _run_eval(game_class, ctor_kwargs, lvl, hp,
                             args.runs, installed, use_installed=False)
            sys.stdout.write(f"{base['solve_rate']*100:.0f}% solved")

        if mode in ("both", "optimized") and has_installed:
            sys.stdout.write(f"  {lvl:<10} optimized ... " if mode == "optimized"
                             else "  │  optimized ... ")
            sys.stdout.flush()
            opt = _run_eval(game_class, ctor_kwargs, lvl, hp,
                            args.runs, installed, use_installed=True)
            sys.stdout.write(f"{opt['solve_rate']*100:.0f}% solved")
        elif mode == "optimized":
            # no installed tools — run with defaults as fallback
            sys.stdout.write(f"  {lvl:<10} optimized (defaults) ... ")
            sys.stdout.flush()
            opt = _run_eval(game_class, ctor_kwargs, lvl, hp,
                            args.runs, installed, use_installed=True)
            sys.stdout.write(f"{opt['solve_rate']*100:.0f}% solved")

        elapsed_parts = []
        if base:
            elapsed_parts.append(f"{base['elapsed']:.1f}s")
        if opt:
            elapsed_parts.append(f"{opt['elapsed']:.1f}s")
        print(f"  ({' + '.join(elapsed_parts)})")
        rows.append((lvl, base, opt))

    # ── Table ────────────────────────────────────────────────────────
    print()
    if mode in ("baseline", "optimized") or not has_installed:
        # Single-variant table
        label = "Optimized" if mode == "optimized" else "Baseline"
        hdr = (f"{'Level':<10}  {label+' Solve%':>16}  {'AvgRet':>7}"
               f"  {'AvgSteps':>9}  {'Bar':>12}")
        print(hdr)
        print("─" * len(hdr))
        for lvl, base, opt in rows:
            stats = opt if mode == "optimized" else base
            print(
                f"{lvl:<10}  {stats['solve_rate']*100:>15.0f}%"
                f"  {stats['avg_ret']:>7.3f}"
                f"  {stats['avg_steps']:>9.1f}"
                f"  {_bar(stats['solve_rate'])}"
            )
        all_stats = [opt if mode == "optimized" else base for _, base, opt in rows]
        total_solved = sum(s["solved"] for s in all_stats)
        total_games  = sum(s["total"]  for s in all_stats)
        print("─" * len(hdr))
        print(f"{'TOTAL':<10}  {total_solved/total_games*100:>15.0f}%  "
              f"({total_solved}/{total_games} games solved)")
    else:
        # Side-by-side baseline vs optimized
        hdr = (
            f"{'Level':<10}  "
            f"{'Base%':>6}  {'BaseRet':>7}  {'BaseSteps':>9}  "
            f"│  "
            f"{'Opt%':>6}  {'OptRet':>7}  {'OptSteps':>9}  "
            f"{'Δsolve':>7}  {'ΔRet':>7}"
        )
        print(hdr)
        print("─" * len(hdr))
        total_b_solved = total_o_solved = 0
        total_games = 0
        for lvl, base, opt in rows:
            d_solve = (opt["solve_rate"] - base["solve_rate"]) * 100
            d_ret   = opt["avg_ret"] - base["avg_ret"]
            trend   = "▲" if d_solve > 0 else ("▼" if d_solve < 0 else "=")
            print(
                f"{lvl:<10}  "
                f"{base['solve_rate']*100:>5.0f}%"
                f"  {base['avg_ret']:>7.3f}"
                f"  {base['avg_steps']:>9.1f}  "
                f"│  "
                f"{opt['solve_rate']*100:>5.0f}%"
                f"  {opt['avg_ret']:>7.3f}"
                f"  {opt['avg_steps']:>9.1f}  "
                f"{d_solve:>+6.0f}%  "
                f"{d_ret:>+7.3f} {trend}"
            )
            total_b_solved += base["solved"]
            total_o_solved += opt["solved"]
            total_games    += base["total"]
        print("─" * len(hdr))
        d_total = (total_o_solved - total_b_solved) / total_games * 100
        print(
            f"{'TOTAL':<10}  "
            f"{total_b_solved/total_games*100:>5.0f}%"
            f"  ({total_b_solved}/{total_games})  "
            f"{'':>9}  │  "
            f"{total_o_solved/total_games*100:>5.0f}%"
            f"  ({total_o_solved}/{total_games})  "
            f"{'':>9}  {d_total:>+6.0f}%"
        )

    print()


if __name__ == "__main__":
    main()
