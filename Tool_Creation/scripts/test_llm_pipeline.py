"""
MCTS + LLM Optimization Pipeline (game-agnostic)

All configuration comes from two files in `MCTS_tools/`:
1. hyperparams/default_hyperparams.py  — engine params (iterations,
   max_rollout_depth, exploration_weight), game identity, and optimization
   settings (phases, num_iters, etc.)
2. training_logic/<TRAINING_LOGIC>.py   — levels, mastery criteria,
   level-selection strategy

The orchestrator (OptimizationRunner.from_config()) reads both files
and drives the iterative LLM optimization loop.
"""

import importlib
import sys
import importlib.util
from pathlib import Path

# ── Setup: add Tool_Creation to sys.path ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import OptimizationRunner, Evaluator
from mcts import MCTSEngine

print(f"Working dir: {Path('.').resolve()}")
print(f"Tool_Creation root: {ROOT}")
print("All imports OK ✓")


# ── Cell 2: Play one baseline game (sanity check) ────────────────────

# Load hyperparams module (single source of config)
_hp_path = ROOT / "MCTS_tools" / "hyperparams" / "default_hyperparams.py"
_spec = importlib.util.spec_from_file_location("hp", str(_hp_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
hp = _mod.get_hyperparams()

ITERS = hp["iterations"]
MAX_DEPTH = hp["max_rollout_depth"]
PHASES = _mod.PHASES
ctor_kwargs = getattr(_mod, "CONSTRUCTOR_KWARGS", {})

# Load game class from config
game_module = importlib.import_module(
    getattr(_mod, "GAME_MODULE", "mcts.games")
)
game_class = getattr(
    game_module, getattr(_mod, "GAME_CLASS", "Sokoban")
)

# Load training logic to get start level
_tl_path = ROOT / "MCTS_tools" / "training_logic" / f"{_mod.TRAINING_LOGIC}.py"
_spec2 = importlib.util.spec_from_file_location("tl", str(_tl_path))
_tl_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_tl_mod)
LEVEL = _tl_mod.START_LEVEL

game = game_class(LEVEL, **ctor_kwargs)
engine = MCTSEngine(game, iterations=ITERS, max_rollout_depth=MAX_DEPTH, logging=True)
baseline = engine.play_game()

base_tag = "SOLVED" if baseline.get("solved") else "UNSOLVED"
print(f"Baseline ({LEVEL}): {base_tag} in {baseline.get('steps', '?')} steps  "
      f"returns={baseline.get('returns', '?')}")
print(f"Hyperparams: iterations={ITERS}, max_depth={MAX_DEPTH}, "
      f"C={hp.get('exploration_weight', 1.41)}")
print(f"Optimize phases: {PHASES}")
print(f"Trace: {baseline.get('log_file', 'N/A')}")


# ── Cell 3: Run iterative optimization ───────────────────────────────

runner = OptimizationRunner.from_config(verbose=True)
summary = runner.run()

best_fns = summary["best_fns"]
print(f"\nbest_fns: { {p: ('set' if f else 'None') for p, f in best_fns.items()} }")
print(f"Final hyperparams: {summary.get('current_hyperparams', {})}")


# ── Cell 4: Multi-Level Evaluation ───────────────────────────────────

best_fns = summary["best_fns"]
levels   = summary["active_levels"] + list(summary.get("mastered_levels", []))
ev       = runner.evaluator

# Build extra_tools from best_fns for optimized eval
opt_tools = {p: f for p, f in best_fns.items() if f is not None} or None

has_optimized = opt_tools is not None
print(f"Best fns: { {p: ('set' if f else 'None') for p, f in best_fns.items()} }")
print(f"Final hyperparams: {summary.get('current_hyperparams', {})}")
print(f"Mastered: {sorted(summary.get('mastered_levels', set()))}")
print(f"Level best scores: {summary.get('level_best_scores', {})}")

if not has_optimized:
    print("\nNo optimized functions adopted — skipping comparative eval.")
else:
    # Only eval on levels visited during optimization (have baselines)
    eval_levels = sorted(ev.level_baselines.keys())
    print(f"\nEvaluating {len(eval_levels)} levels (n=3 each)…")

    rows = []
    for lvl in eval_levels:
        avg_b, sr_b, steps_b, _, t_b = ev.multi_eval(None, lvl, n=3, logging=False)
        avg_o, sr_o, steps_o, _, t_o = ev.multi_eval(
            None, lvl, n=3, logging=False, extra_tools=opt_tools,
        )
        rows.append((lvl, sr_b, sr_o, avg_b, avg_o, steps_b, steps_o, t_b, t_o))
        print(f"{lvl}: baseline={avg_b:.3f} ({sr_b:.0%})  optimized={avg_o:.3f} ({sr_o:.0%})  "
              f"[{t_b:.1f}s + {t_o:.1f}s]")

    print(f"\n{'Level':<10} {'Base Solve%':>12} {'Opt Solve%':>12} "
          f"{'Base AvgRet':>12} {'Opt AvgRet':>12} "
          f"{'Base Steps':>11} {'Opt Steps':>11}")
    print("─" * 82)
    for lvl, sr_b, sr_o, avg_b, avg_o, steps_b, steps_o, *_ in rows:
        print(f"{lvl:<10} {sr_b*100:>11.0f}% {sr_o*100:>11.0f}% "
              f"{avg_b:>12.3f} {avg_o:>12.3f} "
              f"{steps_b:>11.1f} {steps_o:>11.1f}")
