"""
MCTS + LLM Optimization Pipeline — Sokoban Macro-Push variant

Identical workflow to test_llm_pipeline.py, but reads configuration
from ``sokoban_macro_hyperparams.py`` instead of ``default_hyperparams.py``.

Usage:
    python scripts/test_llm_pipeline_macro.py [--output OUTPUT_FILE]
    Default output file: pipeline_macro_output.txt (in project root)
"""

import argparse
import atexit
import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

# ── Setup: add Tool_Creation to sys.path ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HYPERPARAMS_FILE = "sokoban_macro_hyperparams.py"


def _parse_args():
    p = argparse.ArgumentParser(description="Run MCTS + LLM optimization pipeline (macro-push)")
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=ROOT / "pipeline_macro_output.txt",
        help="Write stdout/stderr to this file as well as the terminal "
             "(default: pipeline_macro_output.txt)",
    )
    return p.parse_args()


class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._path = path
        self._file = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# Parse args and tee output to file
_args = _parse_args()
_tee_stdout = _Tee(sys.stdout, _args.output)
_tee_stderr = _Tee(sys.stderr, _args.output)
sys.stdout = _tee_stdout
sys.stderr = _tee_stderr


def _close_output_files():
    if hasattr(sys.stdout, "close"):
        try:
            sys.stdout.close()
        except Exception:
            pass
    if hasattr(sys.stderr, "close"):
        try:
            sys.stderr.close()
        except Exception:
            pass


atexit.register(_close_output_files)

# Clear __pycache__ in MCTS_tools to avoid hangs from stale/corrupt .pyc files
# (e.g. expansion/__pycache__ can cause exec_module to hang)
for _d in (ROOT / "MCTS_tools").rglob("__pycache__"):
    shutil.rmtree(_d, ignore_errors=True)

from orchestrator import OptimizationRunner, Evaluator
from mcts import MCTSEngine

print(f"Output also saved to: {_args.output}")
print(f"Working dir: {Path('.').resolve()}")
print(f"Tool_Creation root: {ROOT}")
print(f"Hyperparams file: {HYPERPARAMS_FILE}")
print("All imports OK ✓")


# ── Cell 2: Play one baseline game (sanity check) ────────────────────

_hp_path = ROOT / "MCTS_tools" / "hyperparams" / HYPERPARAMS_FILE
_spec = importlib.util.spec_from_file_location("hp", str(_hp_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
hp = _mod.get_hyperparams()

ITERS = hp["iterations"]
MAX_DEPTH = hp["max_rollout_depth"]
PHASES = _mod.PHASES
ctor_kwargs = getattr(_mod, "CONSTRUCTOR_KWARGS", {})

game_module = importlib.import_module(
    getattr(_mod, "GAME_MODULE", "mcts.games")
)
game_class = getattr(
    game_module, getattr(_mod, "GAME_CLASS", "SokobanMacro")
)

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

runner = OptimizationRunner.from_config(
    hyperparams_file=HYPERPARAMS_FILE, verbose=True
)
summary = runner.run()

best_fns = summary["best_fns"]
print(f"\nbest_fns: { {p: ('set' if f else 'None') for p, f in best_fns.items()} }")
print(f"Final hyperparams: {summary.get('current_hyperparams', {})}")


# ── Cell 4: Multi-Level Evaluation ───────────────────────────────────

# Use current_fns (coherent set after all iterations) not best_fns (per-phase historical mix)
current_fns = summary["current_fns"]
levels   = summary["active_levels"] + list(summary.get("mastered_levels", []))
ev       = runner.evaluator

opt_tools = {p: f for p, f in current_fns.items() if f is not None} or None

has_optimized = opt_tools is not None
print(f"Current fns (optimized eval): { {p: ('set' if f else 'None') for p, f in current_fns.items()} }")
print(f"Final hyperparams: {summary.get('current_hyperparams', {})}")
print(f"Mastered: {sorted(summary.get('mastered_levels', set()))}")
print(f"Level best scores: {summary.get('level_best_scores', {})}")

eval_levels = sorted(ev.level_baselines.keys())
if not has_optimized:
    print("\nNo optimized functions adopted — comparison will show baseline vs baseline.")
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
