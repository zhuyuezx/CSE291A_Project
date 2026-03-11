# Plan: Multiple Smoke Tests + 5 Repair Attempts

## Problem

1. **Single smoke test**: The optimizer runs one call per phase. A function can pass for one state but fail for another (e.g. `legal_actions()` empty, different state shape).
2. **One repair then skip**: `max_repair_attempts` defaults to 1. If the first repair fails, we give up immediately.

## Goal

1. Run **multiple smoke tests** (different states/inputs) per phase.
2. **Repair up to 5 times** until all tests pass.

---

## Part 1: Multiple Smoke Tests

### Test scenario generation

Add a helper that builds several test scenarios from `state_factory`:

```python
def _build_smoke_test_scenarios(
    state_factory: Callable,
    phase: str,
    num_scenarios: int = 4,
) -> list[tuple[list, str]]:
    """
    Build (args_list, label) for each scenario.
    Returns e.g. [(args_0, "initial"), (args_1, "after_1_move"), ...]
    """
```

**Scenarios per phase:**

| Phase          | Scenario 0        | Scenario 1–3                                      |
|----------------|-------------------|---------------------------------------------------|
| selection      | root=Node(s0)     | root=Node(s1), Node(s2), Node(s3) — varied states |
| expansion      | node=Node(s0)     | node=Node(s1), Node(s2), Node(s3)                 |
| simulation     | state=s0          | state=s1, s2, s3                                  |
| backpropagation| node=Node(s0), r=0.5 | node=Node(s1), r=0.0; Node(s2), r=1.0; Node(s3), r=0.3 |

**State variety:** `s0 = state_factory()`, then `s1 = clone(s0) + 1 random legal move`, `s2 = clone(s0) + 2 moves`, etc. Stop if terminal or no legal actions. Use `random.seed(42)` for reproducibility.

### Changes to `_smoke_test`

- Accept `state_factory` and build scenarios via `_build_smoke_test_scenarios`.
- Loop over scenarios; on first failure, return `(None, False, error, tb)`.
- If all pass, return `(fn, True, None, None)`.
- When `state_factory is None`, keep current behavior: load only, no calls.

### File: `LLM/optimizer.py`

- Add `_build_smoke_test_scenarios`.
- Add `NUM_SMOKE_SCENARIOS = 4` (or 3–5).
- Refactor `_smoke_test` to run all scenarios.
- Ensure `_repair` receives the first failure’s error/tb for the repair prompt.

---

## Part 2: Repair Up to 5 Times

### Changes

1. **`LLM/optimizer.py`**
   - Change `max_repair_attempts` default from `1` to `5`.
   - Keep repair loop behavior: if a repair passes all smoke tests, return success; otherwise continue.

2. **`orchestrator/runner.py`**
   - When creating `Optimizer`, pass `max_repair_attempts` from config.
   - Add to hyperparams module: `MAX_REPAIR_ATTEMPTS = 5` (or read from `getattr(hp_mod, "MAX_REPAIR_ATTEMPTS", 5)`).

3. **`MCTS_tools/hyperparams/default_hyperparams.py`** (and macro variant)
   - Add `MAX_REPAIR_ATTEMPTS = 5`.

### Optional: Repair prompt improvement

- Include which scenario failed (e.g. `"Scenario 'after_2_moves' failed"`).
- Keep repair prompt concise; scenario label is optional.

---

## Implementation order

| Step | Task | File(s) |
|------|------|---------|
| 1 | Add `_build_smoke_test_scenarios` | `LLM/optimizer.py` |
| 2 | Refactor `_smoke_test` to run multiple scenarios | `LLM/optimizer.py` |
| 3 | Add `MAX_REPAIR_ATTEMPTS` to hyperparams | `default_hyperparams.py`, `sokoban_macro_hyperparams.py` |
| 4 | Change `max_repair_attempts` default to 5 | `LLM/optimizer.py` |
| 5 | Pass `max_repair_attempts` from runner when creating Optimizer | `orchestrator/runner.py` |
| 6 | Smoke test `_load_installed_tools` in runner (optional) | `orchestrator/runner.py` — `_smoke_test_fn` could use same multi-scenario logic |

---

## Edge cases

- **Empty legal_actions**: When building `s1`, `s2`, … if `legal_actions()` is empty, skip that scenario or use the previous state. Avoid `random.choice([])`.
- **State factory not available**: `state_factory is None` → keep current behavior (load only, no calls).
- **Hyperparams phase**: `get_hyperparams()` has no state; keep single smoke test (call it).
- **Reproducibility**: Use `random.seed(42)` before building scenario states so runs are deterministic.

---

## Summary

| Before | After |
|--------|-------|
| 1 smoke test per phase | 4 scenarios per phase |
| 1 repair attempt | 5 repair attempts |
| Skip on first repair failure | Retry repair up to 5 times until all tests pass |
