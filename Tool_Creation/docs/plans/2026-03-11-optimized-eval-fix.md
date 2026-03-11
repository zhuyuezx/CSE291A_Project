# Fix Plan: Optimized Eval Worse Than Baseline

## Problem

The final evaluation shows optimized results **worse** than baseline on some levels (e.g. level8, level9, level10: baseline 100%, optimized 0%). Two root causes:

1. **best_fns vs current_fns mismatch** — The final eval uses `best_fns`, but adoption updates `current_fns` more often. Tools accepted via `comp >= reject_floor` (without new best) update only `current_fns`, not `best_fns`. So the final eval can use an **incoherent mix** of tools from different iterations.
2. **Per-phase optimization ignores cross-phase synergy** — Each phase is optimized independently on different levels. A selection tuned for level5 may conflict with an expansion tuned for level7 when run together.

---

## Root Cause 1: best_fns vs current_fns

### Current logic (runner.py)

```python
if comp > prev_level_best:
    self.best_fns[opt_phase] = fn
    self.current_fns[opt_phase] = fn
elif comp >= reject_floor:
    self.current_fns[opt_phase] = fn   # best_fns NOT updated
else:
    self.current_fns[opt_phase] = self.best_fns[opt_phase]
```

- **best_fns** is updated only when `comp > prev_level_best`.
- **current_fns** is updated on both "new best" and "accept (floor)".
- The pipeline’s final eval uses `best_fns` (test_llm_pipeline.py line 155).

So when we accept a tool with `comp >= floor` but not a new best, `current_fns` gets the new tool but `best_fns` keeps the old one. The final eval uses `best_fns`, which can be a mix of tools from different iterations that were never evaluated together.

### Fix 1: Use current_fns for final eval

**File:** `scripts/test_llm_pipeline.py`

Change:
```python
opt_tools = {p: f for p, f in best_fns.items() if f is not None} or None
```
to:
```python
# Use current_fns (coherent set after all iterations) not best_fns (per-phase historical mix)
opt_tools = {p: f for p, f in summary["current_fns"].items() if f is not None} or None
```

**Rationale:** `current_fns` is the tool set after all iterations. It is the coherent combination we actually ended with. `best_fns` mixes tools from different iterations and can be inconsistent.

---

## Root Cause 2: Per-phase tools can harm each other

### How the 4 phases interact

| Phase | Role | Interaction |
|-------|------|-------------|
| **Selection** | Picks which node to expand (UCB1) | Uses `node.visits`, `node.value` from backprop; favors nodes with high value/visits |
| **Expansion** | Adds one child; orders actions | Produces children that selection will later choose among |
| **Simulation** | Rollout from leaf; returns reward | Reward flows into backprop; affects value used by selection |
| **Backpropagation** | Updates visits/value up the tree | Feeds selection’s UCB1 scores |

Changing one phase changes the behavior of others. For example:
- A new **expansion** that orders actions differently changes which children exist and how selection explores.
- A new **backpropagation** that weights rewards differently changes which branches selection favors.
- A new **selection** can favor different branches than the expansion/simulation were tuned for.

### Current adoption logic

We evaluate: `new_fn(opt_phase) + extra_tools(current_fns \ opt_phase)` on the **current level only**. We do not check impact on other levels.

So we can:
- Accept a backpropagation that helps level6 but hurts level8.
- Accept an expansion that helps level7 but hurts level9.
- End up with a combination that is worse than baseline on several levels.

### Fix 2: Cross-level regression check (optional)

When adopting a new tool (`comp >= reject_floor` or `comp > prev_level_best`), run a short check on 2–3 other levels (e.g. previously mastered or high baseline). If we see a large regression (e.g. 100% → 0%), either:
- Reject the adoption, or
- Adopt but log a warning.

**Implementation sketch:**
```python
# After deciding to adopt (comp >= floor or comp > prev)
if adopted:
    sample_levels = [l for l in ev.level_baselines if l != cur_level][:3]
    for sl in sample_levels:
        _, sr, _, _, _ = ev.multi_eval(fn, sl, n=2, phase=opt_phase, extra_tools=extra_tools)
        bl_sr = ev.level_baselines[sl]["solve_rate"]
        if bl_sr >= 0.5 and sr < 0.5:  # significant regression
            if self.verbose:
                print(f"  ⚠️ Regression on {sl}: {bl_sr:.0%} → {sr:.0%}")
            # Option: revert, or just warn
```

---

## Fix 3: Align best_fns with current_fns on floor-accept (optional)

To keep `best_fns` and `current_fns` in sync when we accept via floor:

```python
elif comp >= reject_floor:
    self.current_fns[opt_phase] = fn
    self.best_fns[opt_phase] = fn   # Keep best_fns in sync
```

Then `best_fns` always matches the last accepted tool set. The final eval can keep using `best_fns`, and it will match `current_fns`.

**Trade-off:** We lose the notion of “best per phase from when it achieved new best.” That may be acceptable if we prefer a single coherent tool set.

---

## Summary of recommended changes

| Priority | Change | File | Effect |
|----------|--------|------|--------|
| **P0** | Use `current_fns` for final eval | `test_llm_pipeline.py` | Final eval uses the coherent tool set we ended with |
| **P1** | Sync `best_fns` with `current_fns` on floor-accept | `orchestrator/runner.py` | Keeps best_fns consistent with current_fns |
| **P2** | Cross-level regression check when adopting | `orchestrator/runner.py` | Reduces adopting tools that hurt other levels |
| **P3** | Document phase interaction | `docs/` or code comments | Clarifies how phases depend on each other |

---

## Phase synergy (for documentation)

The four phases form a pipeline:

1. **Selection** chooses a leaf using UCB1: `value/visits + C * sqrt(log(parent_visits)/visits)`.
2. **Expansion** adds one child; action order affects which branch is explored first.
3. **Simulation** runs a rollout and returns a reward.
4. **Backpropagation** propagates reward and visit counts up the tree.

For good performance:
- Selection should favor promising branches (high value).
- Expansion should order actions so that promising ones are tried early.
- Simulation should produce rewards that reflect true state quality.
- Backpropagation should aggregate rewards in a way that guides selection.

Changing one phase can help on some levels and hurt on others. Cross-level checks and using a single coherent tool set (`current_fns`) help avoid regressions.
