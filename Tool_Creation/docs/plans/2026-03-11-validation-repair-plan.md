# Plan: Validation-Failure Repair

## Problem

When the LLM generates code with a **validation error** (e.g. wrong parameter name: `node` instead of `root` for selection), the pipeline returns immediately and never attempts repair:

```
Validation failed: ["Function 'default_selection' param 0 is 'node', expected 'root'..."]
Eval: SKIPPED (smoke test failed or error)
```

The repair loop is only triggered when the **smoke test** fails (code installs and loads, but crashes when called). Validation failures occur before install, so repair is never reached.

## Root Cause

In `LLM/optimizer.py`, the flow is:

1. Parse & validate
2. **If validation fails → `return out`** (no repair)
3. Install
4. Smoke test (with repair loop)

Validation failures are "hard stops" with no recovery path.

## Plan

### 1. Add Validation-Failure Repair Loop

When validation fails, instead of returning immediately:

- Run a repair loop (up to `max_repair_attempts`) similar to smoke-test repair
- Send a targeted prompt to the LLM with:
  - The broken code
  - The validation errors (e.g. "param 0 is 'node', expected 'root'")
  - The expected signature for the phase (from `EXPECTED_SIGNATURES`)
  - Instruction: fix the signature/code to match the expected API

### 2. Implementation Sketch

**New method: `_repair_validation(parsed, validation_errors, llm_result, attempt)`**

- Build prompt: "Your code failed validation. Errors: ... Expected signature: ... Fix and return the corrected code."
- Query LLM
- Parse response, re-validate
- If valid → proceed to install + smoke test
- If still invalid and attempts remain → retry

**Modify `run()` flow:**

```python
validation = self.manager.validate(parsed, phase=self.target_phase)
if not validation["valid"]:
    # NEW: Attempt validation repair
    for attempt in range(self.max_repair_attempts):
        repair_result = self._repair_validation(
            parsed, validation["errors"], llm_result, attempt
        )
        if repair_result is None:
            continue
        repair_parsed = repair_result["parsed"] or parse(repair_result["response"])
        parsed.update(repair_parsed)
        validation = self.manager.validate(parsed, phase=self.target_phase)
        if validation["valid"]:
            break
    if not validation["valid"]:
        out["error"] = f"Validation failed after {self.max_repair_attempts} repair attempts."
        return out
# Continue to install + smoke test...
```

### 3. Repair Prompt Content

For validation repair, the prompt should include:

- Broken code
- Validation errors (exact messages)
- Expected signature: `EXPECTED_SIGNATURES[phase]`
- Phase-specific note: "For selection, first param must be named `root` (not `node`). For expansion/backpropagation, first param must be `node`."
- Same structured output format

### 4. Shared Repair Logic (Optional)

Consider extracting common repair logic:

- `_repair_validation()` — fix signature/parse errors
- `_repair_smoke()` (existing `_repair`) — fix runtime errors

Both use similar structure: build prompt, query, parse, validate/retest.

### 5. Files to Modify

| File | Change |
|------|--------|
| `LLM/optimizer.py` | Add `_repair_validation()`; wrap validation-fail block in repair loop |
| `LLM/tool_manager.py` | Export `EXPECTED_SIGNATURES` (already imported by optimizer) |

### 6. Edge Cases

- **Parse failure in repair response**: Treat as repair failure, retry
- **Repair produces different validation error**: Retry with new errors in prompt
- **max_repair_attempts**: Use same limit as smoke-test repair (default 5)

### 7. Expected Signatures Reference

```
selection:       ["root", "exploration_weight"]
expansion:       ["node"]
simulation:      ["state", "perspective_player", "max_depth"]
backpropagation: ["node", "reward"]
```

The LLM often uses `node` for selection (semantically similar to `root`) but the engine expects `root`. The repair prompt should explicitly state the required parameter names.
