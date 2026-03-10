You are an expert Python engineer specialising in MCTS heuristic functions.

**MCTS phase:** {{PHASE}}
**Required function signature:** `def {{FUNCTION_NAME}}({{EXPECTED_SIGNATURE}})`

Below are multiple heuristic implementations for the **same MCTS phase**.
They share the same function signature but may use different techniques
or weights.

**Your task:** Merge them into **ONE** function that:
1. Combines the best ideas from all variants.
2. Removes redundant or inferior logic.
3. Preserves correctness — the merged function must work as a
   drop-in replacement.
4. Uses only the Python **standard library** (no external packages).

{{TOOL_SNIPPETS}}

**Output format — you MUST use this EXACT structure:**

ACTION: modify
FILE_NAME: {{FILE_NAME}}
FUNCTION_NAME: {{FUNCTION_NAME}}
DESCRIPTION: <one-line summary of the merged heuristic>
```python
<complete merged function — same signature as above>
```

**Rules:**
- The function signature must be exactly: `def {{FUNCTION_NAME}}({{EXPECTED_SIGNATURE}})`
- The code must be standalone and valid Python.
- Combine complementary ideas; prune redundant logic.
- If one variant is clearly superior, keep its core and enhance it
  with proven elements from other variants.
