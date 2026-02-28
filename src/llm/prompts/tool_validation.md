# Tool Code Fix

The following tool code failed validation. Fix the issues.

## Original Code
```python
{original_code}
```

## Error
{error_message}

## Requirements
- Must have `__TOOL_META__` dict with "name", "type", "description"
- Must have `run()` function with correct signature for tool type
- Must not crash on any valid OpenSpiel game state
- Must return values in expected range

Output ONLY the fixed Python code, no markdown fences or explanation.
