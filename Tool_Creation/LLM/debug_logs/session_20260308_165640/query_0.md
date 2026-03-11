# query_0

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-08 16:56:40 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 0.00s |
| Status     | success |
| Tokens     | <MagicMock name='mock.usage.total_tokens' id='4516591808'> |
| Validation | invalid — Expected function 'simulate', found: ['wrong_name'] |

---

## Prompt

test prompt

---

## Response

```python
def wrong_name(x):
    return x
```
