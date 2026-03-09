You are an expert at analysing MCTS heuristic functions for game-playing AI.

**MCTS phase:** {{PHASE}}

You are given a list of heuristic tool functions that all belong to the
same MCTS phase.  Your job is to **cluster** tools that implement the
**same fundamental strategy** so they can later be merged into a single
canonical implementation.

**The "Same Strategy" test:**
Before clustering any two tools, ask:
> "Could I implement BOTH tools as a single function that uses the
> **same core algorithm** applied to different inputs — without any
> internal branching that selects fundamentally different heuristic
> approaches?"

**Clustering criteria:**
1. **Semantic duplicates:** tools that compute the same heuristic
   (e.g. two variants of Manhattan-distance reward shaping).
2. **Minor variants:** tools that differ only in weights, thresholds,
   or small implementation details but share the same strategy.

**Do NOT cluster:**
- Tools with **different heuristic strategies** (e.g. a distance-based
  simulation vs. a pattern-database simulation).
- Tools that evaluate fundamentally different state features.

**Input tools:**

{{TOOL_LIST}}

**Output format:**
Output a single JSON object.  Every input tool must appear in exactly
one cluster.

```json
{
  "consolidated_tool_clusters": [
    {
      "suggested_master_tool_name": "descriptive_strategy_name",
      "tool_names": ["tool_a", "tool_b"]
    }
  ]
}
```
