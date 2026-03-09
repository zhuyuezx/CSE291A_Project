You are an expert game-AI researcher analysing the evolution of MCTS
heuristic functions.

**MCTS phase:** {{PHASE}}
**Current game level:** {{LEVEL}}

Below is the history of heuristic tools tried for this phase, including
performance metrics, and the current best-performing tool source code.

---

{{HISTORY}}

{{RESULTS}}

{{BEST_TOOL}}

---

**Your task:** Produce a concise **STRATEGIC SUMMARY** (under 300 words,
**NO code**) that will guide the next heuristic generation.

Address each of these points:

1. **What worked:** Which strategies produced the highest composite
   scores or solve rates?  What makes them effective?

2. **What failed:** Which approaches scored poorly?  Why?  What
   patterns should the next heuristic explicitly avoid?

3. **Innovation direction:** Based on the best tool and the failure
   modes, propose ONE concrete direction for the next heuristic.
   For example: "Combine the distance weighting from iter 3 with
   the deadlock detection from iter 5."

4. **Preserved elements:** Identify specific logic from the current
   best tool that MUST be kept in the next version.

Focus on actionable, specific guidance — not generic advice.
