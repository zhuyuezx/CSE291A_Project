# Game State, Tools, and Heuristics: A Unified Model

This doc addresses two problems:
1. **Fragmentation** — Game state, tools, and heuristics feel disconnected; it's unclear what tools to generate and why they plug into heuristics.
2. **Latency** — Relying on a slow LLM (e.g. Yunjue-style) to evaluate every prompt and generate tools takes minutes and doesn't scale.

---

## 1. The Only Interface MCTS Has: Heuristic Slots

MCTS never sees "tools." It only calls **four heuristic functions** at fixed moments:

| Slot | Input | Output | When used |
|------|--------|--------|-----------|
| `evaluation(state, player)` | state, perspective player | `float \| None` | During simulation: "Is this state good enough to stop rolling out?" |
| `rollout_policy(state)` | state | `action` | During simulation: "Which action to take in the rollout?" |
| `exploration_weight(root_visits)` | visit count | `float` | Selection: UCB1 constant |
| `action_priority(state, actions)` | state, legal actions | reordered `list[action]` | Expansion: which action to try first |

So **any** improvement (whether from a "tool" or raw code) must **become** one of these four callables. There is no other connection point.

---

## 2. What "Tools" Should Mean (No Fragmentation)

A **tool** here is not "whatever the LLM invents." It is:

- **Input:** Always the **game state** (and maybe `perspective_player` for evaluation).
- **Output:** Something one of the heuristic slots can consume:
  - A **number** → feeds into `evaluation`
  - An **action** → feeds into `rollout_policy` (or used to rank actions for `action_priority`)
  - A **score per action** → feeds into `action_priority` or a policy

So:

- **Game state** = the single source of truth (board, positions, legal moves).
- **Tool** = a function **state → value** or **state → action** (or state → scores over actions). It answers one clear question about the state.
- **Heuristic** = the slot MCTS calls; it is **implemented by** calling one or more tools and mapping their outputs to the slot’s return type.

Connection in one line:

**State → Tool(s) → value/action → Heuristic return value → MCTS uses it.**

If a piece of code doesn’t take state and produce value/action (or something trivially convertible), it doesn’t help MCTS and shouldn’t be called a “tool” in this pipeline.

---

## 3. For a Puzzle: What Tools Actually Help the “Next Step”?

The “next step” = **choose the next action** from the current state. MCTS does that by:

- **Selection/Expansion:** which node to expand (guided by `action_priority` and UCB).
- **Simulation:** from a leaf, run a rollout; at each step, `rollout_policy(state)` picks an action; optionally `evaluation(state)` stops early with a value.

So tools that help the next step are exactly those that answer:

1. **“How good is this state?”** → number  
   → Used inside `evaluation(state, player)` so MCTS can stop rollouts early or bias the tree.  
   Examples: `distance_to_goal(state)`, `boxes_on_targets(state) / num_targets`, `-manhattan(state)`.

2. **“Which action is best from this state?”** (or “rank actions”)  
   → Used inside `rollout_policy(state)` or `action_priority(state, actions)`.  
   Examples: “action that minimizes distance_to_goal”, “actions that increase boxes_on_targets first”.

So for a puzzle:

- **Subproblem 1:** “How far is state from goal?” → tool returns number → **connects to** `evaluation` (e.g. `1/(1 + distance)`).
- **Subproblem 2:** “Is state deadlocked?” → tool returns bool/number → **connects to** `evaluation` (e.g. return 0 if deadlock).
- **Subproblem 3:** “Which action reduces distance most?” → tool returns action (or scores) → **connects to** `rollout_policy` or `action_priority`.

There is no extra “why connect?” — the heuristic is the **only** place MCTS gets guidance. Tools that don’t feed into a heuristic are unused. So we **define** tools so their output type matches a slot (number for evaluation, action for rollout, ordering for action_priority), and the “connection” is: **the heuristic’s implementation calls the tool(s).**

---

## 4. A Concrete Mapping (Puzzle Example)

**Predefined tool signatures (state-in, value/action-out):**

```text
# For evaluation slot
distance_to_goal(state) -> float      # 0 = at goal, higher = worse
progress(state) -> float              # e.g. boxes on targets / total
is_deadlock(state) -> bool            # True => evaluation should return 0

# For rollout_policy / action_priority
best_action_by_distance(state) -> action   # greedy: action minimizing distance
rank_actions_by_progress(state, actions) -> list[action]
```

**Heuristics as thin wrappers (connection in code):**

```python
def evaluation(state, perspective_player):
    if is_deadlock(state):
        return 0.0
    return 1.0 / (1.0 + distance_to_goal(state))

def rollout_policy(state):
    return best_action_by_distance(state)  # or random if tool fails
```

So: **game state** is input to tools; **tools** return numbers/actions; **heuristics** are the only interface and they just call tools. No conceptual gap.

---

## 5. Reducing Latency and Fragmentation (Practical)

### 5.1 Don’t let the LLM “invent” tools from scratch every time

- **Predefine a small tool library per game:** e.g. `distance_to_goal`, `is_deadlock`, `progress`, `best_action_by_distance` with **fixed signatures** (state → float or state → action).
- **Implement them once** (by you or by one LLM call per tool, then freeze). No “minutes per prompt” in the inner loop.
- **Let the LLM only choose/combine:** given the list of available tools and their types, the LLM only writes the **heuristic body** that calls them (e.g. “use distance_to_goal and is_deadlock to return a float”). That’s a small, fast prompt (or even a template).

### 5.2 One-shot tool generation, many fast evaluations

- **Phase 1 (slow, rare):** One or few LLM calls to **generate** tool implementations (or confirm which of the predefined ones to use). Store them in a registry.
- **Phase 2 (fast, repeated):** No LLM. MCTS runs with heuristics that **call** those tools; you only evaluate win/solve rate. Optimization loop can try different **combinations** or **weights** without calling the LLM again (e.g. grid search over weights, or a tiny local search).

### 5.3 “Connection” as a single, small prompt

- Give the LLM: (1) game description, (2) **list of available tools** with signatures and one-line descriptions, (3) slot contract (“you must return float in [0,1]”), (4) a few example states or traces.
- Ask: “Write **only** the body of `evaluation(state, perspective_player)` that calls these tools and returns a float.” No “invent a tool” — only “connect these tools to this slot.” Shorter, more focused, fewer failures and faster.

### 5.4 Optional: small model for combination, big model only for new tools

- **Combination/heuristic body:** small, fast model (or template).
- **New tool design:** only when you add a new game or new subproblem; then use the slow model once and cache the result.

---

## 6. Summary

| Problem | Approach |
|--------|----------|
| **“Tools and heuristics don’t connect”** | Define tools as **state → value/action** only. Heuristics are the **only** interface MCTS has; they are implemented by calling tools. So tools must be designed to feed a slot (evaluation = numbers, rollout/priority = action or action order). |
| **“What tools for the next step?”** | Precisely: (1) state quality → number (for `evaluation`), (2) best/ranked action (for `rollout_policy` / `action_priority`). Those are the only ways to improve the next move. |
| **“Why connect tools to heuristics?”** | Because MCTS only sees heuristics. Any tool that doesn’t feed into a heuristic is unused. The connection is literal: heuristic code calls the tool and returns its result (or a simple function of it). |
| **“LLM too slow”** | Predefine tool set and signatures; implement once (or rarely). LLM only **combines** tools into heuristic bodies (small prompt). Do many evaluations (games) without LLM; use slow LLM only for rare “new tool” or “new game” steps. |

This keeps the pipeline clear: **state → tools (state → value/action) → heuristics (call tools, return what MCTS needs) → MCTS.** No fragmentation, and latency stays in check by minimizing and batching LLM use.
