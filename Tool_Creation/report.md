# Sokoban Value Estimator: Heuristic Placement Report

## 1. The MCTS Engine Architecture

The engine (`mcts/mcts_engine.py`) runs the standard four-phase loop on every iteration:

```
Select → Expand → Simulate → Backpropagate
```

Each phase is a **pluggable function** loaded from `MCTS_tools/<phase>/` via `tool_config.json`. The engine hot-swaps these at runtime via `set_tool(phase, fn)`. The LLM optimizer currently targets `["simulation", "expansion"]` (see `MCTS_tools/hyperparams/default_hyperparams.py:33`).

---

## 2. The Proposed Heuristic

From `sokoban_astar.ipynb`, the combined Manhattan-distanceV value estimator is:
V
```
h(state) = player_to_closest_box + sum_over_boxes(min_box_to_any_target)
```

`SokobanState` already exposes half of this at `mcts/games/sokoban.py:253`:

```python
def total_box_distance(self) -> int:
    """Sum of min Manhattan distance from each box to nearest target."""
```

The player-to-closest-box component is missing from the state API but trivially computable inline:

```python
pr, pc = state.player
player_dist = min(abs(pr - br) + abs(pc - bc) for br, bc in state.boxes)
```

---

## 3. Where the Heuristic File Lives in the 4 Stages

### Stage 1 — Selection (`MCTS_tools/selection/`)

**Current:** `default_selection.py` — pure UCB1 walk.
**Heuristic role:** UCB1's exploit term is `child.value / child.visits`, which averages backpropagated simulation returns. The heuristic does not live here directly, but a modified selection could bias toward children whose *state* has smaller heuristic distance — acting as a **prior** over actions before visits accumulate.
**File path if added:** `MCTS_tools/selection/sokoban_selection.py`
**Fit:** Weak fit. This phase operates on tree nodes, not raw states, and changing UCB1 risks breaking the exploration-exploitation balance.

---

### Stage 2 — Expansion (`MCTS_tools/expansion/`)

**Current:** `default_expansion.py` — pops an arbitrary untried action (LIFO order from `legal_actions()`).
**Heuristic role:** The heuristic can rank untried actions before popping, prioritising pushes that reduce box distance. This is **action ordering / heuristic expansion**:

```python
# order untried actions by resulting heuristic cost (ascending)
node._untried_actions.sort(key=lambda a: _heuristic_after(node.state, a), reverse=True)
action = node._untried_actions.pop()  # cheapest cost first
```

**File path:** `MCTS_tools/expansion/sokoban_expansion.py`
**Fit:** Good fit. Expansion ordering directly narrows which branch MCTS explores first. The LLM optimizer already targets this phase.

---

### Stage 3 — Simulation (`MCTS_tools/simulation/`)

**Current:** `default_simulation.py` — pure random rollout returning `state.returns()[player]` (0.0 or 1.0).
**Heuristic role:** **Primary target.** Instead of (or in addition to) random rollout, return a shaped reward that reflects how close the current state is to solved:

```python
def sokoban_simulation(state, perspective_player, max_depth):
    total_dist = state.total_box_distance()
    pr, pc = state.player
    player_dist = min(abs(pr - br) + abs(pc - bc) for br, bc in state.boxes) if state.boxes else 0
    h = total_dist + player_dist
    return 1.0 / (1.0 + h)   # in (0, 1]; 1.0 = solved
```

This gives MCTS a **dense reward signal** instead of the sparse {0, 1} terminal signal, which is the main reason random rollout fails on hard Sokoban levels.

**File path:** `MCTS_tools/simulation/sokoban_simulation.py`
**Fit:** Best fit. The simulation phase is called on every MCTS iteration and its return value is the sole reward signal backpropagated through the tree. The LLM optimizer already targets this phase.

---

### Stage 4 — Backpropagation (`MCTS_tools/backpropagation/`)

**Current:** `default_backpropagation.py` — walks up the tree adding the reward to each ancestor's `.value`.
**Heuristic role:** The reward already arrives shaped from the simulation stage. One could add a **state-value correction** here (e.g. mixing the simulation reward with a static heuristic evaluated at the leaf), but this is non-standard and couples the backprop logic to Sokoban-specific state structure.
**File path if added:** `MCTS_tools/backpropagation/sokoban_backpropagation.py`
**Fit:** Weak fit. The standard role of backprop is reward aggregation, not evaluation. Putting domain heuristics here obscures the architecture.

---

## 4. Recommended File Location

```
MCTS_tools/
├── simulation/
│   ├── default_simulation.py          ← current: random rollout
│   └── sokoban_simulation.py          ← NEW: Manhattan heuristic value estimator
└── expansion/
    ├── default_expansion.py           ← current: LIFO pop
    └── sokoban_expansion.py           ← NEW: heuristic-ordered expansion (optional)
```

Both files are within the phases the LLM optimizer already targets (`PHASES = ["simulation", "expansion"]` in `default_hyperparams.py:33`). No changes to the engine, game interface, or training logic are needed.

---

## 5. Integration with the LLM Optimizer Pipeline

The LLM optimizer (`PromptBuilder`, `orchestrator/evaluator.py`) follows a three-step loop:

| Step | What happens |
|------|-------------|
| **Analyse** | LLM reads current tool source + gameplay traces, identifies weaknesses |
| **Generate** | LLM proposes improved code for `sokoban_simulation.py` |
| **Critique** | LLM reviews the draft, fixes bugs, confirms reward spread |

The evaluator then runs `EVAL_RUNS=3` games, computes `composite_score = 0.6 × solve_rate + 0.4 × avg_returns`, and accepts the new file if it improves over baseline. The heuristic-shaped simulation reward directly improves `avg_returns` (denser signal) and, once the shaped reward guides the tree toward solutions, `solve_rate` rises too.

The `PromptBuilder` includes domain hints encouraging exactly this kind of change:

> *"Add distance-based scoring factors"* — `LLM/prompt_builder.py:399`

---

## 6. Summary

| Phase | File | Role of heuristic | Priority |
|-------|------|-------------------|----------|
| **Simulation** | `MCTS_tools/simulation/sokoban_simulation.py` | Replaces sparse {0,1} terminal reward with `1/(1 + player_dist + box_dist)` — dense value estimator | **Primary** |
| **Expansion** | `MCTS_tools/expansion/sokoban_expansion.py` | Orders untried actions by heuristic cost so MCTS explores promising pushes first | Secondary |
| Selection | — | Not recommended; risks breaking UCB1 balance | Low |
| Backpropagation | — | Not recommended; couples reward aggregation to domain logic | Low |

The heuristic code belongs in **`MCTS_tools/simulation/sokoban_simulation.py`** as a shaped-reward function, and optionally in **`MCTS_tools/expansion/sokoban_expansion.py`** as an action-ordering function. Both are already in the LLM optimizer's target phase list and will be iteratively refined across optimization runs.
