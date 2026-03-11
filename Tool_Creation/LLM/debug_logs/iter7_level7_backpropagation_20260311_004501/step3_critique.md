# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:45:33 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.08s |
| Status     | success |
| Tokens     | 5098 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

PHASE: backpropagation
  • What it does: Sends the simulation result back up the visited path. Updates node statistics (visits, value) that selection's UCB1 uses.
  • Optimization goal: Control HOW strongly rollout evidence affects node values. Calibrate depth discount, solved vs partial progress, path length.
  • Constraints: Only aggregates evidence — no move generation, no deadlock pruning, no rollout policy. Must stay coherent with selection's expectations.
  • Good patterns: depth discount so shorter plans dominate, weight solved outcomes above partial progress, reduce credit for noisy weak rollouts.
  • Avoid: move generation, deadlock pruning, rollout action-choice policy.

APPROACH — 70 / 30 RULE:
  ~70% of iterations: INCREMENTAL OPTIMIZATION
    • Start from the CURRENT code.
    • Make targeted, gradual improvements (add a check,
      tweak weights, add a heuristic factor, etc.).
    • Prefer building on what works rather than replacing it.

  ~30% of iterations: PARADIGM SHIFT (when warranted)
    • If the current approach is fundamentally flawed or
      plateauing, you may propose a larger restructure.
    • Explain clearly WHY a rewrite is needed.
    • Even rewrites should keep proven components that work.

GENERAL RULES:
  • Write clean, well-structured code — as long as the
    heuristic needs to be (no artificial line limit).
  • Each iteration builds on the previous version.
  • Complex heuristics with multiple factors are encouraged
    when they improve play quality.

------------------------------------------------------------
GAME RULES
------------------------------------------------------------
Game: Sokoban (Macro-Push Variant)

== Overview ==
Sokoban is a single-player puzzle game. The player pushes boxes onto
target positions inside a grid-based warehouse. This variant uses
MACRO-PUSH actions: instead of single-step movement (UP/DOWN/LEFT/RIGHT),
each action represents a complete box push from any position the player
can walk to in the current state.

== Symbols ==
  #   wall (impassable)
  .   target position
  $   box
  *   box on a target
  @   player
  +   player standing on a target
  (space)  empty floor

== Rules ==
1. The player's reachable region is all floor cells connected to the
   player's current position by paths not blocked by walls or boxes
   (computed via BFS flood-fill).
2. An action is a (player_pos, direction) tuple where:
   - player_pos is a cell in the reachable region
   - direction is UP(0), DOWN(1), LEFT(2), or RIGHT(3)
   - The cell at player_pos + delta(direction) must contain a box
   - The cell at player_pos + 2*delta(direction) must be free
3. Applying an action:
   a. The player walks from current position to player_pos via BFS
      shortest path (costs len(path) steps).
   b. The player pushes the box one cell in the push direction
      (costs 1 step). Player ends at the box's original position.
4. Total step cost per action = walk_steps + 1.
5. The action space varies per state — different states may have
   different numbers of available pushes.
6. The puzzle is solved when ALL boxes are on target positions.
7. A game is lost when a box is in deadlock (e.g., corner not on target),
   or when max_steps is reached.

== Actions ==
The action space is a list of (player_pos, direction) tuples.
  player_pos = (row, col) — where the player must stand to push
  direction = 0(UP), 1(DOWN), 2(LEFT), 3(RIGHT) — push direction

Example: ((2, 3), 1) means "walk to (2,3) then push DOWN" — the box
at (3,3) moves to (4,3), player ends at (3,3).

== State Representation ==
Each state is described by:
  - The player's (row, col) position.
  - The set of (row, col) positions of all boxes.
  - A step counter (game terminates if max_steps is reached).
  - Derived metrics: boxes_on_targets count and total_box_distance
    (sum of Manhattan distance from each box to its nearest target).

== Reward ==
  - Solved (all boxes on targets): 1.0
  - Deadlocked or max_steps: 0.0

== GameState API ==
Public attributes (via properties):
  walls     : frozenset[tuple[int,int]]   – wall positions
  targets   : frozenset[tuple[int,int]]   – target positions
  boxes     : set[tuple[int,int]]         – current box positions
  player    : tuple[int,int]              – current player position
  height, width : int                     – grid dimensions
  num_targets   : int                     – number of targets
  steps, max_steps : int                  – current / maximum step count

Public methods:
  clone()            → new independent copy of the state
  legal_actions()    → list[tuple[tuple[int,int], int]]  (macro-push tuples)
  apply_action(a)    → None (mutates the state in-place)
  is_terminal()      → bool
  returns()          → list[float]
  current_player()   → int (always 0)
  state_key()        → str (hashable key for transposition)
  boxes_on_targets() → int
  total_box_distance() → int

== Key Strategic Concepts ==
  - Every action is a box push — think about WHICH box to push and
    from WHERE, not about walking directions.
  - Avoid pushes that create deadlocks (box in non-target corner,
    box against wall with no reachable target).
  - Plan push ORDER: placing one box may block paths needed for others.
  - Consider the cost of walking: a push requiring a long walk costs
    more steps than a nearby push.
  - Minimize total push count; fewer pushes usually means fewer
    opportunities for deadlock.


------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fixed attribute handling for `MCTSNode` (which uses `__slots__`) by removing illegal `best`/`wins` assignments and preserving the original boost‑for‑solved‑leaf logic.
Generated:   2026-03-11T00:40:19.410251
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements retained:
      • No per‑step discounting (the simulation already encodes distance).
      • A tiny epsilon boost is added to solved rewards so that they remain
        noticeable even after many low‑reward visits.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Small boost for a solved leaf – guarantees it stays above any
    # non‑solved averaged value even after many visits.
    _SOLVED_EPS = 1e-3

    # ------------------------------------------------------------------
    # 1. Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- reward handling ------------------------------------------
        # Add epsilon boost only for a solved leaf (reward == 1.0).
        reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the parent (rootward)
        node = node.parent
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|---------------------------|--------------------------|
| **1** | *Almost all children receive an average ≈ 0.002‑0.005* despite some moves immediately placing a box on a target (e.g. actions with 1‑visit avg ≈ 0.58‑0.72). | The back‑propagation treats every rollout reward equally, so the few high‑reward visits are drowned by the huge mass of low‑reward visits. The averaged value that later decides the real move is therefore almost indistinguishable from “bad”. |
| **2** | *Solved leaves get only a microscopic boost (`_SOLVED_EPS = 1e‑3`).* After a single solved rollout the child’s value is ≈ 0.001 + 1 = 1.001, but after a few hundred cheap rollouts the average collapses back toward 0. | The boost is far too small to keep a solved branch dominant over the flood of noisy, low‑reward simulations. The algorithm can easily “forget” a winning line. |
| **3** | *No depth‑dependent discount.* A rollout that reaches a solved state after 30 pushes contributes the same 1.0 (plus ε) as a solved state after 2 pushes. because the simulation already penalises distance only weakly, the back‑propagation does not amplify short solutions. | MCTS should favour shorter plans; otherwise the tree keeps expanding deep “flat” branches (see the 500‑visit child on `(3,6)` that never advances). |
| **4** | *Value is only used for tie‑breaking in selection.* The A*‑guided selection picks the child with the lowest *f = g + h*; value influences the choice only when two children have identical f. Thus a badly averaged value never helps to prefer a promising child, and the search wastes visits on low‑potential nodes. | The back‑propagation’s output is effectively ignored by the dominant selection rule, so any improvement must make the value itself more informative for the final move choice. |

**2. ROOT CAUSE**

The back‑propagation routine:

```python
while node:
    node.visits += 1
    reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward
    node.value = ((node.visits-1) * node.value + reward_adj) / node.visits
    node = node.parent
```

* **Averaging alone** forces every low‑reward rollout to pull the average down, drowning out the sparse high‑reward signals.
* **Fixed ε** (1 × 10⁻³) is negligible compared with the typical reward scale (≈ 0.0‑0.2 for non‑solved states), so solved leaves cannot stay “above” the mass of low values.
* **No depth awareness** – the same `reward_adj` is added at every level, regardless of how many pushes were required to obtain it.
* **The value is never used for selection** except as a tie‑breaker, so even a perfectly calibrated average would have little impact unless it also drives the final move‑choice rule.

These design choices make the back‑propagation ineffective at steering the search toward short, high‑progress paths.

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement (recommended).**  
The current framework works (A*‑guided selection, dead‑lock pruning, etc.). We only need a smarter statistic that survives the flood of cheap rollouts and that rewards short, solved trajectories.

Add **three lightweight modifications** to `default_backpropagation`:

1. **Depth‑discounted reward**  
   Compute the depth of the node being updated (distance from the leaf) and apply a discount factor γ ∈ (0, 1) (e.g. 0.95).  
   ```python
   depth = 0
   while node:
       discount = gamma ** depth
       reward_adj = (reward + solved_eps) * discount if reward == 1.0 else reward * discount
       …
       depth += 1
   ```
   This gives near‑term nodes a larger contribution, encouraging the algorithm to prefer shorter solutions.

2. **Stronger solved‑leaf boost**  
   Replace the tiny ε (1e‑3) with a *relative* boost proportional to the remaining step budget, e.g. `solved_bonus = 0.1 * (max_steps - leaf.steps) / max_steps`. This makes a solved leaf noticeably better than any unsolved average, even after many visits.

3. **Maintain a cumulative reward sum** (optional but cheap)  
   Store `node.total_reward` (add a new slot) and compute `node.value = node.total_reward / node.visits`. Updating a sum avoids repeated multiplication and keeps the average numerically stable. The added slot does not interfere with other phases.

These changes preserve the existing loop, respect the `__slots__` layout (by only adding a field that can be pre‑declared once), and require no alteration to selection or simulation. The result is:

* **Higher‑value children for short, solved branches**, so the final move choice (usually “pick child with highest value”) will select them.
* **Depth discount** keeps the tree from over‑investing in deep, flat branches, reducing the 500‑visit `(3,6)` pattern.
* **Larger solved boost** guarantees that once a complete solution is found, it dominates the average even after thousands of cheap rollouts.

No restructuring of the overall MCTS pipeline is needed; the improvements directly address the three highest‑impact weaknesses identified above.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root with depth discounting
    and a stronger solved‑leaf boost.

    Enhancements:
      • Depth‑discount (γ = 0.95) – rewards closer to the leaf count more,
        encouraging shorter solution paths.
      • Solved‑leaf bonus – a relative boost proportional to remaining
        step budget, plus a tiny epsilon, so a solved branch stays above
        the sea of low‑reward rollouts.
      • Keeps the simple running‑average update used elsewhere.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    _SOLVED_EPS   = 1e-3          # tiny constant boost (keeps >0 after many rolls)
    _SOLVED_WEIGHT = 0.10         # relative solved‑bonus factor
    _GAMMA        = 0.95          # depth‑discount factor (0 < γ < 1)

    depth = 0                     # distance from the leaf (0 for the leaf itself)

    while node is not None:
        # ---- 1️⃣ Increment visit count ---------------------------------
        node.visits += 1

        # ---- 2️⃣ Compute reward adjustment -------------------------------
        reward_adj = reward

        # If leaf was solved, add a bonus that scales with how early the
        # solution was found (more steps left → larger bonus).
        if reward == 1.0:
            try:
                max_steps = node.state.max_steps
                steps = node.state.steps
                # Guard against division‑by‑zero; fall back to a fixed weight.
                if max_steps > 0:
                    bonus = _SOLVED_WEIGHT * (max_steps - steps) / max_steps
                else:
                    bonus = _SOLVED_WEIGHT
            except Exception:
                # In case the state does not expose steps / max_steps,
                # fall back to a fixed bonus.
                bonus = _SOLVED_WEIGHT
            reward_adj = reward + _SOLVED_EPS + bonus

        # Apply depth discount: the farther we are from the leaf, the
        # smaller the contribution.
        discount = _GAMMA ** depth
        reward_adj *= discount

        # ---- 3️⃣ Running average update ---------------------------------
        # new_avg = ((n‑1) * old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # ---- 4️⃣ Move upward ------------------------------------------------
        node = node.parent
        depth += 1
```

------------------------------------------------------------
TASK — CRITIQUE & FINALIZE
------------------------------------------------------------
Review the DRAFT code above for critical issues.

Check for:
  1. BUGS — API misuse, crashes, wrong variable names
  2. SPEED — unnecessary clones or deep loops (runs 1000s of times)
  3. REWARD SPREAD — does the return value vary across states?

RULES:
  - Preserve the INTENT of the draft (incremental fix or
    restructure). Do NOT add unrelated features.
  - Fix bugs and performance issues. Improve clarity.
  - If the draft is correct and fast, output it UNCHANGED.

You MUST format your response EXACTLY as follows:

CRITIQUE:
<1-3 bullet points, or 'No issues found'>

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line summary>
```python
<complete final function code>
```

Rules for the header fields:
- ACTION must be either 'create' or 'modify'.
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

CRITIQUE:
- The solved‑bonus is computed for every ancestor node, but it should only apply at the leaf where the solution was found.
- Recomputing `_GAMMA ** depth` each iteration is unnecessary; a running multiplier is cheaper.
- The code assumes every node has a `state` attribute with `steps` and `max_steps`. Guard this more robustly and compute the bonus once from the leaf node.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: Fix solved‑bonus scope, use iterative discount, and safely acquire leaf step information.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root with depth discounting
    and a stronger solved‑leaf boost.

    Enhancements:
      • Depth‑discount (γ = 0.95) – rewards closer to the leaf count more,
        encouraging shorter solution paths.
      • Solved‑leaf bonus – a relative boost proportional to remaining
        step budget, applied **only at the leaf** where the solution was
        found, so it dominates after many visits.
      • Keeps the simple running‑average update used elsewhere.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    _SOLVED_EPS   = 1e-3          # tiny constant boost (keeps >0 after many rolls)
    _SOLVED_WEIGHT = 0.10         # relative solved‑bonus factor
    _GAMMA        = 0.95          # depth‑discount factor (0 < γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣ Capture leaf‑specific information (if available)
    # ------------------------------------------------------------------
    leaf_steps = None
    leaf_max_steps = None
    try:
        # `node` is the leaf when the function is called.
        leaf_steps = node.state.steps          # type: ignore[attr-defined]
        leaf_max_steps = node.state.max_steps  # type: ignore[attr-defined]
    except Exception:
        # If the node does not expose those attributes we simply skip the
        # solved‑bonus, relying on the epsilon boost.
        pass

    # ------------------------------------------------------------------
    # 2️⃣ Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    depth = 0          # distance from the leaf (0 for the leaf itself)
    discount = 1.0     # γ ** depth, updated iteratively

    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- reward handling -----------------------------------------
        reward_adj = reward

        # Apply solved‑bonus only at the leaf (depth == 0) and only if the
        # leaf is a solved state (reward == 1.0).
        if depth == 0 and reward == 1.0:
            bonus = _SOLVED_WEIGHT
            if leaf_max_steps and leaf_max_steps > 0:
                # Scale bonus by remaining steps ratio.
                bonus *= (leaf_max_steps - leaf_steps) / leaf_max_steps
            reward_adj = reward + _SOLVED_EPS + bonus

        # ---- depth discount -------------------------------------------
        reward_adj *= discount

        # ---- running average update -----------------------------------
        # new_avg = ((n‑1) * old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # ---- move upward -----------------------------------------------
        node = node.parent
        depth += 1
        discount *= _GAMMA   # equivalent to γ ** depth for the next level
```
