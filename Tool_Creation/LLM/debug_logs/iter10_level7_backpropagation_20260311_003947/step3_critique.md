# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:40:14 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 6.30s |
| Status     | success |
| Tokens     | 4798 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

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
Description: No changes needed; the draft implementation is correct and efficient.
Generated:   2026-03-11T00:35:53.297559
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Improvements over the previous version:
      • Do **not** overwrite the A*‑style visited table used during
        expansion. The table stores a g‑score based on pushes + walk
        cost, while the leaf node only knows the total step count.
        Updating it here caused mismatched pruning.  We therefore
        leave the visited dict untouched (it is already maintained
        by ``default_expansion``).
      • Apply an exponential discount based on the *actual number of
        steps* taken from the root, not on abstract push depth.  This
        yields a smoother penalty (γ‑discount) that respects long
        walks but does not overly punish deep but necessary solutions.
      • Keep ``node.value`` as a running **average** instead of a raw
        sum.  This removes the hidden bias introduced by mixing
        discounted and undiscounted rewards and allows the selector
        to use ``node.value`` directly.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Discount factor per step (tuned for Sokoban macro‑push)
    _GAMMA = 0.995

    # ------------------------------------------------------------------
    # 1. Locate the root node to obtain the step offset.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits, discounted average value.
    #    The visited‑pruning table is deliberately left unchanged.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- depth‑based discount --------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        step_depth = max(0, node_steps - root_steps)   # >= 0
        discount = _GAMMA ** step_depth

        # ---- running average of discounted reward ---------------------
        #   new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward * discount) / node.visits

        # Move toward the root
        node = node.parent
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (trace evidence) | Why it hurts the search |
|------|--------------------------|--------------------------|
| 1️⃣ | All children in the first three moves have **avg ≈ 0.0** despite thousands of visits (e.g. `((3,4),0) v=229, avg=0.000`). The only non‑zero averages are tiny (0.036‑0.065) and they belong to actions that do **not** lead to a solution (the final state still has only 1/3 boxes on targets). | The back‑propagated reward that reaches the root is essentially zero, so the selector has no signal to prefer the few promising pushes. |
| 2️⃣ | Simulation returns a **smooth exponential** (`exp(-0.3 · h)`) for non‑terminal states. For a typical heuristic `h≈5‑8` this yields **<0.2**. When averaged with the many *0.0* leaves (dead‑locks, step‑limit) the value collapses to ≈0. | The value function becomes indistinguishable from the default “no information” value, so the A*‑guided selector cannot break ties on quality. |
| 3️⃣ | Back‑propagation **discounts** the leaf reward by `_GAMMA**step_depth` (0.995^d). Even with depth ≈ 20 the discount ≈ 0.90, trimming the already tiny reward further. | The discount amplifies the loss of signal and makes deep but promising branches look worse than shallow dead‑ends. |
| 4️⃣ | The node stores only a **single average** (`node.value`). A single solved leaf (reward = 1.0) is quickly drowned by many low‑reward visits, because the average is *linear* in the number of visits. | The algorithm never recognises that a branch has *ever* reached a solution, so it keeps exploring sub‑optimal branches. |

**2. ROOT CAUSE**

The back‑propagation routine is mathematically correct but **mis‑aligned with the needs of a deterministic puzzle**:

* It treats every leaf as a stochastic payoff, averaging away the *binary* win signal that is most informative for Sokoban.
* The exponential discount (per step) is unnecessary – the simulation already encodes distance‑to‑goal through `h`. Discounting only attenuates the already‑small heuristic‑derived reward.
* By storing only the mean, the backup discards the *maximum* (or win count) information that would allow the selector to prefer branches that have ever produced a win, even if they have been visited many times without reward.

Consequently, the selector (which selects by `f = g + h` and uses `value/visits` only as a tie‑breaker) receives essentially a flat value landscape, leading to the observed uniform visitation and failure to solve level 7.

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement** (the problem is fixable without rewriting the whole MCTS pipeline).

We will augment the back‑propagation routine with three lightweight changes that directly address the weaknesses above while preserving the existing A*‑guided selection and expansion code.

| Change | Rationale |
|--------|-----------|
| **a. Keep a “wins” counter** (`node.wins += 1` when `reward == 1.0`). | Gives each node a binary success metric that survives averaging. |
| **b. Replace the discounted average with a *max‑reward* backup** (`node.best = max(node.best, reward)`). Keep the running average for compatibility, but let the selector (via tie‑break) also see `node.best`. | Guarantees that a branch that has ever solved the puzzle retains a high value, preventing it from being drowned out. |
| **c. Remove the step‑based discount** (`discount = 1.0`). | The simulation’s heuristic already penalises distance; extra discount only erodes signal. |
| **d. Optional – small epsilon boost for wins** (e.g., `reward = 1.0 + 1e‑3` when solved). | Guarantees that a solved leaf’s contribution outranks any non‑solved leaf even after averaging. |

Implementation sketch (no code required):

```text
while node:
    node.visits += 1
    if reward == 1.0:          # solved leaf
        node.wins   = getattr(node, "wins",   0) + 1
        node.best   = max(getattr(node, "best", 0.0), 1.0)
        reward_adj = 1.0 + 1e-3          # tiny boost
    else:
        reward_adj = reward
    # running average (unchanged)
    node.value = ((node.visits-1)*node.value + reward_adj) / node.visits
    node = node.parent
```

**Why this works**

* The **wins / best** fields give the selector a higher‑resolution signal when `f` scores are equal; nodes that have ever led to a solution will dominate tie‑breaks, steering search toward the proven path.
* Removing the discount stops the systematic attenuation of the already‑small heuristic reward, so a solved leaf propagates a stronger signal.
* The epsilon boost ensures that a solved leaf’s contribution to the mean never collapses to zero, preserving its influence even after many non‑solved rollouts.

These changes are **local** (only the back‑propagation file), keep the current A*‑guided selection/exansion untouched, and directly address the observed weakness (near‑zero averages and loss of win information). The expected outcome is a sharper value landscape, faster identification of promising pushes, and ultimately a higher solve rate on level 7 and the remaining unsolved levels.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements:
      • Remove per‑step discounting (the simulation already encodes distance).
      • Keep a per‑node ``wins`` counter and ``best`` value so that any node
        that has ever produced a solved leaf retains a strong signal.
      • Add a tiny epsilon boost to solved rewards to prevent them from being
        washed out by many low‑reward visits when computing the running average.
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

        # ---- win / best tracking ---------------------------------------
        if reward == 1.0:                     # leaf is a solved state
            # Increment win count (use getattr to avoid AttributeError)
            node.wins = getattr(node, "wins", 0) + 1
            # Record the best reward seen at this node
            node.best = max(getattr(node, "best", 0.0), 1.0)
            # Apply epsilon boost for the averaging step
            reward_adj = 1.0 + _SOLVED_EPS
        else:
            # No win – keep existing best if any
            node.best = max(getattr(node, "best", 0.0), reward)
            reward_adj = reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the root
        node = node.parent
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
- No issues found

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: No changes required; the function correctly implements the incremental improvements.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements:
      • Remove per‑step discounting (the simulation already encodes distance).
      • Keep a per‑node ``wins`` counter and ``best`` value so that any node
        that has ever produced a solved leaf retains a strong signal.
      • Add a tiny epsilon boost to solved rewards to prevent them from being
        washed out by many low‑reward visits when computing the running average.
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

        # ---- win / best tracking ---------------------------------------
        if reward == 1.0:                     # leaf is a solved state
            # Increment win count (use getattr to avoid AttributeError)
            node.wins = getattr(node, "wins", 0) + 1
            # Record the best reward seen at this node
            node.best = max(getattr(node, "best", 0.0), 1.0)
            # Apply epsilon boost for the averaging step
            reward_adj = 1.0 + _SOLVED_EPS
        else:
            # No win – keep existing best if any
            node.best = max(getattr(node, "best", 0.0), reward)
            reward_adj = reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the root
        node = node.parent
```
