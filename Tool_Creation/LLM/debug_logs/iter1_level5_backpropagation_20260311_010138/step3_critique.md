# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:02:04 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.25s |
| Status     | success |
| Tokens     | 5449 |
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
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with A* g-score sync:
for every node on the path from leaf to root, update

    visited[state_key] = min(g, existing)

so that the expansion phase's visited check always reflects the shallowest
depth at which each state has been reached — matching A*'s invariant that
visited[key] holds the best known path cost.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root + sync A* visited g-scores.

    For each node on the walk:
      • node.visits += 1
      • node.value  += reward  (sign-flipped at opponent nodes for 2-player games;
                                 Sokoban is single-player so sign never flips)
      • visited[state_key] = min(current_g, known_best_g)
    """
    # Determine root player perspective (same convention as default_backpropagation)
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    visited = _ag.get_visited()

    while node is not None:
        node.visits += 1

        mover = node.parent.state.current_player() if node.parent else perspective
        node.value += reward if mover == perspective else -reward

        # ── A* g-sync ─────────────────────────────────────────────────
        key = node.state.state_key()
        g   = _ag.node_depth(node)
        if key not in visited or visited[key] > g:
            visited[key] = g

        node = node.parent
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (observed in trace or likely on other levels) | Why it hurts play |
|------|------------------------------------------------------|-------------------|
| **1** | **No depth‑discount on the back‑propagated reward** – every ancestor receives the *full* leaf reward, even if the solution required many pushes.  In the trace the first three moves receive > 0.96 average value despite the leaf being 7 pushes away.  This inflates the value of deep branches and can make the search favour longer, higher‑cost plans on harder levels. |
| **2** | **Uniform treatment of solved vs. merely‑good roll‑outs** – the reward coming from `default_simulation` already differentiates progress, but back‑propagation does not amplify a *fully solved* leaf.  Consequently a solved rollout (reward ≈ 1.0) and a high‑but‑incomplete rollout (reward ≈ 0.9) are propagated identically, making the algorithm less eager to terminate early when a solution is found. |
| **3** | **Redundant sign‑flip logic for a single‑player game** – the `mover`/`perspective` gymnastics add overhead and obscure the core update.  While not fatal, it increases the chance of a bug (e.g., using the wrong player when the tree is re‑rooted). |
| **4** | **No explicit handling of “partial‑progress” credit beyond the simulation** – once a rollout is back‑propagated, nodes that already have several boxes on targets do not get extra boost proportional to that progress; the information is lost in the averaging of `value / visits`. |

The most damaging of these is **(1) the lack of a depth‑discount**, because it directly conflicts with the “shorter plans dominate” pattern recommended for back‑propagation.  

---

**2. ROOT CAUSE**  

The current `default_backpropagation` performs:

```python
while node is not None:
    node.visits += 1
    node.value  += reward   # possibly sign‑flipped
    visited[key] = min(g, visited.get(key, INF))
    node = node.parent
```

* It adds the *raw* `reward` to every ancestor, irrespective of how many edges away the leaf is (`g` is only used for the A* visited sync, not for the value update).  
* It does not examine the leaf’s `state.is_terminal()` flag, so a solved leaf (`reward == 1.0`) is treated the same as a leaf that merely has a high heuristic reward.  
* The sign‑flip code is kept for a two‑player template, but Sokoban is single‑player, so the extra branching is unnecessary and can hide bugs when the tree is re‑rooted after a move.  

Thus the algorithm propagates *undiscounted, undifferentiated* evidence, which biases the average value toward deep, noisy rollouts and blurs the distinction between “solved” and “almost solved”.

---

**3. PROPOSED APPROACH**  

**Strategy A – Incremental refinement** (recommended).  
The back‑propagation step can be improved with two small, well‑contained changes that directly address weaknesses 1 and 2 while preserving the proven A*‑g‑sync and the rest of the pipeline.

1. **Depth‑discounted reward**  
   *Compute the distance* `d` from the current node to the leaf (e.g., `leaf_depth – node_depth`).  
   *Apply a discount* `γ^d` (`γ` ∈ (0, 1), e.g., 0.95) before adding to `node.value`.  
   This makes each push contribute less to the value the farther it is from the successful outcome, encouraging the tree to prefer short, low‑cost solutions.

2. **Solved‑leaf amplification**  
   Detect whether the leaf state is terminal and solved (`leaf.state.boxes_on_targets() == leaf.state.num_targets`).  
   If so, boost the reward (e.g., `reward = 1.0 + SOLVED_BONUS` with `SOLVED_BONUS` ≈ 0.2) **before** discounting.  
   The bonus propagates (still discounted) so ancestors of a genuine solution receive a stronger signal than ancestors of merely‑good rollouts.

3. **Simplify sign handling**  
   Since `current_player()` is always `0`, drop the `mover` logic and always add `reward` (or its discounted version).  This reduces noise and clarifies intent.

4. **Leave A* visited sync untouched** – it already correctly records the shallowest `g`.

*Implementation sketch (no code required):*  

* At the start, store `leaf_depth = _ag.node_depth(leaf_node)` (the incoming `node`).  
* Walk upward, compute `d = leaf_depth - _ag.node_depth(node)`.  
* `discount = γ ** d`.  
* If `leaf_is_solved`, set `base = 1.0 + SOLVED_BONUS` else `base = reward`.  
* Update `node.value += discount * base`.  

All other lines (visits increment, visited‑sync) remain unchanged.  

These tweaks are lightweight, keep the existing A*‑guided selection/expansion intact, and directly target the most harmful bias observed in the trace and likely on harder levels.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with:
  • Depth‑discounted reward propagation – rewards diminish with distance
    from the leaf, encouraging shorter solution plans.
  • Solved‑leaf bonus – when the leaf state is a solved terminal, a small
    extra bonus is added before discounting, making genuine solutions
    stand out from high‑but‑incomplete rollouts.
  • Simplified sign handling (Sokoban is single‑player).

In addition, the A* visited‑g‑score sync is preserved so that expansion
continues to respect the shallowest‑known path cost.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a rollout reward from a leaf node up to the root.

    Enhancements:
      * depth_discount (γ) reduces the impact of evidence the farther
        the ancestor is from the leaf (γ ≈ 0.95 by default).
      * solved_bonus adds a small extra reward for truly solved leaf
        states, ensuring they are preferred over merely‑good rollouts.
    """
    # ------------------------------------------------------------------
    # Configuration parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (0 < γ < 1)
    SOLVED_BONUS = 0.20   # extra reward for a leaf that actually solves the puzzle

    # ------------------------------------------------------------------
    # Determine the depth of the leaf node once – used for discounting.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)

    # Detect whether the leaf is a *solved* terminal state.
    leaf_state = leaf_node.state
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )
    # Base reward for this leaf (may be boosted if solved).
    base_reward = reward + SOLVED_BONUS if is_solved else reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying depth discount and updating A* g‑scores.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Compute how far this node is from the leaf (0 for leaf itself).
        cur_depth = _ag.node_depth(cur)
        distance_from_leaf = leaf_depth - cur_depth

        # Apply exponential discount.
        discounted_reward = (GAMMA ** distance_from_leaf) * base_reward

        # Sokoban is single‑player → always add (no sign flip).
        cur.value += discounted_reward

        # ── A* g‑score sync (unchanged) ─────────────────────────────────
        key = cur.state.state_key()
        g = cur_depth
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent.
        cur = cur.parent
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
No issues found

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: No changes needed – the function already implements the intended incremental improvements correctly.
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with:
  • Depth‑discounted reward propagation – rewards diminish with distance
    from the leaf, encouraging shorter solution plans.
  • Solved‑leaf bonus – when the leaf state is a solved terminal, a small
    extra bonus is added before discounting, making genuine solutions
    stand out from high‑but‑incomplete rollouts.
  • Simplified sign handling (Sokoban is single‑player).

In addition, the A* visited‑g‑score sync is preserved so that expansion
continues to respect the shallowest‑known path cost.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a rollout reward from a leaf node up to the root.

    Enhancements:
      * depth_discount (γ) reduces the impact of evidence the farther
        the ancestor is from the leaf (γ ≈ 0.95 by default).
      * solved_bonus adds a small extra reward for truly solved leaf
        states, ensuring they are preferred over merely‑good rollouts.
    """
    # ------------------------------------------------------------------
    # Configuration parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (0 < γ < 1)
    SOLVED_BONUS = 0.20   # extra reward for a leaf that actually solves the puzzle

    # ------------------------------------------------------------------
    # Determine the depth of the leaf node once – used for discounting.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)

    # Detect whether the leaf is a *solved* terminal state.
    leaf_state = leaf_node.state
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )
    # Base reward for this leaf (may be boosted if solved).
    base_reward = reward + SOLVED_BONUS if is_solved else reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying depth discount and updating A* g‑scores.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Compute how far this node is from the leaf (0 for leaf itself).
        cur_depth = _ag.node_depth(cur)
        distance_from_leaf = leaf_depth - cur_depth

        # Apply exponential discount.
        discounted_reward = (GAMMA ** distance_from_leaf) * base_reward

        # Sokoban is single‑player → always add (no sign flip).
        cur.value += discounted_reward

        # ── A* g‑score sync (unchanged) ─────────────────────────────────
        key = cur.state.state_key()
        g = cur_depth
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent.
        cur = cur.parent
```
