# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:07:07 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.42s |
| Status     | success |
| Tokens     | 5533 |
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

**1. KEY WEAKNESSES**

| Rank | Symptom (evidence from Trace #1) | Why it hurts performance |
|------|----------------------------------|---------------------------|
| 1️⃣ | **All pushes receive the same leaf reward** – every leaf that reaches a solved state returns 1.0, regardless of how many macro‑pushes or walk‑steps were needed. In the trace the *average* Q‑values of the chosen child quickly climb to ≈0.99, even though the solution required 7 pushes and many walk‑steps. This makes the search blind to solution length and can favour “long‑winded” pushes that look good heuristically but inflate the real step cost. |
| 2️⃣ | **Back‑propagation uses a plain average** (`node.value += reward`). When a node is visited many times via cheap but dead‑ending rollouts (e.g. moves that create a dead‑lock later) the average is dragged down, suppressing promising branches that would otherwise be optimal. In the first three moves we already see a handful of children with *v=1, avg≈0.14* – these low‑value branches pollute the average of their parents. |
| 3️⃣ | **Visited‑sync stores only tree depth** (`_ag.node_depth(node)`). Depth counts pushes, but a macro‑push can cost dozens of walk‑steps. The A*‑guided expansion therefore thinks a state reached in 3 pushes is cheap even if it required 30 extra walking steps, leading to premature pruning of truly cheaper‑cost states. |
| 4️⃣ | **No discount for depth** – the reward is propagated unchanged up the whole path. In a deterministic single‑player setting the *best* reachable value (or a depth‑penalised value) is more informative than an unscaled average. |

The combined effect is a tree that quickly concentrates visits on a branch that reaches the goal, but without any incentive to minimise the true step count. In harder levels this can cause the algorithm to converge on sub‑optimal, excessively long solutions or to get stuck behind dead‑locks that are not penalised early enough.

---

**2. ROOT CAUSE**

The back‑propagation routine does three things:

1. **Visits & plain sum** – `node.value += reward`. No consideration of *how far* the leaf was from the root.
2. **Depth‑only g‑sync** – `g = _ag.node_depth(node)`. The *g* used by the A* visited table is a push count, not the actual cost (`walk_steps + 1`) that the Sokoban‑Macro variant defines.
3. **No discount / no max‑backup** – the routine treats every simulation as an independent stochastic rollout, yet the simulation phase is deterministic and already returns a heuristic‑based *shaped* reward.

Thus the algorithm treats all solved leaves equally and updates the visited table with an underestimate of the true cost, which is precisely why the expansion phase prunes states based on the wrong metric and why the Q‑values are inflated regardless of solution length.

---

**3. PROPOSED APPROACH (Strategy A – Incremental)**  

The current architecture (A*‑guided selection/expansion, heuristic‑only simulation) works; we only need to make the backup reflect **solution quality** (i.e. total step cost) and to keep the value estimate *optimistic* rather than diluted by averages.

**a. Depth‑aware reward scaling**  
When back‑propagating, compute the *true cumulative cost* `c` from the leaf up to the current node (the sum of `walk_steps + 1` for each macro‑push). This can be derived from the state’s `steps` attribute:  

```
c = leaf_state.steps - node.state.steps
```

Replace the raw `reward` with a **discounted** version, e.g.:

```
discount = 0.99   # or 1.0 - epsilon
scaled_reward = reward * (discount ** c)
```

This makes longer paths receive a slightly lower payoff, steering the tree toward shorter solutions while still preserving the dense heuristic signal.

**b. Prefer max‑backup over averaging**  
Because the simulation is deterministic, the best value reachable from a node is the most informative. Change the update to keep the *maximum* of all seen rollouts:

```
node.value = max(node.value, scaled_reward)   # store best seen
node.visits += 1
```

(Alternatively keep both sum and max and use `node.value = node.max_value` for UCB‑style selection.)

**c. Sync visited with true cost, not depth**  
During back‑propagation, after computing `c`, update the A* visited table with the **minimum observed cost**:

```
if key not in visited or visited[key] > c:
    visited[key] = c
```

This aligns the expansion’s pruning logic with the exact macro‑push cost metric, preventing premature discarding of low‑cost states that happen deeper in the tree.

**d. Minor safety tweak** – retain the existing sign‑flipping logic (irrelevant for single‑player) and the root‑player detection; they do no harm.

These four small edits keep the overall A*‑guided framework untouched, require only local changes in `default_backpropagation`, and directly address the highest‑impact weaknesses: lack of cost awareness, averaging dilution, and mismatched g‑scores. The result should be faster convergence to *shorter* solutions and fewer wasted visits on low‑value branches.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS backpropagation with cost‑aware reward scaling.

Improvements:
  • Use the true step‑cost (state.steps) rather than tree depth for A* g‑sync.
  • Apply a discount based on the distance (in steps) from the leaf to each
    ancestor node, so longer paths receive a slightly lower payoff.
  • Keep the original sign‑flipping logic for completeness (single‑player
    games never flip).
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
    Backpropagate a (depth‑discounted) reward from leaf to root and keep the
    A* visited table up‑to‑date with the *minimum observed step cost* for each
    state.

    For each node on the upward walk:
      • node.visits   += 1
      • node.value    += scaled_reward   (scaled by how many steps separate
                                          the leaf from this node)
      • visited[state_key] = min(observed_cost, existing)

    The discount factor penalises long solution paths, encouraging the search
    to favour shorter macro‑push sequences while still retaining the dense
    heuristic information from the simulation phase.
    """
    # ------------------------------------------------------------------
    # Determine root player perspective (same convention as original code)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # ------------------------------------------------------------------
    # Capture the leaf's total step count once – this is the absolute cost
    # from the start of the episode to the leaf.
    # ------------------------------------------------------------------
    leaf_steps = node.state.steps

    # Discount factor (0 < d <= 1). Values close to 1 keep the reward mostly
    # intact while still penalising longer paths.
    DISCOUNT = 0.99

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    while node is not None:
        # Increment visit count.
        node.visits += 1

        # Determine which player “moved” to reach this node.
        if node.parent is None:
            mover = perspective
        else:
            mover = node.parent.state.current_player()

        # Compute the step distance from the leaf to the current node.
        # This is the additional cost incurred after this node to reach the leaf.
        cost_from_node = leaf_steps - node.state.steps  # non‑negative int

        # Apply a depth‑based discount to the raw reward.
        scaled_reward = reward * (DISCOUNT ** cost_from_node)

        # Update the value. For single‑player Sokoban we never flip sign,
        # but we keep the original logic for completeness.
        node.value += scaled_reward if mover == perspective else -scaled_reward

        # ------------------------------------------------------------------
        # Sync the A* visited table with the *actual* step cost.
        # The true cost to reach this state from the start of the episode is
        # node.state.steps (because each macro‑push incremented `steps` by its
        # walk cost + 1). Store the smallest cost ever observed.
        # ------------------------------------------------------------------
        key = node.state.state_key()
        observed_cost = node.state.steps
        if key not in visited or visited[key] > observed_cost:
            visited[key] = observed_cost

        # Move to parent.
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
- No issues found.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: No changes needed; the draft implementation is correct and efficient.
```python
"""
A*-guided MCTS backpropagation with cost‑aware reward scaling.

Improvements:
  • Use the true step‑cost (state.steps) rather than tree depth for A* g‑sync.
  • Apply a discount based on the distance (in steps) from the leaf to each
    ancestor node, so longer paths receive a slightly lower payoff.
  • Keep the original sign‑flipping logic for completeness (single‑player
    games never flip).
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
    Backpropagate a (depth‑discounted) reward from leaf to root and keep the
    A* visited table up‑to‑date with the *minimum observed step cost* for each
    state.

    For each node on the upward walk:
      • node.visits   += 1
      • node.value    += scaled_reward   (scaled by how many steps separate
                                          the leaf from this node)
      • visited[state_key] = min(observed_cost, existing)

    The discount factor penalises long solution paths, encouraging the search
    to favour shorter macro‑push sequences while still retaining the dense
    heuristic information from the simulation phase.
    """
    # ------------------------------------------------------------------
    # Determine root player perspective (same convention as original code)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    perspective = root.state.current_player()

    # ------------------------------------------------------------------
    # Capture the leaf's total step count once – this is the absolute cost
    # from the start of the episode to the leaf.
    # ------------------------------------------------------------------
    leaf_steps = node.state.steps

    # Discount factor (0 < d <= 1). Values close to 1 keep the reward mostly
    # intact while still penalising longer paths.
    DISCOUNT = 0.99

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Walk up the tree, updating statistics.
    # ------------------------------------------------------------------
    while node is not None:
        # Increment visit count.
        node.visits += 1

        # Determine which player “moved” to reach this node.
        if node.parent is None:
            mover = perspective
        else:
            mover = node.parent.state.current_player()

        # Compute the step distance from the leaf to the current node.
        # This is the additional cost incurred after this node to reach the leaf.
        cost_from_node = leaf_steps - node.state.steps  # non‑negative int

        # Apply a depth‑based discount to the raw reward.
        scaled_reward = reward * (DISCOUNT ** cost_from_node)

        # Update the value. For single‑player Sokoban we never flip sign,
        # but we keep the original logic for completeness.
        node.value += scaled_reward if mover == perspective else -scaled_reward

        # ------------------------------------------------------------------
        # Sync the A* visited table with the *actual* step cost.
        # The true cost to reach this state from the start of the episode is
        # node.state.steps (because each macro‑push incremented `steps` by its
        # walk cost + 1). Store the smallest cost ever observed.
        # ------------------------------------------------------------------
        key = node.state.state_key()
        observed_cost = node.state.steps
        if key not in visited or visited[key] > observed_cost:
            visited[key] = observed_cost

        # Move to parent.
        node = node.parent
```
