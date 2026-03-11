# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:21:12 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 6.32s |
| Status     | success |
| Tokens     | 4988 |
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

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|--------------------------|--------------------------|
| 1️⃣ | *All children are visited 500 times with the same tiny average value (≈0.167).* The tree never branches – every iteration re‑expands the same single line of pushes. | The algorithm prunes almost every alternative action during **expansion**; once a state is marked “visited” the next g‑score is considered *better* and the action is discarded forever. |
| 2️⃣ | *`visited` is updated with `g = _ag.node_depth(node)`* (push‑count only) while **expansion** stores `total_g = parent_depth + walk + 1` (push‑count + walk cost). | The pruning test `if key in visited and visited[key] <= total_g: prune` is overly aggressive because `visited` contains a *smaller* cost (pushes only) than the true cost used for comparison. Many legitimate successors are rejected simply because they require a longer walk, even if they lead to a better heuristic. |
| 3️⃣ | *Back‑propagation does not influence selection:* selection uses `f = g + h` and never looks at `node.value`. | Even if the value were propagated correctly, it would not steer the search; the bottleneck is the premature pruning caused by the mismatched g‑scores. |
| 4️⃣ | *Reward is added unchanged for every node on the path.* This inflates early nodes’ values but, again, values are ignored by the A*‑guided selector. | No measurable effect on the observed behaviour, but it wastes a tiny amount of work. |

**2. ROOT CAUSE**

The **core defect** is the inconsistency between the *cost metric* used for pruning in **expansion** and the *cost metric* stored during **back‑propagation**:

*Expansion* computes  
```python
total_g = parent_depth + walk + 1          # pushes + walking steps
```
and saves `visited[key] = total_g`.

*Back‑propagation* writes  
```python
g = _ag.node_depth(node)                   # pushes only
visited[key] = min(visited.get(key, ∞), g)
```

Because `node_depth` ≤ `total_g`, the visited table almost always contains a *lower* cost than the one being compared later, causing `visited[key] <= total_g` to be true and the action to be pruned permanently. The search therefore collapses to the first branch it discovers and never explores alternatives – exactly what the trace shows.

**3. PROPOSED APPROACH – A (incremental)**  

Fix the mismatch by making back‑propagation store the *same* cost metric that expansion uses. The `GameState` already tracks the cumulative step count (`state.steps`), which equals the sum of all walk steps plus one per push. Therefore:

1. **Compute the true g‑score** for each node as the *total number of steps taken from the root*:
   ```python
   g = node.state.steps - root.state.steps   # or simply node.state.steps if root.steps == 0
   ```
2. **Update the visited table with that g‑score** instead of `node_depth`:
   ```python
   if key not in visited or visited[key] > g:
       visited[key] = g
   ```
3. (Optional but cheap) expose a helper `node_total_cost(node)` that returns `node.state.steps` so the code stays readable.

*Why this works:*  
Now `visited[key]` contains the same metric (`pushes + walks`) that expansion compares against `total_g`. An action will be pruned only if a *cheaper* path (in total steps) to the same state has already been found, exactly mimicking A*’s closed‑list invariant. This eliminates the systematic over‑pruning, allows many previously discarded pushes to be explored, and should break the “single‑line” pattern seen in the trace.

**Additional low‑cost refinements (still incremental):**

* **Depth‑aware reward discount** – multiply `reward` by `γ**node.state.steps` (γ≈0.99) before back‑propagating. Although selection ignores values, a more meaningful value can help any future switch back to a UCB‑based selector without harming the current A*‑guided path.
* **Store `node.total_g` on creation** (e.g., `node.g = parent.g + walk + 1`) so that back‑propagation can read it directly, avoiding reliance on `state.steps` if the environment ever changes.

These tweaks keep the proven A*‑guided selection and simulation intact, only repairing the pruning invariant that is currently crippling exploration on level 6.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS backpropagation.

Standard MCTS backprop (visits + value) augmented with A* g-score sync:
for every node on the path from leaf to root, update

    visited[state_key] = min(g, existing)

where *g* is the true accumulated step cost (pushes + walk steps) stored
in ``state.steps``.  This matches the ``total_g`` metric used during
expansion, preventing over‑pruning.  Additionally the propagated reward
is optionally discounted by a small factor γ per step to give deeper
nodes a slightly lower contribution (useful if a value‑based selector
is ever re‑introduced).
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Optional

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_backpropagation(node, reward: float, gamma: Optional[float] = 0.99) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1/(1+h) or terminal return).
    gamma : float, optional
        Discount factor per step.  If supplied, the reward is multiplied by
        ``gamma ** depth`` where *depth* is the number of steps taken from
        the root to the current node.  This makes deeper nodes contribute
        slightly less, which can improve value‑based selection without
        affecting the current A*‑guided selector.

    The function updates:
        * node.visits
        * node.value   (adds discounted reward)
        * _ag.visited[state_key]  – the minimum total step cost ``g`` seen
          for that state.
    """
    # ------------------------------------------------------------------
    # 1. Determine the root of the tree (needed for perspective & g‑calc)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    visited = _ag.get_visited()

    # cache root step count once – normally zero but keep generic
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        # Determine which player made the move that led to *node*.
        # For single‑player Sokoban this is always the same player,
        # but we keep the generic formulation.
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # Apply optional discount based on total steps from the root.
        # ``node.state.steps`` holds pushes + walk steps.
        node_steps = getattr(node.state, "steps", 0)
        depth = node_steps - root_steps                      # total steps so far
        discounted_reward = reward * (gamma ** depth) if gamma else reward

        node.value += discounted_reward if mover == perspective else -discounted_reward

        # ---- A* g‑score sync -------------------------------------------
        # Use the same metric that expansion records: total steps (pushes+walks).
        # ``node.state.steps`` already equals that total cost.
        g = node_steps - root_steps   # cost from root to this node

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # move up the tree
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
- The function signature adds a third parameter `gamma`; existing MCTS code expects `default_backpropagation(node, reward)` and will raise a `TypeError`.
- The optional discount factor is unnecessary for the current A*‑guided selector and adds extra computation.
- Minor: unused import `Optional` and extra comment complexity.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: Restore original API and ensure the visited table stores the same total‑step cost used during expansion.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1.0 for solved, 0.0 for dead‑end).

    The function updates:
        * node.visits
        * node.value   (adds reward, respecting player perspective)
        * _ag.visited[state_key] – the minimum total step cost ``g`` seen
          for that state, where ``g`` equals the number of steps taken
          from the root (pushes + walk steps). This matches the metric
          used in expansion and prevents over‑pruning.
    """
    # ------------------------------------------------------------------
    # 1. Find the root of the tree (needed for perspective & cost offset)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    visited = _ag.get_visited()

    # Cache root step count (normally 0, but keep generic)
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # No discounting needed for the current A*‑guided selector.
        node.value += reward if mover == perspective else -reward

        # ---- A* g‑score sync -------------------------------------------
        # Total steps from the root to this node (pushes + walking steps).
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move up the tree
        node = node.parent
```
