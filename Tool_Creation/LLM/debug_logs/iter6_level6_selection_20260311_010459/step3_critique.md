# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:05:31 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 7.86s |
| Status     | success |
| Tokens     | 6059 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: selection).

PHASE: selection
  • What it does: Walks down the tree from root to a leaf. Chooses which existing branch to explore next. Must balance exploration (UCB) and exploitation.
  • Optimization goal: Improve how we RANK existing nodes — favor promising branches, deprioritize dead ends. Your heuristic adjusts node scores used by UCB1.
  • Constraints: Called very often. Keep it CHEAP — no multi-step rollouts, no deep deadlock simulation. Rank nodes, don't simulate.
  • Good patterns: bonus for more boxes on targets, bonus for lower box distance, penalize obvious deadlocks, novelty bonus for under-visited nodes.
  • Avoid: expensive rollout logic, final reward shaping (that belongs in simulation).

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
TARGET HEURISTIC TO IMPROVE (selection)
------------------------------------------------------------
```python
"""
A*-guided MCTS selection.

Replaces UCB1 with best-first (min f = g + h) node selection, mirroring
A*'s heapq.heappop(pq). Detects the start of each new MCTS search and
resets the shared visited table.
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state (one instance shared across all 4 phase files) ──
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk: at each level select the child with min f = g + h.

    Mirrors A*'s heapq.heappop(pq) — always descend toward the lowest-cost
    frontier node rather than the highest UCB1 score.

    Tie-breaking: among equal f-scores, prefer the child with fewer visits
    to encourage coverage of equally-promising branches.

    Also detects when a new MCTS search begins (root state key changed) and
    resets the shared visited dict for the new search.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    # Walk to the actual root of the tree passed in.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    # Reset when:
    #   • visits == 0  → brand-new root (first iteration of a fresh search,
    #                     including after smoke-test pollution)
    #   • key changed  → different game/level
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ── Walk tree: min-f child selection ─────────────────────────────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best      = None
        best_f    = math.inf
        best_q    = -math.inf  # avg value — tie-break when f is equal

        for child in node.children.values():
            g = _ag.node_depth(child)
            h = _ag.h_sokoban(child.state)
            f = g + h
            q = child.value / child.visits if child.visits > 0 else 0.0
            if f < best_f or (f == best_f and q > best_q):
                best   = child
                best_f = f
                best_q = q

        if best is None:
            break
        node = best

    return node
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Rank | Symptom (from the trace) | Why it matters |
|------|---------------------------|----------------|
| 1️⃣ | **No exploration term** – the `exploration_weight` argument is never used; selection is a pure *best‑first* (`min (g+h)`) walk. After a few visits the same branch (e.g., `((2, 7),2)`) gets 500 visits and is never abandoned, even though its heuristic‑driven path stalls with 1/2 boxes on target. | MCTS collapses to a greedy A* search. With only 500 iterations the tree never samples alternative pushes that could break the dead‑end. |
| 2️⃣ | **Heuristic h is blind to dead‑lock risk** – `h_sokoban` (or `h_sokoban_box_only`) only measures Manhattan distance, not whether a push creates a corner‑deadlock or blocks other boxes. The selection keeps following pushes that temporarily lower the distance (e.g., moving the second box into a “tight” corridor) but later makes the last box unreachable. | The algorithm repeatedly descends into states that look cheap but are actually unsolvable, as seen after Move 7 where a box is on target but the remaining box is trapped in a 1‑step dead‑end. |
| 3️⃣ | **Tie‑breaking only on average Q** – when `f` scores are equal the code prefers the child with higher *average value*; no consideration of *visit count* or *potential* (e.g., distance reduction). This further entrenches heavily visited nodes. | After the first few expansions the same children dominate the selection, preventing the tree from exploring low‑visit “wild‑card” actions that could solve the puzzle. |
| 4️⃣ | **g + h is not discounted for walk‑cost** – `g` is node depth (number of pushes) while each macro‑push can have a large walk cost. A path with many cheap pushes may have the same `g` as a path with few expensive pushes, misleading the selection. | The search favors long walk‑heavy sequences (the trace shows many steps between pushes) that waste iterations. |

**2. ROOT CAUSE**

The selection function implements *pure A*‑style best‑first*:

```python
g = _ag.node_depth(child)
h = _ag.h_sokoban(child.state)
f = g + h
```

* It never incorporates the `exploration_weight` argument, so the classic UCB term `c·√(ln(N)/n)` is missing.  
* The heuristic used (`h_sokoban`) only counts box‑to‑target Manhattan distance; dead‑lock detection lives in expansion, not in selection, so a child that looks cheaper can still be a dead‑end.  
* The only tie‑breaker is `q = child.value / child.visits`, which favours already‑visited nodes and does not push the algorithm toward under‑explored actions.  

Consequently the tree quickly becomes a single deep chain of high‑visit nodes, and MCTS can’t recover to explore alternative pushes that would free the remaining box.

**3. PROPOSED APPROACH**

**Strategy A – Incremental improvement (recommended).**  
The current framework (A*‑guided selection with shared visited table) works well for pruning obvious dead‑locks and keeping `g`‑scores consistent. We only need to *re‑introduce a cheap exploration bias* and *soft‑penalize pushes that increase the distance*.

**Targeted modifications**

1. **Add a UCB‑style exploration bonus** (still cheap) to the score used for picking the child:  

   ```
   parent_visits = node.visits
   child_visits   = child.visits
   explore = exploration_weight * sqrt(log(parent_visits + 1) / (child_visits + 1))
   ```

2. **Combine exploitation, heuristic, and exploration** into a single scalar (higher = better) instead of pure `min f`. A convenient formulation is a *modified UCT* where the heuristic acts as an additional prior:

   ```
   # exploitation term (average value)
   explo = child.value / child.visits if child.visits > 0 else 0.0

   # heuristic term – treat lower h as higher prior
   # scale h to [0,1] using 1/(1+h) (same as simulation reward)
   prior = 1.0 / (1.0 + h)

   # final score (maximize)
   score = explo + exploration_weight * explore + beta * prior
   ```

   `beta` (e.g., 0.3) lets Manhattan closeness influence the decision without overwhelming exploration.

3. **Dead‑lock‑aware penalty** (still O(1)): compute the *delta* in total box distance between parent and child. If `delta < 0` (distance increased) subtract a small constant `penalty` (e.g., 0.1) from the score. This discourages pushes that move a box away from any target.

4. **Tie‑break on visit count**: if two children have identical scores, pick the one with fewer visits (`child_visits`). This favours novelty when scores are equal.

5. **Keep the reset logic & shared visited table** unchanged – they already guarantee consistent A* pruning across searches.

**Why this suffices**

* The added exploration term directly uses the `exploration_weight` argument that was previously ignored, restoring the core MCTS balance.  
* By turning the heuristic into a *prior* (`1/(1+h)`) we retain the cheap distance information while allowing the algorithm to ignore a slightly higher `h` if the value estimate is promising.  
* The distance‑delta penalty is a negligible cost (one integer subtraction) yet immediately discourages moves that push a box away, reducing the chance of walking into the dead‑end observed after Move 7.  
* All changes stay O(1) per child, preserving the requirement that selection be called millions of times.

With these tweaks the tree will continue to follow low‑distance pushes **but** will also periodically sample alternative branches, giving the search a chance to discover the final push that completes level 6. If after a few more iterations the solve rate improves, we can fine‑tune `beta`, `penalty`, and `exploration_weight` without any structural rewrite.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS selection with exploration and dead‑lock‑aware penalty.

Enhancements:
  * Uses the provided `exploration_weight` to add a cheap UCB‑style
    exploration term (c * sqrt(log(N_parent) / (n_child+1))).
  * Treats the Manhattan‑distance heuristic as a prior:
        prior = 1 / (1 + h)   (higher when boxes are nearer targets).
  * Applies a small penalty when a child’s heuristic distance is larger
    than its parent’s (i.e., the push moves a box away from any target).
  * Tie‑breaks equal scores by preferring the child with fewer visits,
    encouraging novelty.
  * Retains the original A* reset logic and shared visited table.
"""

from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

# ── Load shared A* state (one instance shared across all 4 phase files) ──
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


def default_selection(node, exploration_weight: float = 1.41):
    """
    Walks down the tree, selecting children using a blend of:

      • Exploitation   – average value (Q)
      • Exploration    – UCB‑style bonus controlled by `exploration_weight`
      • Heuristic prior – 1/(1+h) where h is the A* box distance heuristic
      • Penalty for increasing h (discourages pushes that move boxes away)

    The child with the **highest** combined score is chosen.
    Ties are broken by the smaller visit count.

    Also detects the start of a new MCTS search and resets the shared
    visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # Parameters for the new scoring term
    BETA = 0.30          # weight of the heuristic prior
    PENALTY = 0.10       # fixed penalty when child.h > parent.h

    # Initial heuristic of the current node (used for delta calculation)
    parent_h = _ag.h_sokoban(node.state)

    # ── Walk tree: select best child at each level ───────────────────────────
    while not node.is_terminal:
        # If there are still actions that have never been expanded, stop here.
        if not node.is_fully_expanded:
            return node

        best_child = None
        best_score = -math.inf
        best_visits = math.inf  # for tie‑breaking

        parent_visits = node.visits

        for child in node.children.values():
            # ---- core statistics ------------------------------------------------
            child_visits = child.visits
            # exploitation term (average value)
            explo = child.value / child_visits if child_visits > 0 else 0.0

            # exploration term (UCB)
            explore = exploration_weight * math.sqrt(
                math.log(parent_visits + 1) / (child_visits + 1)
            )

            # heuristic prior (higher is better)
            h = _ag.h_sokoban(child.state)
            prior = 1.0 / (1.0 + h)   # matches simulation reward scaling

            # penalty if this push increased the total box distance
            penalty = PENALTY if h > parent_h else 0.0

            # combined score (higher is better)
            score = explo + explore + BETA * prior - penalty

            # ---- choose best ----------------------------------------------------
            if (score > best_score) or (
                math.isclose(score, best_score) and child_visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child_visits

        if best_child is None:
            # No children? break out – this should not happen, but guard anyway.
            break

        # Move down the tree; update parent_h for the next iteration.
        node = best_child
        parent_h = _ag.h_sokoban(node.state)

    return node
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
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: No changes required; the draft implementation is correct and efficient.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    Walks down the tree, selecting children using a blend of:

      • Exploitation   – average value (Q)
      • Exploration    – UCB‑style bonus controlled by `exploration_weight`
      • Heuristic prior – 1/(1+h) where h is the A* box distance heuristic
      • Penalty for increasing h (discourages pushes that move boxes away)

    The child with the **highest** combined score is chosen.
    Ties are broken by the smaller visit count.

    Also detects the start of a new MCTS search and resets the shared
    visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # Parameters for the new scoring term
    BETA = 0.30          # weight of the heuristic prior
    PENALTY = 0.10       # fixed penalty when child.h > parent.h

    # Initial heuristic of the current node (used for delta calculation)
    parent_h = _ag.h_sokoban(node.state)

    # ── Walk tree: select best child at each level ───────────────────────────
    while not node.is_terminal:
        # If there are still actions that have never been expanded, stop here.
        if not node.is_fully_expanded:
            return node

        best_child = None
        best_score = -math.inf
        best_visits = math.inf  # for tie‑breaking

        parent_visits = node.visits

        for child in node.children.values():
            # ---- core statistics ------------------------------------------------
            child_visits = child.visits
            # exploitation term (average value)
            explo = child.value / child_visits if child_visits > 0 else 0.0

            # exploration term (UCB)
            explore = exploration_weight * math.sqrt(
                math.log(parent_visits + 1) / (child_visits + 1)
            )

            # heuristic prior (higher is better)
            h = _ag.h_sokoban(child.state)
            prior = 1.0 / (1.0 + h)   # matches simulation reward scaling

            # penalty if this push increased the total box distance
            penalty = PENALTY if h > parent_h else 0.0

            # combined score (higher is better)
            score = explo + explore + BETA * prior - penalty

            # ---- choose best ----------------------------------------------------
            if (score > best_score) or (
                math.isclose(score, best_score) and child_visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child_visits

        if best_child is None:
            # No children? break out – this should not happen, but guard anyway.
            break

        # Move down the tree; update parent_h for the next iteration.
        node = best_child
        parent_h = _ag.h_sokoban(node.state)

    return node
```
