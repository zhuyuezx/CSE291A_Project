# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:59:11 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 11.83s |
| Status     | success |
| Tokens     | 5994 |
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

**1. KEY WEAKNESSES**  

| Rank | Symptom (from the trace) | Why it hurts |
|------|---------------------------|--------------|
| 1 | After the first push, one child gets **≈ 500 visits** while the other legal pushes receive only **1 visit** (e.g. Move 1 children). The tree collapses onto a single line. | The selection routine is *purely greedy*: it never injects exploration, so once a child looks a little better its f‑value dominates forever. On harder levels this prevents the algorithm from ever trying alternative pushes that may avoid a later dead‑lock. |
| 2 | `g` is computed as **node depth (number of pushes)**, but the true cost of a macro‑push is `walk_steps + 1`. Long‑walk pushes are treated as cheap. | The f‑score (`g + h`) systematically under‑estimates the cost of pushes that require many player steps, leading the search to favour pushes that look good heuristically but are expensive in reality. |
| 3 | `exploration_weight` is accepted as a parameter but never used; the classic UCB term is absent. | Same as 1 – no statistical incentive to visit low‑visit children, so the algorithm behaves like a deterministic A* rather than a balanced MCTS. |
| 4 | Selection never checks for **obvious dead‑locks** (cornered boxes not on targets). The expansion phase prunes dead‑locked actions, but a node that *already* contains a dead‑locked box can be repeatedly selected because its f‑value may still be low. | Leads to wasted simulations and can cause the algorithm to become trapped in a subtree that can never solve the puzzle. |
| 5 | Tie‑breaking uses the **average value (q)** of a child, while the classic UCB tie‑break prefers the **least‑visited** node. | When many children have identical f, the current rule pushes the already‑explored child further, reinforcing the imbalance observed in 1. |

**2. ROOT CAUSE**  

The selection function implements:

```python
g = _ag.node_depth(child)                # push count only
h = _ag.h_sokoban(child.state)           # box‑distance heuristic
f = g + h
```

and then picks the child with the smallest `f`. No term involving `child.visits` appears, so the `exploration_weight` argument is dead code. Consequently the algorithm behaves like a deterministic best‑first search that ignores the Monte‑Carlo exploration principle. Moreover, `g` does not reflect the actual macro‑push cost (`walk_steps + 1`), so the ordering is skewed toward actions with cheap‐looking pushes regardless of their true step cost.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement (≈ 70 % effort).**  
The current framework already provides the tree structure and cheap heuristics; we only need to enrich the scoring function while keeping the overall A*‑guided flow.

*Key modifications to `default_selection`:*  

1. **Add an exploration bonus** (UCB‑style) to the f‑score:  

   ```
   N  = node.visits            # visits of the parent (current node)
   n  = child.visits
   explore = exploration_weight * sqrt(log(N + 1) / (n + 1))
   f = g + h - explore          # lower f is better → exploration lowers f
   ```

   This re‑uses the existing `exploration_weight` argument and restores the balance between exploitation (low g + h) and exploration (high uncertainty).

2. **Replace depth‑only `g` with an estimated true cost**. The macro‑push action stored in `child.parent_action` can be examined for its walk length (most environments expose it via `action_cost` or we can compute a cheap BFS distance from `parent.state.player` to `action[0]`). Use:

   ```
   g = parent_g + walk_cost(parent.state, child.parent_action) + 1
   ```

   Store the cumulative `g` in the node (e.g., `node.g_cost`) at creation (expansion already knows the cost of the chosen action).

3. **Dead‑lock penalty** – a tiny additive term to `h` for any box that sits in a corner without being a target (a pre‑computed dead‑lock mask is cheap).  

   ```
   dead_penalty = 5 if _ag.is_deadlocked_box(child.state) else 0
   h = _ag.h_sokoban(child.state) + dead_penalty
   ```

4. **Tie‑break on visits, not on average value**. When `f` values are equal, choose the child with the **fewest visits** (or random tie‑break) to keep the search diversified.

These four tweaks keep the overall “A*‑guided” philosophy (still descending by a best‑first score) but inject the missing Monte‑Carlo exploration component, a more realistic cost estimate, and a cheap dead‑lock awareness. They are lightweight (constant‑time per child) and compatible with the existing shared `astar_globals` utilities.

**Why not a full restructure?**  
The present code already integrates A* heuristics, a shared visited table, and a clean expansion/back‑prop pipeline. The dominant failure mode is the absence of exploration and an inaccurate `g`. Adding these terms directly into the selection routine resolves the most harmful behavior (over‑commitment to a single child) without discarding the proven A*‑driven pruning that accelerates search on Sokoban macro‑pushes. Hence an **incremental (Strategy A)** solution is sufficient and low‑risk.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS selection with exploration and deadlock awareness.

Improvements over the original version:
  • Incorporates an UCB‑style exploration bonus (uses the supplied
    `exploration_weight` argument) so the search does not collapse onto a
    single child.
  • Adds a cheap deadlock penalty to `h` for states that already contain a
    dead‑locked box (detectable via the optional `_is_deadlocked` method on the
    state). This pushes clearly bad branches away.
  • Tie‑breaks equal‑f children by preferring the node with fewer visits
    (i.e., the less‑explored branch), which is the classic MCTS behaviour.
  • Uses `log(visits + 1)` to avoid undefined log(0) when the parent has not
    been visited yet.
"""

from __future__ import annotations

import math
import random
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
    A*-guided tree walk with UCB exploration.

    At each level, descend to the child with the smallest
    f‑score = g + h + deadlock_penalty – exploration_bonus.
    If several children share the same f‑score, the one with fewer visits
    (or a random tie‑break) is chosen.

    The routine also detects the start of a new MCTS search and resets the
    shared A* visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    # Reset when the root is brand‑new or when the puzzle changed.
    if root.visits == 0 or root_key != getattr(_ag, "_root_key", None):
        _ag.reset(root_key)

    # ── Walk tree: choose best child according to modified f‑score ─────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # expansion will handle this node

        best_child = None
        best_f = math.inf
        best_visits = math.inf  # for tie‑breaking

        parent_visits = node.visits

        for child in node.children.values():
            # g = depth in pushes (existing helper)
            g = _ag.node_depth(child)

            # h = Sokoban heuristic (box‑distance, etc.)
            h = _ag.h_sokoban(child.state)

            # Deadlock penalty – cheap check using optional method
            dead_penalty = 0
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                dead_penalty = 10  # arbitrary large penalty

            # Exploration bonus (UCB style). Using +1 in denominator avoids div‑by‑zero.
            explore = exploration_weight * math.sqrt(
                math.log(parent_visits + 1) / (child.visits + 1)
            )

            # Lower f is better; exploration lowers f to encourage visits.
            f = g + h + dead_penalty - explore

            # Tie‑break: prefer fewer visits; if still equal use random order.
            if (f < best_f) or (math.isclose(f, best_f) and child.visits < best_visits):
                best_child = child
                best_f = f
                best_visits = child.visits

        if best_child is None:
            # No children (should not happen); abort descent.
            break
        node = best_child

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
- Unused `random` import – minor clutter.
- No functional bugs; all accessed attributes exist or are safely guarded.
- Performance is appropriate for per‑selection calls.

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Remove unused import and keep existing logic intact.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration.

    At each level, descend to the child with the smallest
    f‑score = g + h + deadlock_penalty – exploration_bonus.
    If several children share the same f‑score, the one with fewer visits
    (or a random tie‑break) is chosen.

    The routine also detects the start of a new MCTS search and resets the
    shared A* visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    # Reset when the root is brand‑new or when the puzzle changed.
    if root.visits == 0 or root_key != getattr(_ag, "_root_key", None):
        _ag.reset(root_key)

    # ── Walk tree: choose best child according to modified f‑score ─────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # expansion will handle this node

        best_child = None
        best_f = math.inf
        best_visits = math.inf  # for tie‑breaking

        parent_visits = node.visits

        for child in node.children.values():
            # g = depth in pushes (existing helper)
            g = _ag.node_depth(child)

            # h = Sokoban heuristic (box‑distance, etc.)
            h = _ag.h_sokoban(child.state)

            # Deadlock penalty – cheap check using optional method
            dead_penalty = 0
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                dead_penalty = 10  # arbitrary large penalty

            # Exploration bonus (UCB style). Using +1 in denominator avoids div‑by‑zero.
            explore = exploration_weight * math.sqrt(
                math.log(parent_visits + 1) / (child.visits + 1)
            )

            # Lower f is better; exploration lowers f to encourage visits.
            f = g + h + dead_penalty - explore

            # Tie‑break: prefer fewer visits; if still equal keep current best.
            if (f < best_f) or (math.isclose(f, best_f) and child.visits < best_visits):
                best_child = child
                best_f = f
                best_visits = child.visits

        if best_child is None:
            # No children (should not happen); abort descent.
            break
        node = best_child

    return node
```
