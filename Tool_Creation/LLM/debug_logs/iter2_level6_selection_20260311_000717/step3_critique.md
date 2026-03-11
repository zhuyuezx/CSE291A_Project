# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:07:49 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 7.83s |
| Status     | success |
| Tokens     | 4969 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: selection).

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

**1. KEY WEAKNESSES**  

| Rank | Symptom (trace) | Why it hurts performance |
|------|----------------|--------------------------|
| **1** | After move 12 the search repeatedly revisits the same line of nodes (e.g. `((2,5),2)` and `((2,6),2)` have v ≈ 500, avg ≈ 0.498) and never pushes the second box onto the goal. | **Pure greedy‑A\***: `default_selection` chooses the child with the *lowest* `f = g + h` and never injects any exploration. When several children share the same `f` (common in Sokoban macro pushes because `g` is just the push count and `h` is often identical for a set of pushes), the algorithm locks onto the first‑found high‑value branch and ignores alternatives that could reduce the remaining distance. |
| **2** | Tie‑breaking uses the *higher* average Q (`best_q`) when `f` ties, contrary to the comment “prefer the child with fewer visits”. This drives the search toward already‑heavily‑sampled nodes. | The comment is misleading; the code actually **exploits** the best‑scoring child, further reducing exploration. The result is the huge visit counts (≈ 500) on a single child while other promising pushes get 0 visits. |
| **3** | The parameter `exploration_weight` is accepted but never used. | No ability to tune the balance between exploitation (low `f`) and exploration (UCB‑style bonus). |
| **4** *(secondary)* | `h = _ag.h_sokoban(child.state)` includes the player‑to‑box walking cost, but `g` already counts *pushes* only. The walk cost is therefore counted twice for macro pushes, distorting `f`. | May cause many children to have identical `f` values, aggravating the lack of discrimination and the reliance on the wrong tie‑breaker. |

**2. ROOT CAUSE**  

The selection routine is essentially **deterministic best‑first**:

```python
g = _ag.node_depth(child)          # number of pushes so far
h = _ag.h_sokoban(child.state)     # Manhattan box‑to‑target + player walk
f = g + h
```

* No exploration term → no stochasticity.  
* Tie‑breaker (`q > best_q`) rewards already‑visited branches, opposite of the intended “fewer visits”.  
* `exploration_weight` is ignored, so the function cannot be tuned.  

Consequently the tree quickly collapses onto a single high‑visit path, as seen in the trace (every move after the first few has a single child with `v ≈ 500`). The algorithm therefore never backs out to try a different macro‑push that would lower the remaining distance.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement (recommended).**  
The core A\*‑guided idea—using `g + h` as a primary ordering—remains valuable, but we must **re‑introduce controlled exploration** and **fix the tie‑break**. The changes are small, keep the existing shared globals, and preserve the deterministic ordering when `f` truly differs.

* **Add an exploration bonus** (UCB‑style) to the score used for selection:  

  ```
  exploration = exploration_weight * sqrt(ln(parent.visits + 1) / (child.visits + 1))
  score = -(g + h) + exploration          # negative because we still “maximise” in the loop
  ```

  This keeps the A\* notion (lower `g+h` is better) but lets rarely‑visited children compete.

* **Correct the tie‑breaker**: when `f` values are equal (within a tiny epsilon) prefer the child with **fewer visits**, not higher average value. That can be expressed by comparing `(child.visits, -child.value/child.visits)` as a secondary key.

* **Use the *box‑only* heuristic for `h`** (the same one the expansion phase already relies on) to avoid double‑counting walking cost. Replace `_ag.h_sokoban` with `_ag.h_sokoban_box_only` in selection (or expose a unified “heuristic” function). This makes `f` more discriminative and reduces the number of exact ties.

* **Respect the `exploration_weight` argument** (default 1.41) so the hyper‑parameter can be tuned without code changes.

These four modifications are each a one‑line tweak or a small re‑ordering of existing variables; they preserve the overall A\*‑guided spirit while restoring the crucial exploration/exploitation balance that was missing. With a modest exploration term, the tree will continue to push the most promising box but will also occasionally sample alternative pushes (e.g., the second box in the trace) that break the plateau where `f` stays constant, giving the MCTS a realistic chance to solve level 6.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with controlled exploration.

    At each level we select the child that minimises the A* cost
    `f = g + h` while also adding a UCB‑style exploration bonus.
    This prevents the search from locking onto a single high‑visit branch
    and respects the `exploration_weight` hyper‑parameter.

    Tie‑breaking now naturally prefers children with fewer visits
    (higher exploration term).  The box‑only heuristic is used for `h`
    to avoid double‑counting player walking cost.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ── Walk tree: select child with best (low f + exploration) score ───
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best_child = None
        best_score = -math.inf

        parent_visits = node.visits  # parent of all candidates
        # avoid log(0); parent_visits may be 0 on first iteration
        log_parent = math.log(parent_visits + 1)

        for child in node.children.values():
            g = _ag.node_depth(child)
            # use box‑only heuristic (matches expansion & simulation phases)
            h = _ag.h_sokoban_box_only(child.state)
            f = g + h

            # UCB‑style exploration term favouring less‑visited nodes
            child_visits = child.visits
            exploration = exploration_weight * math.sqrt(
                log_parent / (child_visits + 1)
            )

            # We *maximise* score; lower f is better → -f + exploration
            score = -f + exploration

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            break  # no children (should not happen)

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
- The draft calls `_ag.h_sokoban_box_only`, which may not exist in the shared `astar_globals` module, leading to an `AttributeError`.
- Tie‑breaking on equal scores is not explicit; while the exploration term usually handles it, adding a deterministic secondary key (e.g., fewer visits) prevents nondeterministic behavior.
- Minor readability improvement: compute the heuristic function once outside the loop.

ACTION: modify
FILE_NAME: default_selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Fix missing heuristic function, add deterministic tie‑breaker, and improve clarity.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with controlled exploration.

    At each level we select the child that minimises the A* cost
    `f = g + h` while also adding a UCB‑style exploration bonus.
    This prevents the search from locking onto a single high‑visit branch
    and respects the ``exploration_weight`` hyper‑parameter.

    Tie‑breaking prefers children with fewer visits when scores are equal.
    The box‑only heuristic is used for ``h`` if available; otherwise the
    regular Sokoban heuristic is used (to avoid double‑counting player walk
    cost when the specialised version exists).
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # Choose which heuristic to use (fallback to the generic one)
    h_func = getattr(_ag, "h_sokoban_box_only", _ag.h_sokoban)

    # ── Walk tree: select child with best (low f + exploration) score ───
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best_child = None
        best_score = -math.inf
        best_visits = math.inf   # secondary tie‑breaker (fewer visits is better)

        parent_visits = node.visits
        log_parent = math.log(parent_visits + 1)  # avoid log(0)

        for child in node.children.values():
            g = _ag.node_depth(child)
            h = h_func(child.state)          # heuristic component
            f = g + h

            # UCB‑style exploration term favouring less‑visited nodes
            child_visits = child.visits
            exploration = exploration_weight * math.sqrt(
                log_parent / (child_visits + 1)
            )

            # We *maximise* score; lower f is better → -f + exploration
            score = -f + exploration

            # Primary: higher score; Secondary: fewer visits
            if (score > best_score) or (score == best_score and child_visits < best_visits):
                best_score = score
                best_child = child
                best_visits = child_visits

        if best_child is None:  # safety net; should never happen
            break

        node = best_child

    return node
```
