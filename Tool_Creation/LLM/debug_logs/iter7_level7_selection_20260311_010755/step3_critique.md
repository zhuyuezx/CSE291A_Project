# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:08:22 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.39s |
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

**1. KEY WEAKNESSES**

| # | Observation (trace) | Why it hurts |
|---|---------------------|--------------|
| 1 | After the first move the child **((3, 6),0)** accumulated **483/500** visits, while all other actions received ≤ 9 visits. | The selection routine never re‑examines alternative pushes; it “locks‑in” a single branch and never discovers the later‑necessary pushes that would finish the level. |
| 2 | The **UCB‑style exploration term is missing** – tie‑breaking only looks at average value *when* `f` ties, which almost never happens. | Pure A* ordering (`f = g + h`) is deterministic and drives the walk straight down the lowest‑heuristic path, ignoring the stochastic benefit of exploring poorly‑visited nodes. |
| 3 | The heuristic `h_sokoban` does **not penalise obvious deadlocks** (cornered boxes, frozen walls). The walk still follows low‑`h` pushes that later become dead‑ends, e.g. the second‑move sequence that leaves the puzzle with 2/3 boxes on targets and a remaining distance 1 but never reaches the final target. | Without a deadlock penalty, `f` can be deceptively low for moves that look good locally but block the last box, preventing the search from ever reaching a solved state. |
| 4 | No **novelty/visit‑penalty** term: nodes with high visitation are not discouraged, so the algorithm repeatedly expands the same child instead of spreading the search budget. | This compounds weakness 1, causing “stagnation” after a few hundred iterations. |

*Impact ranking*: 1 (over‑exploitation) → 2 (missing exploration term) → 3 (deadlock‑blind heuristic) → 4 (lack of visit‑penalty).

---

**2. ROOT CAUSE**

The selection function is a **direct A*‑style best‑first walk**:

```python
best_f = g + h
if f < best_f or (f == best_f and q > best_q):
    best = child
```

* It **ignores the classic UCB1 exploration factor** (`c * sqrt(log(N)/n)`) that MCTS relies on to balance exploitation vs. exploration.
* The only tie‑breaker is the **average value `q`**, which rarely affects the decision because `f` values differ for almost every child.
* The heuristic `h_sokoban` is a pure distance‑to‑goal estimate; it does **not inflate** the cost of deadlocked successors, so a deadlock can appear attractive (low distance) and dominate the search.
* Consequently the tree collapses onto the child with the smallest `g+h`, which quickly becomes the heavily visited branch seen in the trace.

---

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinement** (recommended).

We keep the useful parts of the current code (shared A* globals, root‑reset logic, `g`/`h` computation) and add lightweight, cheap modifications that directly address the top weaknesses:

1. **Hybrid score = f − exploration_bonus**  
   Compute an exploration term `e = C * sqrt(log(parent_visits) / child_visits)` (with a modest constant `C ≈ 0.5`).  
   Choose the child with **minimal `score = g + h – e`**. This re‑introduces true MCTS exploration while still biasing towards low heuristic cost.

2. **Deadlock penalty in `h`**  
   If a child state is detected deadlocked via `state._is_deadlocked()`, add a large constant `DEADLOCK_PENALTY` (e.g., +10) to its `h` before scoring. This pushes deadlocked nodes out of the best‑first path without needing a full rollout.

3. **Visit‑penalty / novelty bonus**  
   When `child.visits` is high, subtract a small term `visit_penalty = V * child.visits` (e.g., `V = 0.001`). This gently discourages repeatedly expanding the same node, encouraging the algorithm to “peek” at less‑explored branches even if their raw `f` is marginally higher.

4. **Tie‑break on visit count** (fallback)  
   If scores are equal, prefer the child with **fewer visits** (instead of the current average‑value tie‑break), reinforcing exploration.

These tweaks are **O(1)** per child, keep the overall cost negligible, and preserve the A*‑guided intuition (low `g+h` still matters) while restoring the essential stochastic exploration that MCTS needs. By penalising deadlocks we also avoid the “local‑distance trap” evident in the trace.

*Why not a full restructure?*  
The existing framework already provides a solid shared A* infrastructure and a clean separation of phases. The primary failure is the **over‑exploitation caused by the pure A* selection**, not a fundamental incompatibility. Adding a lightweight exploration term and deadlock awareness fixes the symptom without discarding the proven heuristic‑driven ordering, making the change safe, fast to test, and easy to tune.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight exploration.

    For each level we pick the child that minimizes

        score = g + h + VISIT_PENALTY * visits - exploration_term

    where
        g = depth from the root,
        h = sokoban heuristic (with a large penalty for dead‑locked states),
        exploration_term = exploration_weight * sqrt(log(parent_visits) / child_visits)
                           (∞ when child_visits == 0, forcing selection of un‑visited nodes).

    Tie‑breaking: if scores are equal (within a tiny epsilon) we prefer the child
    with fewer visits, encouraging novelty.

    The function also detects the beginning of a new MCTS search and resets the
    shared A* state accordingly.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # constants for the new scoring
    DEADLOCK_PENALTY = 10.0      # pushes dead‑locked nodes out of the best‑first path
    VISIT_PENALTY = 0.001       # slight bias against over‑visited nodes
    EPS = 1e-9                  # tolerance for tie‑breaking on floating point

    # ── Walk tree: choose child with minimal hybrid score ───────────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best_child = None
        best_score = math.inf
        best_visits = math.inf  # used for tie‑break when scores equal

        parent_visits = max(node.visits, 1)  # avoid log(0)

        for child in node.children.values():
            # g‑cost (depth)
            g = _ag.node_depth(child)

            # heuristic h, with deadlock penalty if applicable
            h = _ag.h_sokoban(child.state)
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                h += DEADLOCK_PENALTY

            # exploration term (UCB‑style); infinite if never visited
            if child.visits == 0:
                exploration = float("inf")
            else:
                exploration = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )

            # visit penalty (encourages novelty)
            visit_pen = VISIT_PENALTY * child.visits

            # final score – lower is better
            score = g + h + visit_pen - exploration

            # pick the minimal score; tie‑break on fewer visits
            if (score < best_score - EPS) or (
                abs(score - best_score) <= EPS and child.visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child.visits

        if best_child is None:
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
No issues found

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: No changes needed; the incremental improvements are correct and efficient.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight exploration.

    For each level we pick the child that minimizes

        score = g + h + VISIT_PENALTY * visits - exploration_term

    where
        g = depth from the root,
        h = sokoban heuristic (with a large penalty for dead‑locked states),
        exploration_term = exploration_weight * sqrt(log(parent_visits) / child_visits)
                           (∞ when child_visits == 0, forcing selection of un‑visited nodes).

    Tie‑breaking: if scores are equal (within a tiny epsilon) we prefer the child
    with fewer visits, encouraging novelty.

    The function also detects the beginning of a new MCTS search and resets the
    shared A* state accordingly.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # constants for the new scoring
    DEADLOCK_PENALTY = 10.0      # pushes dead‑locked nodes out of the best‑first path
    VISIT_PENALTY = 0.001       # slight bias against over‑visited nodes
    EPS = 1e-9                  # tolerance for tie‑breaking on floating point

    # ── Walk tree: choose child with minimal hybrid score ───────────
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # hand off to expansion

        best_child = None
        best_score = math.inf
        best_visits = math.inf  # used for tie‑break when scores equal

        parent_visits = max(node.visits, 1)  # avoid log(0)

        for child in node.children.values():
            # g‑cost (depth)
            g = _ag.node_depth(child)

            # heuristic h, with deadlock penalty if applicable
            h = _ag.h_sokoban(child.state)
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                h += DEADLOCK_PENALTY

            # exploration term (UCB‑style); infinite if never visited
            if child.visits == 0:
                exploration = float("inf")
            else:
                exploration = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )

            # visit penalty (encourages novelty)
            visit_pen = VISIT_PENALTY * child.visits

            # final score – lower is better
            score = g + h + visit_pen - exploration

            # pick the minimal score; tie‑break on fewer visits
            if (score < best_score - EPS) or (
                abs(score - best_score) <= EPS and child.visits < best_visits
            ):
                best_child = child
                best_score = score
                best_visits = child.visits

        if best_child is None:
            break
        node = best_child

    return node
```
