# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:02:39 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.15s |
| Status     | success |
| Tokens     | 5711 |
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

| Rank | Symptom (trace) | Why it hurts |
|------|-----------------|--------------|
| **1** | The root always selects the same child: `((3,3),0)` gets **495/500** visits, all other actions only 1‑4 visits. After three pushes the search never tries the alternative pushes that would place the third box. | **Pure exploitation** – the selector never explores low‑visit branches, so once a path looks “good” it dominates the tree and the algorithm “gets stuck” with 2/3 boxes on target. |
| **2** | The tie‑breaker prefers the child with the **higher average value** (`q`). After a few simulations the best‑looking child also has the highest `q`, reinforcing the bias described above. | Reinforces the same exploitation loop; a child with a single lucky high rollout can monopolise the search. |
| **3** | `exploration_weight` is an unused argument; the classic UCB term (`c*sqrt(ln(N)/n)`) never appears. | The algorithm cannot recover from a misleading early heuristic, even if the heuristic (`h_sokoban`) is fairly informative. |
| **4** | The heuristic `h = _ag.h_sokoban(state)` does not explicitly penalise *obvious dead‑locks* (e.g., a box in a corner not on a target). The selector may therefore descend into branches that are doomed, wasting visits. | Causes high‑visit wasteful branches that look cheap in `g+h` because `g` is the same for all children and `h` may be small for a dead‑locked push (the box is already near a target). |
| **5** | Tie‑break on equal `f` uses **higher** `q`. | Even when `f` is identical (common early on), the branch with a few good rollouts wins, further reducing diversity. |

**2. ROOT CAUSE**  

The **selection routine** is essentially an A* best‑first walk:

```python
for child in node.children:
    g = node_depth(child)
    h = h_sokoban(child.state)
    f = g + h
    q = child.value / child.visits
    if f < best_f or (f == best_f and q > best_q):
        best = child
```

* No exploration term (`c * sqrt(log(parent.visits)/child.visits)`) → pure exploitation.  
* `exploration_weight` argument is never used, so the function cannot be tuned.  
* The tie‑breaker (`q > best_q`) favours already‑visited, high‑value children, making the bias self‑reinforcing.  
* Because all children share the same `g` (= depth + 1), the decision reduces to “pick the child with lowest heuristic *or* highest average value”. Early stochastic noise in `value` dominates, and the tree collapses to a single line (as seen in the trace).  

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement** (recommended).  
The current framework (A*‑guided walk + shared A* globals) works well for pruning illegal states and keeping cheap `h` evaluations. We only need to **re‑introduce a light exploration pressure** and tweak the tie‑breaker so that low‑visit nodes are still considered.

* **Blend UCB into the score:**  
  Compute a *selection score* `s = (g + h) - exploration_weight * sqrt(log(parent.visits) / child.visits)`.  
  – When a child is unvisited (`visits == 0`), treat the exploration term as a large negative bonus (or set `visits = 1e‑9`) so that unseen actions are favoured.  
  – The existing `exploration_weight` parameter can now control the exploration‑exploitation trade‑off without code changes elsewhere.

* **Dead‑lock penalty:**  
  If `child.state` (or the resulting state after the push) is dead‑locked (`hasattr(state, "_is_deadlocked") and state._is_deadlocked()`), add a constant large penalty (e.g., `+10_000`) to the score. This keeps the A* pruning but makes the selector actively avoid those branches even if they have a low `h`.

* **Tie‑breaking on *lower* visits:**  
  When `s` values are equal (unlikely but possible early on), prefer the child with **fewer visits** (`child.visits`) rather than higher `q`. This encourages **novelty** and prevents a single high‑value child from monopolising the search.

* **Keep the reset logic** for new MCTS searches unchanged – it is essential for the shared visited table.

These changes are a few extra arithmetic operations per selection (log, sqrt, a constant penalty) and therefore stay cheap. They preserve the successful A*‑guided pruning while giving the tree the ability to explore alternative pushes, which is exactly what the trace shows is missing.

**Why not a full restructure?**  
The current A*‑guided selection already provides valuable domain‑specific guidance (depth + box/target heuristic, dead‑lock pruning). The failure mode is purely the lack of exploration, not a conceptual flaw in the A* idea. Adding a lightweight UCB term restores the classic MCTS balance without discarding the proven pruning and shared‑state infrastructure, so an incremental fix is sufficient and safer.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight UCB exploration.

    At each level we descend to the child that minimises
        score = (g + h + deadlock_penalty) - exploration_bonus

    where:
        g = depth of the child (A* path cost so far)
        h = sokoban heuristic for the child state
        deadlock_penalty = large constant added if the child state is a dead‑lock
        exploration_bonus = exploration_weight * sqrt( ln(N_parent) / n_child )
            – unvisited children receive an infinite bonus, guaranteeing they are tried.
    Tie‑breaking prefers the child with **fewer visits** when scores are equal,
    encouraging novelty.

    The function also detects the start of a new MCTS search (root state key
    change) and resets the shared A* visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ── Walk tree: select child using f + exploration ───────────────────
    DEADLOCK_PENALTY = 1_000_000  # effectively discards dead‑locked branches

    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # expansion will handle the leaf

        best_child = None
        best_score = math.inf
        best_visits = None

        # parent visit count for the UCB term (add 1 to avoid log(0))
        parent_visits = max(node.visits, 1)

        for child in node.children.values():
            # A* components
            g = _ag.node_depth(child)
            h = _ag.h_sokoban(child.state)
            f = g + h

            # dead‑lock penalty (if we can detect it)
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                f += DEADLOCK_PENALTY

            # exploration bonus – unvisited nodes get infinite boost
            if child.visits == 0:
                explore_bonus = float('inf')
            else:
                explore_bonus = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )

            # lower score is better
            score = f - explore_bonus

            # Choose the best child (lower score) – with tie‑break on fewer visits
            if score < best_score - 1e-12:
                best_child = child
                best_score = score
                best_visits = child.visits
            elif abs(score - best_score) <= 1e-12:
                # equal score → prefer the less‑visited node
                if child.visits < (best_visits if best_visits is not None else math.inf):
                    best_child = child
                    best_visits = child.visits

        if best_child is None:
            break  # safety fallback
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
FILE_NAME: default_selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: No changes needed; the implementation is correct and efficient.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight UCB exploration.

    At each level we descend to the child that minimises
        score = (g + h + deadlock_penalty) - exploration_bonus

    where:
        g = depth of the child (A* path cost so far)
        h = sokoban heuristic for the child state
        deadlock_penalty = large constant added if the child state is a dead‑lock
        exploration_bonus = exploration_weight * sqrt( ln(N_parent) / n_child )
            – unvisited children receive an infinite bonus, guaranteeing they are tried.
    Tie‑breaking prefers the child with **fewer visits** when scores are equal,
    encouraging novelty.

    The function also detects the start of a new MCTS search (root state key
    change) and resets the shared A* visited table.
    """
    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ── Walk tree: select child using f + exploration ───────────────────
    DEADLOCK_PENALTY = 1_000_000  # effectively discards dead‑locked branches

    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # expansion will handle the leaf

        best_child = None
        best_score = math.inf
        best_visits = None

        # parent visit count for the UCB term (add 1 to avoid log(0))
        parent_visits = max(node.visits, 1)

        for child in node.children.values():
            # A* components
            g = _ag.node_depth(child)
            h = _ag.h_sokoban(child.state)
            f = g + h

            # dead‑lock penalty (if we can detect it)
            if hasattr(child.state, "_is_deadlocked") and child.state._is_deadlocked():
                f += DEADLOCK_PENALTY

            # exploration bonus – unvisited nodes get infinite boost
            if child.visits == 0:
                explore_bonus = float('inf')
            else:
                explore_bonus = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )

            # lower score is better
            score = f - explore_bonus

            # Choose the best child (lower score) – with tie‑break on fewer visits
            if score < best_score - 1e-12:
                best_child = child
                best_score = score
                best_visits = child.visits
            elif abs(score - best_score) <= 1e-12:
                # equal score → prefer the less‑visited node
                if child.visits < (best_visits if best_visits is not None else math.inf):
                    best_child = child
                    best_visits = child.visits

        if best_child is None:
            break  # safety fallback
        node = best_child

    return node
```
