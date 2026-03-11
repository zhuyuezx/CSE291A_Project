# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:08:14 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 7.94s |
| Status     | success |
| Tokens     | 8591 |
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
MCTS TOOL FUNCTIONS (all 4 phases)
------------------------------------------------------------

--- selection ◀ TARGET ---
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

--- expansion ---
```python
"""
A*-guided MCTS expansion.

Mirrors A*'s inner loop:
    for action in legal_actions(state):
        if deadlocked(next): skip
        if visited[key] <= g_new: skip      ← visited check
        visited[key] = g_new                ← record best g
        heappush(pq, (g+h, ...))            ← pick lowest-h action

Permanently removes pruned actions (deadlock / already-visited-better) from
_untried_actions so they are never reconsidered. Among surviving actions
picks the one with the lowest h (smallest distance-to-goal).
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


def default_expansion(node):
    """
    Expand the best untried action using A* visited + heuristic filtering.

    Steps:
      0. If this tree has a new root (new MCTS move), reset shared visited.
      1. Compute g_new = depth(node) + 1.
      2. For every untried action: deadlock → prune; visited[key] <= g_new → prune;
         else score by h_sokoban(next_state).
      3. Pick action with lowest h; write visited[key] = g_new.
      4. Fallback: if all pruned, expand first remaining untried action.
    """
    # Reset A* visited at the start of each MCTS search (each move). Otherwise
    # we prune states in the current tree using g-scores from previous trees.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    g_new   = _ag.node_depth(node) + 1
    visited = _ag.get_visited()

    scored  = []   # (h, action, next_state, key)
    pruned  = []   # actions to remove permanently

    for action in list(node._untried_actions):
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ── A*: deadlock prune ────────────────────────────────────────
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            pruned.append(action)
            continue

        # ── A*: visited check ─────────────────────────────────────────
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            pruned.append(action)
            continue

        # Use box-only heuristic to match sokoban_astar.ipynb A* ordering
        scored.append((_ag.h_sokoban_box_only(next_state), action, next_state, key))

    # Remove all permanently-pruned actions
    for a in pruned:
        node._untried_actions.remove(a)

    if scored:
        # Pick action with min h; break ties by action order to match A* determinism
        scored.sort(key=lambda x: (x[0], x[1]))
        _, chosen_action, chosen_state, chosen_key = scored[0]
        node._untried_actions.remove(chosen_action)
        visited[chosen_key] = g_new
    else:
        # All actions pruned — fallback: expand first remaining action
        # (avoids stall when entire frontier is already visited)
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state  = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key    = chosen_state.state_key()
            visited[chosen_key] = g_new
        else:
            # Nothing left — return existing child (safety net)
            if node.children:
                return next(iter(node.children.values()))
            # Absolute fallback: return the node itself (terminal-like)
            return node

    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

--- simulation ---
```python
"""
A* heuristic evaluation — no rollout.

In A*, there is no simulation: the heuristic h(state) IS the leaf evaluation.
This phase replaces the random rollout with a direct heuristic score:

    reward = 1.0 / (1.0 + h)     h = sum(box_to_nearest_target)  [box-only, match notebook]

Values are in (0, 1]:  solved state → 1.0,  farther state → closer to 0.
The shaped reward gives MCTS a dense gradient toward the goal instead of
the sparse {0, 1} terminal signal from random rollouts.
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


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A* heuristic evaluation at the leaf node — no rollout.

    Returns:
        1.0                    if the state is solved (h = 0)
        state.returns()[p]     if terminal (deadlock / step-limit)
        1.0 / (1.0 + h)        otherwise  (h = A* Manhattan heuristic)
    """
    if state.is_terminal():
        return state.returns()[perspective_player]

    # Use box-only heuristic to match sokoban_astar.ipynb and expansion ordering
    h = _ag.h_sokoban_box_only(state)
    if h == 0:
        return 1.0
    return 1.0 / (1.0 + h)
```

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: 
Generated:   2026-03-11T01:07:54.675390
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a leaf reward up the tree with stronger discounting
    and step‑cost awareness.

    Improvements over the previous version:
      * Uses an absolute depth discount (γ ** depth) rather than a
        leaf‑relative discount, sharply penalising deeper nodes.
      * Reduces γ to 0.70, making long plans contribute far less value.
      * Applies a linear penalty proportional to the total step count
        of the leaf state (walk + push costs), discouraging expensive
        macro‑pushes.
      * Increases the solved‑state bonus to 0.5 so a true solution
        clearly out‑scores near‑solved leaves.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be adjusted without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.70          # depth‑discount factor (stronger decay)
    SOLVED_BONUS = 0.50   # extra reward for a solved leaf
    STEP_PENALTY = 0.01   # penalty per step taken in the leaf state

    # ------------------------------------------------------------------
    # Lazy‑import shared A* globals (avoids circular imports).
    # ------------------------------------------------------------------
    import importlib.util
    import sys
    from pathlib import Path

    _KEY = "astar_globals"
    if _KEY not in sys.modules:
        _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
        _s = importlib.util.spec_from_file_location(_KEY, str(_p))
        _m = importlib.util.module_from_spec(_s)
        sys.modules[_KEY] = _m
        assert _s.loader is not None  # safeguard for static checkers
        _s.loader.exec_module(_m)
    _ag = sys.modules[_KEY]

    # ------------------------------------------------------------------
    # Gather leaf information once.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)          # absolute depth from root
    leaf_state = leaf_node.state

    # Detect solved terminal states.
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )

    # Apply step‑cost penalty to the raw simulation reward.
    # The simulation reward is already in (0, 1]; we subtract a small
    # amount proportional to the number of steps taken to reach the leaf.
    penalised_reward = reward - STEP_PENALTY * getattr(leaf_state, "steps", 0)
    if penalised_reward < 0.0:
        penalised_reward = 0.0

    # Add solved bonus if appropriate and clamp the final base reward.
    base_reward = penalised_reward + (SOLVED_BONUS if is_solved else 0.0)
    # Upper bound: solved leaf should not exceed 1.0 + SOLVED_BONUS.
    max_reward = 1.0 + SOLVED_BONUS
    if base_reward > max_reward:
        base_reward = max_reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying absolute depth discount.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Absolute depth from the root for this node.
        cur_depth = _ag.node_depth(cur)

        # Discount reward exponentially with depth.
        discounted_reward = (GAMMA ** cur_depth) * base_reward

        # Single‑player game → always additive.
        cur.value += discounted_reward

        # ---- Synchronise A* g‑score (unchanged) ----
        key = cur.state.state_key()
        g = cur_depth
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent.
        cur = getattr(cur, "parent", None)
```

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
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level7
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level7 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 8 levels): 0.7500

Per-level best composites so far:
  level1: best=1.0000 (baseline=1.0000) [MASTERED]
  level10: best=1.0000 (baseline=1.0000) [MASTERED]
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level4: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level7: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level6', 'level7', 'level8', 'level9']
Mastered levels: ['level1', 'level10', 'level2', 'level3', 'level4', 'level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 6 [level6] [expansion]: composite=0.0000, solve_rate=0%, eval_time=0.4s, desc= ✗ rejected
  Iter 6 [level6] [simulation]: composite=0.0000, solve_rate=0%, eval_time=0.8s, desc= ✗ rejected
  Iter 6 [level6] [backpropagation]: composite=0.0000, solve_rate=0%, eval_time=0.5s, desc= ✗ rejected

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
TASK — IMPROVE THE HEURISTIC
------------------------------------------------------------
Improve the 'selection' function above based on your
analysis. Decide which approach fits best:

OPTION A — Incremental optimization (~70% of the time):
  Start from the CURRENT code and make targeted improvements.
  Examples:
  • Add a deadlock check (box in corner with no target)
  • Add distance-based scoring factors
  • Switch from random rollout to greedy lookahead
  • Add early-termination for stuck states
  • Tune weights or scoring formula
  • Combine multiple heuristic signals

OPTION B — Paradigm shift (~30% of the time):
  If the current approach is fundamentally limited, propose
  a different strategy. Keep any components that work well.
  State clearly in DESCRIPTION why a restructure is needed.

How the 'selection' phase works in MCTS:
  - Receives (root, exploration_weight). Returns the chosen child index.
  - Adjusts UCB scores or selection policy to favor promising branches.
  - Called at every level during tree descent — keep it FAST.
  - Must produce a valid child index; selection drives which branch gets expanded next.

CONSTRAINTS:
  • Same function signature as the current code.
  • Standalone function, standard library only.
  • SELF-CONTAINED: define ALL helper functions inline.
    Do NOT use undefined names (e.g. _ag, _bfs_distance).
  • node is MCTSNode (uses __slots__). Use ONLY node.state,
    node.parent, node.children, node._untried_actions,
    node.visits, node.value. Do NOT assign new attributes.
  • Code can be as long as the heuristic requires — no
    artificial line limit. Prioritize quality over brevity.

You MUST format your response EXACTLY as follows:

ACTION: modify
FILE_NAME: <filename>.py
FUNCTION_NAME: <entry_point_function_name>
DESCRIPTION: <one-line: what you changed and why>
```python
<your complete function code here>
```

Rules for the header fields:
- ACTION must be either 'create' (brand new tool) or 'modify'
  (improving the existing tool shown above).
- FILE_NAME must end in .py and contain only [a-z0-9_].
- FUNCTION_NAME must match the main function defined in the code.
- The code block must be valid Python that can run standalone.

---

## Response

ACTION: modify
FILE_NAME: selection.py
FUNCTION_NAME: default_selection
DESCRIPTION: Added exploration term (UCB‑style), deadlock penalty, and visit‑penalty to the A*‑guided selection to restore exploration and avoid deadlocked branches.
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
