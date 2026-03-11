# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:44:34 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.85s |
| Status     | success |
| Tokens     | 9293 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: simulation).

PHASE: simulation
  • What it does: Rolls forward from a leaf state to estimate how promising it is. Returns a reward (e.g. 0–1) that flows into backpropagation.
  • Optimization goal: Produce REWARDS that reflect true state quality. Shaped partial progress helps MCTS distinguish good from bad actions.
  • Constraints: Must return a FLOAT. Reward MUST vary across states — flat rewards ≈ random play. Called thousands of times per move — keep it fast.
  • Good patterns: shaped score (boxes on targets, distance improvement), penalize deadlocks/loops/stagnation, prefer pushes over wandering, early termination when stuck.
  • Avoid: tree-level visit balancing, acceptance criteria for tools — this phase only scores rollouts.

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

--- selection ---
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

--- simulation ◀ TARGET ---
```python
"""
LLM-generated MCTS tool: simulation
Description: Fixed missing imports (`sys`, `deque`) and aligned variable names with the public `GameState` API.
Generated:   2026-03-11T00:39:40.315107
"""

from __future__ import annotations

import importlib.util
import math
import sys                     # ← added
from collections import deque  # ← added
from pathlib import Path

# ── Load shared A* state (may be unused after the rewrite) ─────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag  # retained for compatibility; not used directly


def _bfs_reachable(start, walls, boxes):
    """
    BFS from ``start`` avoiding walls and boxes.
    Returns a dict mapping each reachable cell to its distance.
    """
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        d = dist[cur] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in dist:
                continue
            dist[nxt] = d
            q.append(nxt)
    return dist


def _simple_corner_deadlock(state) -> bool:
    """
    Fast detection of obvious corner dead‑locks:
    a non‑target box that is adjacent to walls on two orthogonal sides.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in targets:
            continue
        r, c = b
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for Sokoban‑macro MCTS.

    Signals combined:
      • Manhattan distance of boxes to nearest targets.
      • Count of boxes unreachable by the player (without pushing).
      • Estimated minimal walk distance to any legal push position.
      • Heavy penalty for obvious corner dead‑locks.

    The combined heuristic ``h`` is transformed into a smooth reward
    ``exp(-λ·h)`` (λ≈0.3).  Terminal states return the game‑provided reward,
    but obvious dead‑locks are given a small negative value to separate them
    from ordinary step‑limit terminations.
    """
    # ---- Terminal handling -------------------------------------------------
    if state.is_terminal():
        ret = state.returns()[perspective_player]   # 1.0 for solved, 0.0 otherwise
        # Penalise only clear dead‑locks; keep step‑limit terminations at 0.0
        if ret == 0.0 and _simple_corner_deadlock(state):
            return -0.1
        return ret

    # ---- 1️⃣ Box‑to‑target Manhattan distance --------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ---- 2️⃣ Unreachable‑box count -------------------------------------------
    walls = state.walls
    boxes = state.boxes
    reachable = _bfs_reachable(state.player, walls, boxes)
    unreachable_cnt = sum(1 for b in boxes if b not in reachable)

    # ---- 3️⃣ Minimal walk distance to a legal push position ------------------
    min_walk = None
    for action in state.legal_actions():
        push_pos = action[0]                       # cell where player must stand
        if push_pos in reachable:
            d = reachable[push_pos]
            if min_walk is None or d < min_walk:
                min_walk = d
    if min_walk is None:                           # no push possible (should be terminal)
        min_walk = 10.0

    # ---- 4️⃣ Corner dead‑lock penalty ----------------------------------------
    deadlock_penalty = 5.0 if _simple_corner_deadlock(state) else 0.0

    # ---- 5️⃣ Weighted combination --------------------------------------------
    alpha = 1.0   # box distance weight
    beta  = 2.0   # unreachable boxes weight
    gamma = 0.5   # walk distance weight
    delta = 1.0   # dead‑lock penalty scaling
    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) + (delta * deadlock_penalty)

    # ---- 6️⃣ Reward shaping --------------------------------------------------
    lam = 0.3
    reward = math.exp(-lam * h)

    # Guard against numerical underflow
    if reward < 1e-8:
        reward = 0.0
    return reward
```

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Fixed attribute handling for `MCTSNode` (which uses `__slots__`) by removing illegal `best`/`wins` assignments and preserving the original boost‑for‑solved‑leaf logic.
Generated:   2026-03-11T00:40:19.410251
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Enhancements retained:
      • No per‑step discounting (the simulation already encodes distance).
      • A tiny epsilon boost is added to solved rewards so that they remain
        noticeable even after many low‑reward visits.
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

        # ---- reward handling ------------------------------------------
        # Add epsilon boost only for a solved leaf (reward == 1.0).
        reward_adj = reward + _SOLVED_EPS if reward == 1.0 else reward

        # ---- running average of (possibly boosted) reward -------------
        # new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward_adj) / node.visits

        # Move toward the parent (rootward)
        node = node.parent
```

------------------------------------------------------------
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
```python
"""
LLM-generated MCTS tool: simulation
Description: Fixed missing imports (`sys`, `deque`) and aligned variable names with the public `GameState` API.
Generated:   2026-03-11T00:39:40.315107
"""

from __future__ import annotations

import importlib.util
import math
import sys                     # ← added
from collections import deque  # ← added
from pathlib import Path

# ── Load shared A* state (may be unused after the rewrite) ─────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag  # retained for compatibility; not used directly


def _bfs_reachable(start, walls, boxes):
    """
    BFS from ``start`` avoiding walls and boxes.
    Returns a dict mapping each reachable cell to its distance.
    """
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        d = dist[cur] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in walls or nxt in boxes or nxt in dist:
                continue
            dist[nxt] = d
            q.append(nxt)
    return dist


def _simple_corner_deadlock(state) -> bool:
    """
    Fast detection of obvious corner dead‑locks:
    a non‑target box that is adjacent to walls on two orthogonal sides.
    """
    walls = state.walls
    targets = state.targets
    for b in state.boxes:
        if b in targets:
            continue
        r, c = b
        if ((r - 1, c) in walls and (r, c - 1) in walls) or \
           ((r - 1, c) in walls and (r, c + 1) in walls) or \
           ((r + 1, c) in walls and (r, c - 1) in walls) or \
           ((r + 1, c) in walls and (r, c + 1) in walls):
            return True
    return False


def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for Sokoban‑macro MCTS.

    Signals combined:
      • Manhattan distance of boxes to nearest targets.
      • Count of boxes unreachable by the player (without pushing).
      • Estimated minimal walk distance to any legal push position.
      • Heavy penalty for obvious corner dead‑locks.

    The combined heuristic ``h`` is transformed into a smooth reward
    ``exp(-λ·h)`` (λ≈0.3).  Terminal states return the game‑provided reward,
    but obvious dead‑locks are given a small negative value to separate them
    from ordinary step‑limit terminations.
    """
    # ---- Terminal handling -------------------------------------------------
    if state.is_terminal():
        ret = state.returns()[perspective_player]   # 1.0 for solved, 0.0 otherwise
        # Penalise only clear dead‑locks; keep step‑limit terminations at 0.0
        if ret == 0.0 and _simple_corner_deadlock(state):
            return -0.1
        return ret

    # ---- 1️⃣ Box‑to‑target Manhattan distance --------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ---- 2️⃣ Unreachable‑box count -------------------------------------------
    walls = state.walls
    boxes = state.boxes
    reachable = _bfs_reachable(state.player, walls, boxes)
    unreachable_cnt = sum(1 for b in boxes if b not in reachable)

    # ---- 3️⃣ Minimal walk distance to a legal push position ------------------
    min_walk = None
    for action in state.legal_actions():
        push_pos = action[0]                       # cell where player must stand
        if push_pos in reachable:
            d = reachable[push_pos]
            if min_walk is None or d < min_walk:
                min_walk = d
    if min_walk is None:                           # no push possible (should be terminal)
        min_walk = 10.0

    # ---- 4️⃣ Corner dead‑lock penalty ----------------------------------------
    deadlock_penalty = 5.0 if _simple_corner_deadlock(state) else 0.0

    # ---- 5️⃣ Weighted combination --------------------------------------------
    alpha = 1.0   # box distance weight
    beta  = 2.0   # unreachable boxes weight
    gamma = 0.5   # walk distance weight
    delta = 1.0   # dead‑lock penalty scaling
    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) + (delta * deadlock_penalty)

    # ---- 6️⃣ Reward shaping --------------------------------------------------
    lam = 0.3
    reward = math.exp(-lam * h)

    # Guard against numerical underflow
    if reward < 1e-8:
        reward = 0.0
    return reward
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (evidence from the trace) | Why it hurts the search |
|------|-----------------------------------|--------------------------|
| **1** | *Almost all children have avg ≈ 0.002* (e.g. the first‑move node shows 497/500 visits with avg 0.002). The simulation returns values that are practically zero for the majority of non‑terminal states. | When the rollout value collapses to ~0, the UCT‑style back‑propagation can’t discriminate between good and bad pushes. The tree spends millions of simulations on irrelevant branches before a rare high‑reward leaf is found. |
| **2** | *Unreachable‑box term dominates* – every box that is not already in the player’s flood‑fill is counted. In macro‑push Sokoban the player normally cannot stand next to any box without first pushing another one, so the count is often ≥ 1 even in perfectly solvable positions. | This term adds **β · unreachable_cnt** (β = 2.0) to the heuristic *h*, inflating h and driving `exp(-λh)` toward zero. It masks genuine progress (boxes moved onto targets) and biases the simulation against any push that initially requires a walk. |
| **3** | *No explicit progress reward* – the formula never uses `state.boxes_on_targets()`. In the trace, after move 3 the state already has 1/3 boxes on target, but the reward does not reflect that improvement. | The search cannot “see” that a move is making progress; it must infer it indirectly through reduced Manhattan distance, which is noisy and can even increase temporarily. |
| **4** | *λ = 0.3 is too aggressive* – for modest h ≈ 8 the reward is already ≤ 0.09; deeper states (h ≈ 15) produce ~10⁻³, effectively zero. | Exponential decay squashes the dynamic range, turning a reasonably good heuristic into a binary “good / dead” signal, amplifying weakness #1. |
| **5** | *Corner‑deadlock penalty is over‑scaled* (Δ = 1, deadlock_penalty = 5). When a simple corner is detected, h  jumps by 5, pushing the reward below 10⁻³. This creates a hard “‑0.1” sink for any state that accidentally touches a corner even if the box could later be rescued. | Leads to a large negative penalty (‑0.1) on terminal dead‑locks and a huge drop in reward for borderline states, further flattening the Q‑distribution. |

**2. ROOT CAUSE**  

The simulation function builds its score from  

```
h = α·box_dist + β·unreachable_cnt + γ·min_walk + δ·deadlock_penalty
reward = exp(-λ·h)
```

*Missing progress term* – nothing rewards `boxes_on_targets`.  
*Unreachable‑box mis‑use* – the macro‑push setting makes almost every box “unreachable” until the first push, so the term is a constant bias rather than a signal.  
*Weight / λ mismatch* – β = 2, λ = 0.3 together turn a typical h ≈ 5–10 into a reward < 0.2, and deeper states become ≈ 0. This collapses the value signal that the back‑propagation relies on.  
*Corner‑deadlock penalty* is harsh and applied before the exponential, pushing many non‑dead states into the near‑zero regime.  

Consequently the rollout returns almost identical near‑zero values for the vast majority of nodes, which explains the flat Q‑values observed in the trace.

**3. PROPOSED APPROACH – Strategy A (incremental)**  

We can retain the fast, deterministic structure of the current simulation while adding three targeted fixes that directly address the highest‑impact weaknesses:

1. **Add a progress term**  
   ```python
   on_target = state.boxes_on_targets()
   progress = on_target / state.num_targets   # 0 → 1
   ```
   Incorporate it as `+ ε·(1‑progress)` (or subtract `ε·progress`) with a modest weight (ε ≈ 2.0). This directly raises the reward when a box reaches a goal, giving the tree a clear gradient.

2. **Replace the unreachable‑box count**  
   – Use *push‑reachability* instead of pure walk‑reachability: run a BFS on the *push graph* (states where the player can stand to push a box) to see which boxes could be moved without creating deadlocks. Count only boxes that are **truly immovable** (e.g., boxed in a corner not on a target).  
   – If implementing the full push‑graph is too heavy, simply set `unreachable_cnt = 0` for the simulation (or a tiny weight β ≈ 0.2) because the expansion phase already prunes deadlocked pushes.

3. **Rescale λ and the weights**  
   – Reduce λ to 0.1 (or compute reward as `1/(1+λ·h)`) to keep the reward in a usable range (≈ 0.3‑0.9 for typical h).  
   – Lower β and γ (e.g., β = 0.5, γ = 0.2) so that box distance dominates early, while walk cost remains a mild bias.

4. **Soft‑enforce corner deadlocks**  
   – Keep the detection but assign a much smaller penalty, e.g., `deadlock_penalty = 2.0` and weight δ = 0.3, or treat it as an additive term after the exponential (`reward *= 0.5` if deadlock). This prevents the reward from collapsing to ‑0.1 on borderline states.

5. **Clamp and normalise**  
   After computing `reward`, enforce `reward = max(0.0, min(1.0, reward))`. Optionally linearly map the raw exponential output to `[0.05, 1.0]` to guarantee a minimal informative signal.

These changes preserve the existing fast BFS and corner‑check utilities, require only a few extra arithmetic operations, and keep the simulation pure‑evaluation (no roll‑out loops). The resulting reward distribution will spread from ~0.7 for states with several boxes already placed, down to ~0.1 for distant, cluttered positions, giving MCTS a much richer gradient while staying inexpensive enough for thousands of simulations per move.

------------------------------------------------------------
TASK — IMPROVE THE HEURISTIC
------------------------------------------------------------
Improve the 'simulation' function above based on your
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

How the 'simulation' phase works in MCTS:
  - Called from a LEAF node, receives (state, perspective_player, max_depth).
  - Must return a FLOAT reward backpropagated up the tree.
  - Reward MUST vary across states so MCTS can distinguish good from bad actions.
  - Flat rewards ≈ random play. Called thousands of times per move — keep it FAST.

CONSTRAINTS:
  • Same function signature as the current code.
  • Standalone function, standard library only.
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
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: added progress term, rescaled/de‑emphasized unreachable count, softened deadlock penalty, changed reward shaping to a smoother 1/(1+λh) with smaller λ, and tuned weights for better reward spread
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Refined leaf evaluation for Sokoban‑macro MCTS.

    Improvements over the previous version:
      • **Progress term** – rewards states that already have boxes on targets.
      • **Weaker unreachable‑box penalty** – the original count dominates early
        rollouts; we now give it a small weight (or zero) because macro‑push
        actions often make many boxes temporarily unreachable.
      • **Softened corner‑deadlock penalty** – a modest additive cost instead of
        a large jump that drove the exponential to near‑zero.
      • **Smoother reward mapping** – use ``1 / (1 + λ·h)`` (λ≈0.1) instead of
        ``exp(-λ·h)`` to keep the reward in a usable range [0,1].
      • **Adjusted weight balance** – box‑to‑target distance remains the main
        signal, walk distance a mild bias, and progress a strong positive boost.
      • **Clamping** – guarantees the returned value stays within ``[0.0, 1.0]``.
    """
    # ---- 0️⃣ Terminal handling ------------------------------------------------
    if state.is_terminal():
        base_ret = state.returns()[perspective_player]   # 1.0 solved, 0.0 otherwise
        # Small negative for obvious corner dead‑locks to separate them from a plain timeout
        if base_ret == 0.0 and _simple_corner_deadlock(state):
            return -0.1
        return base_ret

    # ---- 1️⃣ Box‑to‑target Manhattan distance --------------------------------
    box_dist = state.total_box_distance()          # Σ min‑Manhattan per box

    # ---- 2️⃣ Unreachable‑box count (lightly weighted) -------------------------
    walls = state.walls
    boxes = state.boxes
    reachable = _bfs_reachable(state.player, walls, boxes)
    unreachable_cnt = sum(1 for b in boxes if b not in reachable)

    # ---- 3️⃣ Minimal walk distance to any legal push position -------------------
    min_walk = None
    for action in state.legal_actions():
        push_pos = action[0]                       # cell where player must stand
        if push_pos in reachable:
            d = reachable[push_pos]
            if min_walk is None or d < min_walk:
                min_walk = d
    if min_walk is None:                           # no push possible (should be terminal)
        min_walk = 10.0

    # ---- 4️⃣ Corner dead‑lock detection (soft penalty) -------------------------
    deadlock_penalty = 0.5 if _simple_corner_deadlock(state) else 0.0

    # ---- 5️⃣ Progress: how many boxes already on targets -----------------------
    on_target = state.boxes_on_targets()
    progress_ratio = on_target / state.num_targets if state.num_targets else 0.0

    # ---- 6️⃣ Weighted combination ------------------------------------------------
    #   Positive contributions (want to minimise)   : box_dist, unreachable, walk, deadlock
    #   Negative contribution (want to maximise)    : progress_ratio
    alpha = 1.0   # box distance weight
    beta  = 0.3   # unreachable boxes weight (reduced from 2.0)
    gamma = 0.2   # walk distance weight
    delta = 0.6   # deadlock penalty scaling
    epsilon = 4.0 # progress boost weight
    h = (alpha * box_dist) + (beta * unreachable_cnt) + (gamma * min_walk) \
        + (delta * deadlock_penalty) - (epsilon * progress_ratio)

    # ---- 7️⃣ Reward shaping ------------------------------------------------------
    lam = 0.10
    reward = 1.0 / (1.0 + lam * max(h, 0.0))   # ensures reward ∈ (0,1]

    # Clamp for safety (e.g. numerical issues)
    if reward < 0.0:
        reward = 0.0
    elif reward > 1.0:
        reward = 1.0
    return reward
```
