# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:09:08 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 16.14s |
| Status     | success |
| Tokens     | 8438 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: expansion).

PHASE: expansion
  • What it does: Creates new child nodes from a frontier node. Decides which actions to materialize into the tree and in what order.
  • Optimization goal: PRUNE bad actions and ORDER remaining actions so promising ones are tried first. Filter deadlocks before they enter the tree.
  • Constraints: Best place for hard constraints. Order actions; optionally filter some entirely. No rollout policies or value aggregation.
  • Good patterns: reject pushes into non-target corners, reject wall deadlocks, prefer pushes that reduce box distance, deprioritize no-op player movement.
  • Avoid: long rollout policies, reward aggregation, node-value update rules.

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
TARGET HEURISTIC TO IMPROVE (expansion)
------------------------------------------------------------
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

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. Key Weaknesses**

| Rank | Symptom (from the trace) | Why it hurts performance |
|------|---------------------------|--------------------------|
| **1** | The tree quickly concentrates on a single macro‑push – action **((3,6), 0)** receives **483/500** visits while every other legal push is explored only once. The average value of that branch (≈0.42) is modest, yet it dominates the search, preventing any exploration of alternative push orders that could free the remaining box. | The expansion routine **prunes too aggressively**: once a state is seen with a g‑score ≤ the current depth, *all* other actions leading to that same state are removed permanently from `_untried_actions`. Because the macro‑push cost is **not** reflected in `g_new` (depth + 1) the algorithm treats a long walk‑push and a nearby push as equal, so the first cheap‑looking action wins and the rest are discarded. |
| **2** | The heuristic used for ordering (`h_sokoban_box_only`) ignores the player’s walking cost and any structural deadlocks. After three pushes the puzzle is at distance 1, but the remaining box is stuck in a corner that is not a target (see final map). | A box‑only Manhattan sum rewards *any* reduction in distance, even when the push creates a classic Sokoban deadlock (box in a non‑target corner or against a wall with no target on that line). Because the expansion never penalises such configurations, the search “thinks” they are promising and spends the majority of its budget on them. |
| **3** | The only dead‑lock test called is `next_state._is_deadlocked()`, which in the current code checks only the most obvious static dead‑locks. The final state is still a *simple* dead‑lock (box wedged in a corner) that slips through. | Missing detection of **simple frozen dead‑locks** (corner, wall‑line, 2‑box lock) means the algorithm expands states that are hopeless, wasting iterations and inflating the value of a bad branch. |

**2. Root Cause**

* **Cost‑agnostic g‑score** – `g_new = node_depth + 1` treats every macro‑push as unit cost, while the real step cost is `walk_steps + 1`. Consequently the visited‑pruning logic (`visited[key] <= g_new`) discards alternatives that would be cheaper in real steps but appear more expensive in depth, leading to over‑pruning and loss of diversity.

* **Over‑simplified heuristic** – scoring solely with `h_sokoban_box_only` favours any reduction in box‑to‑target Manhattan distance, ignoring (a) the player’s walking distance and (b) structural dead‑lock patterns. Hence pushes that look good heuristically actually worsen solvability.

* **Permanent removal of pruned actions** – once an action is discarded because its resulting state was previously visited with a lower g, it is never reconsidered, even if later box moves change the context (e.g., freeing a corridor). This “once‑pruned‑forever” policy eliminates potentially useful reorderings.

**3. Proposed Approach – A (Incremental Improvements)**  

The weaknesses are orthogonal and can be remedied by modest extensions to the existing expansion routine; a full rewrite is unnecessary.

| Modification | Rationale & Expected Effect |
|--------------|------------------------------|
| **a. Use true accumulated step cost as g** – compute `g_new = node.state.steps + walk_cost(action) + 1` (the `walk_cost` can be obtained from the BFS length the state computes for the macro‑push). Replace the depth‑only `g_new`. | Visited pruning now respects the actual cost of reaching a state, preventing cheap‑looking actions from eclipsing cheaper‑walking pushes. It also reduces premature pruning of actions that would achieve the same box configuration with fewer steps. |
| **b. Enrich the scoring heuristic** – compute `h = box_distance + α * walk_cost(action)`, where `α` (≈0.5) balances box‑only distance against the immediate walking effort. | Actions that require a long walk receive a higher score, pushing them later in the expansion order. This steers the tree toward pushes that are both distance‑reducing *and* cheap to reach. |
| **c. Add lightweight static dead‑lock detection** – before scoring, reject a next state if any box satisfies: <br>  • is in a non‑target corner (adjacent to walls on two orthogonal sides); <br>  • lies on a wall line where no target exists in the same row/column (simple “wall‑lock”). Implement these checks directly in the expansion loop (no external module needed). | Eliminates the majority of hopeless pushes that currently survive the `_is_deadlocked` test, keeping the search focused on genuinely solvable branches. |
| **d. Change permanent pruning to *temporary* per‑node** – keep the `pruned` list only for the current expansion call (i.e., remove the actions from `_untried_actions` *only* when they are expanded). Do **not** delete actions that fail the visited check; instead leave them for future expansion attempts when the cumulative cost may improve. | Retains alternative push orders that could become optimal after other boxes move, preserving diversity without exploding the branching factor. |

These four tweaks preserve the overall A*‑guided structure (visited dictionary, ordering by `h`) while fixing the cost mis‑alignment, adding a more discriminating heuristic, and strengthening dead‑lock filtering. They directly target the top‑ranked weaknesses observed in the trace and should increase both the exploration of promising pushes and the overall solve rate on level 7 (and the remaining unsolved levels).

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
"""
A*-guided MCTS expansion – incremental improvements.

Key enhancements:
  * Use the real macro‑push cost (walk distance + 1) as g‑score for
    visited‑pruning, instead of the shallow `depth+1`.
  * Enrich the ordering heuristic:
        h = box_distance + α * walk_distance
    (α balances player walking effort against box‑only Manhattan distance).
  * Add cheap static dead‑lock detection (corner and wall‑line dead‑locks)
    and permanently discard those actions.
  * Do **not** permanently discard actions that fail the visited check;
    they stay in `_untried_actions` for possible later expansion when the
    context changes.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from collections import deque
from typing import Tuple, List, Set

# ── Load shared A* state ──────────────────────────────────────────────
_KEY = "astar_globals"
if _KEY not in sys.modules:
    _p = Path(__file__).resolve().parent.parent / "shared" / "astar_globals.py"
    _s = importlib.util.spec_from_file_location(_KEY, str(_p))
    _m = importlib.util.module_from_spec(_s)
    sys.modules[_KEY] = _m
    _s.loader.exec_module(_m)
import astar_globals as _ag


# ----------------------------------------------------------------------
# Helper utilities – kept inside the function file for self‑containment.
# ----------------------------------------------------------------------
def _bfs_walk_len(state, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> int:
    """Return shortest walk length from start to goal avoiding walls & boxes.
    Returns a large number if unreachable (treated as dead‑lock later)."""
    if start == goal:
        return 0
    walls: Set[Tuple[int, int]] = state.walls
    boxes: Set[Tuple[int, int]] = state.boxes
    height, width = state.height, state.width

    q = deque([start])
    dist = {start: 0}
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in walls or (nr, nc) in boxes:
                continue
            if (nr, nc) in dist:
                continue
            if (nr, nc) == goal:
                return d + 1
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    # unreachable
    return sys.maxsize


def _is_simple_deadlocked(state) -> bool:
    """Detect corner dead‑locks and wall‑line dead‑locks for non‑target boxes."""
    walls = state.walls
    targets = state.targets
    h, w = state.height, state.width

    for (br, bc) in state.boxes:
        if (br, bc) in targets:
            continue  # box already on a target is safe

        # ==== corner dead‑lock (two orthogonal walls) ====
        up = (br - 1, bc) in walls
        down = (br + 1, bc) in walls
        left = (br, bc - 1) in walls
        right = (br, bc + 1) in walls
        if (up and left) or (up and right) or (down and left) or (down and right):
            return True

        # ==== wall‑line dead‑lock (box on wall with no target on same line) ====
        if br == 0:  # top wall
            if not any(t[0] == 0 for t in targets):
                return True
        if br == h - 1:  # bottom wall
            if not any(t[0] == h - 1 for t in targets):
                return True
        if bc == 0:  # left wall
            if not any(t[1] == 0 for t in targets):
                return True
        if bc == w - 1:  # right wall
            if not any(t[1] == w - 1 for t in targets):
                return True

    return False


def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements over the previous version:
      * Uses true macro‑push cost (walk steps + 1) as g‑score for visited
        pruning.
      * Orders actions by a richer heuristic:
          h = box_distance + α * walk_distance   (α = 0.5)
      * Performs lightweight static dead‑lock detection and permanently
        removes those actions.
      * Does **not** permanently discard actions that fail the visited check;
        they stay in `_untried_actions` for future attempts.
    """
    # --------------------------------------------------------------
    # 0. Reset shared A* visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # --------------------------------------------------------------
    # 1. Compute the g‑score for this node using the real step count.
    # --------------------------------------------------------------
    # current state's step count (already accumulated)
    cur_steps = getattr(node.state, "steps", 0)
    # g_new will be the total steps after performing the candidate action
    # (walk_len + 1 for the push). We will compute walk_len per action later.
    visited = _ag.get_visited()

    # --------------------------------------------------------------
    # 2. Scan untried actions, prune dead‑locks permanently,
    #    score the rest, but keep visited‑pruned actions for later.
    # --------------------------------------------------------------
    alpha = 0.5                      # weighting for walk distance in heuristic
    scored: List[Tuple[float, Tuple[Tuple[int, int], int],
                      object, str, int]] = []   # (h, action, next_state, key, walk_len)
    deadlocked_to_remove: List[Tuple[Tuple[int, int], int]] = []

    for action in list(node._untried_actions):
        # action = ((pr, pc), direction)
        player_pos, direction = action
        # Compute walk length from current player location to the required position.
        walk_len = _bfs_walk_len(node.state,
                                 node.state.player,
                                 player_pos)
        if walk_len == sys.maxsize:
            # unreachable – treat as deadlocked (cannot execute this push)
            deadlocked_to_remove.append(action)
            continue

        # Apply action on a cloned state.
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ---- a) Simple static dead‑lock detection (permanent) ----
        if hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked():
            deadlocked_to_remove.append(action)
            continue
        if _is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ---- b) visited pruning (temporary) ----
        # Real g for this transition:
        g_new = cur_steps + walk_len + 1
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # Do NOT remove from _untried_actions; just skip scoring.
            continue

        # ---- c) heuristic scoring ----
        box_h = _ag.h_sokoban_box_only(next_state)   # box‑only Manhattan distance
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # Permanently remove dead‑locked actions.
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h).  If none, fall back.
    # --------------------------------------------------------------
    if scored:
        # Sort by heuristic, then deterministic action order.
        scored.sort(key=lambda x: (x[0], x[1]))
        h_best, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        # Remove ONLY the chosen action from the untried list.
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        # Record best g for this state in the shared visited map.
        visited[chosen_key] = chosen_g
    else:
        # No scored actions (either all dead‑locked or visited‑pruned).
        # Fallback: expand any remaining untried action (if any).
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            # Compute its g for visited bookkeeping.
            walk_len = _bfs_walk_len(node.state,
                                     node.state.player,
                                     chosen_action[0])
            chosen_g = cur_steps + walk_len + 1
            visited[chosen_key] = chosen_g
        else:
            # No actions left – return an existing child if present,
            # otherwise the node itself (terminal‑like safety net).
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # 4. Create the child node and register it.
    # --------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
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
- The fallback branch overwrites the shared `visited` entry with a potentially higher `g` value, weakening the pruning logic.
- Minor redundancy: walk length is recomputed for the chosen action after it has already been calculated in the loop (when the action was visited‑pruned). This is acceptable but can be streamlined.
- Overall logic and API usage are correct; no crashes expected.

ACTION: modify
FILE_NAME: expansion.py
FUNCTION_NAME: default_expansion
DESCRIPTION: Prevent overwriting a better g‑score in the visited map during the fallback expansion and streamline walk‑length handling.
```python
def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements:
      * Uses true macro‑push cost (walk steps + 1) as g‑score for visited
        pruning.
      * Orders actions by a richer heuristic:
          h = box_distance + α * walk_distance   (α = 0.5)
      * Performs lightweight static dead‑lock detection and permanently
        removes those actions.
      * Does **not** permanently discard actions that fail the visited check;
        they stay in `_untried_actions` for future attempts.
      * In the fallback case, the visited map is only updated if the new
        g‑score is better than any previously recorded one.
    """
    # --------------------------------------------------------------
    # 0. Reset shared A* visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if getattr(_ag, "_root_key", None) != root_key:
        _ag.reset(root_key)

    # --------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # --------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)
    visited = _ag.get_visited()

    # --------------------------------------------------------------
    # 2. Scan untried actions.
    # --------------------------------------------------------------
    alpha = 0.5  # weight for walk distance in the heuristic
    scored = []  # (h, action, next_state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action

        # walk distance from current player location to the required position
        walk_len = _bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- a) static dead‑lock detection (permanent) -----
        if (hasattr(next_state, "_is_deadlocked") and next_state._is_deadlocked()) \
                or _is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ----- b) visited pruning (temporary) -----
        g_new = cur_steps + walk_len + 1
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for possible later expansion
            continue

        # ----- c) heuristic scoring -----
        box_h = _ag.h_sokoban_box_only(next_state)   # box‑only Manhattan distance
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h).  If none, fall back.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        h_best, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No scored actions – fall back to any remaining untried action.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            player_pos, _ = chosen_action
            walk_len = _bfs_walk_len(node.state, node.state.player, player_pos)
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            chosen_g = cur_steps + walk_len + 1
            # update visited only if this g is an improvement
            if (chosen_key not in visited) or (visited[chosen_key] > chosen_g):
                visited[chosen_key] = chosen_g
        else:
            # No actions left – return an existing child or the node itself.
            if node.children:
                return next(iter(node.children.values()))
            return node

    # --------------------------------------------------------------
    # 4. Create and register the child node.
    # --------------------------------------------------------------
    from mcts.node import MCTSNode
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

