# step2_generation

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:27:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.77s |
| Status     | success |
| Tokens     | 9665 |
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
LLM-generated MCTS tool: expansion
Description: 
Generated:   2026-03-11T01:09:23.036760
"""

def default_expansion(node):
    """
    Expand the most promising untried macro‑push action.

    Improvements (preserved):
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
    import sys
    from collections import deque
    from mcts.node import MCTSNode

    # ------------------------------------------------------------------
    # Helper 1: BFS walk length (player moves only, boxes are static)
    # ------------------------------------------------------------------
    def bfs_walk_len(state, start, goal):
        """Return shortest walk length from start to goal avoiding walls & boxes.
        If unreachable, return sys.maxsize."""
        if start == goal:
            return 0
        walls = state.walls
        boxes = state.boxes
        width, height = state.width, state.height

        # simple 4‑direction moves
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in walls or (nx, ny) in boxes:
                    continue
                if (nx, ny) == goal:
                    return nd
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, nd))
        return sys.maxsize

    # ------------------------------------------------------------------
    # Helper 2: Very cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def is_simple_deadlocked(state):
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        # pre‑compute wall adjacency for quick checks
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # check four corner patterns
            if ((bx + 1, by) in walls or (bx - 1, by) in walls) and \
               ((bx, by + 1) in walls or (bx, by - 1) in walls):
                # Two perpendicular walls -> corner
                # Need both a horizontal and a vertical wall adjacent
                horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
                vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
                if horiz and vert:
                    return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Per‑root visited dictionary (persistent across calls)
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # --------------------------------------------------------------
    # 0. Reset shared visited dictionary if we are at a new root.
    # --------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._root_key = root_key

    visited = default_expansion._visited

    # --------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # --------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # --------------------------------------------------------------
    # 2. Scan untried actions.
    # --------------------------------------------------------------
    alpha = 0.5                     # weight for walk distance in the heuristic
    scored = []                     # (h, action, next_state, key, g_new)
    deadlocked_to_remove = []

    for action in list(node._untried_actions):
        player_pos, direction = action

        # a) walk distance from current player location to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # ----- static dead‑lock detection (permanent) -----
        if is_simple_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # ----- visited pruning (temporary) -----
        g_new = cur_steps + walk_len + 1   # real cost to reach this macro‑state
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            # keep the action for potential later expansion
            continue

        # ----- heuristic scoring -----
        # box‑only Manhattan distance (provided by GameState)
        box_h = next_state.total_box_distance()
        h = box_h + alpha * walk_len
        scored.append((h, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # --------------------------------------------------------------
    # 3. Choose the best scored action (lowest h). If none, fall back.
    # --------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        _, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # No scored actions – fall back to any remaining untried action.
        if node._untried_actions:
            chosen_action = node._untried_actions.pop()
            player_pos, _ = chosen_action
            walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
            chosen_state = node.state.clone()
            chosen_state.apply_action(chosen_action)
            chosen_key = chosen_state.state_key()
            chosen_g = cur_steps + walk_len + 1
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
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: 
Generated:   2026-03-11T01:07:22.346016
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched leaf evaluation for MCTS simulation.

    Returns:
        1.0                                    if the state is solved,
        state.returns()[p]                     if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1]    based on box distance,
                                                player walk cost, and target progress.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """
        Compute the shortest‑path distance from `start` to every reachable
        cell on the board, treating both walls and boxes as obstacles.
        Returns a dict {(r, c): distance}.
        """
        frontier = deque([start])
        distances = {start: 0}
        # 4‑neighbour moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while frontier:
            cur = frontier.popleft()
            cur_dist = distances[cur]

            for dr, dc in moves:
                nr, nc = cur[0] + dr, cur[1] + dc
                # stay inside the board
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                # cannot step onto a wall or a box
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in distances:
                    distances[nxt] = cur_dist + 1
                    frontier.append(nxt)
        return distances

    # ------------------------------------------------------------------
    # Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # `returns()` returns a sequence indexed by player id.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    # The public API provides `total_box_distance()` which returns the
    # sum of Manhattan distances from each box to its nearest target.
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 2️⃣ Player walk cost to the *closest* push position (w)
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    # Compute distances from the player's current location.
    distances = bfs_distances(state.player, walls, boxes, height, width)

    # Determine the minimal distance to any push position that is legal.
    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    # If there is no reachable push position, treat the walk cost as 0.
    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 3️⃣ Fraction of boxes already on targets (f)
    # ------------------------------------------------------------------
    # `boxes_on_targets()` returns the number of boxes that sit on targets.
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 4️⃣ Combine the three components.
    #    α = walk‑cost weight, γ = target‑completion bonus weight.
    # ------------------------------------------------------------------
    ALPHA = 0.1   # penalty for long walks
    GAMMA = 0.4   # stronger bonus for having boxes placed

    # Core reward from distance & walk cost (never exceeds 1.0).
    base = 1.0 / (1.0 + h + ALPHA * w)

    # Add the target‑completion bonus.
    reward = base + GAMMA * f

    # Clamp to the valid range.
    reward = max(0.0, min(1.0, reward))

    # Special case: all boxes on targets → perfect reward.
    if h == 0:
        reward = 1.0

    return reward
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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**

| Impact | Observation (trace) | Why it hurts |
|--------|---------------------|--------------|
| **① No exploration term** – the algorithm is pure greedy. | After the first expansion the child `((4,4),0)` gets 497/500 visits and dominates every later decision. The same pattern repeats at every depth (e.g., move 5: `((2,2),3)` 491/500). | With `exploration_weight` never used, the tree collapses onto a single branch; promising alternatives never receive enough roll‑outs to be discovered. |
| **② Tie‑break on average value reinforces the dominant branch**. | When two children have identical `f = g+h`, the code selects the one with the higher `q = value/visits`. The dominant child already has a higher `q`, so it wins all future ties, locking the search. | This creates a positive feedback loop – the “winner” gets more visits → higher `q` → wins more ties → never challenged. |
| **③ Heuristic `h_sokoban` is too coarse** (in the shared module it is essentially *total box distance*). | All children in a given level often share the same Manhattan‑sum (e.g., after move 3 the total distance drops from 7→5 regardless of which push is taken). Consequently `f` values are identical for many actions, forcing the algorithm to rely on the `q`‑tie‑break above. | The heuristic does not reward *boxes on targets* or penalise *long walks* / *simple dead‑locks*, so it cannot discriminate between a good push (creates a target) and a neutral push. |
| **④ Depth `g` does not reflect true macro‑push cost**. | `g` is taken from `_ag.node_depth(child)`, i.e. number of tree edges, while a macro‑push may cost 1 (step) or many walk steps. The selection therefore treats a costly push the same as a cheap one. | The planner can be steered into high‑cost pushes that do not improve the board, wasting roll‑outs. |
| **⑤ Static dead‑lock detection lives only in expansion**. | If a child’s state is a dead‑lock that escaped the expansion filter (e.g., a corner created later), its `h` remains low; the selection keeps descending into it, wasting iterations (see move 6 where the path stalls with distance 3). | No penalty at selection time means the search can still follow a dead‑end branch for many iterations. |

**2. ROOT CAUSE**

The `default_selection` function implements:

```python
f = g + h
if f < best_f or (f == best_f and q > best_q):
    best = child
```

* The *exploration term* from UCB1 (`c * sqrt(log(N)/n)`) is completely omitted, so the algorithm never intentionally samples low‑visit nodes.  
* `q` is used only as a tie‑breaker, which magnifies exploitation bias when `f` values are equal—a frequent situation because `h_sokoban` provides little granularity.  
* `g` is a pure depth count, not the accumulated macro‑push cost (`state.steps`).  
* No additional progress signal (boxes on targets) or dead‑lock penalty is incorporated into `h`.  

Consequently the search quickly converges to the first “good enough” child and stays there, even though other actions could increase the number of boxes on targets (as seen at moves 2–4 where a different push would have placed a second box earlier).

**3. PROPOSED APPROACH – Strategy A (incremental)**  

We can keep the overall A*‑style best‑first walk but **augment the scoring** so that it regains exploration, respects true macro cost, and values progress/dead‑lock avoidance.

* **Blend a lightweight UCB term** into the score:  
  `score = g + h - β * progress_bonus + exploration_weight * sqrt(log(parent_visits) / child_visits)`  
  – `exploration_weight` is the same parameter already passed (default 1.41).  
  – This restores systematic sampling of under‑visited children while still favouring low `f`.

* **Add a progress bonus** that rewards boxes already on targets:  
  `progress_bonus = η * (boxes_on_target / num_targets)` (η≈0.5).  
  This makes pushes that create a target lower the overall score, encouraging the tree to move toward solved states.

* **Penalise long macro‑pushes** using the true cost stored in the state (`state.steps` or the walk length used in expansion):  
  `g = child.state.steps` instead of pure depth. This differentiates cheap pushes from expensive ones.

* **Dead‑lock penalty** derived from the same cheap corner check used in expansion:  
  `h += κ * deadlock_indicator` (κ large, e.g. 5). This pushes dead‑locked branches upward in `f`, making them unlikely to be chosen.

* **Tie‑break on fewer visits only when scores are identical** (remove the `q`‑based tie‑break). Let the UCB term handle exploration; `q` can stay out of selection entirely.

With these modest changes the selection function still walks down a best‑first path, but it now:

1. **Explores** low‑visit alternatives regularly, preventing the 495/500 dominance seen in the trace.  
2. **Prioritises** actions that actually increase the number of placed boxes, addressing the plateau at moves 4‑6.  
3. **Respects macro‑push cost**, avoiding wasteful long walks that stall progress.  
4. **Avoids known dead‑locks** early, freeing iterations for productive branches.

Because the current code already provides `node.state`, `node.parent.visits`, and a simple heuristic function, implementing the above extensions requires only a few extra calculations and keeps the overall A*‑guided structure intact. This incremental refinement should yield a markedly higher discovery rate of solutions without the overhead of a full redesign.

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
DESCRIPTION: Added lightweight UCB exploration term, progress bonus for boxes on targets, true macro‑push cost (state.steps) as g, and dead‑lock penalty; removed q‑based tie‑break to reduce exploitation bias.
```python
def default_selection(node, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with lightweight UCB exploration and Sokoban‑specific
    progress / dead‑lock signals.

    - g  : accumulated macro‑push cost (state.steps) – real cost so far.
    - h  : base heuristic from shared module (e.g., total box distance).
    - p  : progress bonus proportional to fraction of boxes already on targets.
    - d  : dead‑lock penalty (simple corner check).
    - u  : UCB‑style exploration term using parent and child visit counts.

    The child with the *lowest* (g + h - p + u) is selected.
    Tie‑break falls back to fewer visits.
    """
    import math

    # ── Detect new MCTS search → reset shared A* state ───────────────
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if root.visits == 0 or root_key != _ag._root_key:
        _ag.reset(root_key)

    # ------------------------------------------------------------------
    # Helper: cheap static dead‑lock detection (corner dead‑locks)
    # ------------------------------------------------------------------
    def _is_deadlocked(state) -> bool:
        """Return True if any non‑target box sits in an immovable corner."""
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            # two orthogonal walls adjacent → corner
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Walk down the tree until we hit a node that needs expansion.
    # ------------------------------------------------------------------
    while not node.is_terminal:
        if not node.is_fully_expanded:
            return node   # let expansion handle it

        best_child = None
        best_score = math.inf
        best_visits = math.inf  # for deterministic tie‑break on fewer visits

        # parent visit count for UCB term (avoid log(0))
        parent_visits = max(1, node.visits)

        for child in node.children.values():
            # ----- g : real macro‑push cost accumulated so far -----
            g = getattr(child.state, "steps", 0)

            # ----- h : base heuristic from shared module -----
            h = _ag.h_sokoban(child.state)

            # ----- dead‑lock penalty -----
            if _is_deadlocked(child.state):
                h += 5.0          # heavy penalty to push dead‑ends up

            # ----- progress bonus (more boxes on targets → lower score) -----
            num_targets = child.state.num_targets
            if num_targets:
                progress = child.state.boxes_on_targets() / num_targets
            else:
                progress = 0.0
            PROGRESS_WEIGHT = 0.5
            progress_bonus = PROGRESS_WEIGHT * progress

            # ----- UCB exploration term -----
            child_visits = child.visits
            # Add a tiny epsilon to avoid division by zero; exploration term
            # vanishes for unvisited nodes (since log(parent_visits)/0 -> large,
            # we treat unvisited as maximal exploration by using 1e-9 denominator).
            exploration = exploration_weight * math.sqrt(
                math.log(parent_visits) / (child_visits + 1e-9)
            )

            # ----- total selection score (lower is better) -----
            score = g + h - progress_bonus + exploration

            # Choose the child with the minimal score; break ties on fewer visits.
            if (score < best_score) or (abs(score - best_score) < 1e-9 and child_visits < best_visits):
                best_child = child
                best_score = score
                best_visits = child_visits

        if best_child is None:
            # Should not happen, but fallback to any child.
            return node
        node = best_child

    return node
```
