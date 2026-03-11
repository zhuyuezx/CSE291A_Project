# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:34:20 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 12.93s |
| Status     | success |
| Tokens     | 11117 |
| Validation | invalid — No function definition found in code. |

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
Description: Incremental heuristic improvements – delta distance penalty, target‑placement bonus, stronger dead‑lock detection, and increased walk‑cost weight.
Generated:   2026-03-11T15:29:43.565517
"""

def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions.

    New features:
      * Penalise pushes that increase total box‑Manhattan distance.
      * Reward pushes that place a box on a target.
      * Detect additional deadlocks: boxes against a wall with no target
        reachable in that direction (wall‑line deadlocks) and keep the
        original corner check.
      * Increase walk‑cost weight (α) to discourage cheap, irrelevant pushes.
    """
    import sys
    from collections import deque
    from mcts.node import MCTSNode

    # ------------------------------------------------------------------
    # Helper 1: BFS walk length (player moves only, boxes are static)
    # ------------------------------------------------------------------
    def bfs_walk_len(state, start, goal):
        """Shortest walk length from start to goal avoiding walls & boxes.
        Returns sys.maxsize if unreachable."""
        if start == goal:
            return 0
        walls = state.walls
        boxes = state.boxes
        height = state.height   # rows
        width = state.width     # columns

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q = deque()
        q.append((start[0], start[1], 0))
        visited = {start}
        while q:
            x, y, d = q.popleft()
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < height and 0 <= ny < width):
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
    # Helper 2: Simple corner dead‑lock detection
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert  = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Wall‑line dead‑lock detection.
    # A box adjacent to a wall where no target exists beyond that wall
    # in the same line ⇒ the box can never be moved to a target.
    # ------------------------------------------------------------------
    def is_wall_line_deadlocked(state, box_pos):
        walls = state.walls
        targets = state.targets
        h = state.height
        w = state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in dirs:
            neighbor = (box_pos[0] + dx, box_pos[1] + dy)
            if neighbor not in walls:
                continue
            # Look past the wall in the same direction for a target.
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # A target exists beyond the wall → not a deadlock.
                    return False
                step += 1
            # No target found in this direction → deadlock.
            return True
        return False

    # ------------------------------------------------------------------
    # Helper 4: Aggregate dead‑lock test (corner + wall‑line)
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 5: Per‑root visited dictionary (persistent across calls)
    # ------------------------------------------------------------------
    if not hasattr(default_expansion, "_visited"):
        default_expansion._visited = {}
        default_expansion._root_key = None

    # Reset visited map when we encounter a new root state.
    root = node
    while root.parent is not None:
        root = root.parent
    root_key = root.state.state_key()
    if default_expansion._root_key != root_key:
        default_expansion._visited.clear()
        default_expansion._root_key = root_key
    visited = default_expansion._visited

    # ------------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # ------------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Scan untried actions, compute heuristic scores.
    # ------------------------------------------------------------------
    ALPHA = 1.0     # weight for walk distance
    BETA  = 1.5     # penalty for increasing box distance
    GAMMA = 0.8     # bonus for pushing a box onto a target

    old_box_dist = node.state.total_box_distance()
    scored = []                     # (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    # direction vectors for macro‑pushes: UP, DOWN, LEFT, RIGHT
    drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for action in list(node._untried_actions):
        player_pos, direction = action

        # a) walk distance to the required push position
        walk_len = bfs_walk_len(node.state, node.state.player, player_pos)
        if walk_len == sys.maxsize:
            deadlocked_to_remove.append(action)
            continue

        # b) apply action on a cloned state
        next_state = node.state.clone()
        next_state.apply_action(action)

        # c) static dead‑lock detection (permanent removal)
        if is_deadlocked(next_state):
            deadlocked_to_remove.append(action)
            continue

        # d) visited pruning – keep better‑g actions
        g_new = cur_steps + walk_len + 1   # macro‑push cost
        key = next_state.state_key()
        if key in visited and visited[key] <= g_new:
            continue

        # e) delta distance penalty
        new_box_dist = next_state.total_box_distance()
        delta = max(0, new_box_dist - old_box_dist)   # >0 only when distance worsens

        # f) target placement bonus
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after  = (moved_box_before[0] + dx, moved_box_before[1] + dy)
        on_target = 1 if moved_box_after in next_state.targets else 0

        # g) final heuristic
        h = (new_box_dist
             + ALPHA * walk_len
             + BETA * delta
             - GAMMA * on_target)

        scored.append((h, walk_len, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 3. Choose best action (lowest h, break ties with shorter walk)
    # ------------------------------------------------------------------
    if scored:
        scored.sort(key=lambda x: (x[0], x[1]))
        h, walk_len, chosen_action, chosen_state, chosen_key, chosen_g = scored[0]
        if chosen_action in node._untried_actions:
            node._untried_actions.remove(chosen_action)
        visited[chosen_key] = chosen_g
    else:
        # Fallback: expand any remaining untried action (if any)
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
            # No actions left – return an existing child (or self)
            if node.children:
                return next(iter(node.children.values()))
            return node

    # ------------------------------------------------------------------
    # 4. Create child node and register it.
    # ------------------------------------------------------------------
    child = MCTSNode(chosen_state, parent=node, parent_action=chosen_action)
    node.children[chosen_action] = child
    return child
```

--- simulation ---
```python
"""
LLM-generated MCTS tool: simulation
Description: No changes needed; the function is correct and efficient.
Generated:   2026-03-11T15:31:56.604000
"""

def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Improved leaf evaluation for MCTS simulation.

    Returns:
        1.0                                 if the state is solved,
        state.returns()[p]                  if terminal (dead‑lock / step‑limit),
        otherwise a shaped reward in (0, 1] based on box distance,
                                              player walk cost, fraction of boxes placed,
                                              and dead‑lock awareness.
    """
    # ------------------------------------------------------------------
    # Helper: BFS distances from a start cell avoiding walls and boxes.
    # ------------------------------------------------------------------
    from collections import deque

    def bfs_distances(start, walls, boxes, height, width):
        """Shortest‑path distances from `start` to every reachable floor cell."""
        q = deque([start])
        dist = {start: 0}
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                nxt = (nr, nc)
                if nxt in walls or nxt in boxes:
                    continue
                if nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        return dist

    # ------------------------------------------------------------------
    # Helper: simple corner dead‑lock detection.
    # ------------------------------------------------------------------
    def is_corner_deadlocked(state):
        """Return True if any non‑target box is stuck in a corner."""
        walls = state.walls
        targets = state.targets
        for bx, by in state.boxes:
            if (bx, by) in targets:
                continue
            horiz = ((bx + 1, by) in walls) or ((bx - 1, by) in walls)
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: wall‑line dead‑lock detection.
    # If a box sits next to a wall and there is no target beyond that wall
    # in the same line, the box can never be moved to a target.
    # ------------------------------------------------------------------
    def is_wall_line_deadlocked(state, box_pos):
        walls = state.walls
        targets = state.targets
        h, w = state.height, state.width
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in dirs:
            neighbor = (box_pos[0] + dx, box_pos[1] + dy)
            if neighbor not in walls:
                continue
            # look beyond the wall in the same direction
            step = 2
            while True:
                check = (box_pos[0] + dx * step, box_pos[1] + dy * step)
                if not (0 <= check[0] < h and 0 <= check[1] < w):
                    break
                if check in walls:
                    break
                if check in targets:
                    # a target exists beyond the wall → not dead‑locked
                    return False
                step += 1
            # no target found → dead‑locked
            return True
        return False

    # ------------------------------------------------------------------
    # Helper: aggregate dead‑lock test (corner + wall‑line).
    # ------------------------------------------------------------------
    def is_deadlocked(state):
        if is_corner_deadlocked(state):
            return True
        for box in state.boxes:
            if box not in state.targets and is_wall_line_deadlocked(state, box):
                return True
        return False

    # ------------------------------------------------------------------
    # 0️⃣ Terminal handling – keep original semantics.
    # ------------------------------------------------------------------
    if state.is_terminal():
        # state.returns() already contains the appropriate value for
        # dead‑lock or step‑limit termination.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣ Immediate dead‑lock check – heavily punish hopeless leaves.
    # ------------------------------------------------------------------
    if is_deadlocked(state):
        # a dead‑locked leaf should never be attractive.
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣ Box‑only Manhattan distance (h).
    # ------------------------------------------------------------------
    h = state.total_box_distance()

    # ------------------------------------------------------------------
    # 3️⃣ Player walk cost to the *closest* legal push position (w).
    # ------------------------------------------------------------------
    walls = state.walls
    boxes = state.boxes
    height, width = state.height, state.width

    distances = bfs_distances(state.player, walls, boxes, height, width)

    min_walk = None
    for push_pos, _dir in state.legal_actions():
        d = distances.get(push_pos)
        if d is None:
            continue
        if (min_walk is None) or (d < min_walk):
            min_walk = d

    w = min_walk if min_walk is not None else 0

    # ------------------------------------------------------------------
    # 4️⃣ Fraction of boxes already on targets (f).
    # ------------------------------------------------------------------
    num_targets = state.num_targets
    placed = state.boxes_on_targets()
    f = placed / num_targets if num_targets > 0 else 0.0

    # ------------------------------------------------------------------
    # 5️⃣ Combine components.
    #    • ALPHA  – stronger walk‑cost penalty,
    #    • GAMMA  – larger bonus for placed boxes,
    #    • exponent – non‑linear scaling to spread values.
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walk distance (more influential)
    GAMMA = 0.9   # bonus for already‑placed boxes
    EXP   = 0.8   # exponent < 1 makes small h differences larger

    # Non‑linear base reward: higher when both h and walk are small.
    base = 1.0 / (1.0 + (h + ALPHA * w) ** EXP)

    # Add progress bonus.
    reward = base + GAMMA * f

    # Clamp to [0, 1] – solved states are handled explicitly below.
    reward = max(0.0, min(1.0, reward))

    # ------------------------------------------------------------------
    # 6️⃣ Perfect solution override.
    # ------------------------------------------------------------------
    if h == 0:
        reward = 1.0

    return reward
```

--- backpropagation ---
```python
"""
LLM-generated MCTS tool: backpropagation
Description: Ensure solved bonus is applied undiscouted to the root and clean up minor redundancies.
Generated:   2026-03-11T15:32:44.153294
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a leaf reward up the tree while preserving useful
    signal for deep plans.

    Changes vs. previous version:
      • Use *leaf‑relative* depth discount instead of absolute depth.
        This lets the root receive a sizable fraction of the leaf
        reward while still favouring shallower ancestors.
      • Raise the discount factor γ to 0.95 (gentler decay).
      • Reduce step‑penalty to 0.001 – walk cost is already reflected in
        the simulation reward, so we avoid over‑penalising.
      • Add the solved‑state bonus **without discount** to the root node,
        guaranteeing a strong spike when a leaf is a solution.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be adjusted without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (gentler decay)
    SOLVED_BONUS = 1.0    # extra reward for a solved leaf (added to root)
    STEP_PENALTY = 0.001  # penalty per step taken in the leaf state

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
        assert _s.loader is not None
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

    # Apply a modest step‑cost penalty (clip at zero).
    steps_taken = getattr(leaf_state, "steps", 0)
    penalised_reward = reward - STEP_PENALTY * steps_taken
    if penalised_reward < 0.0:
        penalised_reward = 0.0

    base_reward = penalised_reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying *leaf‑relative* depth discount.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Current depth from the root.
        cur_depth = _ag.node_depth(cur)

        # Remaining depth to leaf (non‑negative by construction).
        remaining = leaf_depth - cur_depth

        # Discount factor for this ancestor.
        decay = GAMMA ** remaining

        # Discounted contribution of the leaf reward.
        discounted = decay * base_reward

        # Accumulate into the node's value.
        cur.value += discounted

        # If this node is the root and the leaf was a solution,
        # add the full solved bonus (undiscounted) to create a strong
        # signal for selection.
        if cur.parent is None and is_solved:
            cur.value += SOLVED_BONUS

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
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level6)
Timestamp:  2026-03-11T15:34:07.750419
Iterations: 500
Solved:     False
Steps:      20
Returns:    [0.0]

  Move 1: action=((2, 7), 2), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/2 | Total distance: 7
    Children: []
  Move 2: action=((2, 6), 2), total_visits=500
    State: Step 4/1000 | Boxes on target: 0/2 | Total distance: 6
    Children: [((2, 6), 2)(v=500, avg=0.000)]
  Move 3: action=((2, 5), 2), total_visits=500
    State: Step 5/1000 | Boxes on target: 0/2 | Total distance: 5
    Children: [((2, 5), 2)(v=500, avg=0.000)]
  Move 4: action=((2, 4), 2), total_visits=500
    State: Step 6/1000 | Boxes on target: 0/2 | Total distance: 6
    Children: [((2, 4), 2)(v=498, avg=0.000), ((1, 3), 1)(v=1, avg=0.000), ((2, 2), 3)(v=1, avg=0.000)]
  Move 5: action=((2, 1), 3), total_visits=500
    State: Step 7/1000 | Boxes on target: 0/2 | Total distance: 5
    Children: [((2, 1), 3)(v=500, avg=0.000)]
  Move 6: action=((2, 2), 3), total_visits=500
    State: Step 14/1000 | Boxes on target: 0/2 | Total distance: 6
    Children: [((2, 2), 3)(v=498, avg=0.000), ((2, 4), 2)(v=1, avg=0.000), ((1, 3), 1)(v=1, avg=0.000)]
  Move 7: action=((1, 4), 1), total_visits=500
    State: Step 15/1000 | Boxes on target: 0/2 | Total distance: 5
    Children: [((1, 4), 1)(v=499, avg=0.000), ((2, 3), 3)(v=1, avg=0.000)]
  Move 8: action=((2, 4), 1), total_visits=500
    State: Step 18/1000 | Boxes on target: 0/2 | Total distance: 4
    Children: [((2, 4), 1)(v=499, avg=0.000), ((4, 4), 0)(v=1, avg=0.000)]
  Move 9: action=((4, 6), 0), total_visits=500
    State: Step 19/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: []
  Move 10: action=((2, 7), 2), total_visits=500
    State: Step 27/1000 | Boxes on target: 1/2 | Total distance: 4
    Children: []
  Move 11: action=((2, 6), 2), total_visits=500
    State: Step 30/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: [((2, 6), 2)(v=500, avg=0.005)]
  Move 12: action=((2, 5), 2), total_visits=500
    State: Step 31/1000 | Boxes on target: 1/2 | Total distance: 2
    Children: [((2, 5), 2)(v=500, avg=0.004)]
  Move 13: action=((2, 4), 2), total_visits=500
    State: Step 32/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: [((2, 4), 2)(v=423, avg=0.004), ((1, 3), 1)(v=1, avg=0.000), ((2, 2), 3)(v=76, avg=0.004)]
  Move 14: action=((2, 1), 3), total_visits=500
    State: Step 33/1000 | Boxes on target: 1/2 | Total distance: 2
    Children: [((2, 1), 3)(v=500, avg=0.003)]
  Move 15: action=((2, 4), 2), total_visits=500
    State: Step 40/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: [((2, 2), 3)(v=141, avg=0.002), ((2, 4), 2)(v=358, avg=0.002), ((1, 3), 1)(v=1, avg=0.000)]
  Move 16: action=((2, 1), 3), total_visits=500
    State: Step 49/1000 | Boxes on target: 1/2 | Total distance: 2
    Children: [((2, 1), 3)(v=500, avg=0.001)]
  Move 17: action=((2, 4), 2), total_visits=500
    State: Step 56/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: [((2, 2), 3)(v=234, avg=0.001), ((2, 4), 2)(v=265, avg=0.001), ((1, 3), 1)(v=1, avg=0.000)]
  Move 18: action=((2, 1), 3), total_visits=500
    State: Step 65/1000 | Boxes on target: 1/2 | Total distance: 2
    Children: [((2, 1), 3)(v=500, avg=0.000)]
  Move 19: action=((2, 2), 3), total_visits=500
    State: Step 72/1000 | Boxes on target: 1/2 | Total distance: 3
    Children: [((2, 2), 3)(v=472, avg=0.000), ((2, 4), 2)(v=27, avg=0.000), ((1, 3), 1)(v=1, avg=0.000)]
  Move 20: action=((1, 4), 1), total_visits=500
    State: Step 73/1000 | Boxes on target: 1/2 | Total distance: 2
    Children: [((1, 4), 1)(v=499, avg=0.000), ((2, 3), 3)(v=1, avg=0.000)]

Final state:
Step 76/1000 | Boxes on target: 1/2 | Total distance: 1
  ####   
###  ####
#   @   #
# # $#  #
# . *#  #
#########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level6
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level6 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 4 levels): 0.5000

Per-level best composites so far:
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level6: best=0.0000 (baseline=0.0000)
  level8: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level4', 'level6', 'level7', 'level8', 'level9']
Mastered levels: ['level3', 'level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level8] [simulation]: composite=0.0000, solve_rate=0%, eval_time=0.6s, desc=No changes needed; the function is correct and efficient. ← accepted
  Iter 1 [level8] [backpropagation]: composite=0.0000, solve_rate=0%, eval_time=0.6s, desc=Ensure solved bonus is applied undiscouted to the root and clean up minor redundancies. ✗ rejected
  Iter 2 [level3] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'selection'
heuristic code, and the gameplay traces above.

Produce a focused analysis with these sections:

1. KEY WEAKNESSES
   What are the main problems causing poor play?
   Cite specific move numbers, Q-value patterns, or state
   observations as evidence. Be specific. Rank by impact.

2. ROOT CAUSE
   WHY does the current code produce this behaviour?
   Point to specific logic or missing logic in the code.

3. PROPOSED APPROACH
   Choose one of these strategies:

   A) INCREMENTAL (~70% of cases): Describe targeted
      modifications to the 'selection' function
      that address the top weakness(es). Build on and
      extend the current code.

   B) RESTRUCTURE (~30% of cases): If the current approach
      is fundamentally limited, describe a different
      strategy. Explain why incremental changes won't
      suffice. Keep proven components that work.

   State which strategy (A or B) you recommend and why.

Keep your analysis under 500 words. Do NOT write code.

---

## Response

**1. KEY WEAKNESSES**  

| # | Observation (from the trace) | Why it hurts performance |
|---|------------------------------|--------------------------|
| 1 | The same macro‑pushes are revisited thousands of times (e.g., `((2,4),2)` appears with v≈500 in many successive moves). | The selection routine keeps descending the same low‑f branch, never giving a chance to explore alternative pushes that might break the dead‑end. |
| 2 | The tree‑walk ignores the `exploration_weight` argument completely. | No UCB‑style term → pure greedy A* (min g + h). When several children share an identical f‑score the tie‑breaker (`fewest visits`) is too weak; the algorithm falls into depth‑first “tunnel” behaviour. |
| 3 | `g` is the pure node depth, not the accumulated **real** cost (walk + push). | A cheap‑looking shallow node may hide a huge walk cost, so the algorithm prefers pushes that look close in depth but actually waste many steps, stalling progress toward the goal. |
| 4 | `h_sokoban` is a static Manhattan‑distance heuristic; it does **not** penalise obvious dead‑locks (corner, wall‑line) or reward a box landing on a target. | The f‑value does not distinguish a push that creates a dead‑lock from one that makes progress, so the greedy descent often walks into dead‑ends (see moves 9–12 where a box is placed on a target but the algorithm keeps wobbling around it). |
| 5 | No novelty/visit‑bonus beyond the simple “fewest visits” tie‑break. | Nodes that have been visited many times still receive the same f, so the algorithm spends many iterations on an already‑exhausted branch. |

The combined effect is **excessive exploitation of a single cheap‑looking line**, leading to a long but unproductive search (76 steps, only 1/2 boxes placed, return 0).

---

**2. ROOT CAUSE**  

The selection code implements a pure best‑first walk:

```python
g = _ag.node_depth(child)          # depth only
h = _ag.h_sokoban(child.state)     # static distance heuristic
f = g + h
```

*No exploration term* (`exploration_weight` is never used).  
*Depth* is not the true accumulated cost, so `f` does not reflect the macro‑push cost.  
`h_sokoban` lacks dead‑lock penalties and target‑placement bonuses, so f‐values of dead‑ended and promising children can be identical, making the tie‑breaker (`fewest visits`) insufficient. Consequently the algorithm repeatedly selects the same child, inflating its visit count while never opening alternative branches.

---

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvement (≈70 % of effort).**  
We keep the overall A*‑guided framework (it already provides a useful heuristic backbone) but augment it with three lightweight, high‑impact tweaks:

1. **Add an explicit exploration component** to the score, preserving the supplied `exploration_weight`.  
   - Compute `explore = exploration_weight * sqrt(log(parent.visits) / child.visits)` (or the classic UCB term).  
   - New score: `f = g + h - explore` (lower is better).  
   This restores a balanced trade‑off without discarding the A* intuition.

2. **Replace pure depth with the real accumulated step cost** (`child.state.steps` or the macro‑push cost).  
   - `g = child.state.steps` (or the stored `g` from the shared A* table).  
   - This makes `f` penalise long walks, steering the search toward pushes that actually advance the solution.

3. **Enrich the heuristic `h` locally** (still cheap, no extra roll‑outs).  
   - Pull the dead‑lock detection already used in expansion/simulation (`is_deadlocked`) and add a small penalty (`+ deadlock_penalty`) when `h_sokoban` would otherwise be identical.  
   - Add a tiny **target‑placement bonus** (`‑ target_bonus` if a push puts a box on a target).  
   - Both constants can be tiny (e.g., 0.1) just enough to break ties in favour of progress.

The modified loop would still iterate over children, compute `g`, `h`, `explore`, and select the child with the smallest `f`. Tie‑breaking can stay “fewest visits” as a secondary fallback.

These three changes are O(1) per child, keep the code simple, and directly address the highest‑impact weaknesses: lack of exploration, inaccurate cost estimation, and indistinguishable dead‑ends. No restructuring of the whole MCTS pipeline is required, and the existing A* globals remain usable.

*Why not a full restructure?*  
A complete switch back to classic UCB1 would discard the valuable domain‑specific distance heuristic already proven useful in other levels (selection performed perfectly on level 3). The problem is not the idea of A*‑guided selection but its overly greedy instantiation; an incremental fix restores exploration while preserving the beneficial distance bias.
