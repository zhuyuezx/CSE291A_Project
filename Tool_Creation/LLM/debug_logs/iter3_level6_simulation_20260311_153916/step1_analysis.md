# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:39:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 19.89s |
| Status     | success |
| Tokens     | 13002 |
| Validation | invalid — No code block found in response. |

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

--- simulation ◀ TARGET ---
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
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
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

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level6)
Timestamp:  2026-03-11T15:39:16.696643
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
  Iter 2 [level3] [selection]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
  Iter 3 [level6] [selection]: composite=0.0000, solve_rate=0%, eval_time=0.4s, desc= ✗ rejected
  Iter 3 [level6] [expansion]: composite=0.0000, solve_rate=0%, eval_time=0.0s, desc=No functional changes needed; retain the incremental improvements as‑is. ✗ rejected

------------------------------------------------------------
TASK — ANALYSIS ONLY (no code)
------------------------------------------------------------
Carefully study the game rules, the current 'simulation'
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
      modifications to the 'simulation' function
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

**1. KEY WEAKNESSES**

| Rank | Symptom (trace) | Why it hurts |
|------|----------------|--------------|
| 1️⃣ | **Near‑zero average Q‑values** on every child (e.g. moves 1‑20 all show `avg=0.000`). The rollout reward that backs these values is effectively *zero* for almost every leaf. |
| 2️⃣ | **Oscillating box‑distance** – the planner willingly raises the total Manhattan distance (e.g. move 4 → step 6: distance 5 → 6, move 6 → step 14: distance 6 → 5). The simulation does not penalise moves that make the situation worse, so the tree drifts without a clear “down‑hill” direction. |
| 3️⃣ | **Over‑aggressive dead‑lock filter** – the `is_deadlocked` test in the simulation prunes many non‑dead states (especially the wall‑line check). When it fires, the leaf reward is forced to `0.0`, drowning out any useful signal. |
| 4️⃣ | **Insufficient progress signal** – the only “positive” term is `GAMMA * f` (fraction of boxes on targets). When only one of two boxes is placed (`f = 0.5`) the bonus (`0.9 * 0.5 = 0.45`) is modest, and the base term `1/(1+(h+αw)^exp)` quickly collapses to < 0.2 for distances ≥ 4, yielding an overall reward ≈ 0.3. After depth‑discount in back‑propagation this becomes ≈ 0.02 and is indistinguishable from zero noise. |
| 5️⃣ | **No loop detection** – the rollout can revisit the same state many times; the reward never drops, so the algorithm wastes visits on cycles instead of progressing. |

**2. ROOT CAUSE**

The simulation returns a *static heuristic* that treats the current leaf as the final outcome:

* It ignores the *direction* of change (Δ distance) – only the absolute Manhattan sum `h` is used. Consequently, pushes that temporarily increase distance are not discouraged.
* The dead‑lock test (`is_deadlocked`) is duplicated from expansion but is stricter (wall‑line detection) and is applied **every rollout**. Because the macro‑push model often leaves a box adjacent to a wall while a target lies behind it (still reachable after another push), the test mis‑labels many positions as dead‑locked, returning `0.0`.
* The reward scaling (`base = 1/(1+(h+αw)^exp)`) is shallow; for realistic distances the base is < 0.2, and after the *leaf‑relative* discount (`γ≈0.95`) the contribution that reaches the root is practically zero. Hence the tree receives almost no gradient and defaults to the A*‑guided selection, which only minimises `g+h` and thus wanders between equally‑costly pushes.
* There is no memory of previously visited states, so rollouts can loop indefinitely, each loop still giving the same low reward, inflating visit counts without improving value.

These issues together explain why the algorithm never lifts a box off the last target and ends with a single box distance = 1 after 500 iterations.

**3. PROPOSED APPROACH – Strategy A (incremental)**  

The simulation heuristic is already cheap; we keep its overall structure but **reshape the reward** so that it yields a clear gradient toward the solution and eliminates spurious zeroing.

1. **Delta‑distance penalty**  
   *Compute* `Δ = new_box_dist – old_box_dist` (where `old_box_dist` is the state’s `total_box_distance()` before the rollout step).  
   Add a term `‑ β * max(0, Δ)` (β≈1.0) to the score. This makes any move that worsens distance negative, steering the tree away from back‑and‑forth pushes.

2. **Target‑placement bonus per push**  
   When a push results in a box landing on a target, add `‑ γ_target` (e.g. `‑0.3`) to the heuristic *before* inversion, or equivalently add `+ γ_target` after the `base` inversion. This makes the rollout reward noticeably higher for the crucial “final‑box” moves.

3. **Stronger walk‑cost weight**  
   Raise `α` from 0.5 to ≈ 0.9 so that distant pushes are penalised more, encouraging the macro‑pushes that are cheap to reach.

4. **Clamp the base with a small offset**  
   Use `base = 1.0 / (1.0 + (h + α*w + β*max(0,Δ) - γ_target*on_target) ** EXP)`  
   where `on_target` is 1 if the just‑performed push placed a box on a target. The subtraction inside the power makes the denominator smaller for good moves, increasing the final reward.

5. **Simplify dead‑lock filter**  
   Keep only the *corner* dead‑lock test (the cheap, always‑correct one). Remove the wall‑line check from the simulation (it belongs in expansion where it is used for pruning). This prevents accidental zero‑rewards on viable states.

6. **Add a visited‑state penalty**  
   Maintain a short‑lived `set` of state keys seen in the current rollout (size ≤ 30). If a state repeats, return a small constant `ε = 0.01` instead of the usual score. This breaks loops without heavy bookkeeping.

7. **Tiny step‑penalty** (optional)  
   Subtract `δ * leaf_state.steps` (δ≈0.001) *after* the base computation, similar to back‑propagation, to favour shorter solutions.

These seven tweaks are **local** to `default_simulation`; they do not disturb the selection, expansion, or back‑propagation code that already works. The heuristic will now:

* give a *positive gradient* when a push reduces distance or lands a box on a target,
* penalise pushes that increase distance (eliminating the back‑and‑forth seen around moves 4‑6),
* stop falsely discarding promising leaves,
* and discourage loops that waste visits.

With these changes the rollout rewards will typically fall in the 0.3‑0.9 range for useful states, survive the depth‑discount, and produce discernible Q‑values, allowing the A*‑guided selection to follow a consistent downhill path and finally solve level 6.
