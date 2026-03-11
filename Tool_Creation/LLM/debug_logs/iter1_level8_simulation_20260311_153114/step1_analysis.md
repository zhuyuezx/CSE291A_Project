# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:31:27 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 13.62s |
| Status     | success |
| Tokens     | 10406 |
| Validation | invalid — No function definition found in code. |

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
TARGET HEURISTIC TO IMPROVE (simulation)
------------------------------------------------------------
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

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level8)
Timestamp:  2026-03-11T15:31:14.059528
Iterations: 500
Solved:     False
Steps:      6
Returns:    [0.0]

  Move 1: action=((4, 4), 0), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/4 | Total distance: 7
    Children: [((4, 4), 0)(v=497, avg=0.240), ((3, 2), 0)(v=1, avg=0.072), ((2, 4), 1)(v=1, avg=0.043), ((1, 2), 1)(v=1, avg=0.022)]
  Move 2: action=((3, 4), 0), total_visits=500
    State: Step 1/1000 | Boxes on target: 0/4 | Total distance: 6
    Children: [((3, 4), 0)(v=495, avg=0.242), ((3, 2), 0)(v=1, avg=0.073), ((3, 4), 2)(v=1, avg=0.073), ((3, 2), 3)(v=1, avg=0.058), ((1, 4), 1)(v=1, avg=0.045), ((1, 2), 1)(v=1, avg=0.032)]
  Move 3: action=((3, 2), 0), total_visits=500
    State: Step 2/1000 | Boxes on target: 1/4 | Total distance: 5
    Children: [((1, 5), 2)(v=1, avg=0.152), ((3, 2), 0)(v=495, avg=0.244), ((3, 4), 2)(v=1, avg=0.142), ((3, 2), 3)(v=1, avg=0.131), ((1, 3), 3)(v=1, avg=0.096), ((1, 2), 1)(v=1, avg=0.086)]
  Move 4: action=((1, 1), 3), total_visits=500
    State: Step 8/1000 | Boxes on target: 1/4 | Total distance: 4
    Children: [((1, 1), 3)(v=495, avg=0.245), ((2, 2), 3)(v=1, avg=0.147), ((3, 2), 3)(v=1, avg=0.140), ((3, 4), 2)(v=1, avg=0.089), ((1, 5), 2)(v=1, avg=0.088), ((2, 4), 2)(v=1, avg=0.082)]
  Move 5: action=((2, 2), 3), total_visits=500
    State: Step 11/1000 | Boxes on target: 2/4 | Total distance: 3
    Children: [((2, 2), 3)(v=491, avg=0.247), ((3, 2), 3)(v=7, avg=0.189), ((3, 4), 2)(v=1, avg=0.154), ((2, 4), 2)(v=1, avg=0.147)]
  Move 6: action=((4, 3), 0), total_visits=500
    State: Step 13/1000 | Boxes on target: 2/4 | Total distance: 3
    Children: [((2, 3), 3)(v=1, avg=0.217), ((2, 3), 1)(v=1, avg=0.179), ((3, 2), 3)(v=1, avg=0.203), ((4, 3), 0)(v=495, avg=0.247), ((2, 5), 2)(v=1, avg=0.161), ((3, 4), 2)(v=1, avg=0.140)]

Final state:
Step 18/1000 | Boxes on target: 2/4 | Total distance: 2
########
#  **..#
#  $$  #
#  @   #
#      #
########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level8
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level8 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 2 levels): 0.5000

Per-level best composites so far:
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level8: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level1', 'level10', 'level2', 'level3', 'level4', 'level6', 'level7', 'level8', 'level9']
Mastered levels: ['level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level8] [selection]: composite=0.0000, solve_rate=0%, eval_time=1.5s, desc= ✗ rejected
  Iter 1 [level8] [expansion]: composite=0.0000, solve_rate=0%, eval_time=0.6s, desc=Incremental heuristic improvements – delta distance penalty, target‑placement bonus, stronger dead‑lock detection, and increased walk‑cost weight. ✗ rejected

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

**1. KEY WEAKNESSES**

| Rank | Symptom (observed in the trace) | Why it hurts performance |
|------|--------------------------------|--------------------------|
| 1️⃣ | *Almost identical Q‑values* – after many visits the average value of the best child is only ≈ 0.24, even after placing two boxes (step 13, `Boxes on target: 2/4`). The algorithm cannot tell the “good” pushes from the mediocre ones, so the tree favours whichever was sampled first. |
| 2️⃣ | *Reward range is compressed* – the current formula yields roughly `0.15 – 0.45` for almost every non‑terminal leaf (e.g. total distance = 7 → 0.19, distance = 2 → 0.34). With the back‑propagation depth discount (γ = 0.70) these numbers shrink further, making selection noise‑dominated. |
| 3️⃣ | *Walk‑cost weight is tiny* (`ALPHA = 0.1`). Macro‑pushes that need a long walk are barely penalised, so the roll‑out rewards a state that is expensive to reach just as much as a cheap one. |
| 4️⃣ | *No explicit penalty for worsening box layout* – the simulation only looks at the *total* Manhattan distance (`h`). A push that keeps `h` unchanged but moves a box into a dead‑end is rewarded the same as a constructive push. |
| 5️⃣ | *Dead‑lock awareness is absent* – terminal detection catches only *already* dead‑locked states. A leaf that is one push away from a corner dead‑lock still receives a high reward, encouraging the tree to explore it. |

The combination of (1) and (2) makes the MCTS value estimates almost flat, which explains why the search keeps expanding the same high‑visit branches (see the huge visit counts on the first child of every node) and never discovers the sequence that solves the puzzle.

---

**2. ROOT CAUSE**

The simulation function is deliberately simple:

```python
base = 1.0 / (1.0 + h + ALPHA * w)      # h = total box distance
reward = base + GAMMA * f                # f = fraction of boxes placed
```

* `ALPHA = 0.1` yields a negligible walk penalty (`w` is usually ≤ 5).  
* `GAMMA = 0.4` adds at most 0.4 when half the boxes are placed (`f = 0.5`).  
* `base` is bounded by `1/(1+h)`, so for any realistic `h` the range is 0.08–0.33.  
* The final `clamp` forces everything into `[0,1]` but never stretches the spread.

Consequently, different board configurations map to a narrow band of rewards, and the subsequent back‑propagation (γ‑depth discount, step penalty) further flattens them. Moreover, the function never checks whether the current layout is *dead‑locked* or whether the last push increased the distance (`delta > 0`). Those missing signals let the rollout reward “good” states that are actually traps.

---

**3. PROPOSED APPROACH**

**Strategy A – Incremental refinement of the simulation heuristic**  

The core idea is to *amplify the signal* that already exists (boxes placed, distance, walk cost) and *inject two missing signals*: (i) a penalty when a push makes the box‑distance **worse**, and (ii) a hard‑negative when the state is *dead‑locked* (corner or wall‑line). All changes are local to `default_simulation`; the rest of the pipeline (selection, expansion, back‑prop) stays untouched.

| Change | Rationale | Expected effect |
|--------|-----------|-----------------|
| **Increase walk weight**: `ALPHA ← 0.5` (or 0.7) | Makes long‑walk pushes visibly cheaper in the reward, steering the tree toward low‑cost macro‑pushes. |
| **Boost target bonus**: `GAMMA ← 0.8` (or 1.0) | Differentiates states that have already placed boxes; a state with 2/4 boxes jumps from ~0.30 to ~0.60, giving the tree a strong incentive to lock in progress. |
| **Add distance‑worsening penalty**: compute `delta = max(0, new_h - old_h)` (using the caller’s `state.total_box_distance()` before/after a hypothetical push). Subtract `BETA * delta` from the reward (`BETA ≈ 0.4`). | Penalises pushes that move a box away from any target, producing a lower reward for regressions. |
| **Dead‑lock check**: reuse the `is_corner_deadlocked` / `is_wall_line_deadlocked` helpers from `expansion`. If the leaf is dead‑locked, return `0.0` immediately (or a tiny negative value). | Prevents the rollout from rewarding traps, cutting off branches that would otherwise dominate due to sheer visitation. |
| **Optional non‑linear scaling**: replace `base = 1/(1+h+ALPHA*w)` with `base = 1.0 / (1.0 + (h + ALPHA*w) ** 0.8)` to stretch differences for small `h`. | Gives a steeper slope when the puzzle is near‑solved, making the last few pushes stand out. |

All modifications keep the function pure, fast (still only BFS once) and return a single float. By widening the reward distribution and explicitly punishing regressions and dead‑locks, the MCTS value estimates will acquire a richer gradient, allowing the UCT‑style (or the A*‑guided selection) to discriminate actions far better. This should raise the average leaf reward, increase visit diversity, and ultimately produce the missing solving sequences for level 8.
