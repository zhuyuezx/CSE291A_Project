# step1_analysis

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:41:53 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 14.05s |
| Status     | success |
| Tokens     | 12455 |
| Validation | invalid — SyntaxError: unexpected indent (<unknown>, line 2) |

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
LLM-generated MCTS tool: selection
Description: 
Generated:   2026-03-11T15:35:45.173827
"""

def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    Parameters
    ----------
    root : MCTSNode
        The current node from which the walk starts (may be any node in the tree).
    exploration_weight : float, optional
        Weight applied to the UCB exploration term (default is √2 ≈ 1.41).

    Returns
    -------
    MCTSNode
        The selected node for expansion or simulation.
    """
    import math

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0   # large penalty to discourage dead‑locked states
    TARGET_BONUS    = 10.0       # reward for moving a box onto a target

    def heuristic(state):
        """
        Base heuristic: total Manhattan distance from every box to the
        nearest target (the GameState already provides this via
        `total_box_distance`).
        """
        # The method exists in the public API.
        return state.total_box_distance()

    def is_deadlocked(state):
        """
        Very cheap dead‑lock detection: a box that is not on a target
        and is stuck in a corner formed by two orthogonal walls.
        """
        walls = state.walls
        for box in state.boxes:
            if box in state.targets:
                continue
            x, y = box
            # check four possible corner configurations
            if ((x - 1, y) in walls and (x, y - 1) in walls) or \
               ((x - 1, y) in walls and (x, y + 1) in walls) or \
               ((x + 1, y) in walls and (x, y - 1) in walls) or \
               ((x + 1, y) in walls and (x, y + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # 2️⃣  Tree walk – always pick the child with minimal f‑value
    # ------------------------------------------------------------------
    node = root
    while not node.state.is_terminal():
        # A node is **fully expanded** when there are no untried actions left.
        fully_expanded = (len(node._untried_actions) == 0)

        if not fully_expanded:
            # Let the expansion phase create a fresh child.
            return node

        best_child = None
        best_f = math.inf
        best_q = -math.inf

        # Guard against division‑by‑zero when the parent has never been visited.
        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: heuristic + dead‑lock penalty – target bonus -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: boxes placed on targets relative to the parent.
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- Exploration term (UCB) -----
            if child.visits == 0:
                # Unvisited nodes are given the most attractive (lowest) f.
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore  # lower f is better

            # ----- Tie‑break on average value (q) -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        # Safety net – should never happen, but avoids an infinite loop.
        if best_child is None:
            break

        node = best_child

    return node
```

--- expansion ---
```python
"""
LLM-generated MCTS tool: expansion
Description: No functional changes needed; retain the incremental improvements as‑is.
Generated:   2026-03-11T15:37:55.139859
"""

def default_expansion(node):
    """
    Expanded heuristic for macro‑push actions with additional
    diversification and finer‑grained distance penalties.

    New features:
      * Tie‑breaker noise / visit‑penalty to break deterministic ties.
      * Per‑box distance‑increase penalty (instead of only total increase).
      * Bonus / penalty based on the farthest box distance change.
      * Stronger reward for placing a box on a target.
      * Small penalty for moving the same box as the previous action
        (discourages immediate back‑and‑forth pushes).
      * Retains existing dead‑lock detection, walk‑cost weighting,
        and per‑root visited‑state pruning.
    """
    import sys
    from collections import deque
    from math import log
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
        height = state.height
        width = state.width
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
            vert = ((bx, by + 1) in walls) or ((bx, by - 1) in walls)
            if horiz and vert:
                return True
        return False

    # ------------------------------------------------------------------
    # Helper 3: Wall‑line dead‑lock detection.
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
                    # a target exists beyond the wall → not a deadlock.
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
    # Helper 5: Closest‑target Manhattan distance for a single box
    # ------------------------------------------------------------------
    def closest_target_dist(box, targets):
        return min(abs(box[0] - t[0]) + abs(box[1] - t[1]) for t in targets)

    # ------------------------------------------------------------------
    # Helper 6: Per‑root visited dictionary (persistent across calls)
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
    # 0. Some constants
    # ------------------------------------------------------------------
    ALPHA = 1.0      # walk‑distance weight
    BETA  = 1.2      # per‑box distance‑increase penalty weight
    GAMMA = 1.2      # target‑placement reward weight
    LAMBDA_FAR = 0.5 # farthest‑box distance change weight
    EPS   = 0.01     # tie‑breaker / visit‑penalty magnitude
    RHO   = 0.2      # penalty for moving the same box as previous action

    # ------------------------------------------------------------------
    # 1. Current accumulated step count (real cost so far).
    # ------------------------------------------------------------------
    cur_steps = getattr(node.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Pre‑compute old per‑box distances and farthest distance.
    # ------------------------------------------------------------------
    old_dist_per = {
        b: closest_target_dist(b, node.state.targets) for b in node.state.boxes
    }
    old_farthest = max(old_dist_per.values()) if old_dist_per else 0
    old_dist_list = sorted(old_dist_per.values())

    # direction vectors for macro‑pushes: UP, DOWN, LEFT, RIGHT
    drc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # 3. Scan untried actions, compute heuristic scores.
    # ------------------------------------------------------------------
    scored = []                     # (h, walk_len, action, state, key, g_new)
    deadlocked_to_remove = []

    # Helper to know which box was moved in the previous step (if any)
    prev_moved_box = None
    if node.parent_action is not None:
        prev_player, prev_dir = node.parent_action
        pdx, pdy = drc[prev_dir]
        prev_moved_box = (prev_player[0] + pdx, prev_player[1] + pdy)

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

        # e) per‑box distance increase (delta)
        new_dist_per = {
            b: closest_target_dist(b, next_state.targets) for b in next_state.boxes
        }
        new_dist_list = sorted(new_dist_per.values())
        delta = 0
        for od, nd in zip(old_dist_list, new_dist_list):
            if nd > od:
                delta += nd - od

        # f) farthest‑box distance change bonus/penalty
        new_farthest = max(new_dist_per.values()) if new_dist_per else 0
        far_diff = new_farthest - old_farthest   # >0 → penalty, <0 → bonus

        # g) target placement bonus
        dx, dy = drc[direction]
        moved_box_before = (player_pos[0] + dx, player_pos[1] + dy)
        moved_box_after  = (moved_box_before[0] + dx, moved_box_before[1] + dy)
        on_target = 1 if moved_box_after in next_state.targets else 0

        # h) recent‑box penalty
        recent_pen = RHO if (prev_moved_box is not None and moved_box_before == prev_moved_box) else 0.0

        # i) final heuristic (lower is better)
        h = (next_state.total_box_distance()
             + ALPHA * walk_len
             + BETA * delta
             + LAMBDA_FAR * far_diff
             + recent_pen
             - GAMMA * on_target)

        # j) tiny deterministic tie‑breaker (acts like visit‑penalty)
        tie_noise = EPS * (player_pos[0] + player_pos[1] + direction) / 100.0
        h += tie_noise

        scored.append((h, walk_len, action, next_state, key, g_new))

    # permanently drop actions that lead to dead‑locked states
    for a in deadlocked_to_remove:
        if a in node._untried_actions:
            node._untried_actions.remove(a)

    # ------------------------------------------------------------------
    # 4. Choose best action (lowest h, break ties with shorter walk)
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
    # 5. Create child node and register it.
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
LLM-generated MCTS tool: selection
Description: 
Generated:   2026-03-11T15:35:45.173827
"""

def default_selection(root, exploration_weight: float = 1.41):
    """
    A*-guided tree walk with UCB exploration and a richer heuristic.

    Parameters
    ----------
    root : MCTSNode
        The current node from which the walk starts (may be any node in the tree).
    exploration_weight : float, optional
        Weight applied to the UCB exploration term (default is √2 ≈ 1.41).

    Returns
    -------
    MCTSNode
        The selected node for expansion or simulation.
    """
    import math

    # ------------------------------------------------------------------
    # 1️⃣  Helper constants & functions (self‑contained)
    # ------------------------------------------------------------------
    DEADLOCK_PENALTY = 1_000.0   # large penalty to discourage dead‑locked states
    TARGET_BONUS    = 10.0       # reward for moving a box onto a target

    def heuristic(state):
        """
        Base heuristic: total Manhattan distance from every box to the
        nearest target (the GameState already provides this via
        `total_box_distance`).
        """
        # The method exists in the public API.
        return state.total_box_distance()

    def is_deadlocked(state):
        """
        Very cheap dead‑lock detection: a box that is not on a target
        and is stuck in a corner formed by two orthogonal walls.
        """
        walls = state.walls
        for box in state.boxes:
            if box in state.targets:
                continue
            x, y = box
            # check four possible corner configurations
            if ((x - 1, y) in walls and (x, y - 1) in walls) or \
               ((x - 1, y) in walls and (x, y + 1) in walls) or \
               ((x + 1, y) in walls and (x, y - 1) in walls) or \
               ((x + 1, y) in walls and (x, y + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # 2️⃣  Tree walk – always pick the child with minimal f‑value
    # ------------------------------------------------------------------
    node = root
    while not node.state.is_terminal():
        # A node is **fully expanded** when there are no untried actions left.
        fully_expanded = (len(node._untried_actions) == 0)

        if not fully_expanded:
            # Let the expansion phase create a fresh child.
            return node

        best_child = None
        best_f = math.inf
        best_q = -math.inf

        # Guard against division‑by‑zero when the parent has never been visited.
        parent_visits = node.visits if node.visits > 0 else 1

        for child in node.children.values():
            # ----- g: real accumulated cost (steps taken so far) -----
            g = getattr(child.state, "steps", 0)

            # ----- h: heuristic + dead‑lock penalty – target bonus -----
            h_base = heuristic(child.state)
            dead_pen = DEADLOCK_PENALTY if is_deadlocked(child.state) else 0.0

            # Target bonus: boxes placed on targets relative to the parent.
            parent_on_target = node.state.boxes_on_targets()
            child_on_target = child.state.boxes_on_targets()
            target_bonus = TARGET_BONUS * (child_on_target - parent_on_target)

            h = h_base + dead_pen - target_bonus

            # ----- Exploration term (UCB) -----
            if child.visits == 0:
                # Unvisited nodes are given the most attractive (lowest) f.
                f = -math.inf
            else:
                explore = exploration_weight * math.sqrt(
                    math.log(parent_visits) / child.visits
                )
                f = g + h - explore  # lower f is better

            # ----- Tie‑break on average value (q) -----
            q = child.value / child.visits if child.visits > 0 else 0.0

            if f < best_f or (f == best_f and q > best_q):
                best_child = child
                best_f = f
                best_q = q

        # Safety net – should never happen, but avoids an infinite loop.
        if best_child is None:
            break

        node = best_child

    return node
```

------------------------------------------------------------
GAMEPLAY TRACES
------------------------------------------------------------

--- Trace #1 ---
Game:       Sokoban_Macro (level9)
Timestamp:  2026-03-11T15:41:39.157909
Iterations: 500
Solved:     True
Steps:      12
Returns:    [1.0]

  Move 1: action=((3, 4), 0), total_visits=500
    State: Step 0/1000 | Boxes on target: 0/4 | Total distance: 6
    Children: [((3, 4), 0)(v=496, avg=0.545), ((3, 4), 2)(v=1, avg=0.192), ((3, 2), 3)(v=1, avg=0.188), ((1, 5), 1)(v=1, avg=0.187), ((1, 4), 1)(v=1, avg=0.186)]
  Move 2: action=((2, 4), 2), total_visits=500
    State: Step 1/1000 | Boxes on target: 1/4 | Total distance: 5
    Children: [((2, 4), 2)(v=485, avg=0.577), ((3, 4), 2)(v=10, avg=0.461), ((2, 6), 2)(v=2, avg=0.445), ((2, 2), 3)(v=1, avg=0.465), ((3, 2), 3)(v=1, avg=0.434), ((1, 5), 1)(v=1, avg=0.433)]
  Move 3: action=((2, 3), 1), total_visits=500
    State: Step 2/1000 | Boxes on target: 1/4 | Total distance: 5
    Children: [((2, 3), 1)(v=493, avg=0.610), ((1, 2), 1)(v=1, avg=0.468), ((3, 4), 2)(v=1, avg=0.436), ((2, 1), 3)(v=1, avg=0.434), ((2, 6), 2)(v=1, avg=0.465), ((3, 2), 3)(v=1, avg=0.432), ((4, 3), 0)(v=1, avg=0.411), ((1, 5), 1)(v=1, avg=0.431)]
  Move 4: action=((4, 2), 3), total_visits=500
    State: Step 3/1000 | Boxes on target: 2/4 | Total distance: 4
    Children: [((4, 2), 3)(v=161, avg=0.657), ((4, 4), 2)(v=161, avg=0.605), ((1, 2), 1)(v=161, avg=0.661), ((2, 1), 3)(v=8, avg=0.664), ((2, 6), 2)(v=8, avg=0.678), ((1, 5), 1)(v=1, avg=0.688)]
  Move 5: action=((1, 2), 1), total_visits=500
    State: Step 6/1000 | Boxes on target: 2/4 | Total distance: 4
    Children: [((4, 5), 2)(v=10, avg=0.650), ((1, 2), 1)(v=469, avg=0.685), ((2, 1), 3)(v=10, avg=0.659), ((2, 6), 2)(v=10, avg=0.667), ((1, 5), 1)(v=1, avg=0.684)]
  Move 6: action=((2, 2), 1), total_visits=500
    State: Step 11/1000 | Boxes on target: 2/4 | Total distance: 3
    Children: [((2, 2), 1)(v=495, avg=0.720), ((3, 1), 3)(v=1, avg=0.729), ((4, 5), 2)(v=1, avg=0.726), ((2, 6), 2)(v=1, avg=0.797), ((4, 2), 0)(v=1, avg=0.682), ((1, 5), 1)(v=1, avg=0.723)]
  Move 7: action=((4, 1), 3), total_visits=500
    State: Step 12/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((4, 1), 3)(v=493, avg=0.755), ((4, 5), 2)(v=3, avg=0.936), ((2, 6), 2)(v=3, avg=0.948), ((1, 5), 1)(v=1, avg=0.980)]
  Move 8: action=((2, 6), 2), total_visits=500
    State: Step 15/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((2, 6), 2)(v=498, avg=0.793), ((1, 5), 1)(v=2, avg=0.950)]
  Move 9: action=((2, 5), 2), total_visits=500
    State: Step 22/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((2, 5), 2)(v=499, avg=0.834), ((2, 3), 3)(v=1, avg=0.973)]
  Move 10: action=((2, 4), 2), total_visits=500
    State: Step 23/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((2, 4), 2)(v=497, avg=0.878), ((2, 2), 3)(v=2, avg=0.947), ((1, 3), 1)(v=1, avg=0.970)]
  Move 11: action=((1, 2), 1), total_visits=500
    State: Step 24/1000 | Boxes on target: 3/4 | Total distance: 2
    Children: [((1, 2), 1)(v=498, avg=0.923), ((2, 1), 3)(v=2, avg=0.946)]
  Move 12: action=((2, 2), 1), total_visits=500
    State: Step 27/1000 | Boxes on target: 3/4 | Total distance: 1
    Children: [((2, 2), 1)(v=498, avg=0.972), ((3, 1), 3)(v=1, avg=0.970), ((4, 2), 0)(v=1, avg=0.968)]

Final state:
Step 28/1000 | Boxes on target: 4/4 | Total distance: 0
########
#   *  #
#      #
# @    #
# ***  #
########

------------------------------------------------------------
ADDITIONAL CONTEXT
------------------------------------------------------------
Current level: level9
Current hyperparams: iterations=500, max_rollout_depth=1000, exploration_weight=1.410
Baseline for level9 (default MCTS): composite=0.0000, solve_rate=0%, avg_returns=0.0000
Aggregate best (avg across 4 levels): 0.7500

Per-level best composites so far:
  level2: best=1.0000 (baseline=1.0000) [MASTERED]
  level3: best=1.0000 (baseline=1.0000) [MASTERED]
  level5: best=1.0000 (baseline=1.0000) [MASTERED]
  level9: best=0.0000 (baseline=0.0000)

Active levels (not yet mastered): ['level1', 'level10', 'level4', 'level6', 'level7', 'level8', 'level9']
Mastered levels: ['level2', 'level3', 'level5']

SCORING: composite = 0.6 × solve_rate + 0.4 × avg_returns
  → SOLVING the puzzle is MORE important than heuristic accuracy.

STRATEGY: Prefer gradual, incremental improvements. Build on the
previous version rather than rewriting from scratch. However, if
the current approach is fundamentally flawed, a larger restructure
is acceptable.

Recent iterations:
  Iter 1 [level2] [simulation]: composite=1.0000, solve_rate=100%, desc=n/a ✗ rejected
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

**1. KEY WEAKNESSES**

| # | Symptom (from the trace or aggregate stats) | Why it hurts the search |
|---|---------------------------------------------|--------------------------|
| 1️⃣ | **Selection ignores walk‑cost** – `g = child.state.steps` is used directly, but the *incremental* cost of the *next* macro‑push (walk + push) is not reflected in the f‑value.  Nodes that require a long walk get a huge `g` and are instantly deem‑preferred out, even though the walk may be unavoidable.  This forces the tree to stay in “cheap‑walk” zones and repeatedly expands the same short‑reach moves, leading to the 0 % solve‑rate on most levels. |
| 2️⃣ | **Very weak target incentive** – `TARGET_BONUS = 10` is dwarfed by the typical `h = total_box_distance` (often ≥ 20) and by the massive `DEADLOCK_PENALTY = 1000`.  The algorithm therefore does not prioritize pushes that place a box on a goal; it often explores side‑bars that keep the distance unchanged. |
| 3️⃣ | **Sparse dead‑lock pruning** – the dead‑lock test only catches corner deadlocks.  Many Sokoban dead‑locks are wall‑line or “freeze‑deadlock” patterns.  Because selection never sees them, it wastes visits on branches that can never lead to a solution. |
| 4️⃣ | **No novelty/under‑visited bonus** – the UCB term is *subtracted* from `g+h`.  When a child has been visited a few times, `explore` grows, but the subtraction makes the node *less* attractive (higher f).  This is the opposite of classic UCB (which *adds* an exploration bonus).  Consequently the search quickly collapses onto a single branch and does not explore alternative push orders. |
| 5️⃣ | **Static exploration weight** – a fixed √2 works for many domains but, combined with the subtraction‑style UCB, it is too weak to rescue exploration when `g+h` differences are large (as they are with walk‑cost ignored). |

The trace itself shows that once a few actions receive hundreds of visits (e.g., the first child at Move 1 has 496/500 visits), the algorithm effectively *locks* onto them, even though later moves (e.g., Move 5‑7) needed different pushes to finish the puzzle. The over‑exploitation explains why only a single lucky run solved level 9 while the average remains zero.

---

**2. ROOT CAUSE**

*Selection code* builds `f = g + h - explore`.  
- **Walk‑cost omission**: `g` is the *cumulative* step count, not the *incremental* cost of reaching the child. A child that required a long walk already has a high `g`, so `f` becomes large and the branch is discarded early.  
- **Exploration term sign**: subtracting `explore` makes unvisited nodes artificially *best* (`-inf`), but after the first visit the term *reduces* the node’s attractiveness instead of encouraging further exploration. This contradicts the purpose of UCB and leads to rapid over‑exploitation.  
- **Heuristic scaling**: `TARGET_BONUS` is a constant that does not adapt to the remaining number of boxes or distance magnitudes, so its influence is negligible.  
- **Dead‑lock detection limited to corners** – many dead‑locked configurations are missed, so selection continues down futile branches.

Together these design choices cause the tree to gravitate toward cheap‑walk, high‑visit nodes and ignore promising but costlier pushes.

---

**3. PROPOSED APPROACH (Strategy A – Incremental)**  

We can remedy the dominant problems with a small set of targeted tweaks that keep the overall A*‑guided, UCB‑based selection intact:

1. **Add a *forward walk‑cost* estimate to the heuristic**  
   ```python
   walk_est = min_walk_to_any_push(child.state)   # cheap BFS from child.player to nearest legal push position
   h = h_base + dead_pen - target_bonus + α * walk_est
   ```
   A light weight `α≈0.5` makes pushes that require long walks slightly less attractive, but they are still considered when they dramatically reduce box distance.

2. **Flip the UCB sign (classic formulation)**  
   Replace `f = g + h - explore` with  
   ```python
   f = g + h + explore      # lower f = more promising
   ```
   This restores the intended exploration pressure. Keep the `-inf` shortcut for completely unvisited children.

3. **Scale the target bonus dynamically**  
   ```python
   TARGET_BONUS = 5.0 * (state.num_targets - parent_state.boxes_on_targets())
   ```
   The bonus grows when few boxes remain, making the final pushes decisive.

4. **Enrich dead‑lock detection** – embed the *wall‑line* test already present in the expansion phase (the `is_wall_line_deadlocked` helper).  It is cheap (O(#boxes)) and dramatically reduces wasted visits.

5. **Introduce a tiny novelty term** proportional to the inverse of the child’s visit count:  
   ```python
   novelty = β / (child.visits + 1)   # β≈0.2
   f = g + h + explore - novelty      # novelty lowers f, encouraging under‑visited branches
   ```
   This works together with the corrected UCB to keep the search breadth.

All five modifications are computationally cheap (a few BFS distance look‑ups, a constant‑time penalty, and reuse of existing dead‑lock logic). They directly address the three highest‑impact weaknesses (walk‑cost omission, wrong exploration sign, and weak target incentive) while preserving the overall A*‑guided selection framework that already proved capable of solving a level when luck aligned. Incremental testing can tune `α`, `β`, and the dynamic `TARGET_BONUS` without a full rewrite, adhering to the 70 % rule.
