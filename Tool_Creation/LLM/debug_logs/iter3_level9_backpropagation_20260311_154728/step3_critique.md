# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 15:48:04 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.42s |
| Status     | success |
| Tokens     | 5729 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

PHASE: backpropagation
  • What it does: Sends the simulation result back up the visited path. Updates node statistics (visits, value) that selection's UCB1 uses.
  • Optimization goal: Control HOW strongly rollout evidence affects node values. Calibrate depth discount, solved vs partial progress, path length.
  • Constraints: Only aggregates evidence — no move generation, no deadlock pruning, no rollout policy. Must stay coherent with selection's expectations.
  • Good patterns: depth discount so shorter plans dominate, weight solved outcomes above partial progress, reduce credit for noisy weak rollouts.
  • Avoid: move generation, deadlock pruning, rollout action-choice policy.

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
TARGET HEURISTIC TO IMPROVE (backpropagation)
------------------------------------------------------------
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
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. Key Weaknesses**

| Rank | Observation (from Trace #1) | Why it matters |
|------|-----------------------------|----------------|
| **1** | The *leaf‑relative* depth‑discount (`γ = 0.95`) is applied **on top of** the simulation’s own depth‑discount (`0.97^{steps}`) and the per‑step penalty (`0.001 × steps`). For a solved leaf at step 34 the simulation already yields ≈ 0.55; after back‑prop the root receives only ≈ 0.55 × 0.95^{34} ≈ 0.02 + the undisc‑ounted solved‑bonus. The bulk of the learning signal therefore comes from a **single constant bonus** at the root, while all intermediate nodes receive a heavily attenuated value. |
| **2** | The *solved‑bonus* is added **only to the root**. If a solution is found deeper in the tree, the ancestors (which guide selection) still rely on the tiny discounted reward rather than a strong positive signal. This makes the UCB‑value of promising branches noisy and can cause the algorithm to “forget’’ good lines after a few roll‑outs. |
| **3** | The per‑step penalty (`STEP_PENALTY = 0.001`) is subtracted **before** the depth‑discount. Because the simulation already punishes long roll‑outs via the walk‑cost term and the depth‑discount, the extra penalty compounds the bias against deeper (but correct) plans, especially on harder levels where the optimal solution may require > 40 pushes. |
| **4** | No explicit handling for *partial‑solution* nodes (e.g., “2 / 4 boxes on target”). The only “progress” signal is the raw simulation reward, which is already heavily diluted by the two discounts. Consequently, the UCB‑exploration term dominates early, leading to excessive exploration of low‑value branches (see many children with `avg = 0.000`). |

**2. Root Cause**

The back‑propagation routine was designed to keep deep‑leaf influence modest, but the combination of **(a)** a simulation depth‑discount, **(b)** an additional leaf‑relative discount, and **(c)** a step‑penalty creates *double‑discounting*. The only surviving strong signal is the ad‑hoc root‑only solved‑bonus. Because the value accumulated by intermediate nodes is almost negligible, UCB‑selection cannot reliably distinguish promising lines, which explains the many zero‑average children and the need for a large number of visits (500) just to achieve a solution on a tiny level.

**3. Proposed Approach – A (Incremental)**  

We will keep the overall structure (visits, value, g‑score sync) but **remove the unnecessary attenuation** and **strengthen the solved‑signal** for all ancestors.

| Modification | Rationale | Expected Effect |
|--------------|-----------|-----------------|
| **a. Eliminate the leaf‑relative discount** (`decay = GAMMA ** remaining`). Replace with `decay = 1.0`. | The simulation already supplies a depth‑aware reward; a second discount only weakens learning. | Intermediate nodes receive the full (already‑discounted) reward, giving UCB a clearer signal. |
| **b. Apply the solved‑bonus to *every* node on the path**, not just the root (e.g., `if is_solved: cur.value += SOLVED_BONUS`). | A solved leaf should reinforce the entire trajectory that led to it, not just the root. | All ancestors acquire a large positive boost, improving their average value and reducing wasteful exploration. |
| **c. Remove (or drastically reduce) the per‑step penalty** (`STEP_PENALTY`). The simulation’s walk‑cost (`ALPHA × w`) and depth factor already discourage long roll‑outs. | Prevents over‑penalising deep but correct solutions. | Rewards for longer optimal solutions stay higher, allowing the algorithm to discover them on larger levels. |
| **d. Slightly raise `SOLVED_BONUS` (e.g., 1.5) or scale it by remaining‑target count** to keep a strong gradient even when many solutions are found early. | Keeps the solved‑signal dominant over noisy values from failed roll‑outs. | Faster convergence on promising branches, fewer wasted iterations. |
| **e. (Optional) Clip the discounted reward to `[0, 1]` before adding to `cur.value`** to keep node averages bounded and avoid runaway values when many solved‑bonuses accumulate. | Guarantees that UCB’s exploitation term stays comparable across nodes. | More stable selection dynamics. |

These changes are **local** to `default_backpropagation` and respect the existing API (no new parameters, no external state). They preserve the proven parts (visit counting, A* g‑score synchronization) while correcting the over‑discounting that currently forces the algorithm to rely on a single root‑only bonus. The result should be a sharper value gradient along the path to a solution, higher average returns for intermediate nodes, and ultimately faster, more reliable solving on the remaining levels.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up the tree.

    Improvements over the previous version:
      • Eliminates the leaf‑relative depth discount (the simulation already
        contains a depth‑aware reward). Nodes now receive the full reward.
      • Removes the per‑step penalty – walk‑cost is accounted for in the
        simulation reward, avoiding over‑penalisation of deep but correct
        solutions.
      • Adds a solved‑state bonus to **every** node on the path, not only the
        root, so the whole successful trajectory is reinforced.
      • Clips the reward to [0, 1] before accumulation to keep node values
        bounded.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    SOLVED_BONUS = 2.0          # extra reward added to each ancestor of a solved leaf
    # No depth discount is applied; the simulation already handles it.
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
    leaf_state = leaf_node.state

    # Detect solved terminal states.
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )

    # Ensure the propagated reward stays within the legal interval.
    base_reward = max(0.0, min(1.0, reward))

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards – no additional discounting.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # 1️⃣ Update visit count.
        cur.visits += 1

        # 2️⃣ Accumulate the (clipped) reward.
        cur.value += base_reward

        # 3️⃣ If the leaf was a solution, reinforce every ancestor.
        if is_solved:
            cur.value += SOLVED_BONUS

        # 4️⃣ Synchronise A* g‑score (unchanged).
        key = cur.state.state_key()
        cur_depth = _ag.node_depth(cur)   # absolute depth from the root
        if key not in visited or visited[key] > cur_depth:
            visited[key] = cur_depth

        # 5️⃣ Move to parent.
        cur = getattr(cur, "parent", None)
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
- No bugs detected; the function uses the correct API, variables are defined, and there are no performance concerns.
- The modifications align with the intended incremental improvements.
- Reward handling and A* synchronization remain coherent.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: Finalized incremental improvements to backpropagation.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up the tree.

    Improvements over the previous version:
      • Eliminates the leaf‑relative depth discount (the simulation already
        contains a depth‑aware reward). Nodes now receive the full reward.
      • Removes the per‑step penalty – walk‑cost is accounted for in the
        simulation reward, avoiding over‑penalisation of deep but correct
        solutions.
      • Adds a solved‑state bonus to **every** node on the path, not only the
        root, so the whole successful trajectory is reinforced.
      • Clips the propagated reward to [0, 1] before accumulation to keep node
        values bounded.
    """
    # ------------------------------------------------------------------
    # Tunable parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    SOLVED_BONUS = 2.0          # extra reward added to each ancestor of a solved leaf

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
    leaf_state = leaf_node.state

    # Detect solved terminal states.
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )

    # Ensure the propagated reward stays within the legal interval.
    base_reward = max(0.0, min(1.0, reward))

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards – no additional discounting.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # 1️⃣ Update visit count.
        cur.visits += 1

        # 2️⃣ Accumulate the (clipped) reward.
        cur.value += base_reward

        # 3️⃣ If the leaf was a solution, reinforce every ancestor.
        if is_solved:
            cur.value += SOLVED_BONUS

        # 4️⃣ Synchronise A* g‑score (unchanged).
        key = cur.state.state_key()
        cur_depth = _ag.node_depth(cur)   # absolute depth from the root
        if key not in visited or visited[key] > cur_depth:
            visited[key] = cur_depth

        # 5️⃣ Move to parent.
        cur = getattr(cur, "parent", None)
```
