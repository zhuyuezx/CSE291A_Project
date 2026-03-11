# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:07:54 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 10.07s |
| Status     | success |
| Tokens     | 6096 |
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
Description: Removed the misplaced `from __future__ import annotations` (which caused a `SyntaxError`) and added a lightweight import guard. The rest of the heuristic logic is unchanged.
Generated:   2026-03-11T01:02:10.854452
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate a rollout reward from a leaf node up to the root.

    Enhancements:
      * depth_discount (γ) reduces the impact of evidence the farther
        the ancestor is from the leaf (γ ≈ 0.95 by default).
      * solved_bonus adds a small extra reward for truly solved leaf
        states, ensuring they are preferred over merely‑good rollouts.
    """
    # ------------------------------------------------------------------
    # Configuration parameters – can be tweaked without breaking the API.
    # ------------------------------------------------------------------
    GAMMA = 0.95          # depth‑discount factor (0 < γ < 1)
    SOLVED_BONUS = 0.20   # extra reward for a leaf that actually solves the puzzle

    # ------------------------------------------------------------------
    # Lazy‑import shared A* globals to avoid circular‑import problems.
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
        assert _s.loader is not None  # for mypy / linters
        _s.loader.exec_module(_m)
    _ag = sys.modules[_KEY]

    # ------------------------------------------------------------------
    # Determine the depth of the leaf node once – used for discounting.
    # ------------------------------------------------------------------
    leaf_node = node
    leaf_depth = _ag.node_depth(leaf_node)

    # Detect whether the leaf is a *solved* terminal state.
    leaf_state = leaf_node.state
    is_solved = (
        leaf_state.is_terminal()
        and leaf_state.boxes_on_targets() == leaf_state.num_targets
    )
    # Base reward for this leaf (may be boosted if solved).
    base_reward = reward + SOLVED_BONUS if is_solved else reward

    visited = _ag.get_visited()

    # ------------------------------------------------------------------
    # Propagate upwards, applying depth discount and updating A* g‑scores.
    # ------------------------------------------------------------------
    cur = leaf_node
    while cur is not None:
        # Increment visit count.
        cur.visits += 1

        # Compute distance from the leaf (0 for leaf itself).
        cur_depth = _ag.node_depth(cur)
        distance_from_leaf = leaf_depth - cur_depth

        # Apply exponential discount.
        discounted_reward = (GAMMA ** distance_from_leaf) * base_reward

        # Sokoban is single‑player → always add (no sign flip).
        cur.value += discounted_reward

        # ── A* g‑score sync (unchanged) ─────────────────────────────────
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

**1. KEY WEAKNESSES**

| # | Symptom (trace) | Why it hurts the search |
|---|----------------|--------------------------|
| A | After 12 moves the tree’s *average Q* values are already ~0.5 – 0.6 while the puzzle is still 1 push away (total distance = 1). The algorithm keeps expanding high‑value branches instead of chasing the *shortest* solution. |
| B | The depth‑discount factor is **γ = 0.95** and applied as γ^(leaf‑depth − node‑depth). For a 10‑push line the leaf still contributes 0.60 × reward, which is huge because the simulation reward is already close to 1 (1/(1+h) ≈ 0.9‑1.0). Long, costly plans therefore look just as attractive as the optimal 1‑push finish. |
| C | The *solved‑bonus* (0.20) is tiny compared with the raw reward (≈ 1.0). A solved leaf gets ≈ 1.20, which is barely higher than the ≈ 0.9‑1.0 values of near‑solved leaves, so the search does not prioritize terminal victories. |
| D | Back‑propagation discounts **only by node depth** (push count) and completely ignores the *actual step cost* of a macro‑push (walk + push). A long walk that costs many steps is treated the same as a cheap push, encouraging the planner to favour actions that look cheap in the heuristic but are expensive in real steps. |

**Impact ranking:**  
1️⃣ Insufficient depth/step discount (A + B) – it makes the value landscape flat, so the A*-guided selection cannot differentiate a 1‑push win from a 15‑push detour.  
2️⃣ Undervalued solved‑bonus (C) – solved states never dominate the UCB‑like value used for tie‑breaking.  
3️⃣ No step‑cost penalty (D) – the planner can waste many steps walking around without penalty, leading to the “stuck” pattern observed after move 12.

---

**2. ROOT CAUSE**

The back‑propagation routine:

```python
distance_from_leaf = leaf_depth - cur_depth
discounted_reward = (GAMMA ** distance_from_leaf) * base_reward
```

* uses **γ = 0.95** (very mild decay) and discounts *relative* to the leaf, not the *absolute* cost incurred to reach the leaf.  
* adds a constant `SOLVED_BONUS` of 0.20, which is negligible against a base reward already near 1.0.  
* never incorporates the actual **step count** (`leaf_state.steps` or the macro‑push walk length) into the reward.  

Consequently, a leaf that is only a few steps from the goal still yields a reward ≈ 0.9, and every ancestor receives ≈ 0.9 × 0.95ⁿ – a value that stays high even after many pushes. The selection routine (A*‑guided) therefore prefers nodes with high average Q even if they lie deep in the tree, preventing the algorithm from “seeing” that a much shorter solution exists.

---

**3. PROPOSED APPROACH – Strategy A (incremental)**  

We can keep the overall structure (single‑player, additive value, A*‑guided selection) and simply **re‑balance the credit the leaf gives to its ancestors**:

1. **Stronger depth discount** – drop γ to **0.70** (or make it configurable). This makes a leaf 10 pushes deep contribute only ~0.03 of its reward, sharply favouring shallow solutions.  

2. **Absolute‑depth discount** – compute discount from the root, not from the leaf:  

   ```python
   discount = GAMMA ** cur_depth          # 0 at root, higher deeper
   discounted_reward = discount * base_reward
   ```

   This penalises every extra push equally, regardless of where the leaf sits, and aligns with the A* notion of g‑cost.

3. **Step‑cost penalty** – after obtaining the raw simulation reward `r = 1/(1+h)`, subtract a small linear term proportional to the leaf’s *total step count* (or macro‑push walk length).  

   ```python
   STEP_PENALTY = 0.01           # tunable
   r -= STEP_PENALTY * leaf_state.steps
   r = max(0.0, r)               # keep non‑negative
   ```

   This directly discourages long walks and pushes.

4. **Boost solved terminals** – raise `SOLVED_BONUS` to **0.5** (or 0.4) and *add* it *after* the step‑penalty so a true solution still receives a clear advantage:  

   ```python
   base_reward = r + (SOLVED_BONUS if is_solved else 0.0)
   ```

5. **Clamp reward** – ensure `base_reward` never exceeds 1.0 + SOLVED_BONUS, preserving a clean ordering (solved > any heuristic value).

These changes are **local** (only the back‑propagation loop) and preserve the rest of the pipeline (selection, expansion, simulation) which already work correctly. By sharply reducing the value of deep, costly leaves and making solved leaves stand out, the A*‑guided selector will be driven toward the minimal‑push path, addressing the flat‑value problem that caused the algorithm to linger at Q≈0.5 without solving level 6.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
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
- No issues found.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
```python
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
