# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:35:53 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 9.43s |
| Status     | success |
| Tokens     | 5226 |
| Validation | valid |

---

## Prompt

============================================================
SYSTEM: MCTS Heuristic Improvement
============================================================
You are an expert game-playing AI researcher.
Your task is to improve a specific MCTS heuristic function
for the game 'sokoban_macro' (phase: backpropagation).

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
Description: Unify visited table with expansion, add depth‑discounted reward, and compute depth efficiently.
Generated:   2026-03-11T00:22:36.441500
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root while keeping the shared
    A*‑style visited table consistent and applying a small depth discount
    to the reward.

    This version:
      • Uses the same visited dictionary that ``default_expansion`` writes
        to (``_EXPANSION_STATE["visited"]``) instead of an ad‑hoc fallback.
      • Applies an exponential discount (default 0.99) based on the node's
        depth from the root, making deeper (more costly) solutions less
        attractive and giving the ``value/visits`` term a useful signal.
      • Computes the depth incrementally while climbing the tree,
        avoiding the O(depth²) cost of repeatedly calling a helper.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the shared visited table used by expansion.
    # ------------------------------------------------------------------
    if "_EXPANSION_STATE" in globals():
        visited = globals()["_EXPANSION_STATE"].setdefault("visited", {})
    else:
        # Fallback – a persistent dict attached to this function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root node (needed for step offset).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits, discounted value,
    #    and the shared visited table (minimum g‑score for each state).
    # ------------------------------------------------------------------
    depth = 0                     # number of pushes from root to current node
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value with depth discount --------------------------------
        discount = 0.99 ** depth   # small exponential penalty for deeper nodes
        node.value += reward * discount

        # ---- A* g‑score sync -------------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps                # total steps from root

        key = node.state.state_key()
        # Record the smallest g‑value seen for this state.
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent and increase depth
        node = node.parent
        depth += 1
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (trace evidence) | Why it hurts the search |
|------|---------------------------|--------------------------|
| **1️⃣** | *Visited‑table mismatch*: expansion stores a “g‑score” as `parent_depth + walk + 1` (push‑count + walk steps). Back‑propagation later overwrites the same entry with `node.state.steps – root_steps` (pure step count). Because the two formulas differ, a state that is cheap in the expansion view may be marked “already explored with a lower g” by back‑propagation and subsequently pruned. This can block promising branches early, especially on larger levels where walk cost dominates. |
| **2️⃣** | *Depth‑discounted reward*: `discount = 0.99 ** depth` is applied on every backup. For a solution that needs > 15 pushes the cumulative discount is ≈ 0.86, pulling the average `value/visits` down and making the node look worse than a shallow but dead‑end path. The solver therefore prefers short, possibly myopic pushes and may miss deeper but necessary sequences. |
| **3️⃣** | *No‑normalisation of accumulated value*: `node.value += reward * discount` is a sum, while the selection phase uses `node.value / node.visits`. Because the discount varies with depth, the average becomes a mixture of heavily discounted and undiscounted rewards, introducing a hidden bias that is hard for UCB‑style (or the A*‑guided) selector to interpret. |
| **4️⃣** | *Fallback handling*: when the visited table is missing (`_EXPANSION_STATE` not in globals) a hidden fallback dict is created. If the module reloads (e.g., during a hot‑reload of the MCTS loop) the two tables diverge, again causing inconsistent pruning. |

**2. ROOT CAUSE**  

The back‑propagation routine was written to *share* the visited dictionary used by expansion, but it records a **different g‑metric** (`steps‑root_steps`) than the one expansion uses (`pushes + walk + 1`). Because both phases read/write the same dict, the second write overwrites the first, so later expansions read a g‑value that does not reflect the true cost used for A*‑style pruning. Consequently many legal actions are discarded prematurely (see the large pruning lists in the expansion code).  

The depth discount is another design choice: it assumes “deeper = more pushes = worse”, yet in Sokoban macro‑push the *walk* component can dominate depth, and a constant 0.99 factor is too aggressive for puzzles that legitimately need > 10 pushes.  

Finally, the fallback visited dict makes the bug intermittent and hard to reproduce, which explains why the trace shows a perfect solve on level 5 (few pushes) but would likely fail on harder levels.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental improvements** (recommended). The core algorithm works; we only need to align the cost metric and temper the discount.

1. **Unified g‑score computation** – Introduce a small helper `calc_g(state, parent_depth, walk)` that both expansion and back‑propagation call. Back‑propagation should recompute `g = parent_depth + walk + 1` using the *same* walk distance that expansion used (the walk from the parent node’s player position to the push cell). The necessary walk can be cached on the node when it is created (e.g., `node._walk_cost`). This removes the inconsistency without changing the pruning logic.

2. **Separate visited tables** – Keep the expansion‑pruning table (`_EXPANSION_STATE["visited"]`) and a *pure* back‑propagation table for value‑only statistics. Back‑propagation only needs to update the first; it should **not** overwrite the pruning entry. Instead, after the backup just store the g‑score in a second dict (`_BP_STATE["g_min"]`) if you still need it for analytics. This prevents accidental “over‑pruning”.

3. **Depth‑aware discount based on real step cost** – Replace `0.99 ** depth` with `gamma ** (node.state.steps - root_steps)`, where `gamma` (e.g., 0.995) discounts proportionally to the *actual* number of steps taken, not the abstract push depth. This yields a smoother penalty that respects long walks but does not cripple solutions that require many pushes.

4. **Normalize value on the fly** – Instead of accumulating a raw sum, update an *average* directly:  
   `node.value = ((node.visits-1) * node.value + reward * discount) / node.visits`.  
   This keeps `node.value` already as an average, eliminating the `value/visits` division in selection and avoiding the mixed‑scale bias.

5. **Guard the fallback** – At module load, force creation of `_EXPANSION_STATE` once and never fall back to a hidden dict. Raise an exception if the dict is missing; this surfaces bugs early.

These targeted changes keep the existing A*‑guided selection and expansion logic intact while fixing the primary source of premature pruning and the excessive depth penalty. The solver will retain its fast solves on easy levels (as shown in the trace) and gain robustness on deeper puzzles where walk costs dominate.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Improvements over the previous version:
      • Do **not** overwrite the A*‑style visited table used during
        expansion. The table stores a g‑score based on pushes + walk
        cost, while the leaf node only knows the total step count.
        Updating it here caused mismatched pruning.  We therefore
        leave the visited dict untouched (it is already maintained
        by ``default_expansion``).
      • Apply an exponential discount based on the *actual number of
        steps* taken from the root, not on abstract push depth.  This
        yields a smoother penalty (γ‑discount) that respects long
        walks but does not overly punish deep but necessary solutions.
      • Keep ``node.value`` as a running **average** instead of a raw
        sum.  This removes the hidden bias introduced by mixing
        discounted and undiscounted rewards and allows the selector
        to use ``node.value`` directly.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Discount factor per step (tuned for Sokoban macro‑push)
    _GAMMA = 0.995

    # ------------------------------------------------------------------
    # 1. Locate the root node to obtain the step offset.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits, discounted average value.
    #    The visited‑pruning table is deliberately left unchanged.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- depth‑based discount --------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        step_depth = max(0, node_steps - root_steps)   # >= 0
        discount = _GAMMA ** step_depth

        # ---- running average of discounted reward ---------------------
        #   new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward * discount) / node.visits

        # Move toward the root
        node = node.parent
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
- No bugs detected: the function uses valid API calls, correct variable names, and avoids crashes.
- The loop is efficient (single upward traversal, O(depth)).
- Reward handling is appropriate; the discounted average is computed correctly.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: No changes needed; the draft implementation is correct and efficient.
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root.

    Improvements over the previous version:
      • Do **not** overwrite the A*‑style visited table used during
        expansion. The table stores a g‑score based on pushes + walk
        cost, while the leaf node only knows the total step count.
        Updating it here caused mismatched pruning.  We therefore
        leave the visited dict untouched (it is already maintained
        by ``default_expansion``).
      • Apply an exponential discount based on the *actual number of
        steps* taken from the root, not on abstract push depth.  This
        yields a smoother penalty (γ‑discount) that respects long
        walks but does not overly punish deep but necessary solutions.
      • Keep ``node.value`` as a running **average** instead of a raw
        sum.  This removes the hidden bias introduced by mixing
        discounted and undiscounted rewards and allows the selector
        to use ``node.value`` directly.
    """
    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    # Discount factor per step (tuned for Sokoban macro‑push)
    _GAMMA = 0.995

    # ------------------------------------------------------------------
    # 1. Locate the root node to obtain the step offset.
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 2. Walk up the tree, updating visits, discounted average value.
    #    The visited‑pruning table is deliberately left unchanged.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- depth‑based discount --------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        step_depth = max(0, node_steps - root_steps)   # >= 0
        discount = _GAMMA ** step_depth

        # ---- running average of discounted reward ---------------------
        #   new_avg = ((n-1)*old_avg + new_value) / n
        node.value = ((node.visits - 1) * node.value + reward * discount) / node.visits

        # Move toward the root
        node = node.parent
```
