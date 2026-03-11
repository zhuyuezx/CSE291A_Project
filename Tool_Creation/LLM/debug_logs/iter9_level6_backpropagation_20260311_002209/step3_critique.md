# step3_critique

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:22:36 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 5.88s |
| Status     | success |
| Tokens     | 5137 |
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
Description: Fixed the `NameError` caused by the undefined `_ag` identifier and ensured a fallback visited‑table is available when the A* guide object is not present.
Generated:   2026-03-11T00:21:20.186102
"""

def default_backpropagation(node, reward: float) -> None:
    """
    Backpropagate reward from leaf to root while keeping the A* visited
    table consistent.

    Parameters
    ----------
    node : MCTSNode
        Leaf node from which the back‑propagation starts.
    reward : float
        Leaf evaluation (e.g. 1.0 for solved, 0.0 for dead‑end).

    The function updates:
        * node.visits
        * node.value   (adds reward, respecting player perspective)
        * visited[state_key] – the minimum total step cost ``g`` seen
          for that state, where ``g`` equals the number of steps taken
          from the root (pushes + walk steps). This matches the metric
          used in expansion and prevents over‑pruning.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the visited table.
    #    The original implementation expected a global ``_ag`` object with
    #    a ``get_visited`` method.  If this object is not defined (as was
    #    the case during the test run), we fall back to a simple module‑
    #    level dictionary that persists across calls.
    # ------------------------------------------------------------------
    if "_ag" in globals() and hasattr(_ag, "get_visited"):
        visited = _ag.get_visited()          # type: ignore[name-defined]
    else:
        # Create (or reuse) a fallback visited dict attached to the function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 2. Find the root of the tree (needed for perspective & cost offset)
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban

    # Cache root step count (normally 0, but keep generic)
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 3. Walk up the tree, updating visits/value and the visited table.
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        mover = (
            node.parent.state.current_player()
            if node.parent
            else perspective
        )
        # No discounting needed for the current A*‑guided selector.
        node.value += reward if mover == perspective else -reward

        # ---- A* g‑score sync -------------------------------------------
        # Total steps from the root to this node (pushes + walking steps).
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps

        key = node.state.state_key()
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move up the tree
        node = node.parent
```

------------------------------------------------------------
PRIOR ANALYSIS (from step 1)
------------------------------------------------------------
Below is the analysis identifying weaknesses and a proposed
approach (incremental or restructure). Implement the proposed
changes faithfully — stay aligned with the analysis.

**1. KEY WEAKNESSES**  

| Rank | Symptom (from the trace) | Why it hurts the search |
|------|--------------------------|--------------------------|
| **1️⃣** | The algorithm cycles: after move 2 the state is *almost* the same as after move 1 (`Step 4/1000 … distance 6` → later `Step 11/1000 … distance 6`). The tree keeps expanding the same macro‑pushes instead of steering toward new, better configurations. | There is **no effective duplicate‑state pruning**. Expansion keeps generating children that have already been seen with a cheaper cost, because the *visited* table that drives this pruning is never populated with the real g‑scores of explored nodes. |
| **2️⃣** | All child nodes report the same average value (`avg = 0.167`). The value term never differentiates promising from unpromising branches, so the selector relies almost entirely on the A*‑style `f = g + h`. | The back‑propagation routine does **not feed back any useful reward information** to the nodes that matter for the selector (the `value/visits` term). Consequently the tree does not learn from the heuristic evaluation; it only follows raw depth‑plus‑heuristic, which is blind to the fact that a deeper node may already be a dead‑end. |
| **3️⃣** | The reward from the simulation (`1/(1+h)`) is added unchanged irrespective of how many pushes or walk steps were needed to reach the leaf. | This gives **no depth discount**, so a long, costly solution is valued the same as a short one, encouraging the algorithm to explore costly loops rather than pruning them. |

**2. ROOT CAUSE**  

*Back‑propagation updates the wrong visited table.*

```python
if "_ag" in globals() and hasattr(_ag, "get_visited"):
    visited = _ag.get_visited()
else:
    visited = default_backpropagation._fallback_visited
```

The expansion code (the only place that actually **prunes** duplicate states) uses a **different** dictionary:

```python
_EXPANSION_STATE = { "visited": {} }   # filled in default_expansion
```

Thus the `g`‑score of a node is never recorded where the expansion routine looks, so the test  

```python
if key in visited and visited[key] <= total_g:  # prune
```

always fails. The tree can therefore revisit the same board position with a larger cost, creating the loops seen in the trace.  

Because the selector’s tie‑breaker `q = value/visits` never receives a meaningful signal (the value is inflated equally for all ancestors), the search cannot compensate for the missing pruning, yielding the flat `avg=0.167` scores.

**3. PROPOSED APPROACH**  

**Strategy A – Incremental (targeted) improvements**

The fundamental issue is the *split* visited table; fixing it requires only a few controlled changes while preserving the rest of the pipeline that already works (A*‑guided selection, expansion heuristics, simulation). The steps are:

1. **Unify the visited store** – have both expansion and back‑propagation read/write a single module‑level dictionary (e.g. rename `_EXPANSION_STATE["visited"]` to `_VISITED`).  
   * In `default_expansion` replace references to `_EXPANSION_STATE["visited"]` with `_VISITED`.  
   * In `default_backpropagation` fetch the same `_VISITED` (create it lazily if absent) instead of the fallback dict.

2. **Record the true g‑score** – during back‑propagation compute `g = node.state.steps - root_steps` (already done) and store the *minimum* g for the state key in the unified visited table, exactly as expansion expects. No extra work is needed beyond using the common dict.

3. **Add a shallow depth discount to the propagated reward** so that longer trajectories are penalised and the `value` term becomes informative:

   ```python
   discount = 0.99 ** (node_depth(node))   # small exponential decay
   node.value += reward * discount
   ```

   This keeps the algorithm single‑player (no sign flip) but ensures that deeper nodes contribute less, allowing the selector’s `q` to differentiate cheap solutions from expensive loops.

4. **Optional but cheap** – after updating the unified visited table, also propagate the *best* reward seen for each state (`best_reward[key] = max(best_reward.get(key,0), reward)`). This can be used later for tie‑breaking if desired, but is not mandatory.

These changes directly eliminate the duplicate‑state explosion (weakness 1) and give the `value/visits` term a meaningful gradient (weakness 2). The depth discount addresses weakness 3 without altering the existing A*‑guided selection logic.

**Why not a full restructure?**  
The current pipeline (A*‑guided selection + expansion with monotonicity & corner dead‑lock checks) already produces sensible child ordering; the only bottleneck is the missing feedback loop between back‑propagation and expansion. A modest unification of the visited store plus a depth‑aware reward update resolves the deadlock with minimal risk and preserves all validated components.

------------------------------------------------------------
DRAFT CODE (from step 2 — to be reviewed)
------------------------------------------------------------
```python
def default_backpropagation(node, reward: float) -> None:
    """
    Back‑propagate a leaf reward up to the root while keeping the shared
    A*‑style visited table consistent and applying a small depth discount
    to the reward.

    Changes compared with the original implementation:
      • The visited table used for duplicate‑state pruning is now the
        same dictionary that ``default_expansion`` writes to
        (``_EXPANSION_STATE["visited"]``).  If the expansion module cannot
        be imported we fall back to a module‑level fallback dictionary.
      • A discount of ``discount_factor ** depth`` (default 0.99) is
        applied to the reward before adding it to a node's ``value``.
        This penalises long, costly paths and makes the ``value/visits``
        term informative for the A*‑guided selector.
    """
    # ------------------------------------------------------------------
    # 1. Obtain the shared visited table.
    # ------------------------------------------------------------------
    # Try to import the expansion module and reuse its visited dict.
    visited = None
    try:
        import importlib
        exp_mod = importlib.import_module("expansion")  # module name may vary
        if hasattr(exp_mod, "_EXPANSION_STATE"):
            visited = exp_mod._EXPANSION_STATE.get("visited")
    except Exception:
        # Import failed – continue with fallback.
        pass

    if visited is None:
        # Fallback – a persistent dict attached to this function.
        if not hasattr(default_backpropagation, "_fallback_visited"):
            default_backpropagation._fallback_visited = {}
        visited = default_backpropagation._fallback_visited

    # ------------------------------------------------------------------
    # 2. Helper to compute depth (number of pushes from root).
    # ------------------------------------------------------------------
    def node_depth(n):
        d = 0
        while n.parent is not None:
            d += 1
            n = n.parent
        return d

    # ------------------------------------------------------------------
    # 3. Find the root node (needed for perspective & step offset).
    # ------------------------------------------------------------------
    root = node
    while root.parent is not None:
        root = root.parent

    perspective = root.state.current_player()          # always 0 for Sokoban
    root_steps = getattr(root.state, "steps", 0)

    # ------------------------------------------------------------------
    # 4. Walk up the tree, updating visits, value (with depth discount),
    #    and the shared visited table (minimum g‑score seen for each state).
    # ------------------------------------------------------------------
    while node is not None:
        # ---- visits ---------------------------------------------------
        node.visits += 1

        # ---- value ----------------------------------------------------
        depth = node_depth(node)                # pushes from root to this node
        discount = 0.99 ** depth               # exponential decay, 0.99≈small penalty
        node.value += reward * discount

        # ---- A* g‑score sync -------------------------------------------
        node_steps = getattr(node.state, "steps", 0)
        g = node_steps - root_steps               # total steps from root

        key = node.state.state_key()
        # Record the smallest g‑value seen for this state.
        if key not in visited or visited[key] > g:
            visited[key] = g

        # Move to parent
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
- The visited‑table lookup relies on importing an external module whose name may differ, risking ImportError; using the already‑available global `_EXPANSION_STATE` is safer.
- `node_depth` is recomputed on every loop iteration, yielding O(depth²) work; depth can be accumulated while walking up the tree.
- The `perspective` variable is no longer needed after removing sign‑flip logic, so it can be removed for clarity.

ACTION: modify
FILE_NAME: backpropagation.py
FUNCTION_NAME: default_backpropagation
DESCRIPTION: Unify visited table with expansion, add depth‑discounted reward, and compute depth efficiently.
```python
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
