# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 01:10:05 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.60s |
| Status     | success |
| Tokens     | 2615 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS simulation function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched heuristic evaluation for a leaf node.

    Returns:
        * state.returns()[p]                         if the state is terminal
        * 0.0                                         if a dead‑lock is detected
        * ((1 + w * progress) / (1 + h) - α * steps) * γ**steps
          otherwise, where
            - h = box‑only Manhattan distance (sum of nearest‑target distances)
            - progress = boxes_on_targets / num_targets
            - w, α, γ are tunable constants (default w=0.5, α=0.01, γ=0.97)

    The formula yields values in (0, 1] for non‑terminal,
    non‑dead‑locked states and heavily penalises deep, costly leaves.
    """
    # ------------------------------------------------------------------
    # Tunable hyper‑parameters – adjust if needed without breaking API.
    # ------------------------------------------------------------------
    PROGRESS_WEIGHT = 0.5   # boost for each fraction of boxes already placed
    STEP_PENALTY = 0.01     # linear penalty per macro‑push step taken
    GAMMA = 0.97            # depth discount factor (γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣  Terminal handling (solved, max‑steps, loss)
    # ------------------------------------------------------------------
    if state.is_terminal():
        # The environment already encodes solved/dead‑lock/step‑limit rewards.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 2️⃣  Early dead‑lock detection (optional helper on the state)
    # ------------------------------------------------------------------
    if hasattr(state, "_is_deadlocked") and callable(state._is_deadlocked):
        try:
            if state._is_deadlocked():
                return 0.0
        except Exception:
            # Defensive: if the dead‑lock check itself fails, ignore it.
            pass

    # ------------------------------------------------------------------
    # 3️⃣  Core heuristic components
    # ------------------------------------------------------------------
    # Box‑only Manhattan distance (same as used by A* expansion)
    h = _ag.h_sokoban_box_only(state)

    # Fraction of boxes already on their targets (0.0 … 1.0)
    progress = (
        state.boxes_on_targets() / state.num_targets
        if getattr(state, "num_targets", 0) > 0 else 0.0
    )

    # Base reward mixes distance and progress.
    # When h == 0 (all boxes on targets) the formula yields >1;
    # we will clamp later.
    base_reward = (1.0 + PROGRESS_WEIGHT * progress) / (1.0 + h)

    # ------------------------------------------------------------------
    # 4️⃣  Apply step‑cost penalty (steps = macro‑push count so far)
    # ------------------------------------------------------------------
    steps = getattr(state, "steps", 0)
    penalised = base_reward - STEP_PENALTY * steps
    if penalised < 0.0:
        penalised = 0.0

    # ------------------------------------------------------------------
    # 5️⃣  Depth discount – deeper leaves become less attractive.
    # ------------------------------------------------------------------
    # Use the same `steps` value as a proxy for depth.  If `max_depth`
    # is supplied (non‑zero) we cap the exponent to avoid under‑flow.
    depth = steps
    if max_depth > 0:
        depth = min(depth, max_depth)

    discounted = penalised * (GAMMA ** depth)

    # ------------------------------------------------------------------
    # 6️⃣  Final clamping – keep reward in a sensible range.
    # ------------------------------------------------------------------
    # Solved states (h == 0) should ideally return 1.0.
    if h == 0:
        return 1.0

    # Ensure we never exceed 1.0 after all adjustments.
    if discounted > 1.0:
        discounted = 1.0

    return discounted
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 432, in _smoke_test
    break
    
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/simulation/default_simulation.py", line 52, in default_simulation
    h = _ag.h_sokoban_box_only(state)
        ^^^
NameError: name '_ag' is not defined


== ACTUAL GameState PUBLIC API (for node.state or state param) ==
  apply_action: method
  boxes: set
  boxes_on_targets: method
  clone: method
  current_player: method
  height: int
  is_terminal: method
  legal_actions: method
  max_steps: int
  num_targets: int
  player: tuple
  returns: method
  state_key: method
  steps: int
  targets: frozenset
  total_box_distance: method
  walls: frozenset
  width: int

CRITICAL: The function must be SELF-CONTAINED. Define ALL helper functions inline. Do NOT use undefined names (e.g. _ag, _bfs_distance). Fix ONLY the broken parts. Keep the heuristic strategy the same.
Return using the SAME structured format.

ACTION: modify
FILE_NAME: default_simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**DESCRIPTION:** Fixed undefined reference to `_ag` by implementing an inline helper that computes the box‑only Manhattan distance using the public `GameState` API. Added necessary safeguards and kept the original heuristic strategy unchanged.  

```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Enriched heuristic evaluation for a leaf node.

    Returns:
        * state.returns()[p]                         if the state is terminal
        * 0.0                                         if a dead‑lock is detected
        * ((1 + w * progress) / (1 + h) - α * steps) * γ**steps
          otherwise, where
            - h = box‑only Manhattan distance (sum of nearest‑target distances)
            - progress = boxes_on_targets / num_targets
            - w, α, γ are tunable constants (default w=0.5, α=0.01, γ=0.97)

    The formula yields values in (0, 1] for non‑terminal,
    non‑dead‑locked states and heavily penalises deep, costly leaves.
    """
    # ------------------------------------------------------------------
    # Tunable hyper‑parameters – adjust if needed without breaking API.
    # ------------------------------------------------------------------
    PROGRESS_WEIGHT = 0.5   # boost for each fraction of boxes already placed
    STEP_PENALTY = 0.01     # linear penalty per macro‑push step taken
    GAMMA = 0.97            # depth discount factor (γ < 1)

    # ------------------------------------------------------------------
    # 1️⃣  Terminal handling (solved, max‑steps, loss)
    # ------------------------------------------------------------------
    if state.is_terminal():
        # The environment already encodes solved/dead‑lock/step‑limit rewards.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 2️⃣  Early dead‑lock detection (optional helper on the state)
    # ------------------------------------------------------------------
    if hasattr(state, "_is_deadlocked") and callable(state._is_deadlocked):
        try:
            if state._is_deadlocked():
                return 0.0
        except Exception:
            # Defensive: if the dead‑lock check itself fails, ignore it.
            pass

    # ------------------------------------------------------------------
    # 3️⃣  Helper: box‑only Manhattan distance (h)
    # ------------------------------------------------------------------
    def box_only_manhattan(state) -> float:
        """
        Sum, over every box, of the Manhattan distance to the closest target.
        If there are no targets (unlikely in a valid Sokoban level) the
        function returns 0.0 to avoid division‑by‑zero later on.
        """
        if not state.boxes or not state.targets:
            return 0.0
        total = 0
        # Pre‑compute target coordinates for speed.
        target_coords = list(state.targets)
        for bx, by in state.boxes:
            # Manhattan distance to each target; keep the minimum.
            min_dist = min(abs(bx - tx) + abs(by - ty) for tx, ty in target_coords)
            total += min_dist
        return float(total)

    h = box_only_manhattan(state)

    # ------------------------------------------------------------------
    # 4️⃣  Fraction of boxes already on their targets (0.0 … 1.0)
    # ------------------------------------------------------------------
    # ``boxes_on_targets`` is a method returning the number of correctly placed boxes.
    boxes_on_targets = state.boxes_on_targets()
    num_targets = getattr(state, "num_targets", 0)
    progress = (
        boxes_on_targets / num_targets
        if num_targets > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 5️⃣  Core heuristic components
    # ------------------------------------------------------------------
    # Base reward mixes distance and progress.
    # When h == 0 (all boxes on targets) the formula yields >1; we will clamp later.
    base_reward = (1.0 + PROGRESS_WEIGHT * progress) / (1.0 + h)

    # ------------------------------------------------------------------
    # 6️⃣  Apply step‑cost penalty (steps = macro‑push count so far)
    # ------------------------------------------------------------------
    steps = getattr(state, "steps", 0)
    penalised = base_reward - STEP_PENALTY * steps
    if penalised < 0.0:
        penalised = 0.0

    # ------------------------------------------------------------------
    # 7️⃣  Depth discount – deeper leaves become less attractive.
    # ------------------------------------------------------------------
    # Use the same `steps` value as a proxy for depth.  If `max_depth`
    # is supplied (non‑zero) we cap the exponent to avoid under‑flow.
    depth = steps
    if max_depth > 0:
        depth = min(depth, max_depth)

    discounted = penalised * (GAMMA ** depth)

    # ------------------------------------------------------------------
    # 8️⃣  Final clamping – keep reward in a sensible range.
    # ------------------------------------------------------------------
    # Solved states (h == 0) should ideally return 1.0.
    if h == 0:
        return 1.0

    # Ensure we never exceed 1.0 after all adjustments.
    if discounted > 1.0:
        discounted = 1.0

    return discounted
```
