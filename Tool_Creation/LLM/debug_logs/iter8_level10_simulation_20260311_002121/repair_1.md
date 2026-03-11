# repair_1

| Field      | Value |
|------------|-------|
| Timestamp  | 2026-03-11 00:22:09 |
| Model      | api-gpt-oss-120b |
| Elapsed    | 8.51s |
| Status     | success |
| Tokens     | 2144 |
| Validation | valid |

---

## Prompt

You previously generated the following sokoban_macro MCTS simulation function, but it raised a runtime error.

== BROKEN CODE ==
```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A*‑guided leaf evaluation.

    Returns:
        1.0                         if the state is solved,
        0.0                         if a dead‑lock is detected,
        state.returns()[p]          if terminal for other reasons,
        exp(-γ * h)                 otherwise,
            where h combines box distance, estimated walk cost,
            and boxes‑on‑target progress.
    """
    # Terminal states (solved, dead‑locked, step limit) keep their original reward.
    if state.is_terminal():
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣  Immediate dead‑lock detection (soft dead‑locks)
    # ------------------------------------------------------------------
    if _simple_corner_deadlock(state):
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣  Core cost components
    # ------------------------------------------------------------------
    # a) Box‑to‑nearest‑target Manhattan sum (already provided)
    box_dist = _ag.h_sokoban_box_only(state)          # integer ≥ 0

    # b) Rough estimate of the walking distance the player must travel
    #    before the next push.  Use Manhattan distance to the closest box
    #    (ignoring walls) – cheap and fast, yet captures long walks.
    player_r, player_c = state.player
    if state.boxes:
        walk_est = min(
            abs(player_r - br) + abs(player_c - bc)
            for (br, bc) in state.boxes
        )
    else:
        walk_est = 0

    # c) Progress term: how many boxes are already on targets
    on_target = state.boxes_on_targets()
    total_targets = getattr(state, "num_targets", len(state.targets))

    # ------------------------------------------------------------------
    # 3️⃣  Combine into a single cost `h`
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walking estimate
    BETA = 2.0    # boost per placed box
    h = (box_dist + ALPHA * walk_est) / (1.0 + BETA * on_target)

    # ------------------------------------------------------------------
    # 4️⃣  Convert cost to a reward in (0, 1]
    # ------------------------------------------------------------------
    GAMMA = 0.20
    reward = math.exp(-GAMMA * h)

    # Perfect solution check – avoid rounding errors.
    if on_target == total_targets and box_dist == 0:
        reward = 1.0

    return reward
```

== RUNTIME ERROR ==
Traceback (most recent call last):
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/LLM/optimizer.py", line 392, in _smoke_test
    result = fn(*test_args)
  File "/Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation/MCTS_tools/simulation/simulation.py", line 26, in default_simulation
    if _simple_corner_deadlock(state):
       ^^^^^^^^^^^^^^^^^^^^^^^
NameError: name '_simple_corner_deadlock' is not defined


== ACTUAL GameState PUBLIC API ==
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

Fix ONLY the broken parts. Keep the heuristic strategy the same.
Return using the SAME structured format.

ACTION: modify
FILE_NAME: simulation.py
FUNCTION_NAME: default_simulation
DESCRIPTION: <one-line description of what you fixed>
```python
<complete corrected function here>
```

---

## Response

**ACTION:** modify  
**FILE_NAME:** simulation.py  
**FUNCTION_NAME:** default_simulation  
**DESCRIPTION:** Fixed undefined helper and mismatched API calls; now uses the public `GameState` interface for dead‑lock detection and distance/target information while preserving the original heuristic.

```python
def default_simulation(state, perspective_player: int, max_depth: int = 0) -> float:
    """
    Direct A*‑guided leaf evaluation.

    Returns:
        1.0                         if the state is solved,
        0.0                         if a dead‑lock is detected,
        state.returns()[p]          if terminal for other reasons,
        exp(-γ * h)                 otherwise,
            where h combines box distance, estimated walk cost,
            and boxes‑on‑target progress.
    """
    import math

    # ------------------------------------------------------------------
    # Helper: very cheap corner dead‑lock detection
    # ------------------------------------------------------------------
    def _simple_corner_deadlock(st) -> bool:
        """
        Returns True if any box lies in a corner formed by two walls
        (and that corner is not a target).  This is a classic soft
        dead‑lock check used in Sokoban heuristics.
        """
        # Pre‑compute walls for quick lookup
        walls = st.walls
        targets = st.targets

        for (br, bc) in st.boxes:
            # If the box is already on a target, it cannot be a dead‑lock.
            if (br, bc) in targets:
                continue

            # Check the four corner configurations.
            # Upper‑left corner
            if ((br - 1, bc) in walls and (br, bc - 1) in walls):
                return True
            # Upper‑right corner
            if ((br - 1, bc) in walls and (br, bc + 1) in walls):
                return True
            # Lower‑left corner
            if ((br + 1, bc) in walls and (br, bc - 1) in walls):
                return True
            # Lower‑right corner
            if ((br + 1, bc) in walls and (br, bc + 1) in walls):
                return True
        return False

    # ------------------------------------------------------------------
    # 0️⃣  Terminal states (solved, dead‑locked, step limit)
    # ------------------------------------------------------------------
    if state.is_terminal():
        # The GameState returns a list/tuple of per‑player rewards.
        return state.returns()[perspective_player]

    # ------------------------------------------------------------------
    # 1️⃣  Immediate dead‑lock detection (soft dead‑locks)
    # ------------------------------------------------------------------
    if _simple_corner_deadlock(state):
        return 0.0

    # ------------------------------------------------------------------
    # 2️⃣  Core cost components
    # ------------------------------------------------------------------
    # a) Box‑to‑nearest‑target Manhattan sum (public API)
    #    total_box_distance() already returns the summed Manhattan distance
    #    from each box to its closest target.
    box_dist = state.total_box_distance()          # integer ≥ 0

    # b) Rough estimate of the walking distance the player must travel
    #    before the next push.  Use Manhattan distance to the closest box
    #    (ignoring walls) – cheap and fast, yet captures long walks.
    player_r, player_c = state.player
    if state.boxes:
        walk_est = min(
            abs(player_r - br) + abs(player_c - bc)
            for (br, bc) in state.boxes
        )
    else:
        walk_est = 0

    # c) Progress term: how many boxes are already on targets
    on_target = len(state.boxes_on_targets())
    total_targets = state.num_targets

    # ------------------------------------------------------------------
    # 3️⃣  Combine into a single cost `h`
    # ------------------------------------------------------------------
    ALPHA = 0.5   # weight for walking estimate
    BETA = 2.0    # boost per placed box
    h = (box_dist + ALPHA * walk_est) / (1.0 + BETA * on_target)

    # ------------------------------------------------------------------
    # 4️⃣  Convert cost to a reward in (0, 1]
    # ------------------------------------------------------------------
    GAMMA = 0.20
    reward = math.exp(-GAMMA * h)

    # Perfect solution check – avoid rounding errors.
    if on_target == total_targets and box_dist == 0:
        reward = 1.0

    return reward
```
