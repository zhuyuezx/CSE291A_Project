# A* (sokoban_astar.ipynb) vs MCTS Tools

This doc summarizes how the notebook A* solver relates to the MCTS tool phases and what was changed so MCTS behavior matches A* (including level 2).

## Logic comparison

| Aspect | sokoban_astar.ipynb | MCTS tools (expansion / simulation) |
|--------|----------------------|-------------------------------------|
| **Heuristic** | `state.total_box_distance()` (box → nearest target only) | Expansion and simulation **both** use `h_sokoban_box_only()` = `total_box_distance()` so reward and expansion order match the notebook. |
| **Visited** | `visited[key] = g` per search; prune when `visited[key] <= g_new` | Shared `astar_globals._visited`; reset at start of each search (each move) in expansion. Same rule: prune when `visited[key] <= g_new`. |
| **Deadlock** | `state._is_deadlocked()` (corner only) before enqueue | Expansion: same `_is_deadlocked()`. Simulation (e.g. simulation_heuristic): can use stricter checks. |
| **state_key** | `P{player}B{sorted(boxes)}` | Same in SokobanState. |
| **Tie-break** | Heap order when f equal | Expansion: sort by `(h, action)` so ties are deterministic. |

## Bug fixed: visited not reset

- **Problem:** `astar_globals._visited` is shared across the whole game. Each MCTS move starts a **new** tree (new root), but we never cleared `_visited`. So we pruned states in the **current** tree using g-scores from **previous** moves, which is incorrect for A*-style expansion.
- **Fix:** In `MCTS_tools/expansion/expansion.py`, at the start of `default_expansion(node)` we walk up to the root; if the root’s `state_key()` is different from `_ag._root_key`, we call `_ag.reset(root_key)`. So every new move gets a fresh visited set for that tree.

## Heuristic alignment (critical for level 2)

- **Notebook:** `manhattan_heuristic(state) = state.total_box_distance()` (box-only).
- **Expansion:** Scores untried actions by `h_sokoban_box_only(next_state)`; picks action with minimum h; ties broken by `(h, action)`.
- **Simulation:** Must use the same heuristic for the reward. **Fix:** `simulation.py` now uses `h_sokoban_box_only(state)` so reward = `1/(1 + total_box_distance())`. Previously it used `h_sokoban` (box+player), so the best action by expansion could get a lower reward than another action, causing the wrong move to be chosen (e.g. level 2 failing).

## Deadlock

- **Game / expansion:** `SokobanState._is_deadlocked()` detects only **corner** deadlocks (box in non-target corner).
- **simulation_heuristic:** Uses a stricter `_simple_deadlock()` (corner + wall-line). So simulation can terminate as deadlock states that expansion still expands. That’s acceptable; expansion stays aligned with the game and the notebook.

## How to re-run

After the fix, run the heuristic eval again (e.g. `python scripts/eval_heuristics.py`). Level 3 and 5 should improve because:

1. Visited is reset per move, so we no longer prune good states using old trees.
2. Expansion uses the same box-only heuristic as the notebook A*.

If you want simulation to use the same deadlock rule as the game, you could call `state._is_deadlocked()` inside the simulation instead of (or in addition to) the custom `_simple_deadlock()`.
