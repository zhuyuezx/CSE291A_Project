# Sokoban Macro-Push State Restructure Plan

## Status: Implementation Plan Ready
**Plan:** `docs/superpowers/plans/2026-03-10-sokoban-macro-push.md`

## Core Idea
Introduce a macro-push action space for Sokoban where actions are box pushes reachable from the player's connected region, rather than individual 4-directional steps.

## Design Decisions

### Architecture
1. **Adapter pattern** — `SokobanMacroState` wraps an inner `SokobanState`. All low-level grid logic (move validation, box pushing, win detection, deadlock detection) stays in `SokobanState`. The adapter adds BFS reachability and macro-push action enumeration on top.

2. **Implements `GameState` directly** — Drop-in replacement for MCTS engine. No engine changes needed.

3. **Registered as `"sokoban_macro"`** — Separate game name alongside existing `"sokoban"`. Training config selects which variant. Enables A/B comparison.

### State Representation
4. **State = (player_pos, boxes, walls)** — Unchanged from original. Player position is tracked exactly. Reachable region equivalence is used **only for action generation**, computed on-the-fly.

5. **`state_key`** — Unchanged: `f"P{player}B{sorted_boxes}"`.

### Action Representation
6. **Actions = `(player_pos, direction)` tuples** — `player_pos` is where the player must stand to execute the push. `direction` is the push direction. Box being pushed is at `player_pos + delta(direction)`. After push, player ends up at box's original position. The `player_pos` in the action serves as a built-in connectedness verification.

7. **Variable action count** — Number of legal actions varies per state depending on box/player configuration.

### Step Counting & Logging
8. **Steps = full path length** — Each macro action costs `len(bfs_path) + 1` steps (walk to push position + the push itself). Preserves original step budget / `max_steps` semantics.

9. **Logging = full individual moves** — Each macro action logs the complete sequence of UP/DOWN/LEFT/RIGHT moves (BFS walk path + final push). Trace format is identical to original — visualization works without adaptation.

### Implementation Details
10. **BFS recomputed every call** — No caching. Grids are small (<100 cells), BFS is microseconds.

11. **Clone wraps inner state** — `clone()` calls `inner.clone()` and wraps in new `SokobanMacroState`. Adapter is stateless.

## Method Specifications

### `legal_actions()`
1. BFS from `inner.player` through cells not in `inner.walls` and not in `inner.boxes` → `reachable_set`
2. For each cell in `reachable_set`, check all 4 directions:
   - If `cell + delta` has a box AND `cell + 2*delta` is free (not wall, not box) → valid push
   - Add `(cell, direction)` to action list
3. Return list of all valid `(player_pos, direction)` tuples

### `apply_action((player_pos, direction))`
1. Assert `player_pos` is in current reachable region
2. BFS shortest path from `inner.player` to `player_pos` → directional move sequence
3. Apply each walk move to inner state sequentially (steps += 1 per move)
4. Apply final push move to inner state (steps += 1, box moves, player at box's old position)
5. Store/return full move sequence for logging

### `is_terminal()`, `returns()`, `__str__()`, `state_key()`
Delegate directly to inner state.

### `clone()`
Clone inner `SokobanState`, wrap in new `SokobanMacroState`.

## Impact on Pluggable Heuristic Tools

### Selection (`selection.py`)
- Replace `node_depth()` with `state.steps` for A* `g` cost — preserves correctness with variable-cost macro-pushes
- `h_sokoban()` still valid (box distance + player distance)

### Simulation heuristic (`simulation_heuristic.py`)
- Push bonus (+1.0) becomes constant (every action is a push) — remove or ignore
- Distance delta and deadlock penalty remain primary differentiators

### Expansion (`expansion.py`)
- Largely unchanged — evaluates resulting state, not action format
- Action format changes from int to tuple

### Backpropagation
- Unchanged — only cares about visits and rewards

### Shared A* globals (`astar_globals.py`)
- `node_depth()` → `state.steps` for `g` cost
- `h_sokoban()`, `h_sokoban_box_only()` work as-is

## Impact on LLM Pipeline

### New game description
- Create `game_infos/sokoban_macro.txt` explaining macro-push mechanics
- Teaches LLM: reachable region concept, tuple action format, variable action count, step counting

### Trace format
- `action_chosen`: tuple string instead of int
- `legal_actions`: list of macro-push tuples
- `children_stats`: tuple-keyed
- LLM reasons about "which box to push from where" — higher-level strategic thinking

### LLM-generated tools
- Must handle tuple actions
- Can reason about push ordering, box clustering, corridor analysis — concepts that map naturally to macro-pushes

## File Plan
| File | Action | Purpose |
|------|--------|---------|
| `mcts/games/sokoban_macro.py` | Create | `SokobanMacroState` adapter + `SokobanMacro` factory |
| `mcts/games/sokoban.py` | Unchanged | Original stays intact |
| `LLM/game_infos/sokoban_macro.txt` | Create | Macro-push game description for LLM |
| `MCTS_tools/selection/selection.py` | Update | `g = state.steps` instead of `node_depth()` |
| `MCTS_tools/simulation/simulation_heuristic.py` | Update | Remove push bonus (constant for macro) |
| `MCTS_tools/expansion/expansion.py` | Update | Handle tuple actions |
| `MCTS_tools/shared/astar_globals.py` | Update | `node_depth()` → `state.steps` |
| `MCTS_tools/hyperparams/default_hyperparams.py` | Update | Add `sokoban_macro` game config |
| `MCTS_tools/training_logic/sokoban_training.py` | Update | Support `sokoban_macro` game name |
| `scripts/run_sokoban.py` | Update | Support `sokoban_macro` option |
| Tests | Create | Tests for `SokobanMacroState` BFS, action enum, apply, clone |
