# Sokoban Macro-Push State Adapter — Design Spec

## Problem
The current Sokoban MCTS implementation uses 4-directional single-step actions (UP/DOWN/LEFT/RIGHT). Most steps are "walking" moves that don't change box positions — only pushes matter for solving the puzzle. This bloats the search tree with strategically irrelevant moves.

## Solution
Introduce `SokobanMacroState`, an adapter that wraps `SokobanState` and exposes a macro-push action space. Each action represents a complete box push reachable from the player's current connected region. Walking is handled internally via BFS.

## Architecture

### Adapter Pattern
`SokobanMacroState` implements `GameState`, wrapping an inner `SokobanState`:
- All mutable state lives in the inner state
- The adapter adds BFS reachability computation and macro-push action enumeration
- Registered as game `"sokoban_macro"` alongside existing `"sokoban"`

### State
Same as original: `(player_pos, boxes, walls)`. Reachable region equivalence is used only for action generation, computed on-the-fly via BFS (no caching).

### Actions
Tuples `(player_pos, direction)` where:
- `player_pos`: cell the player must occupy to execute the push (must be in reachable region)
- `direction`: push direction (UP=0, DOWN=1, LEFT=2, RIGHT=3)
- Box at `player_pos + delta(direction)`, destination at `player_pos + 2*delta(direction)`
- After push: player at box's original position

Variable number of actions per state.

### Step Counting
Full path length: `len(bfs_walk_path) + 1` per macro action. Preserves `max_steps` budget semantics.

### Logging
Full individual move sequence (original 4-direction format) logged per macro action. Trace format identical to original for visualization compatibility.

## Method Specs

### `legal_actions()`
1. BFS flood-fill from player through non-wall, non-box cells → reachable set
2. For each reachable cell, for each direction: if adjacent cell in that direction has a box AND the cell beyond is free → `(cell, direction)` is a valid action
3. Return all valid tuples

### `apply_action((player_pos, direction))`
1. Assert `player_pos` in reachable region
2. BFS shortest path player → player_pos → walk move sequence
3. Apply walk moves sequentially to inner state
4. Apply push move to inner state
5. Return full move sequence for logging

### Delegated methods
`is_terminal()`, `returns()`, `__str__()`, `state_key()`, `num_players()` → inner state

### `clone()`
Clone inner state, wrap in new `SokobanMacroState`.

## Heuristic Tool Updates
- Selection: `g = state.steps` (not tree depth)
- Simulation heuristic: push bonus irrelevant (all actions are pushes)
- Expansion: handle tuple actions
- A* globals: use `state.steps` for g-cost

## LLM Pipeline Updates
- New `game_infos/sokoban_macro.txt` describing macro-push mechanics
- Trace format: tuple-based action keys
- LLM-generated tools work with tuple actions

## Files Changed
| File | Change |
|------|--------|
| `mcts/games/sokoban_macro.py` | New — adapter + factory |
| `LLM/game_infos/sokoban_macro.txt` | New — game description |
| `MCTS_tools/selection/selection.py` | g = state.steps |
| `MCTS_tools/simulation/simulation_heuristic.py` | Remove push bonus |
| `MCTS_tools/expansion/expansion.py` | Tuple actions |
| `MCTS_tools/shared/astar_globals.py` | state.steps for g |
| `MCTS_tools/hyperparams/default_hyperparams.py` | sokoban_macro config |
| `MCTS_tools/training_logic/sokoban_training.py` | sokoban_macro support |
| `scripts/run_sokoban.py` | sokoban_macro option |
| Tests | New — macro state tests |
