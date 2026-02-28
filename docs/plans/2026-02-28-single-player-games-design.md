# Design: Single-Player Games Integration

**Date:** 2026-02-28
**Project:** CSE 291A - AI Agents, UCSD Winter 2026

## Overview

Add 4 single-player games (`pathfinding`, `morpion_solitaire`, `2048`, `zork`) as a transfer-validation and evaluation-clarity tier. Single-player games eliminate the opponent-strength confound in win-rate evaluation, giving cleaner performance signals. They also serve as **transfer probes**: each game is chosen to directly exercise a specific tool type developed on the existing 2-player games.

**Game ladder (full):**
```
Connect Four → Quoridor → [pathfinding, morpion_solitaire, 2048] → Chess → Zork
```

---

## Section 1: Core Abstraction — `GameMeta` + `GameAdapter` Update

### `GameMeta` dataclass

Add to `src/games/adapter.py`:

```python
@dataclass
class GameMeta:
    name: str
    is_single_player: bool
    min_return: float       # raw return floor
    max_return: float       # raw return ceiling
    metric_name: str        # "win_rate" | "avg_score" | "success_rate"
    max_sim_depth: int      # rollout depth cap
```

### `GameAdapter.normalize_return(raw: float) -> float`

Maps any raw game return to `[-1, 1]`:

```python
def normalize_return(self, raw: float) -> float:
    span = self.meta.max_return - self.meta.min_return
    return 2.0 * (raw - self.meta.min_return) / span - 1.0
```

Called everywhere `returns()` is consumed: `_simulate()` in the engine, `measure()` in the evaluator.

### `src/games/meta_registry.py`

```python
GAME_META = {
    "connect_four":      GameMeta("connect_four",      False, -1.0,    1.0,   "win_rate",      42),
    "quoridor":          GameMeta("quoridor",           False, -1.0,    1.0,   "win_rate",      200),
    "chess":             GameMeta("chess",              False, -1.0,    1.0,   "win_rate",      200),
    "pathfinding":       GameMeta("pathfinding",        True,   0.0,    1.0,   "success_rate",  500),
    "morpion_solitaire": GameMeta("morpion_solitaire",  True,   0.0,   35.0,  "avg_score",      35),
    "2048":              GameMeta("2048",                True,   0.0, 20000.0, "avg_score",    1000),
    "zork":              GameMeta("zork",               True,   0.0,  350.0,  "avg_score",     500),
}
```

`metadata.json` stores `performance_delta` in normalized `[-1, 1]` units for all games, making cross-game tool impact directly comparable.

---

## Section 2: The 4 Single-Player Games

### Game 1: `pathfinding`

- **OpenSpiel name:** `pathfinding`
- **Actions:** 5 (up/down/left/right/stay), max 1000 steps
- **Signal:** `returns()[0]` = 1.0 on goal reached, 0.0 otherwise → normalized `{-1, +1}`
- **Transfer probe:** Path-evaluation tools from Quoridor (BFS distance, obstacle avoidance, directional action filter)
- **Expected LLM tools:** distance-to-goal evaluator, backtrack-pruning filter, greedy-path rollout
- **Baseline:** random agent (rarely succeeds), vanilla MCTS

### Game 2: `morpion_solitaire`

- **OpenSpiel name:** `morpion_solitaire`
- **Actions:** 460 (grid placements), max 35 steps
- **Signal:** `returns()[0]` = pieces placed (0–35) → normalize via `(raw/35)*2 - 1`
- **Transfer probe:** `action_filter` tools from Quoridor (large action space pruning) + line-extension tools from Connect Four
- **Expected LLM tools:** line-completion filter, density evaluator, greedy-line rollout
- **Baseline:** random agent (~12–15 pieces avg), vanilla MCTS

### Game 3: `2048`

- **OpenSpiel name:** `2048`
- **Actions:** 4 (slide directions), max 8192 steps
- **Signal:** `returns()[0]` = cumulative score (0–20000+) → soft cap normalize at 20000
- **Transfer probe:** Spatial evaluator tools from Connect Four (positional bias, corner preference)
- **Expected LLM tools:** empty-cell count evaluator, monotonicity evaluator, merge-opportunity rollout
- **Baseline:** random agent (~400 avg score), vanilla MCTS

### Game 4: `zork` (Pinnacle)

- **Integration:** Custom `ZorkAdapter` wrapping `frotz` (Z-machine interpreter) via subprocess
- **Actions:** Enumerated command vocabulary (~50–100 commands), indexed as integers, mapped to strings for frotz input
- **Signal:** `returns()[0]` = Zork score (0–350) → normalize via `GameMeta(min=0, max=350)`
- **Transfer probe:** All prior tools, especially path evaluation and action filtering (text-based)
- **Expected LLM tools:** room-exit filter, item-pickup evaluator, objective-proximity evaluator
- **Baseline:** random command agent (score ≈ 0–5), vanilla MCTS

**`ZorkAdapter` key design:**
- `clone_state()` snapshots frotz state via `save`/`restore` commands
- Each `apply_action` forks from a saved snapshot — prevents state corruption across MCTS simulations
- `legal_actions()` returns fixed vocabulary filtered by fast text heuristic: directions present in room text always included; object commands only if object name appears in room text
- `str(state)` returns current room description + inventory + score line (tool interface for text-parsing tools)

**tool_pool structure (final):**
```
tool_pool/
├── global/
├── connect_four/
├── quoridor/
├── chess/
├── pathfinding/
├── morpion_solitaire/
├── 2048/
├── zork/
└── metadata.json
```

---

## Section 3: Evaluation Framework Updates

### Unified `PerformanceResult`

```python
@dataclass
class PerformanceResult:
    game: str
    metric_name: str        # "win_rate" | "avg_score" | "success_rate"
    raw_value: float        # e.g. 14.3 avg pieces, 0.62 win rate
    normalized_value: float # always [-1, 1], comparable across games
    n_games: int
```

`Evaluator.measure(game_name, agent, n_games) -> PerformanceResult` replaces the win-rate-only interface. Single-player games average `normalize_return()` across episodes. Two-player games keep existing win-rate logic.

### Evaluation Table (per game)

Same 4-column structure for all games; metric label varies per game:

| Agent | 100 sims | 500 sims | 1000 sims | 5000 sims |
|---|---|---|---|---|
| Random | — | — | — | — |
| Vanilla MCTS | — | — | — | — |
| MCTS + transferred tools | — | — | — | — |
| MCTS + evolved tools | — | — | — | — |

### Transfer Evaluation Protocol

New evaluation axis measuring cross-game tool transfer speed:

```
For each transfer chain:
  1. cold_start:   normalized score with vanilla MCTS on target game
  2. transferred:  normalized score with tools from source game (no target training)
  3. fully_trained: normalized score after full tool evolution on target game
  4. speed_to_threshold: episodes to reach 80% of fully_trained score
```

**Transfer chains:**

| Source | Target | Tool type probed |
|---|---|---|
| `quoridor` | `pathfinding` | path-evaluation, action filter |
| `connect_four` | `morpion_solitaire` | line detection, action filter |
| `connect_four` | `2048` | spatial evaluator |
| `[all games]` | `zork` | all transferable tools |

---

## Implementation Phases

### Phase A: `GameMeta` + OpenSpiel single-player games
1. Add `GameMeta` dataclass and `meta_registry.py`
2. Update `GameAdapter` with `normalize_return()` and `meta` field
3. Update engine `_simulate()` to use `normalize_return()`
4. Add `pathfinding`, `morpion_solitaire`, `2048` to meta registry
5. Create `tool_pool/{pathfinding,morpion_solitaire,2048}/` directories
6. Write 1–2 hand-authored tools per game to seed the pool
7. Update `Evaluator` to use `PerformanceResult` with unified metric

### Phase B: Zork adapter
1. Install `frotz` and verify subprocess control
2. Implement `ZorkAdapter` with save/restore clone semantics
3. Build command vocabulary and text-heuristic `legal_actions()` filter
4. Add `zork` to meta registry and `tool_pool/zork/`
5. Write 1–2 seed tools (room-exit filter, item evaluator)
6. Integration test: vanilla MCTS on Zork reaches score > random

### Phase C: Transfer evaluation
1. Run transfer chains after tool evolution on source games
2. Measure `speed_to_threshold` for each chain
3. Produce transfer tables for paper
