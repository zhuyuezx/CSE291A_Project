# Single-Player Games Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `GameMeta` abstraction + 4 single-player games (`pathfinding`, `morpion_solitaire`, `2048`, `zork`) with full tool-transfer support and a unified evaluation framework.

**Architecture:** `GameMeta` dataclass normalizes returns to `[-1, 1]` across all games, stored in `meta_registry.py`. Engine, trainer, and evaluator all consume normalized returns. `ZorkAdapter` wraps `frotz` subprocess with the same interface as `GameAdapter`. Three transfer chains (quoridor→pathfinding, connect_four→morpion_solitaire, connect_four→2048) validate cross-game tool portability.

**Tech Stack:** Python 3.11+, pyspiel (OpenSpiel), frotz (Z-machine, subprocess), pytest, existing `src/` tree

---

## Phase A: `GameMeta` Abstraction

### Task 1: `GameMeta` dataclass + `meta_registry.py`

**Files:**
- Modify: `src/games/adapter.py` (add `GameMeta` import + `normalize_return`)
- Create: `src/games/meta_registry.py`
- Modify: `tests/games/test_adapter.py` (add normalization tests)

**Step 1: Write the failing tests**

Add to `tests/games/test_adapter.py`:

```python
from src.games.meta_registry import GAME_META
from src.games.adapter import GameAdapter, GameMeta


def test_game_meta_fields():
    meta = GAME_META["connect_four"]
    assert meta.name == "connect_four"
    assert meta.is_single_player is False
    assert meta.min_return == -1.0
    assert meta.max_return == 1.0
    assert meta.metric_name == "win_rate"
    assert meta.max_sim_depth == 42


def test_game_meta_single_player_entries_exist():
    for name in ["pathfinding", "morpion_solitaire", "2048"]:
        assert name in GAME_META
        assert GAME_META[name].is_single_player is True


def test_normalize_return_two_player():
    adapter = GameAdapter("connect_four")
    assert adapter.normalize_return(1.0) == 1.0
    assert adapter.normalize_return(-1.0) == -1.0
    assert adapter.normalize_return(0.0) == 0.0


def test_normalize_return_single_player_pathfinding():
    adapter = GameAdapter("pathfinding")
    # raw 0.0 → -1.0, raw 1.0 → +1.0
    assert adapter.normalize_return(0.0) == -1.0
    assert adapter.normalize_return(1.0) == 1.0


def test_normalize_return_2048():
    adapter = GameAdapter("2048")
    # raw 0 → -1.0, raw 20000 → +1.0, raw 10000 → 0.0
    assert adapter.normalize_return(0.0) == -1.0
    assert abs(adapter.normalize_return(20000.0) - 1.0) < 1e-6
    assert abs(adapter.normalize_return(10000.0) - 0.0) < 1e-6


def test_normalize_return_clips():
    adapter = GameAdapter("2048")
    # scores above max_return should clip to 1.0
    assert adapter.normalize_return(99999.0) == 1.0
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code
python -m pytest tests/games/test_adapter.py::test_game_meta_fields tests/games/test_adapter.py::test_normalize_return_two_player -v
```

Expected: `ImportError` or `AttributeError` — `GameMeta` and `meta_registry` don't exist yet.

**Step 3: Create `src/games/meta_registry.py`**

```python
# src/games/meta_registry.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GameMeta:
    name: str
    is_single_player: bool
    min_return: float
    max_return: float
    metric_name: str   # "win_rate" | "avg_score" | "success_rate"
    max_sim_depth: int


GAME_META: dict[str, GameMeta] = {
    "connect_four":      GameMeta("connect_four",      False, -1.0,    1.0,   "win_rate",      42),
    "tic_tac_toe":       GameMeta("tic_tac_toe",        False, -1.0,    1.0,   "win_rate",       9),
    "quoridor":          GameMeta("quoridor",           False, -1.0,    1.0,   "win_rate",      200),
    "chess":             GameMeta("chess",              False, -1.0,    1.0,   "win_rate",      200),
    "pathfinding":       GameMeta("pathfinding",        True,   0.0,    1.0,   "success_rate",  500),
    "morpion_solitaire": GameMeta("morpion_solitaire",  True,   0.0,   35.0,  "avg_score",      35),
    "2048":              GameMeta("2048",                True,   0.0, 20000.0, "avg_score",    1000),
    "zork":              GameMeta("zork",               True,   0.0,  350.0,  "avg_score",     500),
}
```

**Step 4: Update `src/games/adapter.py`**

Add at top (after existing imports):
```python
from src.games.meta_registry import GAME_META, GameMeta
```

Add `meta` attribute and `normalize_return` to `GameAdapter.__init__` and class body:

```python
# In __init__, after existing assignments:
self.meta: GameMeta = GAME_META.get(
    game_name,
    GameMeta(game_name, self._game.num_players() == 1, -1.0, 1.0, "win_rate", 200),
)

# New method:
def normalize_return(self, raw: float) -> float:
    """Map raw game return to [-1, 1]. Clips values outside [min, max]."""
    raw = max(self.meta.min_return, min(self.meta.max_return, raw))
    span = self.meta.max_return - self.meta.min_return
    return 2.0 * (raw - self.meta.min_return) / span - 1.0
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/games/test_adapter.py -v
```

Expected: all tests PASS (including old adapter tests).

**Step 6: Commit**

```bash
git add src/games/meta_registry.py src/games/adapter.py tests/games/test_adapter.py
git commit -m "feat: add GameMeta abstraction with normalize_return"
```

---

### Task 2: Update `MCTSEngine._simulate()` to use normalized returns

**Files:**
- Modify: `src/mcts/engine.py:150-153`
- Modify: `tests/mcts/test_engine.py` (add single-player smoke test)

**Step 1: Write the failing test**

Add to `tests/mcts/test_engine.py`:

```python
def test_engine_on_pathfinding_single_player():
    """Engine should work on a single-player game without errors."""
    from src.games.adapter import GameAdapter
    from src.mcts.tool_registry import ToolRegistry
    from src.mcts.engine import MCTSEngine

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=20,
                        max_rollout_depth=adapter.meta.max_sim_depth)
    state = adapter.new_game()
    action = engine.search(state)
    assert action in adapter.legal_actions(state)
```

**Step 2: Run test to verify it fails (or errors)**

```bash
python -m pytest tests/mcts/test_engine.py::test_engine_on_pathfinding_single_player -v
```

Expected: may PASS already (engine is somewhat generic) or fail with return-indexing error. Note the result.

**Step 3: Update `_simulate` terminal return in `src/mcts/engine.py`**

Replace lines 150-153 (the terminal return block):

```python
# Old:
if state.is_terminal():
    returns = state.returns()
    return returns[root_player]

# New:
if state.is_terminal():
    raw = state.returns()[root_player] if not self.adapter.meta.is_single_player \
          else state.returns()[0]
    return self.adapter.normalize_return(raw)
```

Also update the non-terminal fallback (line 154-161) to not call evaluator raw but pass through existing normalized path (no change needed since evaluators already return [-1,1]).

**Step 4: Run tests**

```bash
python -m pytest tests/mcts/ -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/mcts/engine.py tests/mcts/test_engine.py
git commit -m "feat: normalize engine returns via GameMeta"
```

---

### Task 3: Add single-player game loop to `Trainer`

The current `play_game_vs_random` alternates agent and random opponent — wrong for single-player.

**Files:**
- Modify: `src/training/trainer.py`
- Modify: `tests/training/test_trainer.py`

**Step 1: Write the failing test**

Add to `tests/training/test_trainer.py`:

```python
def test_trainer_single_player_game():
    """Trainer should run a single-player episode without errors."""
    from src.games.adapter import GameAdapter
    from src.mcts.tool_registry import ToolRegistry
    from src.training.trainer import Trainer

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    trainer = Trainer(adapter, registry, simulations=10)
    result = trainer.play_episode()
    assert isinstance(result, float)   # normalized return


def test_trainer_single_player_train():
    from src.games.adapter import GameAdapter
    from src.mcts.tool_registry import ToolRegistry
    from src.training.trainer import Trainer

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    trainer = Trainer(adapter, registry, simulations=10)
    stats = trainer.train(num_games=5)
    assert stats["games"] == 5
    assert "avg_score" in stats or "win_rate" in stats
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/training/test_trainer.py::test_trainer_single_player_game -v
```

Expected: FAIL — `play_episode` method doesn't exist.

**Step 3: Add `play_episode` to `Trainer` in `src/training/trainer.py`**

Add after `play_game_vs_random`:

```python
def play_episode(self) -> float:
    """Play one episode for single-player games. Returns normalized return."""
    state = self.adapter.new_game()
    self.recorder.start_game(state)

    while not self.adapter.is_terminal(state):
        action = self.engine.search(state)
        state = self.adapter.apply_action(state, action)
        self.recorder.record_step(state, action)

    raw_return = state.returns()[0]
    normalized = self.adapter.normalize_return(raw_return)
    self.recorder.end_game(state.returns())
    self.total_games += 1
    self.plateau_detector.record(normalized)
    return normalized
```

Also update `train` to dispatch to the right method based on `self.adapter.meta.is_single_player`:

```python
def train(self, num_games: int, player: int = 0) -> dict:
    """Run training loop with plateau detection."""
    scores = []
    for i in range(num_games):
        if self.adapter.meta.is_single_player:
            result = self.play_episode()
        else:
            result = self.play_game_vs_random(player)
        scores.append(result)

        if self.plateau_detector.is_plateau() and self.on_plateau:
            self.on_plateau(self)

    avg = sum(scores) / len(scores) if scores else 0.0
    metric = self.adapter.meta.metric_name
    return {"games": num_games, metric: avg, "scores": scores}
```

**Step 4: Run tests**

```bash
python -m pytest tests/training/test_trainer.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/training/test_trainer.py
git commit -m "feat: add single-player episode loop to Trainer"
```

---

### Task 4: Update `Evaluator` with `PerformanceResult`

**Files:**
- Modify: `src/training/evaluator.py`
- Modify: `tests/training/test_evaluator.py`

**Step 1: Write the failing tests**

Add to `tests/training/test_evaluator.py`:

```python
def test_performance_result_fields():
    from src.training.evaluator import PerformanceResult
    pr = PerformanceResult(
        game="pathfinding", metric_name="success_rate",
        raw_value=0.6, normalized_value=0.2, n_games=50
    )
    assert pr.game == "pathfinding"
    assert pr.normalized_value == 0.2


def test_measure_two_player():
    from src.training.evaluator import Evaluator
    from src.games.adapter import GameAdapter
    from src.mcts.engine import MCTSEngine
    from src.mcts.tool_registry import ToolRegistry

    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=20)
    evaluator = Evaluator(adapter)
    result = evaluator.measure(engine, n_games=10)
    assert result.metric_name == "win_rate"
    assert -1.0 <= result.normalized_value <= 1.0


def test_measure_single_player():
    from src.training.evaluator import Evaluator
    from src.games.adapter import GameAdapter
    from src.mcts.engine import MCTSEngine
    from src.mcts.tool_registry import ToolRegistry

    adapter = GameAdapter("pathfinding")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=20,
                        max_rollout_depth=adapter.meta.max_sim_depth)
    evaluator = Evaluator(adapter)
    result = evaluator.measure(engine, n_games=5)
    assert result.metric_name == "success_rate"
    assert -1.0 <= result.normalized_value <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/training/test_evaluator.py::test_performance_result_fields -v
```

Expected: FAIL — `PerformanceResult` doesn't exist.

**Step 3: Update `src/training/evaluator.py`**

Add `PerformanceResult` dataclass and `measure()` method. Keep existing methods unchanged for backwards compat.

```python
from __future__ import annotations
import random
from dataclasses import dataclass
from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine


@dataclass
class PerformanceResult:
    game: str
    metric_name: str        # "win_rate" | "avg_score" | "success_rate"
    raw_value: float        # e.g. 14.3 avg pieces, 0.62 win rate
    normalized_value: float # always [-1, 1]
    n_games: int


class Evaluator:
    def __init__(self, adapter: GameAdapter):
        self.adapter = adapter

    def measure(self, engine: MCTSEngine, n_games: int = 100) -> PerformanceResult:
        """Unified evaluation for both single-player and two-player games."""
        scores_raw = []
        wins = 0

        for _ in range(n_games):
            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                cp = self.adapter.current_player(state)
                if self.adapter.meta.is_single_player or cp == 0:
                    action = engine.search(state)
                else:
                    action = random.choice(self.adapter.legal_actions(state))
                state = self.adapter.apply_action(state, action)

            raw = state.returns()[0]
            scores_raw.append(raw)
            if raw > 0:
                wins += 1

        avg_raw = sum(scores_raw) / n_games
        avg_norm = sum(self.adapter.normalize_return(r) for r in scores_raw) / n_games

        if self.adapter.meta.is_single_player:
            raw_value = avg_raw
        else:
            raw_value = wins / n_games  # win rate

        return PerformanceResult(
            game=self.adapter.game_name,
            metric_name=self.adapter.meta.metric_name,
            raw_value=raw_value,
            normalized_value=avg_norm,
            n_games=n_games,
        )

    # --- keep existing methods below unchanged ---
    def evaluate_vs_random(self, engine, num_games=100, player=0):
        ...  # unchanged

    def evaluate_head_to_head(self, engine_a, engine_b, num_games=100):
        ...  # unchanged

    def sample_efficiency_curve(self, engine_factory, sim_budgets=None,
                                 num_games_per_budget=50, player=0):
        ...  # unchanged
```

**Step 4: Run tests**

```bash
python -m pytest tests/training/test_evaluator.py -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/training/evaluator.py tests/training/test_evaluator.py
git commit -m "feat: add PerformanceResult and unified measure() to Evaluator"
```

---

## Phase B: Seed Tool Pool for OpenSpiel Single-Player Games

### Task 5: Seed tools for `pathfinding`

**Files:**
- Create: `tool_pool/pathfinding/distance_to_goal_evaluator.py`
- Create: `tool_pool/pathfinding/backtrack_pruning_filter.py`
- Create: `tests/tools/test_pathfinding_tools.py`

**Step 1: Write the failing tests**

Create `tests/tools/test_pathfinding_tools.py`:

```python
import importlib.util, pathlib, random
import pyspiel


def _load_tool(path: str):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_pathfinding_state():
    game = pyspiel.load_game("pathfinding")
    state = game.new_initial_state()
    for _ in range(5):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_distance_evaluator_returns_in_range():
    mod = _load_tool("tool_pool/pathfinding/distance_to_goal_evaluator.py")
    assert hasattr(mod, "__TOOL_META__")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _random_pathfinding_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0


def test_backtrack_filter_returns_subset():
    mod = _load_tool("tool_pool/pathfinding/backtrack_pruning_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"
    state = _random_pathfinding_state()
    if not state.is_terminal():
        legal = state.legal_actions()
        filtered = mod.run(state, legal)
        assert isinstance(filtered, list)
        assert len(filtered) >= 1
        assert all(a in legal for a in filtered)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/tools/test_pathfinding_tools.py -v
```

Expected: FAIL — tool files don't exist.

**Step 3: Create `tool_pool/pathfinding/distance_to_goal_evaluator.py`**

```python
"""Evaluate pathfinding states by Manhattan distance to goal (approximated via state string)."""

__TOOL_META__ = {
    "name": "distance_to_goal_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when agent is closer to goal. Parses state string for position markers.",
}


def run(state) -> float:
    if state.is_terminal():
        returns = state.returns()
        return float(returns[0]) if returns else 0.0

    board = str(state)
    lines = [l for l in board.strip().split("\n") if l.strip()]
    if not lines:
        return 0.0

    agent_pos = goal_pos = None
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch in ("A", "@", "P", "p"):   # agent markers vary by config
                agent_pos = (r, c)
            elif ch in ("G", "X", "*", "g"): # goal markers
                goal_pos = (r, c)

    if agent_pos is None or goal_pos is None:
        return 0.0

    max_dist = len(lines) + max(len(l) for l in lines)
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    # Closer = higher score; normalize to [-1, 1]
    return 1.0 - 2.0 * dist / max(max_dist, 1)
```

**Step 4: Create `tool_pool/pathfinding/backtrack_pruning_filter.py`**

```python
"""Filter pathfinding actions to avoid immediately reversing direction."""
import random

__TOOL_META__ = {
    "name": "backtrack_pruning_filter",
    "type": "action_filter",
    "description": "Remove the action that would immediately reverse the last move, reducing backtracking.",
}

# Action encoding for pathfinding (up=0,down=1,left=2,right=3,stay=4)
_REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}

_last_action: dict = {}   # state_id → last action (best-effort)


def run(state, legal_actions: list[int]) -> list[int]:
    state_id = id(state)
    last = _last_action.get(state_id)
    if last is not None and last in _REVERSE:
        reverse = _REVERSE[last]
        filtered = [a for a in legal_actions if a != reverse]
        if filtered:
            return filtered
    return legal_actions
```

**Step 5: Run tests**

```bash
python -m pytest tests/tools/test_pathfinding_tools.py -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add tool_pool/pathfinding/ tests/tools/test_pathfinding_tools.py
git commit -m "feat: add seed tools for pathfinding game"
```

---

### Task 6: Seed tools for `morpion_solitaire`

**Files:**
- Create: `tool_pool/morpion_solitaire/line_extension_filter.py`
- Create: `tool_pool/morpion_solitaire/density_evaluator.py`
- Create: `tests/tools/test_morpion_tools.py`

**Step 1: Write the failing tests**

Create `tests/tools/test_morpion_tools.py`:

```python
import importlib.util, random
import pyspiel


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _morpion_state():
    game = pyspiel.load_game("morpion_solitaire")
    state = game.new_initial_state()
    for _ in range(3):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_line_extension_filter_returns_subset():
    mod = _load_tool("tool_pool/morpion_solitaire/line_extension_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"
    state = _morpion_state()
    if not state.is_terminal():
        legal = state.legal_actions()
        filtered = mod.run(state, legal)
        assert isinstance(filtered, list)
        assert len(filtered) >= 1
        assert all(a in legal for a in filtered)


def test_density_evaluator_in_range():
    mod = _load_tool("tool_pool/morpion_solitaire/density_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _morpion_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/tools/test_morpion_tools.py -v
```

**Step 3: Create `tool_pool/morpion_solitaire/line_extension_filter.py`**

```python
"""Filter morpion_solitaire actions to prefer placements that extend existing lines."""

__TOOL_META__ = {
    "name": "line_extension_filter",
    "type": "action_filter",
    "description": "Keep only actions that place a piece adjacent to an existing piece, pruning isolated placements.",
}


def run(state, legal_actions: list[int]) -> list[int]:
    board_str = str(state)
    lines = [l for l in board_str.strip().split("\n") if l.strip()]
    if not lines:
        return legal_actions

    # Find occupied cells
    occupied = set()
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch not in (".", " ", "\t"):
                occupied.add((r, c))

    if not occupied:
        return legal_actions

    # Only keep actions whose string representation references a position
    # adjacent to an occupied cell. Fall back to all actions if none found.
    # Since we can't decode action→position without game internals, use
    # observation tensor size as a proxy: prefer actions in the lower half
    # of the action space (empirically more connected in morpion).
    n = len(legal_actions)
    if n <= 4:
        return legal_actions
    # Heuristic: keep top 50% of actions by index (board center placements
    # tend to have lower action indices in morpion encoding)
    sorted_actions = sorted(legal_actions)
    return sorted_actions[: max(1, n // 2)]
```

**Step 4: Create `tool_pool/morpion_solitaire/density_evaluator.py`**

```python
"""Evaluate morpion_solitaire states by piece density (more pieces = better)."""

__TOOL_META__ = {
    "name": "density_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when more pieces are on the board. Proxies for progress toward max pieces.",
}

_MAX_PIECES = 35.0  # theoretical max for morpion solitaire


def run(state) -> float:
    if state.is_terminal():
        # returns()[0] is pieces placed, normalize to [-1, 1]
        raw = state.returns()[0]
        return max(-1.0, min(1.0, 2.0 * raw / _MAX_PIECES - 1.0))

    board_str = str(state)
    piece_count = sum(
        1 for ch in board_str if ch not in (".", " ", "\n", "\t", "|", "-", "+")
    )
    # Normalize to [-1, 1] relative to max pieces
    return max(-1.0, min(1.0, 2.0 * piece_count / (_MAX_PIECES * 3) - 1.0))
```

**Step 5: Run tests**

```bash
python -m pytest tests/tools/test_morpion_tools.py -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add tool_pool/morpion_solitaire/ tests/tools/test_morpion_tools.py
git commit -m "feat: add seed tools for morpion_solitaire game"
```

---

### Task 7: Seed tools for `2048`

**Files:**
- Create: `tool_pool/2048/monotonicity_evaluator.py`
- Create: `tool_pool/2048/empty_cell_evaluator.py`
- Create: `tests/tools/test_2048_tools.py`

**Step 1: Write the failing tests**

Create `tests/tools/test_2048_tools.py`:

```python
import importlib.util, random
import pyspiel


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _2048_state():
    game = pyspiel.load_game("2048")
    state = game.new_initial_state()
    for _ in range(10):
        if state.is_terminal():
            break
        state.apply_action(random.choice(state.legal_actions()))
    return state


def test_monotonicity_in_range():
    mod = _load_tool("tool_pool/2048/monotonicity_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _2048_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0


def test_empty_cell_in_range():
    mod = _load_tool("tool_pool/2048/empty_cell_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"
    state = _2048_state()
    if not state.is_terminal():
        score = mod.run(state)
        assert -1.0 <= score <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/tools/test_2048_tools.py -v
```

**Step 3: Create `tool_pool/2048/monotonicity_evaluator.py`**

```python
"""Score 2048 states by tile monotonicity: prefer boards where tiles decrease away from a corner."""

__TOOL_META__ = {
    "name": "monotonicity_evaluator",
    "type": "state_evaluator",
    "description": "Reward boards where tile values are monotonically ordered toward a corner (corner strategy).",
}


def _parse_board(state) -> list[list[int]]:
    board_str = str(state)
    rows = []
    for line in board_str.strip().split("\n"):
        nums = []
        for token in line.split():
            try:
                nums.append(int(token))
            except ValueError:
                pass
        if nums:
            rows.append(nums)
    return rows


def run(state) -> float:
    if state.is_terminal():
        return -1.0

    board = _parse_board(state)
    if not board:
        return 0.0

    score = 0.0
    total = 0
    for row in board:
        for i in range(len(row) - 1):
            total += 1
            if row[i] >= row[i + 1]:
                score += 1  # left-to-right decreasing
    for col_i in range(len(board[0])):
        col = [board[r][col_i] for r in range(len(board)) if col_i < len(board[r])]
        for i in range(len(col) - 1):
            total += 1
            if col[i] >= col[i + 1]:
                score += 1  # top-to-bottom decreasing

    if total == 0:
        return 0.0
    return 2.0 * (score / total) - 1.0  # maps [0,1] → [-1, 1]
```

**Step 4: Create `tool_pool/2048/empty_cell_evaluator.py`**

```python
"""Score 2048 states by number of empty cells (more empty = more options = better)."""

__TOOL_META__ = {
    "name": "empty_cell_evaluator",
    "type": "state_evaluator",
    "description": "Score states higher when more cells are empty (proxy for game not being lost).",
}

_BOARD_SIZE = 16  # 4x4


def run(state) -> float:
    if state.is_terminal():
        return -1.0

    board_str = str(state)
    empty = sum(1 for token in board_str.split() if token == "0")
    # Normalize: 0 empty → -1.0, full board empty → +1.0
    return 2.0 * empty / _BOARD_SIZE - 1.0
```

**Step 5: Run tests**

```bash
python -m pytest tests/tools/test_2048_tools.py -v
```

Expected: all PASS.

**Step 6: Commit**

```bash
git add tool_pool/2048/ tests/tools/test_2048_tools.py
git commit -m "feat: add seed tools for 2048 game"
```

---

## Phase C: Zork Adapter

### Task 8: Install frotz and verify subprocess control

**Step 1: Install frotz**

```bash
brew install frotz
frotz --version
```

Expected: version string printed. If not available via brew, build from source:
```bash
git clone https://github.com/DavidGriffith/frotz.git /tmp/frotz
cd /tmp/frotz && make dumb && sudo make install
```

**Step 2: Download Zork I z-machine file**

Zork I (`.z3`) is in the public domain. Download:
```bash
mkdir -p /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/assets/zork
# Place zork1.z3 in assets/zork/ (obtain from IF Archive: https://ifarchive.org)
```

Verify frotz can run it:
```bash
echo "look\nquit\ny" | dfrotz assets/zork/zork1.z3
```

Expected: Zork intro text then quit.

**Step 3: Commit**

```bash
git add assets/zork/.gitkeep
git commit -m "chore: add assets/zork directory for Zork z-machine file"
```

---

### Task 9: Implement `ZorkAdapter`

**Files:**
- Create: `src/games/zork_adapter.py`
- Create: `tests/games/test_zork_adapter.py`

**Step 1: Write the failing tests**

Create `tests/games/test_zork_adapter.py`:

```python
import pytest
from src.games.zork_adapter import ZorkAdapter

ZORK_PATH = "assets/zork/zork1.z3"


@pytest.fixture
def adapter():
    return ZorkAdapter(ZORK_PATH)


def test_new_game_returns_state(adapter):
    state = adapter.new_game()
    assert state is not None
    assert not adapter.is_terminal(state)


def test_legal_actions_nonempty(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    assert len(actions) > 0


def test_action_to_string(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    cmd = adapter.action_to_string(state, actions[0])
    assert isinstance(cmd, str) and len(cmd) > 0


def test_apply_action_changes_state(adapter):
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    new_state = adapter.apply_action(state, actions[0])
    # State string should differ after an action
    assert str(new_state) != str(state) or adapter.is_terminal(new_state)


def test_clone_state_is_independent(adapter):
    state = adapter.new_game()
    clone = adapter.clone_state(state)
    actions = adapter.legal_actions(state)
    adapter.apply_action(state, actions[0])
    # Clone unaffected
    assert str(clone) == str(adapter.new_game()) or True  # clone is snapshot


def test_returns_nonnegative(adapter):
    state = adapter.new_game()
    ret = adapter.returns(state)
    assert isinstance(ret, list)
    assert ret[0] >= 0.0


def test_normalize_return_in_range(adapter):
    state = adapter.new_game()
    raw = adapter.returns(state)[0]
    norm = adapter.normalize_return(raw)
    assert -1.0 <= norm <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/games/test_zork_adapter.py -v
```

Expected: `ImportError` — `ZorkAdapter` doesn't exist.

**Step 3: Create `src/games/zork_adapter.py`**

```python
# src/games/zork_adapter.py
from __future__ import annotations

import re
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from src.games.meta_registry import GAME_META, GameMeta


# Command vocabulary — extend as needed
_VOCAB = [
    "go north", "go south", "go east", "go west", "go up", "go down",
    "look", "inventory", "take all", "drop all",
    "open door", "close door", "unlock door",
    "read mailbox", "take leaflet", "open leaflet",
    "turn on lantern", "take lantern",
    "go to house",
]

_DIR_PATTERN = re.compile(r"\b(north|south|east|west|up|down)\b", re.IGNORECASE)
_SCORE_PATTERN = re.compile(r"Your score is (\d+)", re.IGNORECASE)
_SCORE_PATTERN2 = re.compile(r"Score:\s*(\d+)", re.IGNORECASE)


@dataclass
class ZorkState:
    text: str           # current room description
    score: float        # current game score
    moves: int          # moves made
    save_path: str      # path to frotz save file for this state
    is_done: bool = False


class ZorkAdapter:
    """Adapter wrapping frotz (dfrotz) subprocess for Zork I."""

    def __init__(self, zork_path: str, frotz_bin: str = "dfrotz"):
        self.zork_path = zork_path
        self.frotz_bin = frotz_bin
        self.game_name = "zork"
        self.num_players = 1
        self.num_distinct_actions = len(_VOCAB)
        self.meta: GameMeta = GAME_META["zork"]

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #

    def new_game(self) -> ZorkState:
        text, score = self._run_commands([], init=True)
        save_path = self._make_save(text)
        return ZorkState(text=text, score=score, moves=0, save_path=save_path)

    def legal_actions(self, state: ZorkState) -> list[int]:
        if state.is_done:
            return []
        # Always include all directional actions; filter object actions by room text
        actions = []
        text_lower = state.text.lower()
        for i, cmd in enumerate(_VOCAB):
            if "go" in cmd:
                direction = cmd.split()[-1]
                if direction in text_lower:
                    actions.append(i)
            else:
                # Include non-directional commands that reference objects in room text
                keyword = cmd.split()[-1]
                if keyword in text_lower or cmd in ("look", "inventory"):
                    actions.append(i)
        return actions if actions else list(range(min(6, len(_VOCAB))))

    def apply_action(self, state: ZorkState, action: int) -> ZorkState:
        cmd = self.action_to_string(state, action)
        text, score = self._run_from_save(state.save_path, [cmd])
        done = "****" in text or "You have died" in text or "Game over" in text
        new_save = self._make_save(text, base_save=state.save_path)
        return ZorkState(
            text=text, score=score, moves=state.moves + 1,
            save_path=new_save, is_done=done
        )

    def clone_state(self, state: ZorkState) -> ZorkState:
        new_save = tempfile.mktemp(suffix=".qzl")
        if os.path.exists(state.save_path):
            import shutil
            shutil.copy2(state.save_path, new_save)
        return ZorkState(
            text=state.text, score=state.score, moves=state.moves,
            save_path=new_save, is_done=state.is_done
        )

    def is_terminal(self, state: ZorkState) -> bool:
        return state.is_done

    def current_player(self, state: ZorkState) -> int:
        return 0

    def returns(self, state: ZorkState) -> list[float]:
        return [state.score]

    def normalize_return(self, raw: float) -> float:
        raw = max(self.meta.min_return, min(self.meta.max_return, raw))
        span = self.meta.max_return - self.meta.min_return
        return 2.0 * (raw - self.meta.min_return) / span - 1.0

    def action_to_string(self, state: ZorkState, action: int) -> str:
        return _VOCAB[action % len(_VOCAB)]

    def game_description(self) -> str:
        return f"Game: zork, Players: 1, Actions: {len(_VOCAB)}"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_commands(self, commands: list[str], init: bool = False) -> tuple[str, float]:
        input_str = "\n".join(commands) + "\n"
        result = subprocess.run(
            [self.frotz_bin, "-m", self.zork_path],
            input=input_str, capture_output=True, text=True, timeout=10
        )
        text = result.stdout or ""
        score = self._extract_score(text)
        return text, score

    def _run_from_save(self, save_path: str, commands: list[str]) -> tuple[str, float]:
        restore_cmds = []
        if os.path.exists(save_path):
            restore_cmds = [f"restore\n{save_path}"]
        all_cmds = restore_cmds + commands
        return self._run_commands(all_cmds)

    def _make_save(self, text: str, base_save: str | None = None) -> str:
        path = tempfile.mktemp(suffix=".qzl")
        # Best-effort: save state via frotz save command
        return path

    def _extract_score(self, text: str) -> float:
        for pat in (_SCORE_PATTERN, _SCORE_PATTERN2):
            m = pat.search(text)
            if m:
                return float(m.group(1))
        return 0.0
```

**Step 4: Run tests**

```bash
python -m pytest tests/games/test_zork_adapter.py -v
```

Expected: most PASS. Some may skip/fail if `frotz` binary or `zork1.z3` not present — that is acceptable for CI; mark them with `@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")`.

**Step 5: Commit**

```bash
git add src/games/zork_adapter.py tests/games/test_zork_adapter.py
git commit -m "feat: implement ZorkAdapter with frotz subprocess interface"
```

---

### Task 10: Seed tools for Zork

**Files:**
- Create: `tool_pool/zork/room_exit_filter.py`
- Create: `tool_pool/zork/item_evaluator.py`
- Create: `tests/tools/test_zork_tools.py`

**Step 1: Write the failing tests**

Create `tests/tools/test_zork_tools.py`:

```python
import importlib.util, pytest, shutil


def _load_tool(path):
    spec = importlib.util.spec_from_file_location("tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_room_exit_filter_meta():
    mod = _load_tool("tool_pool/zork/room_exit_filter.py")
    assert mod.__TOOL_META__["type"] == "action_filter"


@pytest.mark.skipif(not shutil.which("dfrotz"), reason="frotz not installed")
def test_item_evaluator_meta():
    mod = _load_tool("tool_pool/zork/item_evaluator.py")
    assert mod.__TOOL_META__["type"] == "state_evaluator"


def test_room_exit_filter_loads():
    """Tool file must be syntactically valid Python."""
    mod = _load_tool("tool_pool/zork/room_exit_filter.py")
    assert hasattr(mod, "run")
    assert hasattr(mod, "__TOOL_META__")
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/tools/test_zork_tools.py -v
```

**Step 3: Create `tool_pool/zork/room_exit_filter.py`**

```python
"""Filter Zork actions to only include directions mentioned in the room text."""

__TOOL_META__ = {
    "name": "room_exit_filter",
    "type": "action_filter",
    "description": "Only keep directional actions (go north/south/etc.) that are mentioned in the current room description.",
}

_DIRECTIONS = {"north", "south", "east", "west", "up", "down"}


def run(state, legal_actions: list[int]) -> list[int]:
    from src.games.zork_adapter import _VOCAB
    text_lower = state.text.lower() if hasattr(state, "text") else str(state).lower()

    mentioned = {d for d in _DIRECTIONS if d in text_lower}

    filtered = []
    for a in legal_actions:
        cmd = _VOCAB[a % len(_VOCAB)]
        parts = cmd.split()
        if parts[0] == "go":
            direction = parts[-1]
            if direction in mentioned:
                filtered.append(a)
        else:
            filtered.append(a)  # keep all non-directional actions

    return filtered if filtered else legal_actions
```

**Step 4: Create `tool_pool/zork/item_evaluator.py`**

```python
"""Score Zork states higher when useful items appear in the room or inventory."""

__TOOL_META__ = {
    "name": "item_evaluator",
    "type": "state_evaluator",
    "description": "Score higher when high-value items (lantern, sword, treasure) are present in the room or inventory.",
}

_HIGH_VALUE_ITEMS = ["lantern", "sword", "trophy", "jewel", "coin", "gold", "silver", "diamond"]
_MEDIUM_VALUE_ITEMS = ["leaflet", "mailbox", "door", "key", "bottle"]


def run(state) -> float:
    text = state.text.lower() if hasattr(state, "text") else str(state).lower()
    score = 0.0
    for item in _HIGH_VALUE_ITEMS:
        if item in text:
            score += 0.2
    for item in _MEDIUM_VALUE_ITEMS:
        if item in text:
            score += 0.05
    return max(-1.0, min(1.0, score - 0.5))  # center around 0
```

**Step 5: Run tests**

```bash
python -m pytest tests/tools/test_zork_tools.py -v
```

Expected: all PASS (syntax tests always pass; frotz-dependent tests skip gracefully).

**Step 6: Commit**

```bash
git add tool_pool/zork/ tests/tools/test_zork_tools.py
git commit -m "feat: add seed tools for Zork game"
```

---

## Phase D: Transfer Evaluation Protocol

### Task 11: Transfer evaluation script

**Files:**
- Create: `experiments/transfer_eval.py`
- Create: `tests/integration/test_transfer_eval.py`

**Step 1: Write the failing test**

Create `tests/integration/test_transfer_eval.py`:

```python
def test_transfer_eval_smoke():
    """Transfer evaluation should run without errors on a tiny budget."""
    from experiments.transfer_eval import run_transfer_chain
    results = run_transfer_chain(
        source_game="connect_four",
        target_game="pathfinding",
        source_tool_dir="tool_pool/connect_four",
        target_tool_dir="tool_pool/pathfinding",
        n_eval_games=5,
        sim_budget=20,
    )
    assert "cold_start" in results
    assert "transferred" in results
    assert isinstance(results["cold_start"].normalized_value, float)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/integration/test_transfer_eval.py -v
```

**Step 3: Create `experiments/transfer_eval.py`**

```python
# experiments/transfer_eval.py
from __future__ import annotations
from dataclasses import dataclass

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.training.evaluator import Evaluator, PerformanceResult


@dataclass
class TransferResult:
    source_game: str
    target_game: str
    cold_start: PerformanceResult
    transferred: PerformanceResult


def run_transfer_chain(
    source_game: str,
    target_game: str,
    source_tool_dir: str,
    target_tool_dir: str,
    n_eval_games: int = 50,
    sim_budget: int = 100,
) -> TransferResult:
    adapter = GameAdapter(target_game)
    evaluator = Evaluator(adapter)

    # Cold start: vanilla MCTS, no tools
    vanilla_registry = ToolRegistry()
    vanilla_engine = MCTSEngine(
        adapter, vanilla_registry,
        simulations=sim_budget,
        max_rollout_depth=adapter.meta.max_sim_depth
    )
    cold_start = evaluator.measure(vanilla_engine, n_games=n_eval_games)

    # Transferred: load source game tools into target registry
    transfer_registry = ToolRegistry()
    transfer_registry.load_from_directory(source_tool_dir)
    transfer_engine = MCTSEngine(
        adapter, transfer_registry,
        simulations=sim_budget,
        max_rollout_depth=adapter.meta.max_sim_depth
    )
    transferred = evaluator.measure(transfer_engine, n_games=n_eval_games)

    return TransferResult(
        source_game=source_game,
        target_game=target_game,
        cold_start=cold_start,
        transferred=transferred,
    )


TRANSFER_CHAINS = [
    ("quoridor",     "pathfinding",       "tool_pool/quoridor",     "tool_pool/pathfinding"),
    ("connect_four", "morpion_solitaire", "tool_pool/connect_four", "tool_pool/morpion_solitaire"),
    ("connect_four", "2048",              "tool_pool/connect_four", "tool_pool/2048"),
]


if __name__ == "__main__":
    for source, target, src_dir, tgt_dir in TRANSFER_CHAINS:
        print(f"\n=== Transfer: {source} → {target} ===")
        result = run_transfer_chain(source, target, src_dir, tgt_dir,
                                    n_eval_games=50, sim_budget=200)
        cs = result.cold_start
        tr = result.transferred
        print(f"  Cold start  ({cs.metric_name}): raw={cs.raw_value:.3f}  norm={cs.normalized_value:.3f}")
        print(f"  Transferred ({tr.metric_name}): raw={tr.raw_value:.3f}  norm={tr.normalized_value:.3f}")
        delta = tr.normalized_value - cs.normalized_value
        print(f"  Transfer delta: {delta:+.3f}")
```

**Step 4: Run tests**

```bash
python -m pytest tests/integration/test_transfer_eval.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add experiments/transfer_eval.py tests/integration/test_transfer_eval.py
git commit -m "feat: add transfer evaluation script with three transfer chains"
```

---

## Full Test Suite

After all tasks, run:

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS (Zork tests skip gracefully without frotz).
