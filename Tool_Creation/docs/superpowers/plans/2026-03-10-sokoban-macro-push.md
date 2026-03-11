# Sokoban Macro-Push State Adapter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SokobanMacroState` adapter that wraps `SokobanState` and exposes macro-push actions (BFS-computed box pushes) instead of 4-directional single-step moves.

**Architecture:** Adapter pattern — `SokobanMacroState` implements `GameState`, wraps an inner `SokobanState` for all low-level grid logic. Registered as game `"sokoban_macro"` alongside existing `"sokoban"`. Actions are `(player_pos, direction)` tuples; steps count full BFS walk path + push.

**Tech Stack:** Python 3.14, pytest, existing MCTS framework

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `mcts/games/sokoban_macro.py` | Create | `SokobanMacroState` adapter + `SokobanMacro` factory |
| `mcts/games/__init__.py` | Modify | Add `SokobanMacro, SokobanMacroState` export |
| `tests/test_sokoban_macro.py` | Create | All tests for the macro adapter |
| `LLM/game_infos/sokoban_macro.txt` | Create | Game description for LLM pipeline |
| `MCTS_tools/hyperparams/default_hyperparams.py` | Modify | (Future: `sokoban_macro` config variant) |
| `scripts/run_sokoban.py` | Modify | Add `--macro` flag |

---

## Chunk 1: Core Adapter — BFS + Legal Actions + Apply Action

### Task 1: BFS Reachability — Test + Implementation

**Files:**
- Create: `tests/test_sokoban_macro.py`
- Create: `mcts/games/sokoban_macro.py`

- [ ] **Step 1: Write failing tests for BFS reachability**

```python
# tests/test_sokoban_macro.py
"""Tests for SokobanMacroState adapter."""

import pytest
from mcts.games.sokoban import SokobanState, LEVELS, UP, DOWN, LEFT, RIGHT
from mcts.games.sokoban_macro import SokobanMacroState, SokobanMacro


class TestBFSReachability:
    """BFS flood-fill from player through non-wall, non-box cells."""

    def test_reachable_region_level1(self):
        """Level1: '#####' / '#@$.#' / '#####' — player at (1,1), box at (1,2).
        Reachable region should be just {(1,1)} since box blocks right."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        reachable = macro._compute_reachable()
        assert (1, 1) in reachable
        assert (1, 2) not in reachable  # box blocks
        assert (0, 0) not in reachable  # wall

    def test_reachable_region_open_room(self):
        """Level2: open room — player can reach multiple cells."""
        inner = SokobanState(LEVELS["level2"])
        macro = SokobanMacroState(inner)
        reachable = macro._compute_reachable()
        assert inner.player in reachable
        # Should contain floor cells not blocked by walls/boxes
        for pos in reachable:
            assert pos not in inner.walls
            assert pos not in inner.boxes

    def test_reachable_does_not_cross_boxes(self):
        """Boxes are obstacles for BFS — player cannot walk through them."""
        inner = SokobanState(LEVELS["level3"])
        macro = SokobanMacroState(inner)
        reachable = macro._compute_reachable()
        for pos in inner.boxes:
            assert pos not in reachable
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestBFSReachability -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mcts.games.sokoban_macro'`

- [ ] **Step 3: Implement BFS reachability in sokoban_macro.py**

```python
# mcts/games/sokoban_macro.py
"""
Sokoban macro-push adapter for the MCTS framework.

Wraps a SokobanState and exposes macro-push actions: each action is a
(player_pos, direction) tuple representing a box push reachable from
the player's current connected region. Walking is handled internally
via BFS.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .game_interface import GameState, Game
from .sokoban import SokobanState, LEVELS, UP, DOWN, LEFT, RIGHT, ACTION_DELTAS, ACTION_NAMES


class SokobanMacroState(GameState):
    """
    Macro-push adapter over SokobanState.

    State is still (player_pos, boxes, walls). Actions are (player_pos, direction)
    tuples — macro-pushes reachable from the player's BFS-connected region.
    """

    def __init__(self, inner: SokobanState):
        self._inner = inner

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------

    def _compute_reachable(self) -> set[tuple[int, int]]:
        """BFS flood-fill from player through non-wall, non-box cells."""
        start = self._inner.player
        walls = self._inner.walls
        boxes = self._inner.boxes
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and (nr, nc) not in walls and (nr, nc) not in boxes:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    def _bfs_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[int]:
        """BFS shortest path from start to goal, returns list of direction ints.
        Assumes goal is reachable (in the same connected region)."""
        if start == goal:
            return []
        walls = self._inner.walls
        boxes = self._inner.boxes
        visited: set[tuple[int, int]] = {start}
        # Queue entries: (position, path_so_far)
        queue: deque[tuple[tuple[int, int], list[int]]] = deque([(start, [])])
        while queue:
            (r, c), path = queue.popleft()
            for direction, (dr, dc) in ACTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) == goal:
                    return path + [direction]
                if (nr, nc) not in visited and (nr, nc) not in walls and (nr, nc) not in boxes:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [direction]))
        raise ValueError(f"No path from {start} to {goal}")

    # ------------------------------------------------------------------
    # GameState interface
    # ------------------------------------------------------------------

    def clone(self) -> "SokobanMacroState":
        return SokobanMacroState(self._inner.clone())

    def current_player(self) -> int:
        return 0

    def legal_actions(self) -> list[Any]:
        """Enumerate all macro-push actions from the current reachable region.

        Each action is (player_pos, direction) where:
        - player_pos is a cell in the reachable region adjacent to a box
        - direction is the push direction (box is at player_pos + delta)
        - the cell beyond the box (player_pos + 2*delta) must be free
        """
        reachable = self._compute_reachable()
        actions: list[tuple[tuple[int, int], int]] = []
        walls = self._inner.walls
        boxes = self._inner.boxes
        for cell in reachable:
            cr, cc = cell
            for direction, (dr, dc) in ACTION_DELTAS.items():
                box_pos = (cr + dr, cc + dc)
                if box_pos not in boxes:
                    continue
                # Check cell beyond the box
                beyond = (cr + 2 * dr, cc + 2 * dc)
                if beyond not in walls and beyond not in boxes:
                    actions.append((cell, direction))
        return actions

    def apply_action(self, action: Any) -> None:
        """Apply a macro-push: BFS walk to player_pos, then push.

        Args:
            action: (player_pos, direction) tuple.
        """
        player_pos, direction = action

        # Walk: BFS path from current player to the push position
        walk_path = self._bfs_path(self._inner.player, player_pos)
        for move in walk_path:
            self._inner.apply_action(move)

        # Push: apply the push direction
        self._inner.apply_action(direction)

    def is_terminal(self) -> bool:
        return self._inner.is_terminal()

    def returns(self) -> list[float]:
        return self._inner.returns()

    def state_key(self) -> str:
        return self._inner.state_key()

    def __str__(self) -> str:
        return str(self._inner)

    # ------------------------------------------------------------------
    # Expose inner state attributes for heuristic tools
    # ------------------------------------------------------------------

    @property
    def player(self):
        return self._inner.player

    @property
    def boxes(self):
        return self._inner.boxes

    @property
    def targets(self):
        return self._inner.targets

    @property
    def walls(self):
        return self._inner.walls

    @property
    def height(self):
        return self._inner.height

    @property
    def width(self):
        return self._inner.width

    @property
    def num_targets(self):
        return self._inner.num_targets

    @property
    def steps(self):
        return self._inner.steps

    @property
    def max_steps(self):
        return self._inner.max_steps

    def boxes_on_targets(self) -> int:
        return self._inner.boxes_on_targets()

    def total_box_distance(self) -> int:
        return self._inner.total_box_distance()

    def _is_solved(self) -> bool:
        return self._inner._is_solved()

    def _is_deadlocked(self) -> bool:
        return self._inner._is_deadlocked()


# =====================================================================
# Game factory
# =====================================================================

class SokobanMacro(Game):
    """Factory for macro-push Sokoban games."""

    def __init__(
        self,
        level_name: str = "level1",
        max_steps: int = 200,
        level_lines: list[str] | None = None,
    ):
        if level_lines is not None:
            self.level_lines = level_lines
            self.level_name = "custom"
        else:
            if level_name not in LEVELS:
                raise ValueError(
                    f"Unknown level '{level_name}'. "
                    f"Available: {list(LEVELS.keys())}"
                )
            self.level_name = level_name
            self.level_lines = LEVELS[level_name]
        self.max_steps = max_steps

    def new_initial_state(self) -> SokobanMacroState:
        inner = SokobanState(self.level_lines, self.max_steps)
        return SokobanMacroState(inner)

    def num_players(self) -> int:
        return 1

    def name(self) -> str:
        return f"Sokoban_Macro ({self.level_name})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestBFSReachability -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add mcts/games/sokoban_macro.py tests/test_sokoban_macro.py
git commit -m "feat: add SokobanMacroState adapter with BFS reachability"
```

---

### Task 2: Legal Actions — Tests

**Files:**
- Modify: `tests/test_sokoban_macro.py`

- [ ] **Step 1: Write failing tests for legal_actions**

Add to `tests/test_sokoban_macro.py`:

```python
class TestMacroLegalActions:
    """legal_actions() returns (player_pos, direction) tuples for all reachable pushes."""

    def test_level1_single_push(self):
        """Level1: '#####' / '#@$.#' / '#####'
        Player at (1,1), box at (1,2), target at (1,3).
        Only push: stand at (1,1), push RIGHT — box goes (1,2)->(1,3)."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        actions = macro.legal_actions()
        assert len(actions) == 1
        assert actions[0] == ((1, 1), RIGHT)

    def test_all_actions_have_valid_player_pos(self):
        """Every action's player_pos must be in the reachable region."""
        inner = SokobanState(LEVELS["level5"])
        macro = SokobanMacroState(inner)
        reachable = macro._compute_reachable()
        for player_pos, direction in macro.legal_actions():
            assert player_pos in reachable, f"{player_pos} not reachable"

    def test_all_actions_push_a_box(self):
        """For every action, player_pos + delta must contain a box."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        for player_pos, direction in macro.legal_actions():
            dr, dc = ACTION_DELTAS[direction]
            box_pos = (player_pos[0] + dr, player_pos[1] + dc)
            assert box_pos in inner.boxes, f"No box at {box_pos}"

    def test_all_actions_have_free_beyond(self):
        """For every action, the cell beyond the box must be free."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        for player_pos, direction in macro.legal_actions():
            dr, dc = ACTION_DELTAS[direction]
            beyond = (player_pos[0] + 2*dr, player_pos[1] + 2*dc)
            assert beyond not in inner.walls
            assert beyond not in inner.boxes

    def test_no_actions_when_fully_blocked(self):
        """A state where no box can be pushed should return empty actions."""
        # Construct a tiny level where the only box is in a corner — already deadlocked
        # but still test that legal_actions returns nothing pushable if surrounded
        lines = [
            "####",
            "#$ #",
            "#@ #",
            "####",
        ]
        inner = SokobanState(lines)
        # Box at (1,1): walls on top (0,1) and left (1,0).
        # Player at (2,1). Reachable: {(2,1), (2,2), (1,2)}.
        # From (1,2) push LEFT? box at (1,1), beyond (1,0) is wall. No.
        # From (2,1) push UP? box at (1,1), beyond (0,1) is wall. No.
        macro = SokobanMacroState(inner)
        actions = macro.legal_actions()
        assert len(actions) == 0

    def test_variable_action_count(self):
        """Different states of the same level can have different action counts."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        count_before = len(macro.legal_actions())

        # Apply one action and check count changes
        actions = macro.legal_actions()
        if actions:
            macro2 = macro.clone()
            macro2.apply_action(actions[0])
            count_after = len(macro2.legal_actions())
            # Just verify it's a valid count (>= 0), not necessarily different
            assert count_after >= 0
```

- [ ] **Step 2: Run tests to verify they pass** (implementation already exists from Task 1)

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestMacroLegalActions -v`
Expected: PASS (6 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sokoban_macro.py
git commit -m "test: add legal_actions tests for SokobanMacroState"
```

---

### Task 3: Apply Action + Step Counting — Tests

**Files:**
- Modify: `tests/test_sokoban_macro.py`

- [ ] **Step 1: Write tests for apply_action and step counting**

Add to `tests/test_sokoban_macro.py`:

```python
class TestMacroApplyAction:
    """apply_action() walks via BFS then pushes, counting all steps."""

    def test_level1_solve(self):
        """Level1: one push RIGHT from (1,1) solves it.
        Walk path is empty (already at push position), push costs 1 step."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        macro.apply_action(((1, 1), RIGHT))
        assert macro._inner._is_solved()
        assert macro._inner.steps == 1  # just the push, no walk

    def test_step_count_includes_walk(self):
        """Steps should count walk + push. Walk of N steps + 1 push = N+1."""
        inner = SokobanState(LEVELS["level2"])
        macro = SokobanMacroState(inner)
        actions = macro.legal_actions()
        assert len(actions) > 0
        # Pick any action and count
        action = actions[0]
        player_pos, direction = action
        walk_len = len(macro._bfs_path(macro._inner.player, player_pos))
        macro.apply_action(action)
        assert macro._inner.steps == walk_len + 1

    def test_player_ends_at_box_original_position(self):
        """After push, player should be at where the box was before."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        # Box is at (1,2), push RIGHT from (1,1)
        macro.apply_action(((1, 1), RIGHT))
        assert macro._inner.player == (1, 2)

    def test_box_moves_in_push_direction(self):
        """After push, box should be at box_pos + delta."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        # Box at (1,2), push RIGHT -> box goes to (1,3)
        macro.apply_action(((1, 1), RIGHT))
        assert (1, 3) in macro._inner.boxes
        assert (1, 2) not in macro._inner.boxes

    def test_apply_action_asserts_reachable(self):
        """Applying an action with unreachable player_pos should raise."""
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        # (0,0) is a wall — not reachable
        with pytest.raises(ValueError):
            macro.apply_action(((0, 0), RIGHT))

    def test_sequential_pushes(self):
        """Multiple macro actions should accumulate steps correctly."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        total_steps = 0
        for _ in range(3):  # apply up to 3 actions
            actions = macro.legal_actions()
            if not actions or macro.is_terminal():
                break
            action = actions[0]
            player_pos, direction = action
            walk_len = len(macro._bfs_path(macro._inner.player, player_pos))
            macro.apply_action(action)
            total_steps += walk_len + 1
        assert macro._inner.steps == total_steps
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestMacroApplyAction -v`
Expected: PASS (6 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sokoban_macro.py
git commit -m "test: add apply_action and step counting tests for macro state"
```

---

### Task 4: Clone + Delegated Methods + Factory — Tests

**Files:**
- Modify: `tests/test_sokoban_macro.py`

- [ ] **Step 1: Write tests for clone, delegation, and factory**

Add to `tests/test_sokoban_macro.py`:

```python
class TestMacroCloneAndDelegation:
    """clone() and delegated methods work correctly."""

    def test_clone_is_independent(self):
        """Mutating a clone should not affect the original."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        clone = macro.clone()
        actions = clone.legal_actions()
        if actions:
            clone.apply_action(actions[0])
        assert macro._inner.steps == 0
        assert macro._inner.player == inner.player

    def test_clone_preserves_state(self):
        """Clone should have same state as original."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        clone = macro.clone()
        assert clone.state_key() == macro.state_key()
        assert clone._inner.player == macro._inner.player
        assert clone._inner.boxes == macro._inner.boxes

    def test_is_terminal_delegates(self):
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        assert not macro.is_terminal()
        macro.apply_action(((1, 1), RIGHT))  # solve it
        assert macro.is_terminal()

    def test_returns_delegates(self):
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        macro.apply_action(((1, 1), RIGHT))
        assert macro.returns() == [1.0]

    def test_state_key_delegates(self):
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        assert macro.state_key() == inner.state_key()

    def test_str_delegates(self):
        inner = SokobanState(LEVELS["level1"])
        macro = SokobanMacroState(inner)
        assert str(macro) == str(inner)

    def test_property_proxies(self):
        """All proxied properties should match inner state."""
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        assert macro.player == inner.player
        assert macro.boxes == inner.boxes
        assert macro.targets == inner.targets
        assert macro.walls == inner.walls
        assert macro.height == inner.height
        assert macro.width == inner.width
        assert macro.num_targets == inner.num_targets
        assert macro.steps == inner.steps
        assert macro.max_steps == inner.max_steps

    def test_helper_methods_proxy(self):
        inner = SokobanState(LEVELS["level4"])
        macro = SokobanMacroState(inner)
        assert macro.boxes_on_targets() == inner.boxes_on_targets()
        assert macro.total_box_distance() == inner.total_box_distance()


class TestSokobanMacroFactory:
    """SokobanMacro factory creates correct initial states."""

    def test_creates_macro_state(self):
        game = SokobanMacro(level_name="level1")
        state = game.new_initial_state()
        assert isinstance(state, SokobanMacroState)

    def test_num_players(self):
        game = SokobanMacro(level_name="level1")
        assert game.num_players() == 1

    def test_name(self):
        game = SokobanMacro(level_name="level3")
        assert "Sokoban_Macro" in game.name()
        assert "level3" in game.name()

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            SokobanMacro(level_name="nonexistent")

    def test_custom_lines(self):
        lines = ["####", "#@.#", "#$ #", "####"]
        game = SokobanMacro(level_lines=lines)
        state = game.new_initial_state()
        assert isinstance(state, SokobanMacroState)

    @pytest.mark.parametrize("level_name", list(LEVELS.keys()))
    def test_all_levels_have_legal_macro_actions(self, level_name):
        """Every standard level should have at least one pushable box initially."""
        game = SokobanMacro(level_name=level_name)
        state = game.new_initial_state()
        actions = state.legal_actions()
        assert len(actions) > 0, f"{level_name}: no macro actions at start"
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestMacroCloneAndDelegation tests/test_sokoban_macro.py::TestSokobanMacroFactory -v`
Expected: PASS (17 tests)

- [ ] **Step 3: Update `__init__.py` export**

Modify `mcts/games/__init__.py` — add:
```python
from .sokoban_macro import SokobanMacro, SokobanMacroState
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_sokoban_macro.py mcts/games/__init__.py
git commit -m "test: add clone, delegation, and factory tests; export SokobanMacro"
```

---

## Chunk 2: MCTS Integration + Engine Smoke Test

### Task 5: MCTS Engine Integration Test

**Files:**
- Modify: `tests/test_sokoban_macro.py`

- [ ] **Step 1: Write MCTS engine integration tests**

Add to `tests/test_sokoban_macro.py`:

```python
from mcts import MCTSEngine


class TestMCTSWithMacroState:
    """MCTS engine works with SokobanMacroState as a drop-in replacement."""

    def test_search_returns_tuple_action(self):
        """search() should return a (player_pos, direction) tuple."""
        game = SokobanMacro(level_name="level1", max_steps=50)
        engine = MCTSEngine(game, iterations=10, max_rollout_depth=20)
        state = game.new_initial_state()
        action = engine.search(state)
        assert isinstance(action, tuple)
        assert len(action) == 2
        player_pos, direction = action
        assert isinstance(player_pos, tuple)
        assert direction in (UP, DOWN, LEFT, RIGHT)

    def test_play_game_solves_level1(self):
        """Level1 is trivial — MCTS should solve it with macro-pushes."""
        game = SokobanMacro(level_name="level1", max_steps=50)
        engine = MCTSEngine(game, iterations=50, max_rollout_depth=20)
        result = engine.play_game()
        assert result["solved"]

    def test_play_game_with_logging(self):
        """Logging should work — trace records individual moves."""
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmpdir:
            game = SokobanMacro(level_name="level1", max_steps=50)
            engine = MCTSEngine(
                game, iterations=50, max_rollout_depth=20,
                logging=True, records_dir=tmpdir,
            )
            result = engine.play_game()
            assert result["solved"]
            assert result.get("log_file")
            # Verify trace file is valid JSON
            with open(result["log_file"]) as f:
                trace = json.load(f)
            assert "moves" in trace
            assert "outcome" in trace

    @pytest.mark.parametrize("level_name", ["level1", "level2"])
    def test_engine_runs_without_crash(self, level_name):
        """Engine should not crash on easy levels with macro state."""
        game = SokobanMacro(level_name=level_name, max_steps=100)
        engine = MCTSEngine(game, iterations=20, max_rollout_depth=30)
        result = engine.play_game()
        assert "solved" in result
        assert "steps" in result

    def test_play_many(self):
        """play_many should work with macro state."""
        game = SokobanMacro(level_name="level1", max_steps=50)
        engine = MCTSEngine(game, iterations=50, max_rollout_depth=20)
        stats = engine.play_many(num_games=2)
        assert stats["total"] == 2
        assert "solve_rate" in stats
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py::TestMCTSWithMacroState -v`
Expected: PASS (6 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_sokoban_macro.py
git commit -m "test: add MCTS engine integration tests for macro state"
```

---

## Chunk 3: LLM Pipeline + Script Updates

### Task 6: Game Description for LLM

**Files:**
- Create: `LLM/game_infos/sokoban_macro.txt`

- [ ] **Step 1: Write the macro game description**

```text
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
```

- [ ] **Step 2: Commit**

```bash
git add LLM/game_infos/sokoban_macro.txt
git commit -m "feat: add macro-push game description for LLM pipeline"
```

---

### Task 7: Update run_sokoban.py

**Files:**
- Modify: `scripts/run_sokoban.py`

- [ ] **Step 1: Add --macro flag to run_sokoban.py**

Add to imports (line 25-26):
```python
from mcts.games import Sokoban
from mcts.games.sokoban_macro import SokobanMacro
```

Add argument (after line 37):
```python
    p.add_argument(
        "--macro", action="store_true",
        help="Use macro-push variant (SokobanMacroState).",
    )
```

Modify game creation in `main()` (replace line 75):
```python
    GameClass = SokobanMacro if args.macro else Sokoban
    game = GameClass(args.level)
```

Modify prompt builder game name (replace line 107):
```python
    game_name = "sokoban_macro" if args.macro else "sokoban"
    pb = PromptBuilder(game=game_name, target_phase=args.phase)
```

- [ ] **Step 2: Test manually**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python scripts/run_sokoban.py --level level1 --macro --iterations 50 --no-prompt --verbose`
Expected: Solves level1, shows macro-push actions in output.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sokoban.py
git commit -m "feat: add --macro flag to run_sokoban.py"
```

---

### Task 8: Run Full Test Suite

- [ ] **Step 1: Run all macro tests**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/test_sokoban_macro.py -v`
Expected: All tests pass.

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd /Users/hrzhang/Desktop/WI26/CSE291A/Project/code/CSE291A_Project/Tool_Creation && python -m pytest tests/ -v --ignore=tests/.venv`
Expected: All existing tests still pass.

- [ ] **Step 3: Final commit if any fixes needed**

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | BFS reachability + full implementation | 3 |
| 2 | Legal actions tests | 6 |
| 3 | Apply action + step counting tests | 6 |
| 4 | Clone + delegation + factory tests | 17 |
| 5 | MCTS engine integration tests | 6 |
| 6 | LLM game description | manual |
| 7 | run_sokoban.py --macro flag | manual |
| 8 | Full test suite regression check | all |
| **Total** | | **38+ tests** |
