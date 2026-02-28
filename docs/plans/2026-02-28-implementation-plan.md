# Self-Evolving Game Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a hybrid RL+LLM system where MCTS starts vanilla and an LLM periodically injects auto-discovered heuristic tools as Python code plugins, enabling MCTS to tackle hard turn-based games without human-designed heuristics.

**Architecture:** Three subsystems — (A) Custom Python MCTS engine with 6 dynamic tool hook points using OpenSpiel for game logic, (B) Yunjue-inspired LLM pipeline that analyzes game traces to generate/validate/promote tool code, (C) Cross-game tool manager that persists tools and handles transfer between Connect Four, Quoridor, and Chess. OpenSpiel's MCTSBot is used as the vanilla baseline.

**Tech Stack:** Python 3.11+, OpenSpiel (pyspiel), OpenAI-compatible LLM API (configurable), PyYAML, pytest

---

## Phase 1: MCTS Engine + Tool Hooks + Hand-Written Tools on Connect Four

### Task 1: Project Scaffolding

**Files:**
- Create: `src/__init__.py`
- Create: `src/mcts/__init__.py`
- Create: `src/tools/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/llm/__init__.py`
- Create: `src/games/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/mcts/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/training/__init__.py`
- Create: `pyproject.toml`
- Create: `conf.yaml`
- Create: `tool_pool/global/.gitkeep`
- Create: `tool_pool/connect_four/.gitkeep`
- Create: `tool_pool/quoridor/.gitkeep`
- Create: `tool_pool/chess/.gitkeep`
- Create: `tool_pool/metadata.json`

**Step 1: Create project structure**

```toml
# pyproject.toml
[project]
name = "self-evolving-game-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "open_spiel",
    "pyyaml",
    "openai",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-timeout"]

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 30
```

```yaml
# conf.yaml
TRACE_ANALYZER:
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  temperature: 0.7

CODE_GENERATOR:
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  temperature: 0.2

TOOL_VALIDATOR:
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  temperature: 0.2
```

```json
// tool_pool/metadata.json
{}
```

Create all `__init__.py` files as empty files. Create `.gitkeep` files in each tool_pool subdirectory.

**Step 2: Verify OpenSpiel is importable**

Run: `python -c "import pyspiel; print(pyspiel.load_game('connect_four'))"`
Expected: Game object printed without error. If not, run `pip install open_spiel`.

**Step 3: Commit**

```bash
git init
git add -A
git commit -m "chore: initial project scaffolding with directory structure"
```

---

### Task 2: MCTS Node

**Files:**
- Create: `src/mcts/node.py`
- Create: `tests/mcts/test_node.py`

**Step 1: Write the failing test**

```python
# tests/mcts/test_node.py
from src.mcts.node import MCTSNode


def test_node_creation():
    node = MCTSNode(state=None, parent=None, action=None)
    assert node.visits == 0
    assert node.value == 0.0
    assert node.children == []
    assert node.parent is None
    assert node.action is None
    assert node.untried_actions is None


def test_node_uct_unexplored():
    """Unexplored nodes should have infinite UCT value."""
    node = MCTSNode(state=None, parent=None, action=None)
    assert node.uct_value(c=1.41) == float("inf")


def test_node_uct_value():
    """UCT = value/visits + c * sqrt(ln(parent_visits) / visits)"""
    import math

    parent = MCTSNode(state=None, parent=None, action=None)
    parent.visits = 100
    child = MCTSNode(state=None, parent=parent, action=0)
    child.visits = 10
    child.value = 7.0

    c = 1.41
    expected = 7.0 / 10 + c * math.sqrt(math.log(100) / 10)
    assert abs(child.uct_value(c) - expected) < 1e-6


def test_node_puct_value():
    """PUCT = value/visits + c * prior * sqrt(parent_visits) / (visits + 1)"""
    import math

    parent = MCTSNode(state=None, parent=None, action=None)
    parent.visits = 100
    child = MCTSNode(state=None, parent=parent, action=0, prior=0.3)
    child.visits = 10
    child.value = 7.0

    c = 1.41
    expected = 7.0 / 10 + c * 0.3 * math.sqrt(100) / (10 + 1)
    assert abs(child.puct_value(c) - expected) < 1e-6


def test_best_child_by_visits():
    parent = MCTSNode(state=None, parent=None, action=None)
    c1 = MCTSNode(state=None, parent=parent, action=0)
    c1.visits = 50
    c2 = MCTSNode(state=None, parent=parent, action=1)
    c2.visits = 100
    c3 = MCTSNode(state=None, parent=parent, action=2)
    c3.visits = 30
    parent.children = [c1, c2, c3]
    assert parent.best_child_by_visits().action == 1


def test_backpropagate():
    root = MCTSNode(state=None, parent=None, action=None)
    child = MCTSNode(state=None, parent=root, action=0)
    grandchild = MCTSNode(state=None, parent=child, action=1)

    grandchild.backpropagate(value=1.0)
    assert grandchild.visits == 1
    assert grandchild.value == 1.0
    assert child.visits == 1
    assert child.value == 1.0
    assert root.visits == 1
    assert root.value == 1.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/mcts/test_node.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mcts.node'`

**Step 3: Write minimal implementation**

```python
# src/mcts/node.py
from __future__ import annotations

import math
from typing import Any


class MCTSNode:
    __slots__ = (
        "state",
        "parent",
        "action",
        "prior",
        "children",
        "visits",
        "value",
        "untried_actions",
    )

    def __init__(
        self,
        state: Any,
        parent: MCTSNode | None,
        action: int | None,
        prior: float = 1.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: list[int] | None = None

    def uct_value(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def puct_value(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * self.prior * math.sqrt(self.parent.visits) / (self.visits + 1)
        return exploitation + exploration

    def best_child_by_visits(self) -> MCTSNode:
        return max(self.children, key=lambda c: c.visits)

    def best_child_by_uct(self, c: float = 1.41) -> MCTSNode:
        return max(self.children, key=lambda ch: ch.uct_value(c))

    def backpropagate(self, value: float) -> None:
        self.visits += 1
        self.value += value
        if self.parent is not None:
            self.parent.backpropagate(value)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/mcts/test_node.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/mcts/node.py tests/mcts/test_node.py
git commit -m "feat: MCTS node with UCT, PUCT, and backpropagation"
```

---

### Task 3: Tool Interface Definitions

**Files:**
- Create: `src/tools/base.py`
- Create: `tests/tools/test_base.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_base.py
from src.tools.base import (
    ToolType,
    ToolMeta,
    validate_tool_meta,
    load_tool_from_file,
)
import tempfile
import os


def test_tool_type_enum():
    assert ToolType.STATE_EVALUATOR == "state_evaluator"
    assert ToolType.ACTION_FILTER == "action_filter"
    assert ToolType.ROLLOUT_POLICY == "rollout_policy"
    assert ToolType.SELECTION_PRIOR == "selection_prior"
    assert ToolType.REWARD_SHAPER == "reward_shaper"
    assert ToolType.MACRO_ACTION == "macro_action"


def test_validate_tool_meta_valid():
    meta = {
        "name": "test_tool",
        "type": "state_evaluator",
        "description": "A test tool",
    }
    result = validate_tool_meta(meta)
    assert result.name == "test_tool"
    assert result.type == ToolType.STATE_EVALUATOR


def test_validate_tool_meta_invalid_type():
    meta = {
        "name": "test_tool",
        "type": "invalid_type",
        "description": "A test tool",
    }
    try:
        validate_tool_meta(meta)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_validate_tool_meta_missing_field():
    meta = {"name": "test_tool"}
    try:
        validate_tool_meta(meta)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_load_tool_from_file():
    tool_code = '''
__TOOL_META__ = {
    "name": "dummy_eval",
    "type": "state_evaluator",
    "description": "Returns 0.5 for any state",
}

def run(state) -> float:
    return 0.5
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(tool_code)
        f.flush()
        try:
            meta, run_fn = load_tool_from_file(f.name)
            assert meta.name == "dummy_eval"
            assert meta.type == ToolType.STATE_EVALUATOR
            assert run_fn(None) == 0.5
        finally:
            os.unlink(f.name)


def test_load_tool_from_file_no_meta():
    tool_code = '''
def run(state):
    return 0.5
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(tool_code)
        f.flush()
        try:
            load_tool_from_file(f.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "__TOOL_META__" in str(e)
        finally:
            os.unlink(f.name)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/tools/base.py
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class ToolType(str, Enum):
    STATE_EVALUATOR = "state_evaluator"
    ACTION_FILTER = "action_filter"
    ROLLOUT_POLICY = "rollout_policy"
    SELECTION_PRIOR = "selection_prior"
    REWARD_SHAPER = "reward_shaper"
    MACRO_ACTION = "macro_action"


@dataclass
class ToolMeta:
    name: str
    type: ToolType
    description: str


def validate_tool_meta(meta: dict) -> ToolMeta:
    required = {"name", "type", "description"}
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"Missing required fields in __TOOL_META__: {missing}")

    try:
        tool_type = ToolType(meta["type"])
    except ValueError:
        valid = [t.value for t in ToolType]
        raise ValueError(
            f"Invalid tool type '{meta['type']}'. Must be one of: {valid}"
        )

    return ToolMeta(name=meta["name"], type=tool_type, description=meta["description"])


def load_tool_from_file(filepath: str) -> tuple[ToolMeta, Callable]:
    spec = importlib.util.spec_from_file_location("tool_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "__TOOL_META__"):
        raise ValueError(
            f"Tool file {filepath} missing __TOOL_META__ dict"
        )
    if not hasattr(module, "run"):
        raise ValueError(f"Tool file {filepath} missing run() function")

    meta = validate_tool_meta(module.__TOOL_META__)
    return meta, module.run
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_base.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/tools/base.py tests/tools/test_base.py
git commit -m "feat: tool interface definitions with ToolType enum and file loader"
```

---

### Task 4: Tool Registry (Dynamic Loading & Dispatch)

**Files:**
- Create: `src/mcts/tool_registry.py`
- Create: `tests/mcts/test_tool_registry.py`

**Step 1: Write the failing test**

```python
# tests/mcts/test_tool_registry.py
import os
import tempfile

from src.mcts.tool_registry import ToolRegistry
from src.tools.base import ToolType


def _write_tool_file(directory: str, name: str, tool_type: str, body: str) -> str:
    code = f'''
__TOOL_META__ = {{
    "name": "{name}",
    "type": "{tool_type}",
    "description": "test tool",
}}

{body}
'''
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def test_register_and_get_tools():
    registry = ToolRegistry()
    registry.register(
        name="eval1",
        tool_type=ToolType.STATE_EVALUATOR,
        run_fn=lambda state: 0.5,
    )
    tools = registry.get_tools(ToolType.STATE_EVALUATOR)
    assert len(tools) == 1
    assert tools[0].name == "eval1"
    assert tools[0].run_fn(None) == 0.5


def test_get_tools_empty():
    registry = ToolRegistry()
    tools = registry.get_tools(ToolType.ACTION_FILTER)
    assert tools == []


def test_load_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_tool_file(
            tmpdir,
            "eval_tool",
            "state_evaluator",
            "def run(state):\n    return 0.42",
        )
        _write_tool_file(
            tmpdir,
            "filter_tool",
            "action_filter",
            "def run(state, legal_actions):\n    return legal_actions[:2]",
        )

        registry = ToolRegistry()
        registry.load_from_directory(tmpdir)

        evals = registry.get_tools(ToolType.STATE_EVALUATOR)
        assert len(evals) == 1
        assert evals[0].run_fn(None) == 0.42

        filters = registry.get_tools(ToolType.ACTION_FILTER)
        assert len(filters) == 1
        assert filters[0].run_fn(None, [1, 2, 3]) == [1, 2]


def test_unregister():
    registry = ToolRegistry()
    registry.register("eval1", ToolType.STATE_EVALUATOR, lambda s: 0.5)
    registry.unregister("eval1")
    assert registry.get_tools(ToolType.STATE_EVALUATOR) == []


def test_list_all():
    registry = ToolRegistry()
    registry.register("eval1", ToolType.STATE_EVALUATOR, lambda s: 0.5)
    registry.register("filter1", ToolType.ACTION_FILTER, lambda s, a: a)
    names = registry.list_all()
    assert set(names) == {"eval1", "filter1"}
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/mcts/test_tool_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/mcts/tool_registry.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.tools.base import ToolType, load_tool_from_file


@dataclass
class RegisteredTool:
    name: str
    tool_type: ToolType
    run_fn: Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        tool_type: ToolType,
        run_fn: Callable,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name, tool_type=tool_type, run_fn=run_fn
        )

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tools(self, tool_type: ToolType) -> list[RegisteredTool]:
        return [t for t in self._tools.values() if t.tool_type == tool_type]

    def list_all(self) -> list[str]:
        return list(self._tools.keys())

    def load_from_directory(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(directory, filename)
            try:
                meta, run_fn = load_tool_from_file(filepath)
                self.register(meta.name, meta.type, run_fn)
            except (ValueError, Exception):
                continue  # skip invalid files
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/mcts/test_tool_registry.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/mcts/tool_registry.py tests/mcts/test_tool_registry.py
git commit -m "feat: tool registry with dynamic loading from directory"
```

---

### Task 5: OpenSpiel Game Adapter

**Files:**
- Create: `src/games/adapter.py`
- Create: `tests/games/__init__.py`
- Create: `tests/games/test_adapter.py`

**Step 1: Write the failing test**

```python
# tests/games/test_adapter.py
import pyspiel
from src.games.adapter import GameAdapter


def test_create_connect_four():
    adapter = GameAdapter("connect_four")
    assert adapter.game_name == "connect_four"
    assert adapter.num_players == 2


def test_new_game_state():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    assert not state.is_terminal()
    assert state.current_player() == 0


def test_legal_actions():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    assert len(actions) == 7  # 7 columns


def test_apply_action():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    actions = adapter.legal_actions(state)
    new_state = adapter.apply_action(state, actions[0])
    assert new_state.current_player() == 1


def test_clone_state():
    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    clone = adapter.clone_state(state)
    adapter.apply_action(state, 0)
    # Clone should not be affected
    assert state.current_player() != clone.current_player() or str(state) != str(clone)


def test_play_random_game_to_terminal():
    import random

    adapter = GameAdapter("connect_four")
    state = adapter.new_game()
    while not adapter.is_terminal(state):
        actions = adapter.legal_actions(state)
        state = adapter.apply_action(state, random.choice(actions))
    returns = adapter.returns(state)
    assert len(returns) == 2
    assert all(isinstance(r, float) for r in returns)


def test_game_description():
    adapter = GameAdapter("connect_four")
    desc = adapter.game_description()
    assert "connect_four" in desc.lower() or len(desc) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/games/test_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/games/adapter.py
from __future__ import annotations

import pyspiel


class GameAdapter:
    """Thin wrapper around OpenSpiel games with a consistent interface."""

    def __init__(self, game_name: str, **params):
        if params:
            param_str = ",".join(f"{k}={v}" for k, v in params.items())
            self._game = pyspiel.load_game(f"{game_name}({param_str})")
        else:
            self._game = pyspiel.load_game(game_name)
        self.game_name = game_name
        self.num_players = self._game.num_players()
        self.num_distinct_actions = self._game.num_distinct_actions()

    def new_game(self):
        return self._game.new_initial_state()

    def legal_actions(self, state) -> list[int]:
        return state.legal_actions()

    def apply_action(self, state, action: int):
        """Apply action to a clone of state (non-mutating)."""
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state

    def clone_state(self, state):
        return state.clone()

    def is_terminal(self, state) -> bool:
        return state.is_terminal()

    def current_player(self, state) -> int:
        return state.current_player()

    def returns(self, state) -> list[float]:
        return state.returns()

    def action_to_string(self, state, action: int) -> str:
        return state.action_to_string(state.current_player(), action)

    def game_description(self) -> str:
        return (
            f"Game: {self.game_name}, "
            f"Players: {self.num_players}, "
            f"Actions: {self.num_distinct_actions}, "
            f"Max length: {self._game.max_game_length()}"
        )

    @property
    def pyspiel_game(self):
        return self._game
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/games/test_adapter.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/games/adapter.py tests/games/__init__.py tests/games/test_adapter.py
git commit -m "feat: OpenSpiel game adapter with consistent interface"
```

---

### Task 6: Core MCTS Engine (Vanilla, No Tools)

**Files:**
- Create: `src/mcts/engine.py`
- Create: `tests/mcts/test_engine.py`

**Step 1: Write the failing test**

```python
# tests/mcts/test_engine.py
import pyspiel
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.games.adapter import GameAdapter


def test_vanilla_mcts_creates_root():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=10, uct_c=1.41)
    state = adapter.new_game()
    action = engine.search(state)
    assert action in adapter.legal_actions(state)


def test_vanilla_mcts_connect_four():
    adapter = GameAdapter("connect_four")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=100, uct_c=1.41)
    state = adapter.new_game()
    action = engine.search(state)
    assert action in adapter.legal_actions(state)


def test_vanilla_mcts_wins_vs_random_tic_tac_toe():
    """MCTS with 200 sims should beat random in tic-tac-toe most of the time."""
    import random

    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=200, uct_c=1.41)

    wins = 0
    num_games = 20
    for _ in range(num_games):
        state = adapter.new_game()
        while not adapter.is_terminal(state):
            if adapter.current_player(state) == 0:
                action = engine.search(state)
            else:
                action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
        if adapter.returns(state)[0] > 0:
            wins += 1

    # MCTS should win at least 80% against random
    assert wins >= 16, f"MCTS only won {wins}/{num_games} against random"


def test_search_returns_policy():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=50, uct_c=1.41)
    state = adapter.new_game()
    action, policy = engine.search_with_policy(state)
    assert action in adapter.legal_actions(state)
    assert isinstance(policy, dict)
    assert action in policy
    # Policy values should sum to ~1.0
    total = sum(policy.values())
    assert abs(total - 1.0) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/mcts/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/mcts/engine.py
from __future__ import annotations

import math
import random
from typing import Any

from src.games.adapter import GameAdapter
from src.mcts.node import MCTSNode
from src.mcts.tool_registry import ToolRegistry
from src.tools.base import ToolType


class MCTSEngine:
    def __init__(
        self,
        adapter: GameAdapter,
        registry: ToolRegistry,
        simulations: int = 1000,
        uct_c: float = 1.41,
        max_rollout_depth: int = 200,
    ):
        self.adapter = adapter
        self.registry = registry
        self.simulations = simulations
        self.uct_c = uct_c
        self.max_rollout_depth = max_rollout_depth

    def search(self, state) -> int:
        action, _ = self.search_with_policy(state)
        return action

    def search_with_policy(self, state) -> tuple[int, dict[int, float]]:
        root = MCTSNode(state=state.clone(), parent=None, action=None)
        root.untried_actions = self._get_actions(root.state)

        for _ in range(self.simulations):
            node = self._select(root)
            node = self._expand(node)
            value = self._simulate(node)
            self._backpropagate(node, value)

        # Build policy from visit counts
        total_visits = sum(c.visits for c in root.children)
        policy = {}
        for child in root.children:
            policy[child.action] = child.visits / total_visits if total_visits > 0 else 0

        best = root.best_child_by_visits()
        return best.action, policy

    def _get_actions(self, state) -> list[int]:
        """Get legal actions, applying action_filter tools if registered."""
        actions = list(state.legal_actions())

        for tool in self.registry.get_tools(ToolType.ACTION_FILTER):
            try:
                actions = tool.run_fn(state, actions)
            except Exception:
                continue  # skip broken tools

        # Add macro actions if any
        for tool in self.registry.get_tools(ToolType.MACRO_ACTION):
            try:
                macro_seq = tool.run_fn(state)
                if macro_seq:
                    # Encode macro as negative action ID (convention)
                    actions.append(-hash(tuple(macro_seq)) % 10_000_000)
            except Exception:
                continue

        return actions

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCT."""
        while node.untried_actions is not None and len(node.untried_actions) == 0 and node.children:
            if node.state.is_terminal():
                return node

            # Apply selection_prior tools for PUCT if available
            priors = self.registry.get_tools(ToolType.SELECTION_PRIOR)
            if priors:
                node = max(node.children, key=lambda c: c.puct_value(self.uct_c))
            else:
                node = max(node.children, key=lambda c: c.uct_value(self.uct_c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand by adding a child for an untried action."""
        if node.state.is_terminal():
            return node
        if node.untried_actions is None:
            node.untried_actions = self._get_actions(node.state)
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        child_state = node.state.clone()
        child_state.apply_action(action)

        child = MCTSNode(state=child_state, parent=node, action=action)
        child.untried_actions = list(child_state.legal_actions()) if not child_state.is_terminal() else []
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Simulate from node to terminal using rollout policy tools."""
        state = node.state.clone()
        current_player = node.state.current_player() if not node.state.is_terminal() else 0
        # Walk up to find the root player
        n = node
        while n.parent is not None:
            n = n.parent
        root_player = n.state.current_player()

        evaluators = self.registry.get_tools(ToolType.STATE_EVALUATOR)
        rollout_policies = self.registry.get_tools(ToolType.ROLLOUT_POLICY)

        depth = 0
        while not state.is_terminal() and depth < self.max_rollout_depth:
            # Check for early cutoff via state evaluator
            if evaluators and depth > 5:
                try:
                    scores = [t.run_fn(state) for t in evaluators]
                    avg_score = sum(scores) / len(scores)
                    if abs(avg_score) > 0.9:
                        return avg_score if state.current_player() == root_player else -avg_score
                except Exception:
                    pass

            legal = state.legal_actions()
            if not legal:
                break

            # Use rollout policy tool or random
            action = None
            if rollout_policies:
                try:
                    tool = random.choice(rollout_policies)
                    action = tool.run_fn(state, legal)
                except Exception:
                    action = None

            if action is None or action not in legal:
                action = random.choice(legal)

            state.apply_action(action)
            depth += 1

        if state.is_terminal():
            returns = state.returns()
            return returns[root_player]

        # Non-terminal (max depth): use evaluator or return 0
        if evaluators:
            try:
                scores = [t.run_fn(state) for t in evaluators]
                return sum(scores) / len(scores)
            except Exception:
                pass
        return 0.0

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate with optional reward shaping."""
        shapers = self.registry.get_tools(ToolType.REWARD_SHAPER)
        if shapers and node.state is not None:
            try:
                shaped_values = [t.run_fn(node.state, value) for t in shapers]
                value = sum(shaped_values) / len(shaped_values)
            except Exception:
                pass

        node.backpropagate(value)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/mcts/test_engine.py -v --timeout=60`
Expected: All 4 tests PASS (the win rate test may take ~10 seconds)

**Step 5: Commit**

```bash
git add src/mcts/engine.py tests/mcts/test_engine.py
git commit -m "feat: MCTS engine with 6 tool hook points and vanilla defaults"
```

---

### Task 7: Hand-Written Tools for Connect Four

**Files:**
- Create: `tool_pool/connect_four/center_column_bias.py`
- Create: `tool_pool/connect_four/threat_detector.py`
- Create: `tool_pool/connect_four/greedy_rollout.py`
- Create: `tests/tools/test_connect_four_tools.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_connect_four_tools.py
import pyspiel
from src.tools.base import load_tool_from_file


def _make_state(moves: list[int] = None):
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    for m in (moves or []):
        state.apply_action(m)
    return state


def test_center_column_bias_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/center_column_bias.py"
    )
    assert meta.type.value == "state_evaluator"


def test_center_column_bias_prefers_center():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/center_column_bias.py"
    )
    state = _make_state()
    score = run_fn(state)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_threat_detector_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/threat_detector.py"
    )
    assert meta.type.value == "action_filter"


def test_threat_detector_returns_subset():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/threat_detector.py"
    )
    state = _make_state()
    legal = state.legal_actions()
    filtered = run_fn(state, legal)
    assert isinstance(filtered, list)
    assert len(filtered) > 0
    assert all(a in legal for a in filtered)


def test_greedy_rollout_loads():
    meta, run_fn = load_tool_from_file(
        "tool_pool/connect_four/greedy_rollout.py"
    )
    assert meta.type.value == "rollout_policy"


def test_greedy_rollout_returns_legal_action():
    _, run_fn = load_tool_from_file(
        "tool_pool/connect_four/greedy_rollout.py"
    )
    state = _make_state()
    legal = state.legal_actions()
    action = run_fn(state, legal)
    assert action in legal
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_connect_four_tools.py -v`
Expected: FAIL — `FileNotFoundError`

**Step 3: Write the three tool files**

```python
# tool_pool/connect_four/center_column_bias.py
"""Evaluate Connect Four states by counting pieces in center columns."""

__TOOL_META__ = {
    "name": "center_column_bias",
    "type": "state_evaluator",
    "description": "Score states higher when the current player has more pieces in center columns. Game-agnostic: uses observation tensor to infer board shape and center.",
}


def run(state) -> float:
    board_str = str(state)
    lines = [l for l in board_str.strip().split("\n") if l.strip()]

    # Count pieces per column for current player
    player = state.current_player()
    if player < 0:
        return 0.0

    # Parse the board string to find piece positions
    # Connect Four board string has 'x' and 'o' characters
    my_char = "x" if player == 0 else "o"
    opp_char = "o" if player == 0 else "x"

    my_center = 0
    opp_center = 0
    total_cols = 0

    for line in lines:
        chars = [c for c in line if c in ("x", "o", ".")]
        if not chars:
            continue
        total_cols = max(total_cols, len(chars))
        center_start = len(chars) // 2 - 1
        center_end = len(chars) // 2 + 1
        for i, c in enumerate(chars):
            if center_start <= i <= center_end:
                if c == my_char:
                    my_center += 1
                elif c == opp_char:
                    opp_center += 1

    if my_center + opp_center == 0:
        return 0.0

    score = (my_center - opp_center) / max(my_center + opp_center, 1)
    return max(-1.0, min(1.0, score))
```

```python
# tool_pool/connect_four/threat_detector.py
"""Filter actions to prioritize winning moves and blocking opponent wins."""

__TOOL_META__ = {
    "name": "threat_detector",
    "type": "action_filter",
    "description": "Prioritize actions that win immediately or block an opponent's immediate win. Falls back to all legal actions if no threats found.",
}


def run(state, legal_actions: list[int]) -> list[int]:
    player = state.current_player()
    if player < 0:
        return legal_actions

    # Check for immediate wins
    winning_moves = []
    for action in legal_actions:
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            returns = child.returns()
            if returns[player] > 0:
                winning_moves.append(action)

    if winning_moves:
        return winning_moves

    # Check for moves that block opponent's immediate win
    # Simulate opponent having the turn by checking each column
    blocking_moves = []
    for action in legal_actions:
        # Check if opponent could win by playing this action
        # We do this by seeing if the next state would give opponent a winning move
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal():
            continue
        opp_legal = child.legal_actions()
        opponent_can_win = False
        for opp_action in opp_legal:
            grandchild = child.clone()
            grandchild.apply_action(opp_action)
            if grandchild.is_terminal() and grandchild.returns()[1 - player] > 0:
                opponent_can_win = True
                break
        if not opponent_can_win:
            blocking_moves.append(action)

    if blocking_moves:
        return blocking_moves

    return legal_actions
```

```python
# tool_pool/connect_four/greedy_rollout.py
"""Biased rollout policy: prefer center columns and blocking moves."""
import random

__TOOL_META__ = {
    "name": "greedy_rollout",
    "type": "rollout_policy",
    "description": "During simulation, with 60% probability play toward center columns or block threats. With 40% probability play randomly.",
}


def run(state, legal_actions: list[int]) -> int:
    if random.random() > 0.6:
        return random.choice(legal_actions)

    player = state.current_player()
    if player < 0:
        return random.choice(legal_actions)

    # Check for immediate wins
    for action in legal_actions:
        child = state.clone()
        child.apply_action(action)
        if child.is_terminal() and child.returns()[player] > 0:
            return action

    # Prefer center columns (for a 7-column board, center is 3)
    num_actions = max(legal_actions) + 1 if legal_actions else 7
    center = num_actions // 2
    # Sort by distance from center
    sorted_actions = sorted(legal_actions, key=lambda a: abs(a - center))
    # Pick from top half with higher probability
    top_half = sorted_actions[: max(1, len(sorted_actions) // 2)]
    return random.choice(top_half)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_connect_four_tools.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tool_pool/connect_four/ tests/tools/test_connect_four_tools.py
git commit -m "feat: hand-written Connect Four tools (center bias, threat detector, greedy rollout)"
```

---

### Task 8: Integration Test — MCTS + Tools vs Vanilla on Connect Four

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_connect_four_with_tools.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_connect_four_with_tools.py
"""
Integration test: MCTS with hand-written tools should beat vanilla MCTS
on Connect Four at equal simulation budgets.
"""
import random
import pyspiel
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.games.adapter import GameAdapter


def _play_game(engine_p0: MCTSEngine, engine_p1: MCTSEngine, adapter: GameAdapter) -> float:
    """Play one game, return player 0's result."""
    state = adapter.new_game()
    while not adapter.is_terminal(state):
        player = adapter.current_player(state)
        if player == 0:
            action = engine_p0.search(state)
        else:
            action = engine_p1.search(state)
        state = adapter.apply_action(state, action)
    return adapter.returns(state)[0]


def test_tools_beat_vanilla():
    """MCTS+tools should win more than lose against vanilla MCTS at 100 sims."""
    adapter = GameAdapter("connect_four")

    vanilla_registry = ToolRegistry()
    tool_registry = ToolRegistry()
    tool_registry.load_from_directory("tool_pool/connect_four")

    vanilla = MCTSEngine(adapter, vanilla_registry, simulations=100, uct_c=1.41)
    with_tools = MCTSEngine(adapter, tool_registry, simulations=100, uct_c=1.41)

    wins = 0
    losses = 0
    draws = 0
    num_games = 20

    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            result = _play_game(with_tools, vanilla, adapter)
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        else:
            result = _play_game(vanilla, with_tools, adapter)
            if result < 0:
                wins += 1
            elif result > 0:
                losses += 1
            else:
                draws += 1

    print(f"\nTools vs Vanilla: {wins}W / {losses}L / {draws}D out of {num_games}")
    # Tools should at least not be significantly worse
    assert wins >= losses, f"Tools lost more than won: {wins}W/{losses}L/{draws}D"


def test_tools_beat_random():
    """MCTS+tools at 50 sims should crush random player."""
    adapter = GameAdapter("connect_four")
    tool_registry = ToolRegistry()
    tool_registry.load_from_directory("tool_pool/connect_four")
    engine = MCTSEngine(adapter, tool_registry, simulations=50, uct_c=1.41)

    wins = 0
    num_games = 20
    for i in range(num_games):
        state = adapter.new_game()
        while not adapter.is_terminal(state):
            player = adapter.current_player(state)
            if player == 0:
                action = engine.search(state)
            else:
                action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
        if adapter.returns(state)[0] > 0:
            wins += 1

    print(f"\nTools+MCTS vs Random: {wins}/{num_games} wins")
    assert wins >= 15, f"Expected at least 15 wins, got {wins}"
```

**Step 2: Run integration tests**

Run: `python -m pytest tests/integration/test_connect_four_with_tools.py -v --timeout=120 -s`
Expected: Both tests PASS (the tools-vs-vanilla test may be close, tools-vs-random should be dominant)

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: integration test - MCTS with tools vs vanilla on Connect Four"
```

---

## Phase 2: LLM Tool Generation Pipeline

### Task 9: LLM Client (Configurable)

**Files:**
- Create: `src/llm/client.py`
- Create: `src/config.py`
- Create: `tests/llm/__init__.py`
- Create: `tests/llm/test_client.py`

**Step 1: Write the failing test**

```python
# tests/llm/test_client.py
import os
from src.config import load_config
from src.llm.client import LLMClient


def test_load_config():
    config = load_config("conf.yaml")
    assert "TRACE_ANALYZER" in config
    assert "model" in config["TRACE_ANALYZER"]


def test_client_creation():
    """Client should be creatable even without a valid API key."""
    client = LLMClient(
        base_url="https://api.example.com/v1",
        model="test-model",
        api_key="test-key",
        temperature=0.7,
    )
    assert client.model == "test-model"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/llm/test_client.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/config.py
from __future__ import annotations

import os

import yaml


def load_config(path: str = "conf.yaml") -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve environment variables in api_key fields
    for section in config.values():
        if isinstance(section, dict) and "api_key" in section:
            key = section["api_key"]
            if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                section["api_key"] = os.environ.get(env_var, key)

    return config
```

```python
# src/llm/client.py
from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: int = 180,
    ):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    @classmethod
    def from_config(cls, config: dict, role: str) -> LLMClient:
        section = config[role]
        return cls(
            base_url=section["base_url"],
            model=section["model"],
            api_key=section.get("api_key", ""),
            temperature=section.get("temperature", 0.7),
            max_retries=section.get("max_retries", 3),
            timeout=section.get("timeout", 180),
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/llm/test_client.py -v`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add src/config.py src/llm/client.py tests/llm/
git commit -m "feat: configurable LLM client with YAML config loading"
```

---

### Task 10: Game Trace Recorder

**Files:**
- Create: `src/training/trace_recorder.py`
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_trace_recorder.py`

**Step 1: Write the failing test**

```python
# tests/training/test_trace_recorder.py
from src.training.trace_recorder import TraceRecorder, GameTrace
from src.games.adapter import GameAdapter
import random


def test_record_game():
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    state = adapter.new_game()
    recorder.start_game(state)

    while not adapter.is_terminal(state):
        action = random.choice(adapter.legal_actions(state))
        state = adapter.apply_action(state, action)
        recorder.record_step(state, action)

    recorder.end_game(adapter.returns(state))
    traces = recorder.get_traces()
    assert len(traces) == 1
    assert len(traces[0].actions) > 0
    assert traces[0].outcome is not None


def test_select_informative_traces():
    """Should select losses and close games preferentially."""
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    # Record several games
    for _ in range(10):
        state = adapter.new_game()
        recorder.start_game(state)
        while not adapter.is_terminal(state):
            action = random.choice(adapter.legal_actions(state))
            state = adapter.apply_action(state, action)
            recorder.record_step(state, action)
        recorder.end_game(adapter.returns(state))

    informative = recorder.select_informative_traces(player=0, n=3)
    assert len(informative) <= 3
    assert all(isinstance(t, GameTrace) for t in informative)


def test_trace_to_string():
    adapter = GameAdapter("tic_tac_toe")
    recorder = TraceRecorder()

    state = adapter.new_game()
    recorder.start_game(state)
    action = adapter.legal_actions(state)[0]
    state = adapter.apply_action(state, action)
    recorder.record_step(state, action)
    recorder.end_game([1.0, -1.0])

    trace = recorder.get_traces()[0]
    text = trace.to_string()
    assert isinstance(text, str)
    assert len(text) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/training/test_trace_recorder.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/training/trace_recorder.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameTrace:
    states: list[str] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    outcome: list[float] | None = None

    def to_string(self) -> str:
        lines = []
        for i, (state_str, action) in enumerate(zip(self.states, self.actions)):
            lines.append(f"Step {i}: Action={action}")
            lines.append(state_str)
            lines.append("")
        if self.outcome:
            lines.append(f"Outcome: {self.outcome}")
        return "\n".join(lines)


class TraceRecorder:
    def __init__(self):
        self._traces: list[GameTrace] = []
        self._current: GameTrace | None = None

    def start_game(self, initial_state) -> None:
        self._current = GameTrace()
        self._current.states.append(str(initial_state))

    def record_step(self, state, action: int) -> None:
        if self._current is None:
            return
        self._current.states.append(str(state))
        self._current.actions.append(action)

    def end_game(self, returns: list[float]) -> None:
        if self._current is None:
            return
        self._current.outcome = returns
        self._traces.append(self._current)
        self._current = None

    def get_traces(self) -> list[GameTrace]:
        return list(self._traces)

    def clear(self) -> None:
        self._traces.clear()

    def select_informative_traces(self, player: int, n: int = 5) -> list[GameTrace]:
        """Select the most informative traces: losses first, then close games."""
        losses = [t for t in self._traces if t.outcome and t.outcome[player] < 0]
        draws = [t for t in self._traces if t.outcome and t.outcome[player] == 0]
        wins = [t for t in self._traces if t.outcome and t.outcome[player] > 0]

        # Prioritize losses, then draws, then wins
        selected = []
        for pool in [losses, draws, wins]:
            for trace in pool:
                if len(selected) >= n:
                    break
                selected.append(trace)

        return selected[:n]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/training/test_trace_recorder.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add src/training/trace_recorder.py tests/training/
git commit -m "feat: game trace recorder with informative trace selection"
```

---

### Task 11: Prompt Templates

**Files:**
- Create: `src/llm/prompts/trace_analysis.md`
- Create: `src/llm/prompts/code_generation.md`
- Create: `src/llm/prompts/tool_validation.md`

**Step 1: Create prompt templates**

```markdown
<!-- src/llm/prompts/trace_analysis.md -->
# Game Trace Analysis

You are an expert game AI designer. Analyze the following game traces where the agent LOST or performed poorly. Identify patterns and propose a heuristic tool that could help the agent play better.

## Game Description
{game_description}

## Current Tools
{current_tools}

## Game Traces (Losses/Poor Performance)
{traces}

## Task
1. Identify WHY the agent is losing (strategic mistakes, missed opportunities, poor move selection)
2. Propose ONE heuristic tool that would address the most critical weakness
3. Specify the tool as JSON:

```json
{
  "name": "snake_case_name",
  "type": "state_evaluator|action_filter|rollout_policy|selection_prior|reward_shaper|macro_action",
  "description": "What this tool does and why it helps",
  "pseudocode": "Step-by-step algorithm description"
}
```

IMPORTANT:
- The tool must be GAME-AGNOSTIC. Use only the OpenSpiel generic API (state.legal_actions(), state.clone(), state.apply_action(), state.is_terminal(), state.returns(), state.current_player(), str(state)).
- Do NOT hardcode board dimensions, piece types, or game-specific rules.
- The tool should be simple and fast (will be called thousands of times per MCTS search).
```

```markdown
<!-- src/llm/prompts/code_generation.md -->
# Tool Code Generation

Write a Python tool file implementing the following heuristic for a game-playing MCTS agent.

## Tool Specification
{tool_spec}

## Required Format

The file MUST follow this exact structure:

```python
__TOOL_META__ = {
    "name": "{tool_name}",
    "type": "{tool_type}",
    "description": "{tool_description}",
}

def run(state{extra_params}) -> {return_type}:
    """
    {tool_description}

    Args:
        state: An OpenSpiel game state object with methods:
            - state.legal_actions() -> list[int]
            - state.clone() -> State
            - state.apply_action(action: int) -> None (mutates in place)
            - state.is_terminal() -> bool
            - state.returns() -> list[float]
            - state.current_player() -> int
            - str(state) -> str (human-readable board)

    Returns:
        {return_description}
    """
    # Implementation here
```

## Tool Type Signatures
- state_evaluator: `run(state) -> float` (range [-1, 1], positive = good for current player)
- action_filter: `run(state, legal_actions: list[int]) -> list[int]` (subset of legal_actions)
- rollout_policy: `run(state, legal_actions: list[int]) -> int` (single action from legal_actions)
- selection_prior: `run(state, legal_actions: list[int]) -> dict[int, float]` (action -> prior probability)
- reward_shaper: `run(state, raw_value: float) -> float` (shaped reward)
- macro_action: `run(state) -> list[int]` (sequence of primitive actions)

## Rules
- GAME-AGNOSTIC: Do NOT import game-specific modules. Do NOT hardcode board sizes.
- FAST: This runs thousands of times per search. Avoid unnecessary cloning or deep loops.
- SAFE: Handle edge cases (empty action lists, terminal states). Never raise exceptions.
- Use only standard library imports (math, random, collections, etc.)

Output ONLY the Python code, no markdown fences or explanation.
```

```markdown
<!-- src/llm/prompts/tool_validation.md -->
# Tool Code Fix

The following tool code failed validation. Fix the issues.

## Original Code
```python
{original_code}
```

## Error
{error_message}

## Requirements
- Must have `__TOOL_META__` dict with "name", "type", "description"
- Must have `run()` function with correct signature for tool type
- Must not crash on any valid OpenSpiel game state
- Must return values in expected range

Output ONLY the fixed Python code, no markdown fences or explanation.
```

**Step 2: Commit**

```bash
git add src/llm/prompts/
git commit -m "feat: LLM prompt templates for trace analysis, code gen, validation"
```

---

### Task 12: Tool Validator

**Files:**
- Create: `src/tools/validator.py`
- Create: `tests/tools/test_validator.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_validator.py
import tempfile
import os
import pyspiel
from src.tools.validator import ToolValidator


def test_validate_valid_tool():
    code = '''
__TOOL_META__ = {
    "name": "test_eval",
    "type": "state_evaluator",
    "description": "Test evaluator",
}

def run(state) -> float:
    return 0.0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=10)
    result = validator.validate_code(code)
    assert result.valid
    assert result.error is None


def test_validate_missing_meta():
    code = '''
def run(state) -> float:
    return 0.0
'''
    validator = ToolValidator(game_name="tic_tac_toe")
    result = validator.validate_code(code)
    assert not result.valid
    assert "__TOOL_META__" in result.error


def test_validate_runtime_crash():
    code = '''
__TOOL_META__ = {
    "name": "crasher",
    "type": "state_evaluator",
    "description": "Crashes on purpose",
}

def run(state) -> float:
    return 1 / 0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert not result.valid
    assert "runtime" in result.error.lower() or "error" in result.error.lower()


def test_validate_bad_return_range():
    code = '''
__TOOL_META__ = {
    "name": "out_of_range",
    "type": "state_evaluator",
    "description": "Returns values outside [-1, 1]",
}

def run(state) -> float:
    return 999.0
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert not result.valid


def test_validate_action_filter():
    code = '''
__TOOL_META__ = {
    "name": "pass_through",
    "type": "action_filter",
    "description": "Returns all actions",
}

def run(state, legal_actions):
    return legal_actions
'''
    validator = ToolValidator(game_name="tic_tac_toe", num_test_states=5)
    result = validator.validate_code(code)
    assert result.valid
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_validator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/tools/validator.py
from __future__ import annotations

import ast
import importlib.util
import random
import tempfile
import os
import time
from dataclasses import dataclass

import pyspiel

from src.tools.base import ToolType, validate_tool_meta


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


class ToolValidator:
    def __init__(
        self,
        game_name: str = "tic_tac_toe",
        num_test_states: int = 50,
        timeout_ms: float = 100.0,
    ):
        self.game_name = game_name
        self.num_test_states = num_test_states
        self.timeout_ms = timeout_ms

    def validate_code(self, code: str) -> ValidationResult:
        # Step 1: Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, error=f"Syntax error: {e}")

        # Step 2: Check __TOOL_META__ exists in AST
        tree = ast.parse(code)
        has_meta = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "__TOOL_META__"
                for t in node.targets
            )
            for node in ast.walk(tree)
        )
        if not has_meta:
            return ValidationResult(
                valid=False, error="Missing __TOOL_META__ dict"
            )

        # Step 3: Load module
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmppath = f.name

        try:
            spec = importlib.util.spec_from_file_location("test_tool", tmppath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return ValidationResult(valid=False, error=f"Import error: {e}")
        finally:
            os.unlink(tmppath)

        # Step 4: Validate meta
        try:
            meta = validate_tool_meta(module.__TOOL_META__)
        except ValueError as e:
            return ValidationResult(valid=False, error=str(e))

        if not hasattr(module, "run"):
            return ValidationResult(valid=False, error="Missing run() function")

        # Step 5: Runtime check on random game states
        game = pyspiel.load_game(self.game_name)
        test_states = self._generate_random_states(game)

        for state in test_states:
            if state.is_terminal() or state.current_player() < 0:
                continue
            try:
                if meta.type == ToolType.STATE_EVALUATOR:
                    result = module.run(state)
                    if not isinstance(result, (int, float)):
                        return ValidationResult(
                            valid=False,
                            error=f"state_evaluator returned {type(result)}, expected float",
                        )
                    if not (-1.0 <= float(result) <= 1.0):
                        return ValidationResult(
                            valid=False,
                            error=f"state_evaluator returned {result}, must be in [-1, 1]",
                        )
                elif meta.type == ToolType.ACTION_FILTER:
                    legal = state.legal_actions()
                    result = module.run(state, legal)
                    if not isinstance(result, list):
                        return ValidationResult(
                            valid=False,
                            error=f"action_filter returned {type(result)}, expected list",
                        )
                elif meta.type == ToolType.ROLLOUT_POLICY:
                    legal = state.legal_actions()
                    result = module.run(state, legal)
                    if result not in legal:
                        return ValidationResult(
                            valid=False,
                            error=f"rollout_policy returned {result}, not in legal actions",
                        )
                elif meta.type == ToolType.REWARD_SHAPER:
                    result = module.run(state, 0.5)
                    if not isinstance(result, (int, float)):
                        return ValidationResult(
                            valid=False,
                            error=f"reward_shaper returned {type(result)}, expected float",
                        )
            except Exception as e:
                return ValidationResult(
                    valid=False, error=f"Runtime error on test state: {e}"
                )

        return ValidationResult(valid=True)

    def _generate_random_states(self, game) -> list:
        states = []
        for _ in range(self.num_test_states):
            state = game.new_initial_state()
            # Play random number of moves
            depth = random.randint(0, 20)
            for _ in range(depth):
                if state.is_terminal():
                    break
                if state.current_player() < 0:
                    break
                action = random.choice(state.legal_actions())
                state.apply_action(action)
            states.append(state)
        return states
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_validator.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/tools/validator.py tests/tools/test_validator.py
git commit -m "feat: tool validator with syntax, meta, runtime, and range checks"
```

---

### Task 13: Tool Generator (LLM Pipeline)

**Files:**
- Create: `src/tools/generator.py`
- Create: `tests/tools/test_generator.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_generator.py
from unittest.mock import MagicMock, patch
from src.tools.generator import ToolGenerator


def test_generator_builds_analysis_prompt():
    """Test that the generator constructs proper prompts from traces."""
    mock_client = MagicMock()
    mock_client.generate.return_value = '{"name": "test_tool", "type": "state_evaluator", "description": "test", "pseudocode": "return 0"}'

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    spec = generator.analyze_traces(
        traces_text="Step 0: Action=4\n...",
        game_description="Tic Tac Toe, 3x3 grid",
        current_tools_desc="No tools loaded",
    )

    assert mock_client.generate.called
    assert spec is not None
    assert "name" in spec


def test_generator_generates_code():
    mock_client = MagicMock()
    mock_client.generate.return_value = '''__TOOL_META__ = {
    "name": "test_tool",
    "type": "state_evaluator",
    "description": "test",
}

def run(state) -> float:
    return 0.0
'''

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    spec = {
        "name": "test_tool",
        "type": "state_evaluator",
        "description": "test",
        "pseudocode": "return 0",
    }
    code = generator.generate_code(spec)
    assert "__TOOL_META__" in code
    assert "def run" in code


def test_generator_full_pipeline_mock():
    """Test the full generate_tool pipeline with mocked LLM."""
    mock_client = MagicMock()

    # First call: trace analysis
    mock_client.generate.side_effect = [
        '{"name": "simple_eval", "type": "state_evaluator", "description": "Returns 0", "pseudocode": "return 0"}',
        # Second call: code generation
        '''__TOOL_META__ = {
    "name": "simple_eval",
    "type": "state_evaluator",
    "description": "Returns 0",
}

def run(state) -> float:
    return 0.0
''',
    ]

    generator = ToolGenerator(
        trace_analyzer_client=mock_client,
        code_generator_client=mock_client,
        validator_client=mock_client,
        game_name="tic_tac_toe",
    )

    result = generator.generate_tool(
        traces_text="Game trace here",
        game_description="Tic Tac Toe",
        current_tools_desc="None",
    )

    assert result is not None
    assert result.valid
    assert "simple_eval" in result.code
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_generator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/tools/generator.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.llm.client import LLMClient
from src.tools.validator import ToolValidator, ValidationResult


@dataclass
class GenerationResult:
    valid: bool
    code: str | None = None
    spec: dict | None = None
    error: str | None = None


PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with open(path) as f:
        return f.read()


class ToolGenerator:
    def __init__(
        self,
        trace_analyzer_client: LLMClient,
        code_generator_client: LLMClient,
        validator_client: LLMClient,
        game_name: str = "tic_tac_toe",
        max_retries: int = 3,
    ):
        self.trace_analyzer = trace_analyzer_client
        self.code_generator = code_generator_client
        self.validator_client = validator_client
        self.game_name = game_name
        self.max_retries = max_retries
        self.tool_validator = ToolValidator(game_name=game_name)

    def analyze_traces(
        self,
        traces_text: str,
        game_description: str,
        current_tools_desc: str,
    ) -> dict | None:
        try:
            prompt_template = _load_prompt("trace_analysis.md")
        except FileNotFoundError:
            prompt_template = (
                "Analyze these game traces and propose a heuristic tool.\n"
                "Game: {game_description}\n"
                "Current tools: {current_tools}\n"
                "Traces:\n{traces}\n"
                "Respond with JSON: {name, type, description, pseudocode}"
            )

        prompt = prompt_template.format(
            game_description=game_description,
            current_tools=current_tools_desc,
            traces=traces_text,
        )

        response = self.trace_analyzer.generate(
            system_prompt="You are a game AI expert. Respond with valid JSON only.",
            user_prompt=prompt,
        )

        # Parse JSON from response
        try:
            # Try to extract JSON from the response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def generate_code(self, spec: dict) -> str:
        tool_type = spec["type"]
        extra_params = ""
        return_type = "float"
        return_desc = "float value"

        type_signatures = {
            "state_evaluator": ("", "float", "Score in [-1, 1]"),
            "action_filter": (
                ", legal_actions: list[int]",
                "list[int]",
                "Subset of legal_actions",
            ),
            "rollout_policy": (
                ", legal_actions: list[int]",
                "int",
                "Single action from legal_actions",
            ),
            "selection_prior": (
                ", legal_actions: list[int]",
                "dict[int, float]",
                "Action to prior probability mapping",
            ),
            "reward_shaper": (
                ", raw_value: float",
                "float",
                "Shaped reward value",
            ),
            "macro_action": ("", "list[int]", "Sequence of primitive actions"),
        }

        if tool_type in type_signatures:
            extra_params, return_type, return_desc = type_signatures[tool_type]

        try:
            prompt_template = _load_prompt("code_generation.md")
        except FileNotFoundError:
            prompt_template = (
                "Write a Python tool implementing: {tool_spec}\n"
                "Name: {tool_name}, Type: {tool_type}\n"
                "Function signature: def run(state{extra_params}) -> {return_type}\n"
                "Output only Python code."
            )

        prompt = prompt_template.format(
            tool_spec=json.dumps(spec, indent=2),
            tool_name=spec["name"],
            tool_type=tool_type,
            tool_description=spec["description"],
            extra_params=extra_params,
            return_type=return_type,
            return_description=return_desc,
        )

        response = self.code_generator.generate(
            system_prompt="You are a Python programmer. Output only valid Python code, no markdown.",
            user_prompt=prompt,
        )

        # Strip markdown fences if present
        code = response.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        return code

    def generate_tool(
        self,
        traces_text: str,
        game_description: str,
        current_tools_desc: str,
    ) -> GenerationResult:
        # Step 1: Analyze traces
        spec = self.analyze_traces(traces_text, game_description, current_tools_desc)
        if spec is None:
            return GenerationResult(valid=False, error="Failed to analyze traces")

        # Step 2: Generate code
        code = self.generate_code(spec)

        # Step 3: Validate (with retries)
        for attempt in range(self.max_retries):
            result = self.tool_validator.validate_code(code)
            if result.valid:
                return GenerationResult(valid=True, code=code, spec=spec)

            # Try to fix with LLM
            try:
                fix_prompt = (
                    f"Fix this tool code. Error: {result.error}\n\n"
                    f"Original code:\n{code}\n\n"
                    "Output only the fixed Python code."
                )
                code = self.validator_client.generate(
                    system_prompt="You are a Python debugger. Output only valid Python code.",
                    user_prompt=fix_prompt,
                )
                code = code.strip()
                if code.startswith("```python"):
                    code = code[len("```python") :].strip()
                if code.endswith("```"):
                    code = code[:-3].strip()
            except Exception:
                pass

        return GenerationResult(
            valid=False,
            code=code,
            spec=spec,
            error=f"Failed validation after {self.max_retries} retries",
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_generator.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add src/tools/generator.py tests/tools/test_generator.py
git commit -m "feat: LLM tool generator with trace analysis, code gen, and validation retry loop"
```

---

### Task 14: Training Loop with Plateau Detection

**Files:**
- Create: `src/training/trainer.py`
- Create: `tests/training/test_trainer.py`

**Step 1: Write the failing test**

```python
# tests/training/test_trainer.py
from src.training.trainer import PlateauDetector


def test_no_plateau_at_start():
    detector = PlateauDetector(window_size=5, improvement_threshold=0.02)
    detector.record(0.3)
    detector.record(0.35)
    detector.record(0.4)
    assert not detector.is_plateau()


def test_plateau_detected():
    detector = PlateauDetector(window_size=3, improvement_threshold=0.02)
    # Fill two windows with same performance
    for _ in range(3):
        detector.record(0.5)
    for _ in range(3):
        detector.record(0.51)  # <2% improvement
    assert detector.is_plateau()


def test_no_plateau_when_improving():
    detector = PlateauDetector(window_size=3, improvement_threshold=0.02)
    for _ in range(3):
        detector.record(0.3)
    for _ in range(3):
        detector.record(0.5)  # big improvement
    assert not detector.is_plateau()


def test_regression_detected():
    detector = PlateauDetector(
        window_size=3, improvement_threshold=0.02, regression_threshold=0.05
    )
    for _ in range(3):
        detector.record(0.6)
    for _ in range(3):
        detector.record(0.5)  # dropped by 0.1 > 0.05
    assert detector.is_plateau()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/training/test_trainer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/training/trainer.py
from __future__ import annotations

import random
from collections import deque
from typing import Callable

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.training.trace_recorder import TraceRecorder


class PlateauDetector:
    def __init__(
        self,
        window_size: int = 50,
        improvement_threshold: float = 0.02,
        regression_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold
        self._history: list[float] = []

    def record(self, win_rate: float) -> None:
        self._history.append(win_rate)

    def is_plateau(self) -> bool:
        if len(self._history) < 2 * self.window_size:
            return False

        prev_window = self._history[-(2 * self.window_size) : -self.window_size]
        curr_window = self._history[-self.window_size :]

        prev_avg = sum(prev_window) / len(prev_window)
        curr_avg = sum(curr_window) / len(curr_window)

        improvement = curr_avg - prev_avg

        # Regression
        if improvement < -self.regression_threshold:
            return True

        # Plateau (no significant improvement)
        if abs(improvement) < self.improvement_threshold:
            return True

        return False

    def current_win_rate(self) -> float | None:
        if not self._history:
            return None
        window = self._history[-self.window_size :]
        return sum(window) / len(window)


class Trainer:
    def __init__(
        self,
        adapter: GameAdapter,
        registry: ToolRegistry,
        simulations: int = 100,
        uct_c: float = 1.41,
        plateau_detector: PlateauDetector | None = None,
        on_plateau: Callable | None = None,
    ):
        self.adapter = adapter
        self.registry = registry
        self.engine = MCTSEngine(adapter, registry, simulations=simulations, uct_c=uct_c)
        self.recorder = TraceRecorder()
        self.plateau_detector = plateau_detector or PlateauDetector()
        self.on_plateau = on_plateau
        self.total_games = 0

    def play_game_vs_random(self, player: int = 0) -> float:
        """Play one game against a random opponent. Returns result for `player`."""
        state = self.adapter.new_game()
        self.recorder.start_game(state)

        while not self.adapter.is_terminal(state):
            current = self.adapter.current_player(state)
            if current == player:
                action = self.engine.search(state)
            else:
                action = random.choice(self.adapter.legal_actions(state))
            state = self.adapter.apply_action(state, action)
            self.recorder.record_step(state, action)

        returns = self.adapter.returns(state)
        self.recorder.end_game(returns)
        self.total_games += 1

        result = 1.0 if returns[player] > 0 else (0.0 if returns[player] == 0 else -1.0)
        self.plateau_detector.record(result)

        return returns[player]

    def train(self, num_games: int, player: int = 0) -> dict:
        """Run training loop with plateau detection."""
        wins = 0
        for i in range(num_games):
            result = self.play_game_vs_random(player)
            if result > 0:
                wins += 1

            if self.plateau_detector.is_plateau() and self.on_plateau:
                self.on_plateau(self)

        return {
            "games": num_games,
            "wins": wins,
            "win_rate": wins / num_games if num_games > 0 else 0,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/training/test_trainer.py -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/training/test_trainer.py
git commit -m "feat: training loop with plateau detection for tool evolution trigger"
```

---

### Task 15: Tool Pool Manager

**Files:**
- Create: `src/tools/manager.py`
- Create: `tests/tools/test_manager.py`

**Step 1: Write the failing test**

```python
# tests/tools/test_manager.py
import json
import os
import tempfile
from src.tools.manager import ToolPoolManager


def _make_tool_file(directory, name, tool_type="state_evaluator"):
    code = f'''
__TOOL_META__ = {{
    "name": "{name}",
    "type": "{tool_type}",
    "description": "Test tool {name}",
}}

def run(state) -> float:
    return 0.0
'''
    path = os.path.join(directory, f"{name}.py")
    with open(path, "w") as f:
        f.write(code)
    return path


def test_save_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ToolPoolManager(pool_dir=tmpdir)
        code = '''
__TOOL_META__ = {
    "name": "new_tool",
    "type": "state_evaluator",
    "description": "A new tool",
}

def run(state) -> float:
    return 0.0
'''
        manager.save_tool("connect_four", "new_tool", code)
        assert os.path.exists(os.path.join(tmpdir, "connect_four", "new_tool.py"))


def test_load_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({"tool1": {"type": "state_evaluator", "origin_game": "ttt"}}, f)

        manager = ToolPoolManager(pool_dir=tmpdir)
        meta = manager.load_metadata()
        assert "tool1" in meta


def test_update_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({}, f)

        manager = ToolPoolManager(pool_dir=tmpdir)
        manager.update_metadata("new_tool", {
            "type": "state_evaluator",
            "origin_game": "connect_four",
            "games_tested": {"connect_four": 5.0},
        })
        meta = manager.load_metadata()
        assert "new_tool" in meta
        assert meta["new_tool"]["origin_game"] == "connect_four"


def test_list_tools_for_game():
    with tempfile.TemporaryDirectory() as tmpdir:
        game_dir = os.path.join(tmpdir, "connect_four")
        os.makedirs(game_dir)
        _make_tool_file(game_dir, "tool_a")
        _make_tool_file(game_dir, "tool_b")

        manager = ToolPoolManager(pool_dir=tmpdir)
        tools = manager.list_tools_for_game("connect_four")
        assert set(tools) == {"tool_a.py", "tool_b.py"}


def test_promote_to_global():
    with tempfile.TemporaryDirectory() as tmpdir:
        game_dir = os.path.join(tmpdir, "connect_four")
        global_dir = os.path.join(tmpdir, "global")
        os.makedirs(game_dir)
        os.makedirs(global_dir)
        _make_tool_file(game_dir, "good_tool")

        manager = ToolPoolManager(pool_dir=tmpdir)
        manager.promote_to_global("connect_four", "good_tool")
        assert os.path.exists(os.path.join(global_dir, "good_tool.py"))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tools/test_manager.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/tools/manager.py
from __future__ import annotations

import json
import os
import shutil


class ToolPoolManager:
    def __init__(self, pool_dir: str = "tool_pool"):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)

    def save_tool(self, game_name: str, tool_name: str, code: str) -> str:
        game_dir = os.path.join(self.pool_dir, game_name)
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, f"{tool_name}.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def load_metadata(self) -> dict:
        meta_path = os.path.join(self.pool_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def update_metadata(self, tool_name: str, info: dict) -> None:
        meta = self.load_metadata()
        meta[tool_name] = info
        meta_path = os.path.join(self.pool_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def list_tools_for_game(self, game_name: str) -> list[str]:
        game_dir = os.path.join(self.pool_dir, game_name)
        if not os.path.exists(game_dir):
            return []
        return [f for f in os.listdir(game_dir) if f.endswith(".py")]

    def promote_to_global(self, game_name: str, tool_name: str) -> None:
        src = os.path.join(self.pool_dir, game_name, f"{tool_name}.py")
        dst_dir = os.path.join(self.pool_dir, "global")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{tool_name}.py")
        shutil.copy2(src, dst)

    def get_all_tools_for_game(self, game_name: str) -> list[str]:
        """Get paths for global + game-specific tools."""
        paths = []
        global_dir = os.path.join(self.pool_dir, "global")
        if os.path.exists(global_dir):
            for f in os.listdir(global_dir):
                if f.endswith(".py"):
                    paths.append(os.path.join(global_dir, f))

        game_dir = os.path.join(self.pool_dir, game_name)
        if os.path.exists(game_dir):
            for f in os.listdir(game_dir):
                if f.endswith(".py"):
                    paths.append(os.path.join(game_dir, f))

        return paths
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tools/test_manager.py -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add src/tools/manager.py tests/tools/test_manager.py
git commit -m "feat: tool pool manager with save, promote, metadata tracking"
```

---

### Task 16: Evaluation Framework

**Files:**
- Create: `src/training/evaluator.py`
- Create: `tests/training/test_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/training/test_evaluator.py
from src.training.evaluator import Evaluator
from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry


def test_evaluate_vs_random():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine = MCTSEngine(adapter, registry, simulations=50)

    evaluator = Evaluator(adapter)
    result = evaluator.evaluate_vs_random(engine, num_games=10, player=0)
    assert "wins" in result
    assert "losses" in result
    assert "draws" in result
    assert "win_rate" in result
    assert result["wins"] + result["losses"] + result["draws"] == 10


def test_evaluate_head_to_head():
    adapter = GameAdapter("tic_tac_toe")
    registry = ToolRegistry()
    engine_a = MCTSEngine(adapter, registry, simulations=50)
    engine_b = MCTSEngine(adapter, registry, simulations=10)

    evaluator = Evaluator(adapter)
    result = evaluator.evaluate_head_to_head(engine_a, engine_b, num_games=10)
    assert "a_wins" in result
    assert "b_wins" in result
    assert result["a_wins"] + result["b_wins"] + result["draws"] == 10
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/training/test_evaluator.py -v --timeout=60`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/training/evaluator.py
from __future__ import annotations

import random

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine


class Evaluator:
    def __init__(self, adapter: GameAdapter):
        self.adapter = adapter

    def evaluate_vs_random(
        self, engine: MCTSEngine, num_games: int = 100, player: int = 0
    ) -> dict:
        wins, losses, draws = 0, 0, 0
        for i in range(num_games):
            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                current = self.adapter.current_player(state)
                if current == player:
                    action = engine.search(state)
                else:
                    action = random.choice(self.adapter.legal_actions(state))
                state = self.adapter.apply_action(state, action)

            result = self.adapter.returns(state)[player]
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games if num_games > 0 else 0,
            "num_games": num_games,
        }

    def evaluate_head_to_head(
        self,
        engine_a: MCTSEngine,
        engine_b: MCTSEngine,
        num_games: int = 100,
    ) -> dict:
        a_wins, b_wins, draws = 0, 0, 0
        for i in range(num_games):
            # Alternate who plays first
            a_player = i % 2
            b_player = 1 - a_player

            state = self.adapter.new_game()
            while not self.adapter.is_terminal(state):
                current = self.adapter.current_player(state)
                if current == a_player:
                    action = engine_a.search(state)
                else:
                    action = engine_b.search(state)
                state = self.adapter.apply_action(state, action)

            result_a = self.adapter.returns(state)[a_player]
            if result_a > 0:
                a_wins += 1
            elif result_a < 0:
                b_wins += 1
            else:
                draws += 1

        return {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "draws": draws,
            "a_win_rate": a_wins / num_games if num_games > 0 else 0,
            "b_win_rate": b_wins / num_games if num_games > 0 else 0,
            "num_games": num_games,
        }

    def sample_efficiency_curve(
        self,
        engine_factory,
        sim_budgets: list[int] = None,
        num_games_per_budget: int = 50,
        player: int = 0,
    ) -> dict[int, float]:
        """Evaluate win rate at different simulation budgets."""
        if sim_budgets is None:
            sim_budgets = [10, 50, 100, 500, 1000]

        curve = {}
        for budget in sim_budgets:
            engine = engine_factory(budget)
            result = self.evaluate_vs_random(engine, num_games_per_budget, player)
            curve[budget] = result["win_rate"]

        return curve
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/training/test_evaluator.py -v --timeout=60`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add src/training/evaluator.py tests/training/test_evaluator.py
git commit -m "feat: evaluation framework with vs-random, head-to-head, and efficiency curves"
```

---

### Task 17: Main Entry Point

**Files:**
- Create: `main.py`

**Step 1: Write the main script**

```python
# main.py
"""
Self-Evolving Game Agent - Main Entry Point

Usage:
  python main.py --game connect_four --mode train --sims 100 --games 50
  python main.py --game connect_four --mode eval --sims 100 --games 50
  python main.py --game connect_four --mode evolve --sims 100 --games 100
"""
import argparse
import json
import os

from src.config import load_config
from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.tools.manager import ToolPoolManager
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer, PlateauDetector


def main():
    parser = argparse.ArgumentParser(description="Self-Evolving Game Agent")
    parser.add_argument("--game", default="connect_four", help="OpenSpiel game name")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "evolve"],
        default="eval",
        help="Run mode",
    )
    parser.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--uct-c", type=float, default=1.41, help="UCT exploration constant")
    parser.add_argument("--tool-pool", default="tool_pool", help="Tool pool directory")
    parser.add_argument("--no-tools", action="store_true", help="Run vanilla MCTS (no tools)")
    args = parser.parse_args()

    adapter = GameAdapter(args.game)
    print(f"Game: {adapter.game_description()}")

    # Load tools
    registry = ToolRegistry()
    if not args.no_tools:
        pool_manager = ToolPoolManager(args.tool_pool)
        tool_paths = pool_manager.get_all_tools_for_game(args.game)
        for path in tool_paths:
            try:
                from src.tools.base import load_tool_from_file
                meta, run_fn = load_tool_from_file(path)
                registry.register(meta.name, meta.type, run_fn)
                print(f"  Loaded tool: {meta.name} ({meta.type.value})")
            except Exception as e:
                print(f"  Failed to load {path}: {e}")

    tools_loaded = registry.list_all()
    print(f"Tools loaded: {len(tools_loaded)} ({', '.join(tools_loaded) if tools_loaded else 'none'})")

    engine = MCTSEngine(adapter, registry, simulations=args.sims, uct_c=args.uct_c)
    evaluator = Evaluator(adapter)

    if args.mode == "eval":
        print(f"\nEvaluating vs random ({args.games} games, {args.sims} sims/move)...")
        result = evaluator.evaluate_vs_random(engine, num_games=args.games)
        print(f"Results: {result['wins']}W / {result['losses']}L / {result['draws']}D")
        print(f"Win rate: {result['win_rate']:.1%}")

    elif args.mode == "train":
        detector = PlateauDetector(window_size=max(10, args.games // 5))
        trainer = Trainer(
            adapter, registry, simulations=args.sims, uct_c=args.uct_c,
            plateau_detector=detector,
        )
        print(f"\nTraining vs random ({args.games} games, {args.sims} sims/move)...")
        result = trainer.train(num_games=args.games)
        print(f"Results: {result['wins']}W / {result['games'] - result['wins']}L")
        print(f"Win rate: {result['win_rate']:.1%}")

        wr = detector.current_win_rate()
        if wr is not None:
            print(f"Current rolling win rate: {wr:.1%}")
        if detector.is_plateau():
            print("PLATEAU DETECTED - tool evolution recommended")

    elif args.mode == "evolve":
        print("\nEvolution mode requires LLM API configuration in conf.yaml")
        print("Not yet fully integrated - use Phase 2 tasks")


if __name__ == "__main__":
    main()
```

**Step 2: Test the entry point**

Run: `python main.py --game tic_tac_toe --mode eval --sims 50 --games 10`
Expected: Win rate should be high (>80%) against random

Run: `python main.py --game connect_four --mode eval --sims 100 --games 10 --no-tools`
Expected: Reasonable win rate against random

Run: `python main.py --game connect_four --mode eval --sims 100 --games 10`
Expected: Should load Connect Four tools and report win rate

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: main entry point with train/eval/evolve modes"
```

---

## Phase 3: Cross-Game Transfer (Quoridor, Chess)

### Task 18: Quoridor Validation

**Step 1:** Verify Quoridor works in OpenSpiel:

```bash
python -c "
import pyspiel
game = pyspiel.load_game('quoridor')
state = game.new_initial_state()
print('Quoridor loaded')
print(f'Actions: {game.num_distinct_actions()}')
print(f'Players: {game.num_players()}')
print(state)
print(f'Legal actions: {len(state.legal_actions())}')
"
```

**Step 2:** Run vanilla MCTS on Quoridor to establish baseline:

```bash
python main.py --game quoridor --mode eval --sims 100 --games 10 --no-tools
```

**Step 3:** Run with Connect Four tools transferred:

```bash
python main.py --game quoridor --mode eval --sims 100 --games 10
```

**Step 4: Commit any fixes**

```bash
git commit -m "test: validate Quoridor baseline and tool transfer from Connect Four"
```

---

### Task 19: Chess Validation

Same as Task 18 but for chess:

```bash
python -c "
import pyspiel
game = pyspiel.load_game('chess')
state = game.new_initial_state()
print('Chess loaded')
print(f'Actions: {game.num_distinct_actions()}')
print(state)
"
```

Then run eval:

```bash
python main.py --game chess --mode eval --sims 50 --games 5 --no-tools
python main.py --game chess --mode eval --sims 50 --games 5
```

**Commit:**

```bash
git commit -m "test: validate Chess baseline and cross-game tool transfer"
```

---

## Phase 4: Full Evaluation Suite

### Task 20: Evaluation Scripts

**Files:**
- Create: `experiments/run_evaluation.py`

**Step 1: Write comprehensive evaluation script**

```python
# experiments/run_evaluation.py
"""
Full evaluation suite:
1. Win rate tables at various sim budgets
2. Sample efficiency curves
3. Cross-game transfer analysis
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.adapter import GameAdapter
from src.mcts.engine import MCTSEngine
from src.mcts.tool_registry import ToolRegistry
from src.tools.manager import ToolPoolManager
from src.tools.base import load_tool_from_file
from src.training.evaluator import Evaluator


GAMES = ["connect_four", "quoridor"]
SIM_BUDGETS = [50, 100, 500, 1000]
GAMES_PER_EVAL = 50
TOOL_POOL_DIR = "tool_pool"


def load_registry(game_name: str, pool_dir: str) -> ToolRegistry:
    registry = ToolRegistry()
    manager = ToolPoolManager(pool_dir)
    for path in manager.get_all_tools_for_game(game_name):
        try:
            meta, run_fn = load_tool_from_file(path)
            registry.register(meta.name, meta.type, run_fn)
        except Exception:
            continue
    return registry


def run_win_rate_table(game_name: str):
    print(f"\n{'='*60}")
    print(f"Win Rate Table: {game_name}")
    print(f"{'='*60}")

    adapter = GameAdapter(game_name)
    evaluator = Evaluator(adapter)

    results = {}
    for sims in SIM_BUDGETS:
        # Vanilla
        vanilla_engine = MCTSEngine(adapter, ToolRegistry(), simulations=sims)
        vanilla_result = evaluator.evaluate_vs_random(vanilla_engine, GAMES_PER_EVAL)

        # With tools
        tool_registry = load_registry(game_name, TOOL_POOL_DIR)
        tool_engine = MCTSEngine(adapter, tool_registry, simulations=sims)
        tool_result = evaluator.evaluate_vs_random(tool_engine, GAMES_PER_EVAL)

        results[sims] = {
            "vanilla": vanilla_result["win_rate"],
            "with_tools": tool_result["win_rate"],
        }

        print(f"  {sims} sims: vanilla={vanilla_result['win_rate']:.1%}, tools={tool_result['win_rate']:.1%}")

    return results


def run_sample_efficiency(game_name: str):
    print(f"\n{'='*60}")
    print(f"Sample Efficiency: {game_name}")
    print(f"{'='*60}")

    adapter = GameAdapter(game_name)
    evaluator = Evaluator(adapter)
    tool_registry = load_registry(game_name, TOOL_POOL_DIR)

    vanilla_curve = evaluator.sample_efficiency_curve(
        lambda sims: MCTSEngine(adapter, ToolRegistry(), simulations=sims),
        SIM_BUDGETS, GAMES_PER_EVAL,
    )
    tool_curve = evaluator.sample_efficiency_curve(
        lambda sims: MCTSEngine(adapter, tool_registry, simulations=sims),
        SIM_BUDGETS, GAMES_PER_EVAL,
    )

    for sims in SIM_BUDGETS:
        print(f"  {sims} sims: vanilla={vanilla_curve[sims]:.1%}, tools={tool_curve[sims]:.1%}")

    return {"vanilla": vanilla_curve, "with_tools": tool_curve}


if __name__ == "__main__":
    all_results = {}
    for game in GAMES:
        all_results[game] = {
            "win_rate_table": run_win_rate_table(game),
            "sample_efficiency": run_sample_efficiency(game),
        }

    output_path = "experiments/results.json"
    os.makedirs("experiments", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
```

**Step 2: Run evaluation**

```bash
python experiments/run_evaluation.py
```

**Step 3: Commit**

```bash
git add experiments/
git commit -m "feat: full evaluation suite with win rates, efficiency curves, and results output"
```

---

## Summary of All Tasks

| # | Task | Phase | Dependencies |
|---|------|-------|-------------|
| 1 | Project scaffolding | 1 | None |
| 2 | MCTS Node | 1 | 1 |
| 3 | Tool interface definitions | 1 | 1 |
| 4 | Tool registry | 1 | 3 |
| 5 | OpenSpiel game adapter | 1 | 1 |
| 6 | Core MCTS engine | 1 | 2, 4, 5 |
| 7 | Hand-written Connect Four tools | 1 | 3 |
| 8 | Integration test (tools vs vanilla) | 1 | 6, 7 |
| 9 | LLM client | 2 | 1 |
| 10 | Game trace recorder | 2 | 5 |
| 11 | Prompt templates | 2 | None |
| 12 | Tool validator | 2 | 3 |
| 13 | Tool generator (LLM pipeline) | 2 | 9, 11, 12 |
| 14 | Training loop with plateau detection | 2 | 6, 10 |
| 15 | Tool pool manager | 2 | 3 |
| 16 | Evaluation framework | 2 | 5, 6 |
| 17 | Main entry point | 2 | All above |
| 18 | Quoridor validation | 3 | 17 |
| 19 | Chess validation | 3 | 17 |
| 20 | Full evaluation suite | 4 | 18, 19 |
