"""
Core MCTS engine.

Standard MCTS loop:  Select -> Expand -> Simulate -> Backpropagate

Each phase is a pluggable "tool" -- a single Python function loaded from
MCTS_tools/<phase>/. Tool paths are specified in tool_config.json so
nothing is hardcoded. The engine supports hot-swap via set_tool() so the
LLM can inject improved versions at runtime.

When logging=True, the engine automatically records every game played
via play_game() / play_many() into JSON trace files under records/.

Tool slots:
    selection(node, exploration_weight)        -> MCTSNode
    expansion(node)                            -> MCTSNode
    simulation(state, player, max_depth)       -> float
    backpropagation(node, reward)              -> None
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .node import MCTSNode
from .games.game_interface import Game, GameState


# ── Load tool configuration from JSON ────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parent / "tool_config.json"

def _load_config() -> dict:
    """Read and return the tool configuration."""
    with open(_CONFIG_PATH) as f:
        return json.load(f)

_CONFIG = _load_config()

# Resolve tools_dir relative to the config file's directory
_TOOLS_DIR = (Path(_CONFIG_PATH).parent / _CONFIG["tools_dir"]).resolve()


def _get_default_path(phase: str) -> Path:
    """Return the absolute path to the default tool file for a phase."""
    phase_cfg = _CONFIG["phases"][phase]
    return _TOOLS_DIR / phase_cfg["folder"] / phase_cfg["default"]


def _load_function_from_file(filepath: str | Path, func_name: str | None = None) -> Callable:
    """
    Load a single function from a Python file.

    If func_name is None, returns the first public function defined
    in the file (by source order).
    """
    filepath = Path(filepath)
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if func_name:
        fn = getattr(module, func_name, None)
        if fn is not None and callable(fn):
            return fn

    # Fallback: first public callable
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj) and not isinstance(obj, type):
            return obj

    raise ImportError(f"No callable found in {filepath}")


# =====================================================================
# Engine
# =====================================================================

class MCTSEngine:
    """
    Monte Carlo Tree Search engine with pluggable phase tools.

    Each MCTS phase (selection, expansion, simulation, backpropagation)
    is a standalone function loaded from MCTS_tools/<phase>/.

    Usage::

        game = Sokoban(level_name="level1")
        engine = MCTSEngine(game, iterations=100)
        action = engine.search(game.new_initial_state())

    Hot-swap a tool::

        engine.set_tool("simulation", my_custom_simulation_fn)
        # or load from file:
        engine.load_tool("simulation", "MCTS_tools/simulation/manhattan_sim.py")
    """

    PHASES = tuple(_CONFIG["phases"].keys())

    def __init__(
        self,
        game: Game,
        iterations: int = 100,
        max_rollout_depth: int = 50,
        exploration_weight: float = 1.41,
        logging: bool = False,
        records_dir: str | Path | None = None,
    ):
        self.game = game
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth
        self.exploration_weight = exploration_weight
        self.logging = logging

        # Logger is created lazily only when logging is enabled
        self._logger = None
        self._records_dir = records_dir
        if self.logging:
            from .trace_logger import TraceLogger
            self._logger = TraceLogger(records_dir=records_dir)

        # Load default tools from paths specified in tool_config.json
        self._tools: dict[str, Callable] = {}
        self._tool_paths: dict[str, str] = {}   # phase -> filepath (for source display)
        for phase in self.PHASES:
            default_path = _get_default_path(phase)
            fn = _load_function_from_file(default_path)
            self._tools[phase] = fn
            self._tool_paths[phase] = str(default_path)

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tool(self, phase: str, fn: Callable) -> None:
        """
        Replace a phase tool with a callable.

        Args:
            phase: One of 'selection', 'expansion', 'simulation', 'backpropagation'.
            fn:    A callable matching the phase's signature.
        """
        if phase not in self.PHASES:
            raise KeyError(
                f"Unknown phase '{phase}'. Available: {list(self.PHASES)}"
            )
        self._tools[phase] = fn
        self._tool_paths[phase] = "(set programmatically)"

    def load_tool(self, phase: str, filepath: str | Path) -> None:
        """
        Load a tool function from a Python file and set it for the given phase.

        Args:
            phase:    One of 'selection', 'expansion', 'simulation', 'backpropagation'.
            filepath: Path to a .py file containing the tool function.
        """
        if phase not in self.PHASES:
            raise KeyError(
                f"Unknown phase '{phase}'. Available: {list(self.PHASES)}"
            )
        fn = _load_function_from_file(filepath)
        self._tools[phase] = fn
        self._tool_paths[phase] = str(filepath)

    def get_tool(self, phase: str) -> Callable:
        """Return the current tool function for a phase."""
        return self._tools[phase]

    def get_tool_source(self) -> dict[str, str]:
        """Return the full source file for each tool (for prompt building).

        Reads the entire .py file so that helper functions, imports, and
        other context defined alongside the main tool function are
        included in LLM prompts.
        """
        sources: dict[str, str] = {}
        for phase, fn in self._tools.items():
            # Prefer reading the full file over inspect.getsource (which
            # only returns the single function body).
            stored_path = self._tool_paths.get(phase)
            source = self._read_tool_file(phase, fn, stored_path)
            if source is not None:
                sources[phase] = source
            else:
                # Last resort: single-function source via inspect
                try:
                    sources[phase] = inspect.getsource(fn)
                except (OSError, TypeError):
                    sources[phase] = f"# source unavailable for {fn.__name__}"
        return sources

    @staticmethod
    def _read_tool_file(phase: str, fn: Callable, stored_path: str | None = None) -> str | None:
        """Try to read the full .py file that defines *fn*."""
        # 1. Try the stored path (from load_tool or __init__)
        if stored_path and os.path.isfile(stored_path):
            return Path(stored_path).read_text(encoding="utf-8")
        # 2. Try inspect.getfile on the function object
        try:
            fpath = inspect.getfile(fn)
            if fpath and os.path.isfile(fpath):
                return Path(fpath).read_text(encoding="utf-8")
        except (OSError, TypeError):
            pass
        # 3. Fall back to the default path from tool_config.json
        try:
            default = _get_default_path(phase)
            if default and os.path.isfile(default):
                return Path(default).read_text(encoding="utf-8")
        except Exception:
            pass
        return None

    def reset_tool(self, phase: str) -> None:
        """Reset a phase tool to its default from tool_config.json."""
        if phase not in self.PHASES:
            raise KeyError(f"Unknown phase '{phase}'.")
        default_path = _get_default_path(phase)
        fn = _load_function_from_file(default_path)
        self._tools[phase] = fn
        self._tool_paths[phase] = str(default_path)

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(self, root_state: GameState) -> Any:
        """
        Run MCTS from root_state and return the best action.

        Returns:
            The action with the most visits from the root.
        """
        _, action = self._search_internal(root_state)
        return action

    def _search_internal(self, root_state: GameState) -> tuple[MCTSNode, Any]:
        """
        Run MCTS and return (root_node, best_action).

        The root node is needed when logging is enabled so we can
        capture per-child statistics without duplicating the search loop.
        """
        root = MCTSNode(root_state.clone())

        select_fn = self._tools["selection"]
        expand_fn = self._tools["expansion"]
        simulate_fn = self._tools["simulation"]
        backprop_fn = self._tools["backpropagation"]

        for _ in range(self.iterations):
            # 1. Selection -- walk tree via UCB1
            node = select_fn(root, self.exploration_weight)

            # 2. Expansion -- add a child if node has untried actions
            if not node.is_terminal:
                node = expand_fn(node)

            # 3. Simulation -- random rollout from the node
            reward = simulate_fn(
                node.state,
                root_state.current_player(),
                self.max_rollout_depth,
            )

            # 4. Backpropagation -- update stats from leaf to root
            backprop_fn(node, reward)

        if not root.children:
            # Fallback: expansion never added children (e.g. tool pruned all).
            actions = root_state.legal_actions()
            best_action = actions[0] if actions else None
        else:
            best_action = root.most_visited_child().parent_action
        return root, best_action

    # ------------------------------------------------------------------
    # High-level play helpers
    # ------------------------------------------------------------------

    def play_game(self, verbose: bool = False) -> dict:
        """
        Play one full game from initial state using MCTS for every move.

        When self.logging is True, a detailed trace JSON file is written
        to the records directory automatically.

        Returns:
            Dict with keys: solved, steps, returns, moves.
            When logging: also includes 'log_file' and 'trace'.
        """
        state = self.game.new_initial_state()
        moves: list[Any] = []

        # Start trace if logging
        if self.logging and self._logger is not None:
            self._logger.begin_game({
                "game": self.game.name(),
                "timestamp": datetime.now().isoformat(),
                "iterations": self.iterations,
                "max_rollout_depth": self.max_rollout_depth,
                "exploration_weight": self.exploration_weight,
                "tools": {
                    phase: self._tool_paths.get(phase, "unknown")
                    for phase in self.PHASES
                },
            })

        while not state.is_terminal():
            if not state.legal_actions():
                break
            root, action = self._search_internal(state)
            
            # Record move trace if logging
            if self.logging and self._logger is not None:
                children_stats = {}
                for a, child in root.children.items():
                    children_stats[str(a)] = {
                        "visits": child.visits,
                        "value": round(child.value, 4),
                        "avg_value": round(child.value / child.visits, 4)
                            if child.visits > 0 else 0.0,
                    }
                self._logger.record_move({
                    "move_number": len(moves) + 1,
                    "player": state.current_player(),
                    "state_before": str(state),
                    "state_key": state.state_key(),
                    "legal_actions": [str(a) for a in state.legal_actions()],
                    "action_chosen": str(action),
                    "root_visits": root.visits,
                    "children_stats": children_stats,
                })

            state.apply_action(action)
            moves.append(action)

            if verbose:
                print(f"Move {len(moves)}: action={action}")
                print(state)
                print()

        solved = state.returns()[0] >= 1.0  # full solve / win only
        result: dict[str, Any] = {
            "solved": solved,
            "steps": len(moves),
            "returns": state.returns(),
            "moves": moves,
        }

        # Finalize trace if logging
        if self.logging and self._logger is not None:
            trace = self._logger.end_game({
                "solved": solved,
                "steps": len(moves),
                "returns": state.returns(),
                "final_state": str(state),
            })
            result["log_file"] = trace.get("log_file")
            result["trace"] = trace

        return result

    def play_many(
        self,
        num_games: int = 10,
        verbose: bool = False,
    ) -> dict:
        """
        Play multiple games and return aggregate stats.

        Returns:
            Dict with: total, solved, solve_rate, avg_steps, results (list).
        """
        results = []
        for i in range(num_games):
            result = self.play_game(verbose=verbose)
            results.append(result)
            if verbose or (i + 1) % max(1, num_games // 5) == 0:
                tag = "solved" if result["solved"] else "unsolved"
                print(f"  Game {i+1}/{num_games}: {tag} in {result['steps']} steps")

        solved_count = sum(1 for r in results if r["solved"])
        total_steps = sum(r["steps"] for r in results)
        return {
            "total": num_games,
            "solved": solved_count,
            "solve_rate": round(solved_count / num_games, 4),
            "avg_steps": round(total_steps / num_games, 1),
            "results": results,
        }
