"""
Core MCTS engine.

Standard MCTS loop:  Select -> Expand -> Simulate -> Backpropagate

Each phase is a pluggable "tool" — a single Python function loaded from
MCTS_tools/<phase>/. The engine ships with defaults and supports hot-swap
via set_tool() so the LLM can inject improved versions at runtime.

Tool slots:
    selection(node, exploration_weight)        -> MCTSNode
    expansion(node)                            -> MCTSNode
    simulation(state, player, max_depth)       -> float
    backpropagation(node, reward)              -> None
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, Callable

from .node import MCTSNode
from .games.game_interface import Game, GameState


# ── Locate the MCTS_tools directory ──────────────────────────────────
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "MCTS_tools"

# Phase name -> subfolder
_PHASE_DIRS = {
    "selection":        _TOOLS_DIR / "selection",
    "expansion":        _TOOLS_DIR / "expansion",
    "simulation":       _TOOLS_DIR / "simulation",
    "backpropagation":  _TOOLS_DIR / "backpropagation",
}

# Default file names (loaded at engine construction)
_DEFAULTS = {
    "selection":        "default_selection.py",
    "expansion":        "default_expansion.py",
    "simulation":       "default_simulation.py",
    "backpropagation":  "default_backpropagation.py",
}


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

    PHASES = ("selection", "expansion", "simulation", "backpropagation")

    def __init__(
        self,
        game: Game,
        iterations: int = 100,
        max_rollout_depth: int = 50,
        exploration_weight: float = 1.41,
    ):
        self.game = game
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth
        self.exploration_weight = exploration_weight

        # Load default tools from MCTS_tools/
        self._tools: dict[str, Callable] = {}
        self._tool_paths: dict[str, str] = {}   # phase -> filepath (for source display)
        for phase in self.PHASES:
            default_path = _PHASE_DIRS[phase] / _DEFAULTS[phase]
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
        """Return the source code of each tool (for prompt building)."""
        sources: dict[str, str] = {}
        for phase, fn in self._tools.items():
            try:
                sources[phase] = inspect.getsource(fn)
            except (OSError, TypeError):
                # If loaded dynamically, read the file directly
                path = self._tool_paths.get(phase)
                if path and os.path.isfile(path):
                    sources[phase] = Path(path).read_text()
                else:
                    sources[phase] = f"# source unavailable for {fn.__name__}"
        return sources

    def reset_tool(self, phase: str) -> None:
        """Reset a phase tool to its default from MCTS_tools/."""
        if phase not in self.PHASES:
            raise KeyError(f"Unknown phase '{phase}'.")
        default_path = _PHASE_DIRS[phase] / _DEFAULTS[phase]
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
        root = MCTSNode(root_state.clone())

        select_fn = self._tools["selection"]
        expand_fn = self._tools["expansion"]
        simulate_fn = self._tools["simulation"]
        backprop_fn = self._tools["backpropagation"]

        for _ in range(self.iterations):
            # 1. Selection — walk tree via UCB1
            node = select_fn(root, self.exploration_weight)

            # 2. Expansion — add a child if node has untried actions
            if not node.is_terminal:
                node = expand_fn(node)

            # 3. Simulation — random rollout from the node
            reward = simulate_fn(
                node.state,
                root_state.current_player(),
                self.max_rollout_depth,
            )

            # 4. Backpropagation — update stats from leaf to root
            backprop_fn(node, reward)

        return root.most_visited_child().parent_action

    # ------------------------------------------------------------------
    # High-level play helpers
    # ------------------------------------------------------------------

    def play_game(self, verbose: bool = False) -> dict:
        """
        Play one full game from initial state using MCTS for every move.

        Returns:
            Dict with keys: solved, steps, returns, moves.
        """
        state = self.game.new_initial_state()
        moves: list[Any] = []

        while not state.is_terminal():
            action = self.search(state)
            state.apply_action(action)
            moves.append(action)

            if verbose:
                print(f"Move {len(moves)}: action={action}")
                print(state)
                print()

        solved = state.returns()[0] > 0  # player 0 won / solved
        return {
            "solved": solved,
            "steps": len(moves),
            "returns": state.returns(),
            "moves": moves,
        }

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
