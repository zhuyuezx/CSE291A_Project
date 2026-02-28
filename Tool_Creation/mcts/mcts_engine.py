"""
Core MCTS engine with pluggable heuristic functions.

This mirrors the connect_four MCTS_TT but is game-agnostic and accepts
callable heuristics that the LLM agent can swap out.
"""

from __future__ import annotations

import time
import random
from typing import Any, Callable

from .game_interface import Game, GameState
from .node import MCTSNode
from .trace_logger import TraceLogger, GameRecord
from . import heuristics as default_heuristics


class MCTSEngine:
    """
    Monte-Carlo Tree Search with transposition table and hot-swappable
    heuristic functions.

    Heuristic slots (the LLM optimises these):
        rollout_policy(state) -> action
        evaluation(state, perspective_player) -> float | None
        exploration_weight(root_visits) -> float
        action_priority(state, actions) -> list[action]
    """

    def __init__(
        self,
        game: Game,
        iterations: int = 1000,
        *,
        rollout_policy: Callable | None = None,
        evaluation: Callable | None = None,
        exploration_weight: Callable | None = None,
        action_priority: Callable | None = None,
        max_rollout_depth: int = 200,
    ):
        self.game = game
        self.iterations = iterations
        self.max_rollout_depth = max_rollout_depth

        # Pluggable heuristics — fall back to defaults
        self.rollout_policy = rollout_policy or default_heuristics.random_rollout_policy
        self.evaluation = evaluation or default_heuristics.null_evaluation
        self.exploration_weight_fn = exploration_weight or default_heuristics.default_exploration_weight
        self.action_priority = action_priority or default_heuristics.default_action_priority

        # Transposition table: state_key → MCTSNode
        self.transposition_table: dict[str, MCTSNode] = {}

        # Trace logger
        self.logger = TraceLogger()

    # ------------------------------------------------------------------
    # Transposition table
    # ------------------------------------------------------------------

    def _get_node(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        parent_action: Any | None = None,
    ) -> MCTSNode:
        key = state.state_key()
        if key not in self.transposition_table:
            node = MCTSNode(state.clone(), parent_action, parent)
            # Apply action ordering heuristic
            node.untried_actions = self.action_priority(
                state, node.untried_actions
            )
            self.transposition_table[key] = node
        return self.transposition_table[key]

    def clear_table(self):
        """Clear the transposition table (call between games if desired)."""
        self.transposition_table.clear()

    # ------------------------------------------------------------------
    # Core MCTS phases
    # ------------------------------------------------------------------

    def search(self, state: GameState) -> Any:
        """
        Run MCTS from *state* and return the best action.

        Phases per iteration:
            1. Selection  — walk the tree using UCB1
            2. Expansion  — add one child
            3. Simulation — rollout to terminal (or use evaluation)
            4. Backprop   — propagate reward up the tree
        """
        root = self._get_node(state)
        perspective = state.current_player()

        for _ in range(self.iterations):
            node = root
            ew = self.exploration_weight_fn(root.visits)

            # 1. Selection
            while node.is_fully_expanded() and not node.state.is_terminal():
                node = node.best_child(ew)

            # 2. Expansion
            if not node.state.is_terminal() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                next_state = node.state.clone()
                next_state.apply_action(action)
                child = self._get_node(next_state, node, action)
                node.children[action] = child
                node = child

            # 3. Simulation / Evaluation
            reward = self._simulate(node.state, perspective)

            # 4. Backpropagation
            temp = node
            while temp is not None:
                temp.visits += 1
                temp.value += reward
                if temp.parent is None or temp == root.parent:
                    break
                temp = temp.parent

        best_action, _ = root.most_visited_child()
        return best_action

    def _simulate(self, state: GameState, perspective: int) -> float:
        """
        Rollout from *state* to a terminal or until the evaluation
        function returns a value.
        """
        sim_state = state.clone()
        depth = 0

        while not sim_state.is_terminal() and depth < self.max_rollout_depth:
            # Check if the evaluation function wants to short-circuit
            val = self.evaluation(sim_state, perspective)
            if val is not None:
                return val

            action = self.rollout_policy(sim_state)
            sim_state.apply_action(action)
            depth += 1

        if sim_state.is_terminal():
            return sim_state.returns()[perspective]
        # Reached depth limit without terminal — treat as draw
        return 0.0

    # ------------------------------------------------------------------
    # High-level game runner with trace logging
    # ------------------------------------------------------------------

    def play_game(
        self,
        mcts_player: int = 0,
        opponent_policy: Callable[[GameState], Any] | None = None,
        clear_table_each_game: bool = False,
        verbose: bool = False,
    ) -> GameRecord:
        """
        Play a full game: MCTS vs opponent.

        Args:
            mcts_player:     Which player index MCTS controls.
            opponent_policy: Callable(state) -> action for the opponent.
                             Defaults to uniform random.
            clear_table_each_game: Whether to wipe the TT before the game.
            verbose:         Print board each turn.

        Returns:
            A GameRecord with the full trace.
        """
        if clear_table_each_game:
            self.clear_table()

        if opponent_policy is None:
            opponent_policy = lambda s: random.choice(s.legal_actions())

        state = self.game.new_initial_state()
        game_rec = self.logger.new_game(self.game.name(), mcts_player)
        t0 = time.time()

        while not state.is_terminal():
            if state.current_player() == mcts_player:
                action = self.search(state)
                # Log the MCTS decision
                root = self._get_node(state)
                TraceLogger.record_move(game_rec, state, action, root)
            else:
                action = opponent_policy(state)

            if verbose:
                print(f"Turn {game_rec.total_moves + 1} | "
                      f"Player {state.current_player()} → action {action}")
                print(state)
                print()

            state.apply_action(action)

        elapsed = time.time() - t0
        TraceLogger.finalise_game(game_rec, state, elapsed)

        if verbose:
            print(f"Game over. Outcome: {game_rec.outcome} "
                  f"(winner: Player {game_rec.winner})")

        return game_rec

    def play_many(
        self,
        num_games: int = 100,
        mcts_player: int = 0,
        opponent_policy: Callable[[GameState], Any] | None = None,
        clear_table_each_game: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Play multiple games and return aggregate stats.

        Returns:
            A dict with total_games, wins, losses, draws, win_rate.
        """
        for i in range(num_games):
            rec = self.play_game(
                mcts_player=mcts_player,
                opponent_policy=opponent_policy,
                clear_table_each_game=clear_table_each_game,
                verbose=verbose,
            )
            if verbose or (i + 1) % max(1, num_games // 10) == 0:
                print(f"  Game {i+1}/{num_games}: "
                      f"winner=P{rec.winner}, outcome={rec.outcome}")

        return self.logger.summary()

    # ------------------------------------------------------------------
    # Heuristic hot-swap
    # ------------------------------------------------------------------

    def set_heuristic(self, name: str, fn: Callable):
        """
        Replace a heuristic function at runtime.

        Args:
            name: One of 'rollout_policy', 'evaluation',
                  'exploration_weight', 'action_priority'.
            fn:   The new callable.
        """
        valid = {
            "rollout_policy", "evaluation",
            "exploration_weight", "action_priority",
        }
        if name not in valid:
            raise ValueError(f"Unknown heuristic '{name}'. Choose from {valid}")
        if name == "exploration_weight":
            self.exploration_weight_fn = fn
        else:
            setattr(self, name, fn)

    def get_heuristic_source(self) -> dict[str, str]:
        """
        Return the source code of the currently active heuristic functions.
        Useful for including in an LLM prompt.

        Falls back to repr() if inspect.getsource fails (e.g. for lambdas
        or dynamically generated code).
        """
        import inspect

        def _safe_source(fn):
            try:
                return inspect.getsource(fn)
            except (OSError, TypeError):
                return f"# [source unavailable — dynamically defined]\n# repr: {fn!r}"

        return {
            "rollout_policy": _safe_source(self.rollout_policy),
            "evaluation": _safe_source(self.evaluation),
            "exploration_weight": _safe_source(self.exploration_weight_fn),
            "action_priority": _safe_source(self.action_priority),
        }
