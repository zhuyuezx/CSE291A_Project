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
            4. Backprop   — propagate reward along the selected path

        Uses explicit path tracking (rather than parent pointers) so that
        transposition-table sharing and graph cycles are handled correctly.
        This is essential for single-player puzzles with reversible moves.
        """
        root = self._get_node(state)
        perspective = state.current_player()

        for _ in range(self.iterations):
            node = root
            path = [root]
            visited_keys = {root.state.state_key()}
            ew = self.exploration_weight_fn(root.visits)

            # 1. Selection — stop on cycle or unexpanded / terminal node
            while node.is_fully_expanded() and not node.state.is_terminal():
                node = node.best_child(ew)
                key = node.state.state_key()
                if key in visited_keys:
                    break                    # cycle in the graph
                visited_keys.add(key)
                path.append(node)

            # 2. Expansion
            if not node.state.is_terminal() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                next_state = node.state.clone()
                next_state.apply_action(action)
                child = self._get_node(next_state, node, action)
                node.children[action] = child
                node = child
                path.append(node)

            # 3. Simulation / Evaluation
            reward = self._simulate(node.state, perspective)

            # 4. Backpropagation — walk the explicit path
            for n in path:
                n.visits += 1
                n.value += reward

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
        clear_table_each_move: bool = False,
        max_game_moves: int | None = None,
        verbose: bool = False,
    ) -> GameRecord:
        """
        Play a full game: MCTS vs opponent (two-player) or MCTS
        solving a puzzle (single-player).

        For single-player puzzles, current_player() always returns 0
        and the opponent branch is never executed.

        Args:
            mcts_player:     Which player index MCTS controls.
            opponent_policy: Callable(state) -> action for the opponent.
                             Defaults to uniform random. Ignored for
                             single-player games.
            clear_table_each_game: Whether to wipe the TT before the game.
            clear_table_each_move: Whether to wipe the TT between moves
                                   (useful for puzzles where tree reuse
                                   can waste memory).
            max_game_moves:  Hard limit on total moves (safety net on top
                             of the game's own is_terminal logic).
            verbose:         Print board each turn.

        Returns:
            A GameRecord with the full trace.
        """
        if clear_table_each_game:
            self.clear_table()

        is_single_player = self.game.num_players() == 1

        if opponent_policy is None and not is_single_player:
            opponent_policy = lambda s: random.choice(s.legal_actions())

        state = self.game.new_initial_state()
        game_rec = self.logger.new_game(self.game.name(), mcts_player)
        t0 = time.time()
        move_count = 0

        while not state.is_terminal():
            if max_game_moves is not None and move_count >= max_game_moves:
                break

            if state.current_player() == mcts_player:
                if clear_table_each_move:
                    self.clear_table()
                action = self.search(state)
                # Log the MCTS decision
                root = self._get_node(state)
                TraceLogger.record_move(game_rec, state, action, root)
            else:
                action = opponent_policy(state)

            if verbose:
                print(f"Turn {move_count + 1} | "
                      f"Player {state.current_player()} → action {action}")
                print(state)
                print()

            state.apply_action(action)
            move_count += 1

        elapsed = time.time() - t0
        TraceLogger.finalise_game(game_rec, state, elapsed)

        if verbose:
            outcome_msg = (f"Solved!" if is_single_player and game_rec.winner == 0
                           else f"Outcome: {game_rec.outcome} (winner: Player {game_rec.winner})")
            print(f"Game over. {outcome_msg}")

        return game_rec

    def play_many(
        self,
        num_games: int = 100,
        mcts_player: int = 0,
        opponent_policy: Callable[[GameState], Any] | None = None,
        clear_table_each_game: bool = False,
        clear_table_each_move: bool = False,
        max_game_moves: int | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Play multiple games and return aggregate stats.

        Returns:
            A dict with total_games, wins, losses, draws, win_rate.
            For single-player puzzles, wins = solved, draws = unsolved.
        """
        is_single_player = self.game.num_players() == 1
        for i in range(num_games):
            rec = self.play_game(
                mcts_player=mcts_player,
                opponent_policy=opponent_policy,
                clear_table_each_game=clear_table_each_game,
                clear_table_each_move=clear_table_each_move,
                max_game_moves=max_game_moves,
                verbose=verbose,
            )
            if verbose or (i + 1) % max(1, num_games // 10) == 0:
                label = ("solved" if is_single_player and rec.winner == 0
                         else "unsolved" if is_single_player
                         else f"winner=P{rec.winner}")
                print(f"  Game {i+1}/{num_games}: {label}, "
                      f"moves={rec.total_moves}")

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
