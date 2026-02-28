"""
Gameplay trace logger for LLM analysis.

Records every decision and outcome so the LLM agent can study what went
wrong (or right) and propose improved heuristic functions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .game_interface import GameState


@dataclass
class MoveRecord:
    """One move in a game."""
    turn: int
    player: int
    state_key: str
    state_str: str
    legal_actions: list[Any]
    chosen_action: Any
    mcts_visits: int          # root visits when decision was made
    top_actions: list[dict]   # [{action, visits, value}, ...] — top-N children


@dataclass
class GameRecord:
    """Full trace of one game."""
    game_id: int
    game_name: str
    mcts_player: int
    outcome: list[float]         # returns() per player
    winner: int | None           # player index or None for draw
    total_moves: int
    moves: list[MoveRecord] = field(default_factory=list)
    elapsed_sec: float = 0.0


class TraceLogger:
    """
    Accumulates GameRecords across multiple games.

    Provides summaries and serialisation for feeding to the LLM.
    """

    def __init__(self):
        self.games: list[GameRecord] = []
        self._game_counter = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def new_game(self, game_name: str, mcts_player: int) -> GameRecord:
        self._game_counter += 1
        rec = GameRecord(
            game_id=self._game_counter,
            game_name=game_name,
            mcts_player=mcts_player,
            outcome=[],
            winner=None,
            total_moves=0,
        )
        self.games.append(rec)
        return rec

    @staticmethod
    def record_move(
        game_rec: GameRecord,
        state: GameState,
        chosen_action: Any,
        root_node,          # MCTSNode (avoid circular import)
        top_n: int = 5,
    ):
        """Append a move to the game record."""
        # Collect top-N children by visits
        top_children = sorted(
            root_node.children.items(),
            key=lambda kv: kv[1].visits,
            reverse=True,
        )[:top_n]
        top_actions = [
            {
                "action": act,
                "visits": child.visits,
                "avg_value": round(child.value / max(child.visits, 1), 4),
            }
            for act, child in top_children
        ]

        move = MoveRecord(
            turn=game_rec.total_moves + 1,
            player=state.current_player(),
            state_key=state.state_key(),
            state_str=str(state),
            legal_actions=state.legal_actions(),
            chosen_action=chosen_action,
            mcts_visits=root_node.visits,
            top_actions=top_actions,
        )
        game_rec.moves.append(move)
        game_rec.total_moves += 1

    @staticmethod
    def finalise_game(game_rec: GameRecord, terminal_state: GameState, elapsed: float):
        """Fill in outcome fields once the game ends."""
        game_rec.outcome = terminal_state.returns()
        game_rec.elapsed_sec = round(elapsed, 3)
        best = max(range(len(game_rec.outcome)), key=lambda i: game_rec.outcome[i])
        if game_rec.outcome[best] > 0:
            game_rec.winner = best
        else:
            game_rec.winner = None  # draw

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate win/loss/draw stats."""
        wins = sum(1 for g in self.games if g.winner == g.mcts_player)
        losses = sum(
            1 for g in self.games
            if g.winner is not None and g.winner != g.mcts_player
        )
        draws = sum(1 for g in self.games if g.winner is None)
        return {
            "total_games": len(self.games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / max(len(self.games), 1), 4),
        }

    def losing_games(self) -> list[GameRecord]:
        """Return only the games the MCTS player lost — most useful to analyse."""
        return [
            g for g in self.games
            if g.winner is not None and g.winner != g.mcts_player
        ]

    # ------------------------------------------------------------------
    # Serialisation (for LLM consumption)
    # ------------------------------------------------------------------

    def to_json(self, include_states: bool = True) -> str:
        """Serialise all games to JSON."""
        data = []
        for g in self.games:
            gd = asdict(g)
            if not include_states:
                for m in gd["moves"]:
                    del m["state_str"]
                    del m["state_key"]
            data.append(gd)
        return json.dumps(data, indent=2, default=str)

    def unsolved_games(self) -> list[GameRecord]:
        """Return games that were not won — unsolved puzzles or draws."""
        return [g for g in self.games if g.winner is None]

    def format_for_llm(self, max_games: int = 5, losses_only: bool = True) -> str:
        """
        Produce a concise text summary suitable for an LLM prompt.

        Args:
            max_games:   Maximum number of games to include in detail.
            losses_only: If True, only include losing/unsolved games.
                         For single-player puzzles, this includes
                         unsolved games (draws) when there are no losses.

        Returns:
            A human-readable report string.
        """
        if losses_only:
            games = self.losing_games()
            if not games:  # No losses — include unsolved (puzzles)
                games = self.unsolved_games()
        else:
            games = self.games
        games = games[:max_games]

        lines = [
            "=" * 60,
            f"MCTS GAMEPLAY TRACE  ({self.summary()})",
            "=" * 60,
        ]

        for g in games:
            lines.append(f"\n--- Game {g.game_id} | Winner: Player {g.winner} "
                         f"| MCTS was Player {g.mcts_player} "
                         f"| Moves: {g.total_moves} | Time: {g.elapsed_sec}s ---")
            for m in g.moves:
                if m.player == g.mcts_player:
                    lines.append(
                        f"  Turn {m.turn}: P{m.player} chose action={m.chosen_action} "
                        f"(visits={m.mcts_visits}, "
                        f"top={m.top_actions[:3]})"
                    )
                    lines.append(f"    Board:\n{_indent(m.state_str, 6)}")
            lines.append(f"  Outcome: {g.outcome}")

        return "\n".join(lines)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())
