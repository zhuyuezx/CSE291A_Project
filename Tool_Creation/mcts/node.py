"""
MCTS Node with transposition-table support.

Closely follows the connect_four MCTSNode_TT structure but is
game-agnostic — works with any GameState.
"""

from __future__ import annotations

import math
from typing import Any

from .game_interface import GameState


class MCTSNode:
    """A single node in the MCTS search tree."""

    __slots__ = (
        "state", "parent_action", "parent",
        "children", "visits", "value", "untried_actions",
    )

    def __init__(
        self,
        state: GameState,
        parent_action: Any | None = None,
        parent: "MCTSNode | None" = None,
    ):
        self.state = state
        self.parent_action = parent_action
        self.parent = parent
        self.children: dict[Any, MCTSNode] = {}
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: list[Any] = (
            state.legal_actions() if state is not None else []
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight: float = 1.4) -> "MCTSNode":
        """Select the child with the highest UCB1 score."""
        best_score = float("-inf")
        best = None
        for child in self.children.values():
            exploit = child.value / child.visits
            explore = math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + exploration_weight * explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None, "best_child called on node with no children"
        return best

    def most_visited_child(self) -> tuple[Any, "MCTSNode"]:
        """Return (action, child) with the most visits — used to pick the final move."""
        return max(self.children.items(), key=lambda item: item[1].visits)
