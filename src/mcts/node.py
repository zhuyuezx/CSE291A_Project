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
