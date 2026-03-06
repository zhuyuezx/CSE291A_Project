"""
MCTS tree node — pure data structure.

Each node stores:
    - visit count and cumulative value
    - parent link and children dict (action -> child node)
    - the game state snapshot
    - unexpanded actions remaining

All MCTS logic (selection, expansion, simulation, backpropagation)
lives in the tool files under MCTS_tools/.
"""

from __future__ import annotations

from typing import Any


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = (
        "state", "parent", "parent_action",
        "children", "_untried_actions",
        "visits", "value",
    )

    def __init__(
        self,
        state,            # GameState — not type-hinted to avoid circular import
        parent: MCTSNode | None = None,
        parent_action: Any = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children: dict[Any, MCTSNode] = {}
        self._untried_actions: list[Any] = list(state.legal_actions())
        self.visits: int = 0
        self.value: float = 0.0       # cumulative value (sum of backprop'd rewards)

    # ── Tree policy helpers ──────────────────────────────────────────

    @property
    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    @property
    def untried_actions(self) -> list[Any]:
        return self._untried_actions

    def most_visited_child(self) -> MCTSNode:
        """Return the child with the most visits (used for final move selection)."""
        return max(self.children.values(), key=lambda c: c.visits)

    def __repr__(self) -> str:
        return (
            f"MCTSNode(visits={self.visits}, "
            f"value={self.value:.2f}, "
            f"children={len(self.children)}, "
            f"untried={len(self._untried_actions)})"
        )
