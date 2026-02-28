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
        # Walk up to find the root player
        n = node
        while n.parent is not None:
            n = n.parent
        root_player = n.state.current_player()

        evaluators = self.registry.get_tools(ToolType.STATE_EVALUATOR)
        rollout_policies = self.registry.get_tools(ToolType.ROLLOUT_POLICY)
        has_aux = hasattr(self.adapter, "aux_reward")

        depth = 0
        while not self.adapter.is_terminal(state) and depth < self.max_rollout_depth:
            # Early cutoff via state evaluator
            if evaluators and depth > 5:
                try:
                    scores = [t.run_fn(state) for t in evaluators]
                    avg_score = sum(scores) / len(scores)
                    if abs(avg_score) > 0.9:
                        return avg_score if self.adapter.meta.is_single_player else (
                            avg_score if state.current_player() == root_player else -avg_score
                        )
                except Exception:
                    pass

            legal = self.adapter.legal_actions(state)
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

        if self.adapter.is_terminal(state):
            raw = self.adapter.returns(state)
            # single-player → raw is [score]; two-player → [p0, p1]
            val = raw[0] if self.adapter.meta.is_single_player else raw[root_player]
            return self.adapter.normalize_return(val)

        # Non-terminal cutoff: prefer aux_reward, then evaluators, then 0
        if has_aux:
            try:
                return self.adapter.aux_reward(state)
            except Exception:
                pass
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
