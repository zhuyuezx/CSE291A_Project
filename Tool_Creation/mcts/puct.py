"""
PUCT helpers for plugging policy priors into MCTS.

These helpers are designed for the existing tool API:
    selection(node, exploration_weight) -> node
    expansion(node) -> node
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable


def _softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    if not xs:
        return []
    t = max(float(temperature), 1e-8)
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _normalize_prior_map(actions: list[Any], prior_map: dict[Any, float]) -> dict[Any, float]:
    vals = [max(0.0, float(prior_map.get(a, 0.0))) for a in actions]
    total = sum(vals)
    if total <= 0:
        u = 1.0 / len(actions) if actions else 0.0
        return {a: u for a in actions}
    return {a: v / total for a, v in zip(actions, vals)}


def make_puct_selection(
    prior_fn: Callable[[Any], dict[Any, float]],
    c_puct: float = 1.5,
) -> Callable:
    """
    Build a selection tool that uses PUCT:
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Args:
        prior_fn:
            Function that returns priors for the current state:
            {action: prior_probability}.
        c_puct:
            Exploration scale for the prior bonus.
    """
    c = float(c_puct)

    def puct_selection(node, exploration_weight: float = 1.41):
        # exploration_weight is accepted for compatibility with engine API.
        del exploration_weight

        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node

            # Ensure existing children have priors.
            priors = _normalize_prior_map(
                list(node.children.keys()),
                prior_fn(node.state),
            )
            for action, child in node.children.items():
                child.prior = priors.get(action, 0.0)

            sqrt_parent_visits = math.sqrt(max(1, node.visits))
            best_child = None
            best_score = -math.inf

            for child in node.children.values():
                q_mean = child.value / child.visits if child.visits > 0 else 0.0
                u_bonus = c * child.prior * sqrt_parent_visits / (1 + child.visits)
                score = q_mean + u_bonus
                if score > best_score:
                    best_score = score
                    best_child = child

            node = best_child

        return node

    return puct_selection


def make_puct_expansion(
    prior_fn: Callable[[Any], dict[Any, float]],
    strategy: str = "greedy",
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> Callable:
    """
    Build an expansion tool with configurable action choice:
        - greedy: max prior
        - sample: sample by prior distribution
        - epsilon_greedy: random action with prob epsilon, else max prior
    """
    strategy = str(strategy)
    if strategy not in {"greedy", "sample", "epsilon_greedy"}:
        raise ValueError(f"Unknown expansion strategy '{strategy}'")
    eps = max(0.0, min(1.0, float(epsilon)))
    local_rng = rng or random.Random()

    def _sample_action(actions: list[Any], probs: dict[Any, float]) -> Any:
        # Weighted sample without numpy dependency.
        r = local_rng.random()
        cum = 0.0
        last = actions[-1]
        for a in actions:
            cum += probs.get(a, 0.0)
            if r <= cum:
                return a
        return last

    def puct_expansion(node):
        if not node.untried_actions:
            return node

        actions = list(node.untried_actions)
        prior_map = _normalize_prior_map(actions, prior_fn(node.state))

        if strategy == "greedy":
            action = max(actions, key=lambda a: prior_map.get(a, 0.0))
        elif strategy == "sample":
            action = _sample_action(actions, prior_map)
        else:  # epsilon_greedy
            if local_rng.random() < eps:
                action = local_rng.choice(actions)
            else:
                action = max(actions, key=lambda a: prior_map.get(a, 0.0))

        node.untried_actions.remove(action)

        child_state = node.state.clone()
        child_state.apply_action(action)

        from mcts.node import MCTSNode

        child = MCTSNode(child_state, parent=node, parent_action=action)
        child.prior = float(prior_map.get(action, 0.0))
        node.children[action] = child
        return child

    return puct_expansion


def make_dqn_prior_fn(
    q_model: Any,
    encode_state_fn: Callable[[Any], Any],
    legal_actions_fn: Callable[[Any], list[Any]] | None = None,
    action_to_index_fn: Callable[[Any], int] | None = None,
    temperature: float = 1.0,
) -> Callable[[Any], dict[Any, float]]:
    """
    Adapt a DQN model (Q-values) into a policy-prior function P(s,a).

    Args:
        q_model:
            Callable model producing per-action Q values.
        encode_state_fn:
            Converts GameState -> model input.
        legal_actions_fn:
            Returns legal actions for a state (default: state.legal_actions()).
        action_to_index_fn:
            Maps action -> Q-value index (default assumes action is int index).
        temperature:
            Softmax temperature for converting Q-values to priors.
    """
    get_legal = legal_actions_fn or (lambda s: list(s.legal_actions()))
    to_index = action_to_index_fn or (lambda a: int(a))

    def prior_fn(state) -> dict[Any, float]:
        actions = get_legal(state)
        if not actions:
            return {}

        model_input = encode_state_fn(state)
        q_values = q_model(model_input)

        # Support tensors/ndarrays/lists without hard dependency on torch/numpy.
        if hasattr(q_values, "detach"):
            q_values = q_values.detach()
        if hasattr(q_values, "cpu"):
            q_values = q_values.cpu()
        if hasattr(q_values, "numpy"):
            q_values = q_values.numpy()
        if hasattr(q_values, "tolist"):
            q_values = q_values.tolist()

        # Flatten common nested shapes: [1, A] -> [A]
        if isinstance(q_values, list) and len(q_values) == 1 and isinstance(q_values[0], list):
            q_values = q_values[0]

        logits = [float(q_values[to_index(a)]) for a in actions]
        probs = _softmax(logits, temperature=temperature)
        return {a: p for a, p in zip(actions, probs)}

    return prior_fn
