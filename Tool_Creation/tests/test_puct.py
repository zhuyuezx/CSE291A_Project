import math
import random

from mcts import MCTSNode
from mcts.games.tic_tac_toe import TicTacToeState
from mcts.puct import make_dqn_prior_fn, make_puct_expansion, make_puct_selection


class TestPUCTExpansion:
    def test_expands_highest_prior_action_first(self):
        state = TicTacToeState()
        root = MCTSNode(state)

        prior = {a: 0.01 for a in state.legal_actions()}
        prior[4] = 0.9

        expand = make_puct_expansion(lambda s: prior)
        child = expand(root)

        assert child.parent_action == 4
        assert 0.0 < child.prior <= 1.0
        assert 4 in root.children

    def test_sample_strategy_respects_rng(self):
        state = TicTacToeState()
        root = MCTSNode(state)
        prior = {a: 0.0 for a in state.legal_actions()}
        prior[2] = 1.0

        expand = make_puct_expansion(
            lambda s: prior,
            strategy="sample",
            rng=random.Random(0),
        )
        child = expand(root)
        assert child.parent_action == 2


class TestPUCTSelection:
    def test_prefers_higher_prior_when_q_is_tied(self):
        root_state = TicTacToeState()
        root = MCTSNode(root_state)

        # Create two expanded children with tied mean value.
        c0_state = root_state.clone()
        c0_state.apply_action(0)
        c0 = MCTSNode(c0_state, parent=root, parent_action=0)
        c0.visits = 20
        c0.value = 10.0

        c1_state = root_state.clone()
        c1_state.apply_action(1)
        c1 = MCTSNode(c1_state, parent=root, parent_action=1)
        c1.visits = 20
        c1.value = 10.0

        root.children = {0: c0, 1: c1}
        root._untried_actions = []
        root.visits = c0.visits + c1.visits

        priors = {0: 0.1, 1: 0.9}
        select = make_puct_selection(lambda s: priors, c_puct=2.0)
        selected = select(root, exploration_weight=1.41)

        assert selected is c1


class TestDQNPriorAdapter:
    def test_maps_q_values_to_legal_action_priors(self):
        state = TicTacToeState()
        state.apply_action(4)
        state.apply_action(0)
        legal = set(state.legal_actions())

        # 9-action Q output; action 4 has high Q but is illegal now.
        q_values = [0.2, 0.5, 0.1, 0.0, 9.9, -0.3, 0.8, 0.4, 0.6]
        q_model = lambda x: q_values
        encode_state = lambda s: s.board

        prior_fn = make_dqn_prior_fn(q_model, encode_state, temperature=1.0)
        prior = prior_fn(state)

        assert set(prior.keys()) == legal
        assert math.isclose(sum(prior.values()), 1.0, rel_tol=1e-6)
        best_action = max(prior, key=prior.get)
        assert best_action == 6
