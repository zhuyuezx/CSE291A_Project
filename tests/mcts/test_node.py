# tests/mcts/test_node.py
from src.mcts.node import MCTSNode


def test_node_creation():
    node = MCTSNode(state=None, parent=None, action=None)
    assert node.visits == 0
    assert node.value == 0.0
    assert node.children == []
    assert node.parent is None
    assert node.action is None
    assert node.untried_actions is None


def test_node_uct_unexplored():
    """Unexplored nodes should have infinite UCT value."""
    node = MCTSNode(state=None, parent=None, action=None)
    assert node.uct_value(c=1.41) == float("inf")


def test_node_uct_value():
    """UCT = value/visits + c * sqrt(ln(parent_visits) / visits)"""
    import math

    parent = MCTSNode(state=None, parent=None, action=None)
    parent.visits = 100
    child = MCTSNode(state=None, parent=parent, action=0)
    child.visits = 10
    child.value = 7.0

    c = 1.41
    expected = 7.0 / 10 + c * math.sqrt(math.log(100) / 10)
    assert abs(child.uct_value(c) - expected) < 1e-6


def test_node_puct_value():
    """PUCT = value/visits + c * prior * sqrt(parent_visits) / (visits + 1)"""
    import math

    parent = MCTSNode(state=None, parent=None, action=None)
    parent.visits = 100
    child = MCTSNode(state=None, parent=parent, action=0, prior=0.3)
    child.visits = 10
    child.value = 7.0

    c = 1.41
    expected = 7.0 / 10 + c * 0.3 * math.sqrt(100) / (10 + 1)
    assert abs(child.puct_value(c) - expected) < 1e-6


def test_best_child_by_visits():
    parent = MCTSNode(state=None, parent=None, action=None)
    c1 = MCTSNode(state=None, parent=parent, action=0)
    c1.visits = 50
    c2 = MCTSNode(state=None, parent=parent, action=1)
    c2.visits = 100
    c3 = MCTSNode(state=None, parent=parent, action=2)
    c3.visits = 30
    parent.children = [c1, c2, c3]
    assert parent.best_child_by_visits().action == 1


def test_backpropagate():
    root = MCTSNode(state=None, parent=None, action=None)
    child = MCTSNode(state=None, parent=root, action=0)
    grandchild = MCTSNode(state=None, parent=child, action=1)

    grandchild.backpropagate(value=1.0)
    assert grandchild.visits == 1
    assert grandchild.value == 1.0
    assert child.visits == 1
    assert child.value == 1.0
    assert root.visits == 1
    assert root.value == 1.0
