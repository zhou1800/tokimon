from agents.delegation import DelegationGraph


def test_cycle_detection():
    graph = DelegationGraph()
    assert graph.add_edge("a", "b") is True
    assert graph.add_edge("b", "c") is True
    assert graph.add_edge("c", "a") is False
