from dedupe import dedupe_sorted
from scoring import score


class OpCounter:
    def __init__(self) -> None:
        self.count = 0

    def tick(self) -> None:
        self.count += 1


def test_dedupe_correctness():
    assert dedupe_sorted([1, 1, 2, 2, 3, 3]) == [1, 2, 3]


def test_dedupe_ops():
    items = list(range(50)) + list(range(50))
    counter = OpCounter()
    dedupe_sorted(items, ops=counter.tick)
    assert counter.count <= len(items) + 5


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
