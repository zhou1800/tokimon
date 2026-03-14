from intervals import merge_intervals
from scoring import score


def test_sorted_merge():
    assert merge_intervals([(1, 3), (2, 4), (5, 6)]) == [(1, 4), (5, 6)]


def test_unsorted_merge():
    assert merge_intervals([(5, 6), (1, 2), (2, 4)]) == [(1, 4), (5, 6)]


def test_touching_merge():
    assert merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
