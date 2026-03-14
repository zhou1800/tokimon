from summary import summarize
from stats import mean, variance
from scoring import score


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_variance_sample():
    assert round(variance([1.0, 2.0, 3.0]), 2) == 1.0


def test_empty_raises():
    try:
        mean([])
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_summary():
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert round(result["variance"], 2) == 1.0


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
