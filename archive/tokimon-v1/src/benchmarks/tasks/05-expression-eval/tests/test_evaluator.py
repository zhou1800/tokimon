import pytest

from evaluator import evaluate
from scoring import score


def test_precedence():
    assert evaluate("1 + 2 * 3") == 7


def test_parentheses():
    assert evaluate("( 1 + 2 ) * 3") == 9


def test_division():
    assert evaluate("8 / 2 + 1") == 5


def test_invalid():
    with pytest.raises(ValueError):
        evaluate("1 +")


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
