from config import load_config
from scoring import score


def test_precedence():
    defaults = {"a": 1, "b": 2}
    file_config = {"b": 3, "c": 4}
    env_config = {"c": 5}
    assert load_config(defaults, file_config, env_config) == {"a": 1, "b": 3, "c": 5}


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
