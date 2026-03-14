from scoring import score


def test_request_signature():
    from client import request

    result = request("/ping", {"ok": True})
    assert result["endpoint"] == "/ping"
    assert result["data"] == {"ok": True}
    assert result["timeout"] == 5


def test_fetch_user():
    from service import fetch_user

    result = fetch_user(42)
    assert result["endpoint"] == "/users/42"
    assert result["data"] == {"id": 42}


def test_score_bounds():
    assert 0.0 <= score() <= 1.0
