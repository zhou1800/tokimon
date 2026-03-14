from __future__ import annotations

from types import SimpleNamespace

import pytest

from self_improve.source import extract_urls, read_goal_input


def test_extract_urls_trims_trailing_punctuation() -> None:
    urls = extract_urls("Read https://example.com/article, then summarize.")
    assert urls == ["https://example.com/article"]


def test_read_goal_input_fetches_goal_url_when_input_missing(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.headers = SimpleNamespace(get=lambda _k, _d=None: None)

        def read(self, _n: int) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_urlopen(_req, timeout: float):
        assert timeout == 15
        return DummyResponse(b"hello")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = read_goal_input("Read https://example.com/a", input_ref=None)
    assert payload.kind == "url"
    assert payload.ref == "https://example.com/a"
    assert payload.content == "hello"


def test_read_goal_input_prefers_explicit_input_over_goal_url(monkeypatch) -> None:
    called = {"count": 0}

    def fake_urlopen(_req, timeout: float):
        called["count"] += 1
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    payload = read_goal_input("Read https://example.com/a", input_ref="inline text")
    assert payload.kind == "text"
    assert called["count"] == 0


def test_read_goal_input_respects_webtool_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_WEB_ORG_ALLOWLIST", "example.com")

    def fake_urlopen(_req, timeout: float):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="allowlist"):
        read_goal_input("Read https://evil.com/a", input_ref=None)
