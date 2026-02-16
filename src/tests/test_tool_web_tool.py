from __future__ import annotations

from types import SimpleNamespace

from tools.web_tool import WebTool


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


def test_web_tool_rejects_non_http_urls() -> None:
    tool = WebTool()
    result = tool.fetch("file:///etc/passwd")
    assert result.ok is False


def test_web_tool_rejects_localhost() -> None:
    tool = WebTool()
    result = tool.fetch("http://localhost:8000/")
    assert result.ok is False


def test_web_tool_allows_domains_in_org_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_WEB_ORG_ALLOWLIST", "example.com")

    def fake_urlopen(req, timeout: float):
        assert timeout == 15.0
        assert req.full_url == "https://example.com/"
        return DummyResponse(b"ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    tool = WebTool()
    result = tool.fetch("https://example.com/")
    assert result.ok is True
    assert result.data["content"] == "ok"


def test_web_tool_blocks_domains_outside_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_WEB_ORG_ALLOWLIST", "example.com")

    def fake_urlopen(_req, timeout: float):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    tool = WebTool()
    result = tool.fetch("https://evil.com/")
    assert result.ok is False
    assert "allowlist" in (result.error or "")


def test_web_tool_request_allowlist_must_subset_org_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_WEB_ORG_ALLOWLIST", "example.com")
    monkeypatch.setenv("TOKIMON_WEB_REQUEST_ALLOWLIST", "evil.com")

    def fake_urlopen(_req, timeout: float):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    tool = WebTool()
    result = tool.fetch("https://example.com/")
    assert result.ok is False
    assert "subset of org allowlist" in (result.error or "")


def test_web_tool_domain_secrets_injected(monkeypatch) -> None:
    monkeypatch.setenv("TOKIMON_WEB_ORG_ALLOWLIST", "example.com")
    monkeypatch.setenv(
        "TOKIMON_WEB_DOMAIN_SECRETS_JSON",
        '{"example.com": {"Authorization": "Bearer $API_KEY"}}',
    )
    monkeypatch.setenv("API_KEY", "supersecret")

    def fake_urlopen(req, timeout: float):
        assert timeout == 15.0
        assert req.get_header("Authorization") == "Bearer supersecret"
        return DummyResponse(b"ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    tool = WebTool()
    result = tool.fetch("https://example.com/")
    assert result.ok is True
    assert "supersecret" not in (result.data.get("content") or "")
