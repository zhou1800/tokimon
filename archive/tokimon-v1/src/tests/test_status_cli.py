from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

import cli
from gateway.server import GatewayConfig, GatewayServer


def _get_json(url: str, *, timeout_s: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_healthz(base_url: str, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            payload = _get_json(f"{base_url}/healthz")
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
            continue
        if payload.get("ok") is True:
            return
        time.sleep(0.05)
    if last_error:
        raise AssertionError(f"healthz not ready: {last_error}") from last_error
    raise AssertionError("healthz not ready")


def test_status_json_sections_are_present_and_stable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")

    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"

        exit_code = cli.main(["status", "--url", url, "--json"])
        assert exit_code == 0
        first = json.loads(capsys.readouterr().out)

        exit_code = cli.main(["status", "--url", url, "--json"])
        assert exit_code == 0
        second = json.loads(capsys.readouterr().out)

        required_sections = {"ok", "doctor", "gateway", "memory", "sessions"}
        assert required_sections.issubset(first.keys())
        assert set(first.keys()) == set(second.keys())

        for payload in (first, second):
            assert isinstance(payload["ok"], bool)
            assert isinstance(payload["doctor"], dict)
            assert isinstance(payload["gateway"], dict)
            assert isinstance(payload["memory"], dict)
            assert isinstance(payload["sessions"], dict)

            assert payload["gateway"]["ok"] is True
            assert payload["gateway"]["url"] == url
            assert isinstance(payload["gateway"]["elapsed_ms"], int)
            assert payload["gateway"]["elapsed_ms"] >= 0

            assert payload["memory"]["ok"] is True
            assert isinstance(payload["sessions"]["count"], int)
            assert isinstance(payload["sessions"]["sessions"], list)

        for section in ("doctor", "gateway", "memory", "sessions"):
            assert set(first[section].keys()) == set(second[section].keys())
    finally:
        server.stop()
