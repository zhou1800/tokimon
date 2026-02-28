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


def test_logs_cli_json_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"

        exit_code = cli.main(["logs", "--url", url, "--json", "--limit", "1"])
        assert exit_code == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert isinstance(payload.get("entries"), list)
        assert len(payload["entries"]) == 1

        entry = payload["entries"][0]
        assert isinstance(entry, dict)
        assert isinstance(entry.get("id"), int)
        assert isinstance(entry.get("ts_ms"), int)
        assert isinstance(entry.get("event"), str)
    finally:
        server.stop()
