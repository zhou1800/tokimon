from __future__ import annotations

import json
import socket
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


def _reserve_port() -> tuple[int, socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1]), sock


def test_cli_health_ok_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"
        exit_code = cli.main(["health", "--url", url, "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "details": {},
            "elapsed_ms": payload["elapsed_ms"],
            "error": None,
            "ok": True,
            "url": url,
        }
        assert isinstance(payload["elapsed_ms"], int)
        assert payload["elapsed_ms"] >= 0
    finally:
        server.stop()


def test_cli_health_unreachable_port_json(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        port, held = _reserve_port()
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    try:
        url = f"ws://127.0.0.1:{port}/gateway"
        exit_code = cli.main(["health", "--url", url, "--json", "--timeout-ms", "200"])
        assert exit_code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["url"] == url
        assert isinstance(payload["elapsed_ms"], int)
        assert payload["elapsed_ms"] >= 0
        assert isinstance(payload["error"], str)
        assert payload["error"]
        assert isinstance(payload["details"], dict)
    finally:
        held.close()
