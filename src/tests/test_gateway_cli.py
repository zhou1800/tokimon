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


def test_gateway_run_parses() -> None:
    parser = cli.build_parser(exit_on_error=False)
    args = parser.parse_args(["gateway", "run"])
    assert args.command == "gateway"
    assert args.gateway_command == "run"


def test_gateway_health_alias_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"

        exit_code = cli.main(["gateway", "health", "--url", url, "--json"])
        assert exit_code == 0
        gateway_payload = json.loads(capsys.readouterr().out)

        exit_code = cli.main(["health", "--url", url, "--json"])
        assert exit_code == 0
        health_payload = json.loads(capsys.readouterr().out)

        for payload in (gateway_payload, health_payload):
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


def test_gateway_call_health_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"

        exit_code = cli.main(["gateway", "call", "health", "--url", url, "--json"])
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {"type": "res", "id": "2", "ok": True, "payload": {"ok": True}}
    finally:
        server.stop()


def test_gateway_probe_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        url = f"ws://{server.host}:{server.port}/gateway"
        exit_code = cli.main(["gateway", "probe", "--url", url, "--json"])
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

