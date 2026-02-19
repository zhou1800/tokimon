from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

from chat_ui.server import ChatUIConfig, ChatUIServer


def _get_json(url: str, *, timeout_s: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, *, timeout_s: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
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


def test_chat_ui_healthz_and_send(tmp_path: Path) -> None:
    config = ChatUIConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=tmp_path)
    try:
        server = ChatUIServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        payload = _post_json(
            f"{server.url}/api/send",
            {"message": "hello", "history": []},
        )
        assert payload["ok"] is True
        assert payload["reply"] == "mock response"
        assert payload["status"] in {"PARTIAL", "SUCCESS", "FAILURE", "BLOCKED"}
    finally:
        server.stop()
