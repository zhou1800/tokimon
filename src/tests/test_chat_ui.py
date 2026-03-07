from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from chat_ui.server import ChatUIConfig, ChatUIServer
import llm.client as llm_client
from llm.client import MockLLMClient


def _get_json(url: str, *, timeout_s: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_html(url: str, *, timeout_s: float = 2.0) -> tuple[str, str]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        content_type = response.headers.get("content-type") or ""
        body = response.read().decode("utf-8", errors="replace")
    return content_type, body


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
    config = ChatUIConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = ChatUIServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server._server.llm_client = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "hello from scripted llm",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            }
        ]
    )
    server.start()
    try:
        _wait_for_healthz(server.url)
        content_type, body = _get_html(f"{server.url}/")
        assert "text/html" in content_type
        assert "<!doctype html" in body.lower()
        payload = _post_json(
            f"{server.url}/api/send",
            {"message": "hello", "history": [], "model": ""},
        )
        assert payload["ok"] is True
        assert payload["reply"] == "hello from scripted llm"
        assert payload["status"] == "SUCCESS"
        assert payload["summary"] == "hello from scripted llm"
        assert isinstance(payload["artifacts"], list)
        assert isinstance(payload["metrics"], dict)
        assert isinstance(payload["next_actions"], list)
        assert isinstance(payload["failure_signature"], str)
        assert isinstance(payload["ui_blocks"], list)
        assert isinstance(payload["run_id"], str)
        assert isinstance(payload["step_id"], str)

        run_root = tmp_path / "runs" / "chat-ui" / f"run-{payload['run_id']}"
        step_result_path = run_root / "artifacts" / "steps" / payload["step_id"] / "step_result.json"
        assert step_result_path.exists()
        step_result = json.loads(step_result_path.read_text())
        assert step_result["status"] == payload["status"]
        assert step_result["summary"] == payload["summary"]
        assert "ui_blocks" in step_result

        metrics_path = run_root / "reports" / "metrics.json"
        dashboard_path = run_root / "reports" / "dashboard.html"
        assert metrics_path.exists()
        assert dashboard_path.exists()
    finally:
        server.stop()


def test_chat_ui_falls_back_from_unsupported_requested_codex_model(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        try:
            out_idx = cmd.index("--output-last-message")
            model_idx = cmd.index("--model")
        except ValueError as exc:  # pragma: no cover
            raise AssertionError("Codex CLI args missing required flags") from exc

        out_path = Path(cmd[out_idx + 1])
        model = cmd[model_idx + 1]
        if model == "gpt-5.3-codex-spark":
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    'ERROR: {"type":"error","status":400,"error":{"type":"invalid_request_error",'
                    '"message":"The \'gpt-5.3-codex-spark\' model is not supported when using Codex '
                    'with a ChatGPT account."}}'
                ),
            )

        out_path.write_text(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "summary": "fallback reply",
                    "artifacts": [],
                    "metrics": {},
                    "next_actions": [],
                    "failure_signature": "",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(llm_client.subprocess, "run", fake_run)

    config = ChatUIConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = ChatUIServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)
        payload = _post_json(
            f"{server.url}/api/send",
            {"message": "hello", "history": [], "model": "gpt-5.3-codex-spark"},
        )
        assert payload["ok"] is True
        assert payload["status"] == "SUCCESS"
        assert payload["reply"] == "fallback reply"
        assert [cmd[cmd.index("--model") + 1] for cmd in calls] == ["gpt-5.3-codex-spark", "gpt-5.4"]
    finally:
        server.stop()


def test_chat_ui_app_wires_keyboard_first_composer_behavior() -> None:
    content = Path("ui/src/App.tsx").read_text(encoding="utf-8")

    assert "useRef<HTMLTextAreaElement | null>(null)" in content
    assert "inputRef.current?.focus();" in content
    assert "ref={inputRef}" in content
    assert "autoFocus" in content
    assert 'e.key !== "Enter"' in content
    assert "e.shiftKey" in content
    assert "e.nativeEvent.isComposing" in content
    assert "submitInput();" in content
