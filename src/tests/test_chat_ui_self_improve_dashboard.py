from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from chat_ui.server import ChatUIConfig, ChatUIServer


def _post_json(url: str, payload: dict, *, timeout_s: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, *, timeout_s: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _git(cwd: Path, args: list[str]) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def test_chat_ui_self_improve_dashboard_streams_session_trace(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")

    master = tmp_path / "master"
    (master / "src" / "tests").mkdir(parents=True, exist_ok=True)
    (master / "src" / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.pytest.ini_options]",
                "testpaths = [\"src/tests\"]",
                "",
            ]
        )
    )
    (master / "src" / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (master / ".gitignore").write_text(".tokimon-tmp/\nsrc/runs/\n__pycache__/\n.pytest_cache/\n")

    _git(master, ["init"])
    _git(master, ["config", "user.email", "test@example.com"])
    _git(master, ["config", "user.name", "Test User"])
    _git(master, ["add", "."])
    _git(master, ["commit", "-m", "init"])

    server = ChatUIServer(ChatUIConfig(host="127.0.0.1", port=0, llm_provider="mock", workspace_dir=master))
    server.start()
    try:
        start = _post_json(
            f"{server.url}/api/self_improve/start",
            {"goal": "No-op", "llm": "mock", "sessions": 1, "batches": 1, "workers": 1, "no_merge": True},
        )
        assert start["ok"] is True
        run_id = start["run_id"]
        assert isinstance(run_id, str) and run_id
        assert start["session_ids"] == ["1-1"]

        stream_url = f"{server.url}/api/self_improve/stream?run_id={run_id}&session_id=1-1"
        found_final = False
        deadline = time.monotonic() + 20.0
        with urllib.request.urlopen(stream_url, timeout=20.0) as response:
            while time.monotonic() < deadline:
                raw = response.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: ") :])
                if payload.get("event_type") == "worker_final":
                    found_final = True
                    break
        assert found_final is True

        status_deadline = time.monotonic() + 30.0
        while time.monotonic() < status_deadline:
            status = _get_json(f"{server.url}/api/self_improve/status?run_id={run_id}")
            assert status["ok"] is True
            if status.get("status") == "COMPLETED":
                break
            time.sleep(0.1)
        else:
            raise AssertionError("self-improve run did not complete")
    finally:
        server.stop()
