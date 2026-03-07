"""Local web chat UI for Tokimon."""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import time
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from artifacts import ArtifactStore
from agents.worker import Worker
from llm.client import ClaudeCLIClient
from llm.client import ClaudeCLISettings
from llm.client import CodexCLIClient
from llm.client import CodexCLISettings
from llm.client import build_llm_client
from observability.reports import build_run_metrics_payload
from observability.reports import normalize_step_metrics
from observability.reports import write_metrics_and_dashboard
from replay import ReplayRecorder
from runs import RunContext, create_run_context, load_run_context
from tools.file_tool import FileTool
from tools.grep_tool import GrepTool
from tools.patch_tool import PatchTool
from tools.pytest_tool import PytestTool
from tools.web_tool import WebTool


_BUILD_MISSING_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Tokimon Chat (UI build missing)</title>
    <style>
      :root { color-scheme: light dark; }
      body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
      main { max-width: 860px; margin: 40px auto; padding: 0 16px; line-height: 1.5; }
      pre { padding: 12px; border: 1px solid #4443; border-radius: 8px; overflow: auto; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
  </head>
  <body>
    <main>
      <h1>Tokimon Chat UI</h1>
      <p>The React frontend build was not found. Build it from the repo root:</p>
      <pre><code>cd ui
npm install
npm run build</code></pre>
      <p>Then restart <code>tokimon chat-ui</code> and reload this page.</p>
      <p>For local development, you can also run the Vite dev server:</p>
      <pre><code>cd ui
npm run dev</code></pre>
    </main>
  </body>
</html>
"""

_CONVERSATION_FILENAME = "conversation.json"
_STEP_ID_RE = re.compile(r"^chat-(\d+)$")


class ConversationNotFoundError(LookupError):
    """Raised when a requested chat thread cannot be found."""


@dataclass
class ConversationState:
    run_context: RunContext
    artifact_store: ArtifactStore
    created_at: str
    updated_at: str
    title: str
    model: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_step_id: str | None = None
    last_step_result: dict[str, Any] | None = None
    step_index: int = 0
    step_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_wall_time_s: float = 0.0
    loaded_at_perf: float = field(default_factory=time.perf_counter, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def thread_id(self) -> str:
        return self.run_context.run_id

    def to_summary(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "run_id": self.run_context.run_id,
            "title": self.title,
            "preview": _conversation_preview(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "model": self.model or "",
        }

    def to_detail(self) -> dict[str, Any]:
        payload = self.to_summary()
        payload.update(
            {
                "messages": [dict(message) for message in self.messages],
                "last_step_id": self.last_step_id,
                "last_step_result": dict(self.last_step_result or {}) if self.last_step_result else None,
            }
        )
        return payload


@dataclass(frozen=True)
class ChatUIConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    llm_provider: str = "codex"
    workspace_dir: Path = field(default_factory=Path.cwd)


class _ChatHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], config: ChatUIConfig) -> None:
        super().__init__(server_address, _ChatUIHandler)
        self.config = config
        self.workspace_dir = config.workspace_dir.resolve()
        self.llm_provider = (config.llm_provider or "").strip().lower()
        self.llm_client = build_llm_client(self.llm_provider, workspace_dir=self.workspace_dir)
        self.tools = {
            "file": FileTool(self.workspace_dir),
            "grep": GrepTool(self.workspace_dir),
            "patch": PatchTool(self.workspace_dir),
            "pytest": PytestTool(self.workspace_dir),
            "web": WebTool(),
        }
        runs_root = self.workspace_dir / "runs" / "chat-ui"
        self.run_context = create_run_context(runs_root)
        self.run_context.write_manifest({"runner": "chat-ui", "workspace_dir": str(self.workspace_dir)})
        self.artifact_store = ArtifactStore(self.run_context.artifacts_dir)
        self._step_lock = threading.Lock()
        self._step_index = 0
        self._run_start = time.perf_counter()
        self._step_metrics: dict[str, dict[str, Any]] = {}

    def _llm_client_for_request(self, model: str | None) -> Any:
        model = (model or "").strip() or None
        if model and self.llm_provider in {"codex", "codex-cli"}:
            settings = replace(CodexCLISettings.from_env(), model=model)
            return CodexCLIClient(self.workspace_dir, settings=settings)
        if model and self.llm_provider in {"claude", "claude-cli"}:
            settings = replace(ClaudeCLISettings.from_env(), model=model)
            return ClaudeCLIClient(self.workspace_dir, settings=settings)
        return self.llm_client

    def handle_send(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        history = history or []
        memory = _history_to_memory(history)
        llm_client = self._llm_client_for_request(model)
        worker = Worker("Chat", llm_client, self.tools)
        with self._step_lock:
            self._step_index += 1
            step_id = f"chat-{self._step_index:04d}"
        replay = ReplayRecorder(
            step_id=step_id,
            worker_role="Chat",
            goal=message,
            inputs={"message": message, "history": history, "model": model},
            memory=memory,
        )
        output = worker.run(
            goal=message,
            step_id=step_id,
            inputs={"message": message, "history": history, "model": model},
            memory=memory,
            replay_recorder=replay,
        )
        raw_ui_blocks = output.data.get("ui_blocks") if isinstance(output.data, dict) else None
        ui_blocks: list[dict[str, Any]] = []
        if isinstance(raw_ui_blocks, list):
            ui_blocks = [block for block in raw_ui_blocks if isinstance(block, dict)]
        step_result: dict[str, Any] = {
            "status": output.status.value,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": str(output.failure_signature or ""),
            "ui_blocks": ui_blocks,
        }
        self.artifact_store.write_step(
            task_id=self.run_context.run_id,
            step_id=step_id,
            artifacts=output.artifacts,
            outputs={"summary": output.summary},
            step_result=step_result,
            replay_record=replay.build(),
        )
        step_metrics = normalize_step_metrics(
            step_id=step_id,
            attempt_id=1,
            status=output.status.value,
            artifacts=output.artifacts,
            raw_metrics=output.metrics,
            failure_signature=output.failure_signature,
        )
        with self._step_lock:
            self._step_metrics[step_id] = step_metrics
            wall_time_s = time.perf_counter() - self._run_start
            steps = [self._step_metrics[key] for key in sorted(self._step_metrics)]
            run_metrics_payload = build_run_metrics_payload(
                run_id=self.run_context.run_id,
                runner="chat-ui",
                wall_time_s=wall_time_s,
                steps=steps,
                tests_passed=None,
                tests_failed=None,
            )
            write_metrics_and_dashboard(self.run_context.reports_dir, run_metrics_payload)
        return {
            "status": output.status.value,
            "reply": output.summary,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": output.failure_signature,
            "ui_blocks": ui_blocks,
            "run_id": self.run_context.run_id,
            "step_id": step_id,
            "step_result": step_result,
        }


class ChatUIServer:
    def __init__(self, config: ChatUIConfig) -> None:
        self._server = _ChatHTTPServer((config.host, int(config.port)), config)
        self._thread = threading.Thread(target=self._server.serve_forever, name="tokimon-chat-ui", daemon=True)

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def run_chat_ui(config: ChatUIConfig) -> None:
    server = ChatUIServer(config)
    server.start()
    print(f"Tokimon Chat UI: {server.url}")
    try:
        server._thread.join()
    except KeyboardInterrupt:
        server.stop()


class _ChatUIHandler(BaseHTTPRequestHandler):
    server: _ChatHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if self._try_serve_frontend(parsed.path):
            return
        if parsed.path == "/":
            self._send_html(_BUILD_MISSING_HTML)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def _frontend_dist_dir(self) -> Path:
        return self.server.workspace_dir / "ui" / "dist"

    def _try_serve_frontend(self, request_path: str) -> bool:
        dist_dir = self._frontend_dist_dir()
        index_path = dist_dir / "index.html"
        if not index_path.exists():
            return False

        if request_path in {"", "/"}:
            self._send_file(index_path)
            return True

        rel = request_path.lstrip("/")
        base = dist_dir.resolve()
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return False

        if candidate.exists() and candidate.is_file():
            self._send_file(candidate)
            return True

        if "." not in Path(rel).name:
            self._send_file(index_path)
            return True

        return False

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        content_type, _ = mimetypes.guess_type(str(path))
        if not content_type:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json", "image/svg+xml"}:
            content_type = f"{content_type}; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/send":
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            self._send_json({"ok": False, "error": "message must be a non-empty string"}, status=HTTPStatus.BAD_REQUEST)
            return
        history = payload.get("history")
        if history is not None and not isinstance(history, list):
            self._send_json({"ok": False, "error": "history must be a list"}, status=HTTPStatus.BAD_REQUEST)
            return
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            self._send_json({"ok": False, "error": "model must be a string"}, status=HTTPStatus.BAD_REQUEST)
            return
        response = self.server.handle_send(message.strip(), history, model=(model or "").strip() or None)
        self._send_json({"ok": True, **response})

    def _read_json(self) -> dict[str, Any]:
        length = self.headers.get("content-length")
        if not length:
            raise ValueError("missing content-length")
        try:
            n = int(length)
        except ValueError as exc:
            raise ValueError("invalid content-length") from exc
        raw = self.rfile.read(max(0, n))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("invalid json body") from exc
        if not isinstance(payload, dict):
            raise ValueError("json body must be an object")
        return payload

    def _send_html(self, html: str) -> None:
        data = (html or "").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _history_to_memory(history: list[dict[str, Any]], *, limit: int = 16, max_chars: int = 10_000) -> list[str]:
    trimmed = history[-limit:] if limit > 0 else []
    lines: list[str] = []
    total = 0
    for msg in trimmed:
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}" if role else content
        total += len(line)
        if total > max_chars:
            lines.append("...(history truncated)")
            break
        lines.append(line)
    return lines
