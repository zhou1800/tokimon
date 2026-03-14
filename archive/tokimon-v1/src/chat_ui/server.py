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
from llm.client import build_llm_client
from llm.client import interactive_codex_settings_from_env
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
        if self.llm_provider in {"codex", "codex-cli"}:
            self.llm_client = CodexCLIClient(
                self.workspace_dir,
                settings=interactive_codex_settings_from_env(),
            )
        else:
            self.llm_client = build_llm_client(self.llm_provider, workspace_dir=self.workspace_dir)
        self.tools = {
            "file": FileTool(self.workspace_dir),
            "grep": GrepTool(self.workspace_dir),
            "patch": PatchTool(self.workspace_dir),
            "pytest": PytestTool(self.workspace_dir),
            "web": WebTool(),
        }
        self.runs_root = self.workspace_dir / "runs" / "chat-ui"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._conversations_lock = threading.Lock()
        self._conversations: dict[str, ConversationState] = {}

    def _llm_client_for_request(self, model: str | None) -> Any:
        model = (model or "").strip() or None
        if model and self.llm_provider in {"codex", "codex-cli"}:
            return CodexCLIClient(
                self.workspace_dir,
                settings=interactive_codex_settings_from_env(model=model),
            )
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
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        state = self._conversation_for_send(thread_id, message=message, model=model)
        with state.lock:
            effective_history = history or _messages_to_history(state.messages)
            memory = _history_to_memory(effective_history)
            llm_client = self._llm_client_for_request(model)
            worker = Worker("Chat", llm_client, self.tools)
            state.step_index += 1
            step_id = f"chat-{state.step_index:04d}"
            replay = ReplayRecorder(
                step_id=step_id,
                worker_role="Chat",
                goal=message,
                inputs={
                    "message": message,
                    "history": effective_history,
                    "model": model,
                    "thread_id": state.thread_id,
                },
                memory=memory,
            )
            output = worker.run(
                goal=message,
                step_id=step_id,
                inputs={
                    "message": message,
                    "history": effective_history,
                    "model": model,
                    "thread_id": state.thread_id,
                },
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
            replay_record = replay.build()
            state.artifact_store.write_step(
                task_id=state.run_context.run_id,
                step_id=step_id,
                artifacts=output.artifacts,
                outputs={"summary": output.summary},
                step_result=step_result,
                replay_record=replay_record,
            )
            step_metrics = normalize_step_metrics(
                step_id=step_id,
                attempt_id=1,
                status=output.status.value,
                artifacts=output.artifacts,
                raw_metrics=output.metrics,
                failure_signature=output.failure_signature,
            )
            state.step_metrics[step_id] = step_metrics
            state.messages.extend(
                [
                    {"role": "user", "content": message},
                    _assistant_log_entry(step_id=step_id, summary=output.summary, status=output.status.value),
                ]
            )
            if not state.title:
                state.title = _conversation_title(message)
            if model and model.strip():
                state.model = model.strip()
            state.last_step_id = step_id
            state.last_step_result = step_result
            state.updated_at = _string_or_empty(replay_record.get("recorded_at")) or state.updated_at or state.created_at
            if not state.created_at:
                state.created_at = state.updated_at
            elapsed_since_last_write = max(0.0, time.perf_counter() - state.loaded_at_perf)
            wall_time_s = state.base_wall_time_s + elapsed_since_last_write
            steps = [state.step_metrics[key] for key in sorted(state.step_metrics)]
            run_metrics_payload = build_run_metrics_payload(
                run_id=state.run_context.run_id,
                runner="chat-ui",
                wall_time_s=wall_time_s,
                steps=steps,
                tests_passed=None,
                tests_failed=None,
            )
            write_metrics_and_dashboard(state.run_context.reports_dir, run_metrics_payload)
            state.base_wall_time_s = wall_time_s
            state.loaded_at_perf = time.perf_counter()
            self._persist_conversation(state)
        return {
            "status": output.status.value,
            "reply": output.summary,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": output.failure_signature,
            "ui_blocks": ui_blocks,
            "thread_id": state.thread_id,
            "run_id": state.run_context.run_id,
            "step_id": step_id,
            "step_result": step_result,
        }

    def list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        for run_root in sorted(self.runs_root.glob("run-*")):
            if not run_root.is_dir():
                continue
            thread_id = run_root.name.removeprefix("run-").strip()
            if not thread_id:
                continue
            state = self._get_or_load_conversation(thread_id)
            if state is None:
                continue
            conversations.append(state.to_summary())
        conversations.sort(key=lambda item: (str(item.get("updated_at") or ""), str(item.get("thread_id") or "")), reverse=True)
        return conversations

    def get_conversation(self, thread_id: str) -> dict[str, Any]:
        state = self._get_or_load_conversation(thread_id)
        if state is None:
            raise ConversationNotFoundError(thread_id)
        return state.to_detail()

    def _conversation_for_send(self, thread_id: str | None, *, message: str, model: str | None) -> ConversationState:
        normalized_thread_id = _string_or_empty(thread_id) or None
        if normalized_thread_id:
            state = self._get_or_load_conversation(normalized_thread_id)
            if state is None:
                raise ConversationNotFoundError(normalized_thread_id)
            return state
        state = self._create_conversation(message=message, model=model)
        with self._conversations_lock:
            self._conversations[state.thread_id] = state
        return state

    def _create_conversation(self, *, message: str, model: str | None) -> ConversationState:
        run_context = create_run_context(self.runs_root)
        run_context.write_manifest({"runner": "chat-ui", "workspace_dir": str(self.workspace_dir)})
        created_at = _run_created_at(run_context.root)
        state = ConversationState(
            run_context=run_context,
            artifact_store=ArtifactStore(run_context.artifacts_dir),
            created_at=created_at,
            updated_at=created_at,
            title=_conversation_title(message),
            model=_string_or_empty(model) or None,
        )
        self._persist_conversation(state)
        return state

    def _get_or_load_conversation(self, thread_id: str) -> ConversationState | None:
        normalized_thread_id = _string_or_empty(thread_id)
        if not normalized_thread_id:
            return None
        with self._conversations_lock:
            existing = self._conversations.get(normalized_thread_id)
        if existing is not None:
            return existing
        run_root = _conversation_root(self.runs_root, normalized_thread_id)
        if run_root is None or not run_root.exists() or not run_root.is_dir():
            return None
        run_context = load_run_context(run_root)
        state = self._load_persisted_conversation(run_context)
        with self._conversations_lock:
            self._conversations[normalized_thread_id] = state
        return state

    def _load_persisted_conversation(self, run_context: RunContext) -> ConversationState:
        payload = _read_json_file(run_context.root / _CONVERSATION_FILENAME)
        if isinstance(payload, dict):
            state = _conversation_state_from_payload(run_context, payload)
        else:
            state = _reconstruct_conversation_state(run_context)
        self._persist_conversation(state)
        return state

    def _persist_conversation(self, state: ConversationState) -> None:
        _write_json_file(
            state.run_context.root / _CONVERSATION_FILENAME,
            {
                "thread_id": state.thread_id,
                "run_id": state.run_context.run_id,
                "title": state.title,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "model": state.model or "",
                "message_count": len(state.messages),
                "messages": [dict(message) for message in state.messages],
                "last_step_id": state.last_step_id,
                "last_step_result": state.last_step_result or {},
                "step_index": state.step_index,
            },
        )


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
        if parsed.path == "/api/conversations":
            self._send_json({"ok": True, "conversations": self.server.list_conversations()})
            return
        if parsed.path.startswith("/api/conversations/"):
            thread_id = _string_or_empty(unquote(parsed.path.removeprefix("/api/conversations/")))
            if not thread_id:
                self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.server.get_conversation(thread_id)
            except ConversationNotFoundError:
                self._send_json({"ok": False, "error": "conversation not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True, **payload})
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
        thread_id = payload.get("thread_id")
        if thread_id is not None and not isinstance(thread_id, str):
            self._send_json({"ok": False, "error": "thread_id must be a string"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            response = self.server.handle_send(
                message.strip(),
                history,
                model=(model or "").strip() or None,
                thread_id=(thread_id or "").strip() or None,
            )
        except ConversationNotFoundError:
            self._send_json({"ok": False, "error": "conversation not found"}, status=HTTPStatus.NOT_FOUND)
            return
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


def _assistant_log_entry(*, step_id: str, summary: str, status: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": "assistant", "content": summary or "(no reply)", "step_id": step_id}
    status_text = _string_or_empty(status)
    if status_text:
        entry["meta"] = f"status={status_text}"
        if status_text.upper() != "SUCCESS":
            entry["error"] = True
    return entry


def _conversation_preview(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = _string_or_empty(message.get("content")) if isinstance(message, dict) else ""
        if content:
            return _trim_text(content, limit=120, fallback="No messages yet")
    return "No messages yet"


def _conversation_title(text: str) -> str:
    return _trim_text(text, limit=72, fallback="New chat")


def _trim_text(text: str, *, limit: int, fallback: str) -> str:
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return fallback
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."


def _string_or_empty(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _messages_to_history(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _string_or_empty(message.get("role"))
        content = _string_or_empty(message.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})
    return history


def _conversation_root(runs_root: Path, thread_id: str) -> Path | None:
    normalized = _string_or_empty(thread_id)
    if not normalized or "/" in normalized or "\\" in normalized:
        return None
    base = runs_root.resolve()
    candidate = (base / f"run-{normalized}").resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate


def _read_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_file(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_step_metrics(run_context: RunContext) -> tuple[dict[str, dict[str, Any]], float]:
    payload = _read_json_file(run_context.reports_dir / "metrics.json")
    if not isinstance(payload, dict):
        return {}, 0.0

    steps_raw = payload.get("steps")
    step_metrics: dict[str, dict[str, Any]] = {}
    if isinstance(steps_raw, list):
        for item in steps_raw:
            if not isinstance(item, dict):
                continue
            step_id = _string_or_empty(item.get("step_id"))
            if not step_id:
                continue
            step_metrics[step_id] = dict(item)

    wall_time_s = 0.0
    run_raw = payload.get("run")
    if isinstance(run_raw, dict):
        raw_wall_time_s = run_raw.get("wall_time_s")
        if isinstance(raw_wall_time_s, (int, float)):
            wall_time_s = float(raw_wall_time_s)
    return step_metrics, wall_time_s


def _run_created_at(run_root: Path) -> str:
    payload = _read_json_file(run_root / "run.json")
    if not isinstance(payload, dict):
        return ""
    return _string_or_empty(payload.get("created_at"))


def _conversation_state_from_payload(run_context: RunContext, payload: dict[str, Any]) -> ConversationState:
    messages = _normalize_conversation_messages(payload.get("messages"))
    step_metrics, wall_time_s = _load_step_metrics(run_context)
    title = _string_or_empty(payload.get("title"))
    if not title and messages:
        title = _conversation_title(_string_or_empty(messages[0].get("content")))
    if not title:
        title = "New chat"
    created_at = _string_or_empty(payload.get("created_at")) or _run_created_at(run_context.root)
    updated_at = _string_or_empty(payload.get("updated_at")) or created_at
    model = _string_or_empty(payload.get("model")) or None
    last_step_id = _string_or_empty(payload.get("last_step_id")) or _latest_step_id(messages, step_metrics)
    last_step_result_raw = payload.get("last_step_result")
    if isinstance(last_step_result_raw, dict) and last_step_result_raw:
        last_step_result = dict(last_step_result_raw)
    else:
        last_step_result = _load_step_result(run_context, last_step_id)

    step_index = _parse_step_index(last_step_id)
    stored_step_index = payload.get("step_index")
    if isinstance(stored_step_index, int):
        step_index = max(step_index, stored_step_index)
    for step_id in step_metrics:
        step_index = max(step_index, _parse_step_index(step_id))

    return ConversationState(
        run_context=run_context,
        artifact_store=ArtifactStore(run_context.artifacts_dir),
        created_at=created_at,
        updated_at=updated_at,
        title=title,
        model=model,
        messages=messages,
        last_step_id=last_step_id or None,
        last_step_result=last_step_result,
        step_index=step_index,
        step_metrics=step_metrics,
        base_wall_time_s=wall_time_s,
    )


def _reconstruct_conversation_state(run_context: RunContext) -> ConversationState:
    messages: list[dict[str, Any]] = []
    title = ""
    created_at = _run_created_at(run_context.root)
    updated_at = created_at
    model: str | None = None
    last_step_id: str | None = None
    last_step_result: dict[str, Any] | None = None
    step_index = 0

    for step_dir in _iter_step_dirs(run_context.artifacts_dir):
        step_id = step_dir.name
        step_index = max(step_index, _parse_step_index(step_id))
        replay_payload = _read_json_file(step_dir / "replay.json")
        step_result_payload = _read_json_file(step_dir / "step_result.json")

        inputs = replay_payload.get("inputs") if isinstance(replay_payload, dict) else {}
        user_message = ""
        if isinstance(inputs, dict):
            user_message = _string_or_empty(inputs.get("message"))
            maybe_model = _string_or_empty(inputs.get("model"))
            if maybe_model:
                model = maybe_model
        if not user_message and isinstance(replay_payload, dict):
            user_message = _string_or_empty(replay_payload.get("goal"))
        if user_message:
            messages.append({"role": "user", "content": user_message})
            if not title:
                title = _conversation_title(user_message)

        if isinstance(step_result_payload, dict):
            summary = _string_or_empty(step_result_payload.get("summary"))
            status = _string_or_empty(step_result_payload.get("status"))
            messages.append(_assistant_log_entry(step_id=step_id, summary=summary, status=status))
            last_step_result = dict(step_result_payload)
            last_step_id = step_id

        recorded_at = _string_or_empty(replay_payload.get("recorded_at")) if isinstance(replay_payload, dict) else ""
        if recorded_at:
            if not created_at:
                created_at = recorded_at
            updated_at = recorded_at

    step_metrics, wall_time_s = _load_step_metrics(run_context)
    last_step_id = last_step_id or _latest_step_id(messages, step_metrics) or None
    if last_step_result is None:
        last_step_result = _load_step_result(run_context, last_step_id)
    for step_id in step_metrics:
        step_index = max(step_index, _parse_step_index(step_id))

    return ConversationState(
        run_context=run_context,
        artifact_store=ArtifactStore(run_context.artifacts_dir),
        created_at=created_at,
        updated_at=updated_at or created_at,
        title=title or "New chat",
        model=model,
        messages=messages,
        last_step_id=last_step_id,
        last_step_result=last_step_result,
        step_index=step_index,
        step_metrics=step_metrics,
        base_wall_time_s=wall_time_s,
    )


def _normalize_conversation_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = _string_or_empty(item.get("role"))
        content = _string_or_empty(item.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        message: dict[str, Any] = {"role": role, "content": content}
        meta = _string_or_empty(item.get("meta"))
        if meta:
            message["meta"] = meta
        if item.get("error") is True:
            message["error"] = True
        step_id = _string_or_empty(item.get("step_id"))
        if step_id:
            message["step_id"] = step_id
        messages.append(message)
    return messages


def _load_step_result(run_context: RunContext, step_id: str | None) -> dict[str, Any] | None:
    normalized_step_id = _string_or_empty(step_id)
    if not normalized_step_id:
        return None
    payload = _read_json_file(run_context.artifacts_dir / normalized_step_id / "step_result.json")
    return dict(payload) if isinstance(payload, dict) else None


def _iter_step_dirs(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists():
        return []
    return sorted(
        (
            path
            for path in artifacts_dir.iterdir()
            if path.is_dir() and _STEP_ID_RE.match(path.name)
        ),
        key=lambda path: (_parse_step_index(path.name), path.name),
    )


def _latest_step_id(messages: list[dict[str, Any]], step_metrics: dict[str, dict[str, Any]]) -> str:
    latest_step_id = ""
    latest_index = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        step_id = _string_or_empty(message.get("step_id"))
        index = _parse_step_index(step_id)
        if index >= latest_index:
            latest_index = index
            latest_step_id = step_id
    for step_id in step_metrics:
        index = _parse_step_index(step_id)
        if index >= latest_index:
            latest_index = index
            latest_step_id = step_id
    return latest_step_id


def _parse_step_index(step_id: str | None) -> int:
    normalized_step_id = _string_or_empty(step_id)
    match = _STEP_ID_RE.match(normalized_step_id)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


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
