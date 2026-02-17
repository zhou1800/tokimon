"""Local web chat UI for Tokimon."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agents.worker import Worker
from llm.client import CodexCLIClient, CodexCLISettings, build_llm_client
from self_improve.orchestrator import SelfImproveOrchestrator, SelfImproveSettings
from self_improve.workspace import can_use_git_merge, git_toplevel
from tools.file_tool import FileTool
from tools.grep_tool import GrepTool
from tools.patch_tool import PatchTool
from tools.pytest_tool import PytestTool
from tools.web_tool import WebTool


_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Tokimon</title>
    <style>
      :root { color-scheme: light dark; }
      body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
      header { display: flex; gap: 14px; align-items: center; padding: 10px 14px; border-bottom: 1px solid #4443; }
      header .title { font-weight: 650; }
      header nav { display: flex; gap: 8px; }
      header button.tab { padding: 6px 10px; border: 1px solid #4446; background: transparent; border-radius: 8px; cursor: pointer; }
      header button.tab.active { background: #4442; }
      main { height: calc(100vh - 49px); }
      section { display: none; height: 100%; }
      section.active { display: grid; }
      #wrap-chat { display: grid; grid-template-rows: 1fr auto; height: 100%; }
      #wrap-self { display: grid; grid-template-rows: auto 1fr; height: 100%; }
      #log { padding: 16px; overflow: auto; }
      .msg { margin: 0 0 12px 0; white-space: pre-wrap; }
      .user { font-weight: 600; }
      .assistant { opacity: 0.95; }
      form { display: grid; grid-template-columns: 1fr auto; gap: 10px; padding: 12px 16px; border-top: 1px solid #4443; }
      textarea { width: 100%; resize: vertical; min-height: 44px; padding: 10px; font: inherit; }
      button { padding: 10px 14px; font: inherit; }
      .meta { opacity: 0.7; font-size: 12px; margin-top: 4px; }
      .error { color: #b00020; }
      .panel { padding: 14px 16px; overflow: auto; }
      .row { display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; margin-top: 10px; }
      .row label { display: grid; gap: 6px; font-size: 12px; opacity: 0.9; }
      .row input, .row select { padding: 8px; font: inherit; }
      .actions { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
      .actions .status { font-size: 12px; opacity: 0.75; }
      .sessions { display: grid; grid-template-rows: auto 1fr; gap: 10px; margin-top: 12px; height: 100%; min-height: 320px; }
      .session-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
      .session-tabs button { padding: 6px 10px; border-radius: 999px; border: 1px solid #4446; background: transparent; cursor: pointer; }
      .session-tabs button.active { background: #4442; }
      .session-panels { border: 1px solid #4443; border-radius: 10px; overflow: hidden; height: 100%; }
      .session-panel { display: none; height: 100%; overflow: auto; padding: 12px; }
      .session-panel.active { display: block; }
      .line { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; white-space: pre-wrap; margin: 0 0 8px 0; }
    </style>
  </head>
  <body>
    <header>
      <div class="title">Tokimon</div>
      <nav>
        <button class="tab active" data-tab="chat">Chat</button>
        <button class="tab" data-tab="self">Self Improve</button>
      </nav>
    </header>
    <main>
      <section id="tab-chat" class="active">
        <div id="wrap-chat">
          <div id="log"></div>
          <form id="form">
            <textarea id="input" placeholder="Type a message…"></textarea>
            <button id="send" type="submit">Send</button>
          </form>
        </div>
      </section>
      <section id="tab-self">
        <div id="wrap-self">
          <div class="panel">
            <div>Self-improve runs execute multiple parallel sessions and stream each session's trace.</div>
            <textarea id="si-goal" placeholder="Self-improve goal (URLs auto-fetched)"></textarea>
            <div class="row">
              <label>LLM
                <select id="si-llm">
                  <option value="codex">codex</option>
                  <option value="mock">mock</option>
                </select>
              </label>
              <label>Sessions
                <input id="si-sessions" type="number" min="1" max="16" value="4" />
              </label>
              <label>Batches
                <input id="si-batches" type="number" min="1" max="8" value="1" />
              </label>
              <label>Workers
                <input id="si-workers" type="number" min="1" max="16" value="4" />
              </label>
              <label>Merge
                <select id="si-merge">
                  <option value="no-merge">no-merge</option>
                  <option value="merge">merge</option>
                </select>
              </label>
            </div>
            <div class="actions">
              <button id="si-start" type="button">Start self-improve</button>
              <div id="si-status" class="status"></div>
            </div>
            <div id="si-meta" class="status"></div>
          </div>
          <div class="panel sessions">
            <div id="si-session-tabs" class="session-tabs"></div>
            <div id="si-session-panels" class="session-panels"></div>
          </div>
        </div>
      </section>
    </main>
    <script>
      // Tabs
      const tabButtons = Array.from(document.querySelectorAll('header button.tab'));
      function showTab(name) {
        for (const btn of tabButtons) btn.classList.toggle('active', btn.dataset.tab === name);
        document.getElementById('tab-chat').classList.toggle('active', name === 'chat');
        document.getElementById('tab-self').classList.toggle('active', name === 'self');
      }
      for (const btn of tabButtons) btn.addEventListener('click', () => showTab(btn.dataset.tab));

      // Chat
      const log = document.getElementById('log');
      const form = document.getElementById('form');
      const input = document.getElementById('input');
      const sendBtn = document.getElementById('send');
      const history = [];

      function append(role, content, meta) {
        const p = document.createElement('div');
        p.className = 'msg ' + role;
        const label = role === 'user' ? 'You' : 'Tokimon';
        p.textContent = label + ': ' + content;
        if (meta) {
          const m = document.createElement('div');
          m.className = 'meta';
          m.textContent = meta;
          p.appendChild(m);
        }
        log.appendChild(p);
        log.scrollTop = log.scrollHeight;
      }

      async function sendMessage(message) {
        append('user', message);
        history.push({role: 'user', content: message});
        sendBtn.disabled = true;
        try {
          const res = await fetch('/api/send', {
            method: 'POST',
            headers: {'content-type': 'application/json'},
            body: JSON.stringify({message, history})
          });
          const data = await res.json();
          const reply = data.reply || data.summary || '';
          const meta = data.status ? ('status=' + data.status) : '';
          append('assistant', reply || '(no reply)', meta);
          history.push({role: 'assistant', content: reply || ''});
        } catch (err) {
          const msg = (err && err.message) ? err.message : String(err);
          append('assistant', 'Error: ' + msg, 'request failed');
          const last = log.lastChild;
          if (last) last.classList.add('error');
        } finally {
          sendBtn.disabled = false;
        }
      }

      form.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = (input.value || '').trim();
        if (!message) return;
        input.value = '';
        sendMessage(message);
      });

      // Self-improve dashboard
      const si = {
        runId: null,
        sessionIds: [],
        sources: new Map(),
        statusTimer: null,
      };

      const siGoal = document.getElementById('si-goal');
      const siLLM = document.getElementById('si-llm');
      const siSessions = document.getElementById('si-sessions');
      const siBatches = document.getElementById('si-batches');
      const siWorkers = document.getElementById('si-workers');
      const siMerge = document.getElementById('si-merge');
      const siStart = document.getElementById('si-start');
      const siStatus = document.getElementById('si-status');
      const siMeta = document.getElementById('si-meta');
      const siSessionTabs = document.getElementById('si-session-tabs');
      const siSessionPanels = document.getElementById('si-session-panels');

      function setStatus(text, isError) {
        siStatus.textContent = text || '';
        siStatus.style.color = isError ? '#b00020' : '';
      }

      function appendLine(panel, text) {
        const p = document.createElement('div');
        p.className = 'line';
        p.textContent = text;
        panel.appendChild(p);
        while (panel.childNodes.length > 2000) panel.removeChild(panel.firstChild);
        panel.scrollTop = panel.scrollHeight;
      }

      function formatTrace(record) {
        const type = record && record.event_type ? record.event_type : 'trace';
        const p = record && record.payload ? record.payload : {};
        if (type === 'worker_model_call') return `MODEL_CALL iter=${p.iteration} role=${p.worker_role} step=${p.step_id}`;
        if (type === 'worker_model_response') return `MODEL_RESPONSE iter=${p.iteration} type=${p.response_type} status=${p.status || ''} summary=${(p.summary || '').slice(0, 240)}`;
        if (type === 'worker_tool_call') {
          const call = p.call || {};
          return `TOOL_CALL iter=${p.iteration} tool=${call.tool} action=${call.action}`;
        }
        if (type === 'worker_tool_result') return `TOOL_RESULT iter=${p.iteration} tool=${p.tool_name} ok=${p.ok} summary=${(p.summary || '').slice(0, 240)}`;
        if (type === 'worker_final') return `FINAL iter=${p.iteration} status=${p.status} summary=${p.summary}`;
        if (type.startsWith('step_')) return `${type} ${JSON.stringify(p)}`;
        return `${type} ${JSON.stringify(p)}`;
      }

      function clearSessions() {
        for (const es of si.sources.values()) {
          try { es.close(); } catch (_) {}
        }
        si.sources.clear();
        siSessionTabs.textContent = '';
        siSessionPanels.textContent = '';
        si.sessionIds = [];
      }

      function setActiveSession(sessionId) {
        for (const btn of Array.from(siSessionTabs.querySelectorAll('button'))) {
          btn.classList.toggle('active', btn.dataset.sessionId === sessionId);
        }
        for (const panel of Array.from(siSessionPanels.querySelectorAll('.session-panel'))) {
          panel.classList.toggle('active', panel.dataset.sessionId === sessionId);
        }
      }

      function addSessionTab(sessionId) {
        const btn = document.createElement('button');
        btn.textContent = sessionId;
        btn.dataset.sessionId = sessionId;
        btn.addEventListener('click', () => setActiveSession(sessionId));
        siSessionTabs.appendChild(btn);

        const panel = document.createElement('div');
        panel.className = 'session-panel';
        panel.dataset.sessionId = sessionId;
        siSessionPanels.appendChild(panel);

        if (siSessionTabs.querySelectorAll('button').length === 1) setActiveSession(sessionId);
        return panel;
      }

      function startSessionStream(runId, sessionId, panel) {
        const url = `/api/self_improve/stream?run_id=${encodeURIComponent(runId)}&session_id=${encodeURIComponent(sessionId)}`;
        appendLine(panel, `Connecting: ${url}`);
        const es = new EventSource(url);
        si.sources.set(sessionId, es);
        es.addEventListener('trace', (evt) => {
          try {
            const record = JSON.parse(evt.data);
            appendLine(panel, formatTrace(record));
          } catch (e) {
            appendLine(panel, `trace parse error: ${e && e.message ? e.message : String(e)}`);
          }
        });
        es.addEventListener('done', (evt) => {
          appendLine(panel, evt.data ? `done: ${evt.data}` : 'done');
          try { es.close(); } catch (_) {}
        });
        es.onerror = () => {
          appendLine(panel, 'stream error (will retry)');
        };
      }

      async function pollSelfImproveStatus(runId) {
        try {
          const res = await fetch(`/api/self_improve/status?run_id=${encodeURIComponent(runId)}`);
          const data = await res.json();
          if (!data || data.ok !== true) return;
          const status = data.status || (data.data && data.data.status) || '';
          const input = data.input || (data.data && data.data.input) || null;
          siMeta.textContent = `run_id=${runId} status=${status} input=${input ? (input.kind || '') : ''}`.trim();
          if (status && status !== 'RUNNING' && si.statusTimer) {
            clearInterval(si.statusTimer);
            si.statusTimer = null;
          }
        } catch (_) {
          // ignore
        }
      }

      async function startSelfImprove() {
        const goal = (siGoal.value || '').trim();
        if (!goal) {
          setStatus('goal is required', true);
          return;
        }
        if (si.statusTimer) {
          clearInterval(si.statusTimer);
          si.statusTimer = null;
        }
        clearSessions();
        setStatus('starting…', false);
        siMeta.textContent = '';

        const payload = {
          goal,
          llm: (siLLM.value || 'mock'),
          sessions: parseInt(siSessions.value || '4', 10),
          batches: parseInt(siBatches.value || '1', 10),
          workers: parseInt(siWorkers.value || '4', 10),
          no_merge: (siMerge.value || 'no-merge') !== 'merge',
        };
        try {
          const res = await fetch('/api/self_improve/start', {
            method: 'POST',
            headers: {'content-type': 'application/json'},
            body: JSON.stringify(payload),
          });
          const data = await res.json();
          if (!data || data.ok !== true) {
            throw new Error((data && data.error) ? data.error : `http ${res.status}`);
          }
          si.runId = data.run_id;
          si.sessionIds = data.session_ids || [];
          setStatus(`running (${si.runId})`, false);
          for (const sessionId of si.sessionIds) {
            const panel = addSessionTab(sessionId);
            startSessionStream(si.runId, sessionId, panel);
          }
          if (si.statusTimer) clearInterval(si.statusTimer);
          si.statusTimer = setInterval(() => pollSelfImproveStatus(si.runId), 1000);
          pollSelfImproveStatus(si.runId);
        } catch (err) {
          const msg = (err && err.message) ? err.message : String(err);
          setStatus(msg, true);
        }
      }

      siStart.addEventListener('click', startSelfImprove);
    </script>
  </body>
</html>
"""


@dataclass(frozen=True)
class ChatUIConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    llm_provider: str = "mock"
    workspace_dir: Path = field(default_factory=Path.cwd)


@dataclass
class SelfImproveRunState:
    run_id: str
    run_root: Path
    master_root: Path
    goal: str
    input_ref: str | None
    sessions: int
    batches: int
    workers: int
    no_merge: bool
    llm_provider: str
    thread: threading.Thread


class _ChatHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], config: ChatUIConfig) -> None:
        super().__init__(server_address, _ChatUIHandler)
        self.config = config
        self.workspace_dir = config.workspace_dir.resolve()
        self.llm_client = build_llm_client(config.llm_provider, workspace_dir=self.workspace_dir)
        self.tools = {
            "file": FileTool(self.workspace_dir),
            "grep": GrepTool(self.workspace_dir),
            "patch": PatchTool(self.workspace_dir),
            "pytest": PytestTool(self.workspace_dir),
            "web": WebTool(),
        }
        self.self_improve_runs: dict[str, SelfImproveRunState] = {}
        self._self_improve_lock = threading.Lock()

    def handle_send(self, message: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
        history = history or []
        memory = _history_to_memory(history)
        worker = Worker("Chat", self.llm_client, self.tools)
        output = worker.run(goal=message, step_id="chat", inputs={"message": message, "history": history}, memory=memory)
        return {
            "status": output.status.value,
            "reply": output.summary,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": output.failure_signature,
        }

    def start_self_improve(
        self,
        *,
        goal: str,
        input_ref: str | None,
        sessions: int,
        batches: int,
        workers: int,
        no_merge: bool,
        llm_provider: str,
    ) -> SelfImproveRunState:
        master_root = git_toplevel(self.workspace_dir) or self.workspace_dir
        if not can_use_git_merge(master_root):
            raise ValueError(
                "self-improve requires a clean git checkout (run from the repo root and ensure `git status --porcelain` is empty)."
            )

        project_root = master_root / "src" if (master_root / "src" / "pyproject.toml").exists() else master_root
        base_dir = project_root / "runs" / "self-improve"
        base_dir.mkdir(parents=True, exist_ok=True)
        existing = {path.name for path in base_dir.glob("run-*") if path.is_dir()}

        settings = SelfImproveSettings(
            sessions_per_batch=int(sessions),
            batches=int(batches),
            max_workers=int(workers),
            merge_on_success=not no_merge,
        )

        def llm_factory(_session_id: str, workspace_dir: Path):
            normalized = (llm_provider or "").strip().lower()
            if normalized in {"codex", "codex-cli"}:
                codex_settings = CodexCLISettings.from_env()
                if "TOKIMON_CODEX_SANDBOX" not in os.environ:
                    codex_settings = replace(codex_settings, sandbox="workspace-write")
                if "TOKIMON_CODEX_APPROVAL" not in os.environ:
                    codex_settings = replace(codex_settings, ask_for_approval="never")
                if "TOKIMON_CODEX_SEARCH" not in os.environ:
                    codex_settings = replace(codex_settings, search=True)
                if "TOKIMON_CODEX_TIMEOUT_S" not in os.environ:
                    codex_settings = replace(codex_settings, timeout_s=240)
                return CodexCLIClient(workspace_dir, settings=codex_settings)
            return build_llm_client(normalized or "mock", workspace_dir=workspace_dir)

        orchestrator = SelfImproveOrchestrator(master_root, llm_factory=llm_factory, settings=settings)

        errors: list[str] = []

        def run_bg() -> None:
            try:
                orchestrator.run(goal, input_ref=input_ref)
            except Exception as exc:  # pragma: no cover
                errors.append(str(exc))

        thread = threading.Thread(
            target=run_bg,
            name=f"tokimon-self-improve-{int(time.time())}",
            daemon=True,
        )
        thread.start()

        run_root = _wait_for_new_run_dir(base_dir, existing, timeout_s=3.0)
        if run_root is None:
            raise ValueError(errors[0] if errors else "timed out waiting for self-improve run directory")
        run_id = run_root.name
        state = SelfImproveRunState(
            run_id=run_id,
            run_root=run_root,
            master_root=master_root,
            goal=goal,
            input_ref=input_ref,
            sessions=int(sessions),
            batches=int(batches),
            workers=int(workers),
            no_merge=bool(no_merge),
            llm_provider=str(llm_provider or ""),
            thread=thread,
        )
        with self._self_improve_lock:
            self.self_improve_runs[run_id] = state
        return state

    def get_self_improve(self, run_id: str) -> SelfImproveRunState | None:
        run_id = str(run_id or "").strip()
        if not run_id:
            return None
        with self._self_improve_lock:
            return self.self_improve_runs.get(run_id)


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
        if parsed.path == "/":
            self._send_html(_INDEX_HTML)
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/self_improve/runs":
            runs = []
            for run in sorted(self.server.self_improve_runs.values(), key=lambda r: r.run_id):
                runs.append(
                    {
                        "run_id": run.run_id,
                        "run_root": str(run.run_root),
                        "goal": run.goal,
                        "llm": run.llm_provider,
                        "sessions": run.sessions,
                        "batches": run.batches,
                        "workers": run.workers,
                        "no_merge": run.no_merge,
                    }
                )
            self._send_json({"ok": True, "runs": runs})
            return
        if parsed.path == "/api/self_improve/status":
            query = parse_qs(parsed.query or "")
            run_id = (query.get("run_id") or [""])[0]
            run = self.server.get_self_improve(run_id)
            if run is None:
                self._send_json({"ok": False, "error": "unknown run_id"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = _read_json_file(run.run_root / "self_improve.json")
            if payload is None:
                self._send_json({"ok": True, "run_id": run.run_id, "status": "RUNNING"})
                return
            if isinstance(payload, dict):
                self._send_json({"ok": True, **payload})
                return
            self._send_json({"ok": True, "run_id": run.run_id, "status": "RUNNING"})
            return
        if parsed.path == "/api/self_improve/stream":
            query = parse_qs(parsed.query or "")
            run_id = (query.get("run_id") or [""])[0]
            session_id = (query.get("session_id") or [""])[0]
            run = self.server.get_self_improve(run_id)
            if run is None:
                self._send_json({"ok": False, "error": "unknown run_id"}, status=HTTPStatus.NOT_FOUND)
                return
            self._stream_self_improve_session(run, session_id=session_id)
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/send", "/api/self_improve/start"}:
            self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/send":
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                self._send_json({"ok": False, "error": "message must be a non-empty string"}, status=HTTPStatus.BAD_REQUEST)
                return
            history = payload.get("history")
            if history is not None and not isinstance(history, list):
                self._send_json({"ok": False, "error": "history must be a list"}, status=HTTPStatus.BAD_REQUEST)
                return
            response = self.server.handle_send(message.strip(), history)
            self._send_json({"ok": True, **response})
            return

        if parsed.path == "/api/self_improve/start":
            goal = payload.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                self._send_json({"ok": False, "error": "goal must be a non-empty string"}, status=HTTPStatus.BAD_REQUEST)
                return
            input_ref = payload.get("input")
            if input_ref is not None and not isinstance(input_ref, str):
                self._send_json({"ok": False, "error": "input must be a string"}, status=HTTPStatus.BAD_REQUEST)
                return
            llm_provider = str(payload.get("llm") or "mock").strip().lower()
            if llm_provider not in {"mock", "codex"}:
                self._send_json({"ok": False, "error": "llm must be one of: mock, codex"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                sessions = int(payload.get("sessions") or 4)
                batches = int(payload.get("batches") or 1)
                workers = int(payload.get("workers") or 4)
            except (TypeError, ValueError):
                self._send_json({"ok": False, "error": "sessions/batches/workers must be integers"}, status=HTTPStatus.BAD_REQUEST)
                return
            sessions = max(1, min(16, sessions))
            batches = max(1, min(8, batches))
            workers = max(1, min(16, workers))
            no_merge = payload.get("no_merge") is True
            try:
                run = self.server.start_self_improve(
                    goal=goal.strip(),
                    input_ref=input_ref.strip() if isinstance(input_ref, str) and input_ref.strip() else None,
                    sessions=max(1, sessions),
                    batches=max(1, batches),
                    workers=max(1, workers),
                    no_merge=bool(no_merge),
                    llm_provider=llm_provider,
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            session_ids = [
                f"{batch_index}-{index}"
                for batch_index in range(1, run.batches + 1)
                for index in range(1, run.sessions + 1)
            ]
            self._send_json(
                {
                    "ok": True,
                    "run_id": run.run_id,
                    "run_root": str(run.run_root),
                    "session_ids": session_ids,
                }
            )
            return

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

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "keep-alive")
        self.end_headers()

    def _sse_send(self, *, event: str, data: Any) -> bool:
        try:
            payload = json.dumps(data, ensure_ascii=False)
            chunk = f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _stream_self_improve_session(self, run: SelfImproveRunState, *, session_id: str) -> None:
        session_id = str(session_id or "").strip()
        if not session_id:
            self._send_json({"ok": False, "error": "session_id is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        session_dir = run.run_root / "sessions" / f"session-{session_id}"
        workspace_dir = session_dir / "workspace"

        self._send_sse_headers()
        if not self._sse_send(event="trace", data={"event_type": "stream_open", "payload": {"session_id": session_id}}):
            return

        deadline = time.monotonic() + 60.0
        trace_path: Path | None = None
        while time.monotonic() < deadline:
            trace_path = _latest_session_trace(workspace_dir)
            if trace_path is not None and trace_path.exists():
                break
            time.sleep(0.1)
        if trace_path is None or not trace_path.exists():
            self._sse_send(event="done", data={"error": "trace not found"})
            return

        last_keepalive = time.monotonic()
        last_line = time.monotonic()
        with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
            while True:
                line = handle.readline()
                if line:
                    last_line = time.monotonic()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not self._sse_send(event="trace", data=record):
                        return
                    continue

                now = time.monotonic()
                if now - last_keepalive >= 10.0:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    last_keepalive = now

                if _session_completed(session_dir) and now - last_line >= 0.5:
                    self._sse_send(event="done", data={"status": "COMPLETED"})
                    return
                time.sleep(0.1)


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


def _wait_for_new_run_dir(base_dir: Path, existing: set[str], *, timeout_s: float) -> Path | None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        candidates = [path for path in base_dir.glob("run-*") if path.is_dir() and path.name not in existing]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            return candidates[-1]
        time.sleep(0.05)
    return None


def _read_json_file(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
        return json.loads(content)
    except Exception:
        return None


def _latest_session_trace(workspace_dir: Path) -> Path | None:
    runs_dir = workspace_dir / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return None
    run_dirs = [path for path in runs_dir.glob("run-*") if path.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1] / "trace.jsonl"


def _session_completed(session_dir: Path) -> bool:
    payload = _read_json_file(session_dir / "session.json")
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().upper()
    return bool(status and status != "RUNNING")
