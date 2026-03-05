"""Tokimon Gateway server: HTTP + WebSocket control plane."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import secrets
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agents.worker import Worker
from llm.client import ClaudeCLIClient
from llm.client import ClaudeCLISettings
from llm.client import CodexCLIClient
from llm.client import CodexCLISettings
from llm.client import build_llm_client
from policy.dangerous_tools import tool_catalog
from tools.file_tool import FileTool
from tools.grep_tool import GrepTool
from tools.patch_tool import PatchTool
from tools.pytest_tool import PytestTool
from tools.web_tool import WebTool

PROTOCOL_VERSION = 1
_TICK_INTERVAL_MS = 15_000
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_WS_FRAME_BYTES = 2_000_000
_MAX_IDEMPOTENCY_ENTRIES = 128
_MAX_LOG_ENTRIES = 512
_PHASE1_WS_METHODS = ("health", "logs.tail", "methods.list", "send", "tools.catalog")


@dataclass(frozen=True)
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    llm_provider: str = "codex"
    workspace_dir: Path = field(default_factory=Path.cwd)
    dangerously_expose: bool = False
    auth_token: str | None = None


class _GatewayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], config: GatewayConfig) -> None:
        super().__init__(server_address, _GatewayHandler)
        self.config = config
        self.workspace_dir = config.workspace_dir.resolve()
        self.llm_provider = (config.llm_provider or "").strip().lower()
        token = config.auth_token if config.auth_token is not None else os.environ.get("TOKIMON_GATEWAY_AUTH_TOKEN")
        token = (token or "").strip() or None
        self.auth_token = token
        self.llm_client = build_llm_client(self.llm_provider, workspace_dir=self.workspace_dir)
        self.tools = {
            "file": FileTool(self.workspace_dir),
            "grep": GrepTool(self.workspace_dir),
            "patch": PatchTool(self.workspace_dir),
            "pytest": PytestTool(self.workspace_dir),
            "web": WebTool(),
        }
        self._log_lock = threading.Lock()
        self._log_entries: deque[dict[str, Any]] = deque(maxlen=_MAX_LOG_ENTRIES)
        self._log_next_id = 1

    def record_log(self, event: str, payload: dict[str, Any] | None = None) -> None:
        event = str(event or "").strip() or "event"
        entry: dict[str, Any] = {
            "id": None,
            "ts_ms": int(time.time() * 1000),
            "event": event,
            "payload": payload or {},
        }
        with self._log_lock:
            entry["id"] = self._log_next_id
            self._log_next_id += 1
            self._log_entries.append(entry)

    def tail_logs(self, *, limit: int, after: int | None = None) -> dict[str, Any]:
        limit = max(1, int(limit))
        with self._log_lock:
            cursor = self._log_next_id - 1
            entries = list(self._log_entries)
        if after is None:
            sliced = entries[-limit:]
        else:
            sliced = [entry for entry in entries if isinstance(entry.get("id"), int) and int(entry["id"]) > after][:limit]
        return {"entries": sliced, "cursor": cursor}

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
        self.record_log(
            "send",
            {"message_chars": len(message), "history_len": len(history), "model": (model or "").strip() or None},
        )
        memory = _history_to_memory(history)
        llm_client = self._llm_client_for_request(model)
        worker = Worker("Gateway", llm_client, self.tools)
        output = worker.run(
            goal=message,
            step_id="gateway.send",
            inputs={"message": message, "history": history, "model": model},
            memory=memory,
        )
        return {
            "status": output.status.value,
            "reply": output.summary,
            "summary": output.summary,
            "artifacts": output.artifacts,
            "metrics": output.metrics,
            "next_actions": output.next_actions,
            "failure_signature": output.failure_signature,
        }


class GatewayServer:
    def __init__(self, config: GatewayConfig) -> None:
        auth_token = config.auth_token if config.auth_token is not None else os.environ.get("TOKIMON_GATEWAY_AUTH_TOKEN")
        auth_token = (auth_token or "").strip() or None
        host = str(config.host or "").strip() or "127.0.0.1"
        if not _is_loopback_host(host):
            if not bool(config.dangerously_expose):
                raise ValueError(
                    f"refusing to bind Gateway to non-loopback host {host!r} without --dangerously-expose "
                    "and TOKIMON_GATEWAY_AUTH_TOKEN configured"
                )
            if not auth_token:
                raise ValueError(
                    f"refusing to bind Gateway to non-loopback host {host!r} without TOKIMON_GATEWAY_AUTH_TOKEN configured"
                )
        normalized = replace(config, host=host, auth_token=auth_token)
        self._server = _GatewayHTTPServer((normalized.host, int(normalized.port)), normalized)
        self._thread = threading.Thread(target=self._server.serve_forever, name="tokimon-gateway", daemon=True)

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


def run_gateway(config: GatewayConfig) -> None:
    server = GatewayServer(config)
    server.start()
    print(f"Tokimon Gateway: {server.url} (ws: {server.url}/gateway)")
    try:
        server._thread.join()
    except KeyboardInterrupt:
        server.stop()


class _GatewayHandler(BaseHTTPRequestHandler):
    server: _GatewayHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json({"ok": True})
            return
        if parsed.path == "/gateway" and _is_ws_upgrade(self.headers):
            self._handle_gateway_ws()
            return
        self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

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

    def _handle_gateway_ws(self) -> None:
        try:
            self.connection.settimeout(30)
        except Exception:
            pass
        key = self.headers.get("sec-websocket-key")
        if not isinstance(key, str) or not key.strip():
            self.send_error(HTTPStatus.BAD_REQUEST, "missing sec-websocket-key")
            return
        accept = base64.b64encode(hashlib.sha1((key.strip() + _WS_GUID).encode("utf-8")).digest()).decode("ascii")
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        try:
            self.wfile.flush()
        except Exception:
            return

        nonce = secrets.token_hex(16)
        ts_ms = int(time.time() * 1000)
        self._ws_send_json({"type": "event", "event": "connect.challenge", "payload": {"nonce": nonce, "ts": ts_ms}})

        connected = False
        idempotency_cache: dict[str, dict[str, Any]] = {}
        require_auth = bool(getattr(self.server, "auth_token", None))

        try:
            first = self._ws_recv_json()
            if first is None:
                return
            req_id, method, params = _parse_req_frame(first)
            if method != "connect":
                self._ws_send_res_error(req_id, "first method must be connect", details={"code": "CONNECT_REQUIRED"})
                return
            connect_errors = _validate_connect_params(params, require_auth=require_auth)
            if connect_errors:
                self._ws_send_res_error(req_id, "invalid connect params", details={"code": "CONNECT_INVALID", "errors": connect_errors})
                return
            echoed_nonce = str(params.get("challenge", {}).get("nonce") or "")
            if not secrets.compare_digest(echoed_nonce, nonce):
                self._ws_send_res_error(req_id, "connect.challenge nonce mismatch", details={"code": "CHALLENGE_MISMATCH"})
                return
            if require_auth:
                auth = params.get("auth") or {}
                credential = ""
                if isinstance(auth, dict):
                    token = auth.get("token")
                    if isinstance(token, str):
                        credential = token
                    else:
                        credential = str(auth.get("credential") or "")
                expected = str(getattr(self.server, "auth_token") or "")
                if not expected or not secrets.compare_digest(credential, expected):
                    self._ws_send_res_error(req_id, "unauthorized", details={"code": "AUTH_INVALID"})
                    return
            min_protocol = int(params["minProtocol"])
            max_protocol = int(params["maxProtocol"])
            if not (min_protocol <= PROTOCOL_VERSION <= max_protocol):
                self._ws_send_res_error(
                    req_id,
                    "protocol version mismatch",
                    details={"code": "PROTOCOL_VERSION_MISMATCH", "server": PROTOCOL_VERSION, "min": min_protocol, "max": max_protocol},
                )
                return
            self._ws_send_json(
                {
                    "type": "res",
                    "id": req_id,
                    "ok": True,
                    "payload": {"type": "hello-ok", "protocol": PROTOCOL_VERSION, "policy": {"tickIntervalMs": _TICK_INTERVAL_MS}},
                }
            )
            try:
                self.server.record_log(
                    "connect.ok",
                    {"client": params.get("client"), "role": params.get("role"), "scopes": params.get("scopes")},
                )
            except Exception:
                pass
            connected = True

            while True:
                frame = self._ws_recv_json()
                if frame is None:
                    return
                req_id, method, params = _parse_req_frame(frame)
                if not connected:
                    self._ws_send_res_error(req_id, "not connected", details={"code": "NOT_CONNECTED"})
                    continue
                if method == "health":
                    self._ws_send_json({"type": "res", "id": req_id, "ok": True, "payload": {"ok": True}})
                    continue
                if method == "methods.list":
                    self._ws_send_json(
                        {
                            "type": "res",
                            "id": req_id,
                            "ok": True,
                            "payload": {"methods": list(_PHASE1_WS_METHODS)},
                        }
                    )
                    continue
                if method == "tools.catalog":
                    self._ws_send_json(
                        {
                            "type": "res",
                            "id": req_id,
                            "ok": True,
                            "payload": {"tools": tool_catalog()},
                        }
                    )
                    continue
                if method == "logs.tail":
                    tail_errors = _validate_logs_tail_params(params)
                    if tail_errors:
                        self._ws_send_res_error(req_id, "invalid logs.tail params", details={"code": "LOGS_TAIL_INVALID", "errors": tail_errors})
                        continue
                    limit = int(params.get("limit") or 200)
                    after = params.get("after")
                    payload = self.server.tail_logs(limit=limit, after=int(after) if isinstance(after, int) else None)
                    self._ws_send_json({"type": "res", "id": req_id, "ok": True, "payload": payload})
                    continue
                if method == "send":
                    send_errors = _validate_send_params(params)
                    if send_errors:
                        self._ws_send_res_error(req_id, "invalid send params", details={"code": "SEND_INVALID", "errors": send_errors})
                        continue
                    idem_key = str(params["idempotencyKey"]).strip()
                    if idem_key in idempotency_cache:
                        self._ws_send_json({"type": "res", "id": req_id, "ok": True, "payload": idempotency_cache[idem_key]})
                        continue
                    message = str(params["message"]).strip()
                    history = params.get("history")
                    if history is not None and not isinstance(history, list):
                        self._ws_send_res_error(req_id, "history must be a list", details={"code": "SEND_INVALID_HISTORY"})
                        continue
                    model = params.get("model")
                    if model is not None and not isinstance(model, str):
                        self._ws_send_res_error(req_id, "model must be a string", details={"code": "SEND_INVALID_MODEL"})
                        continue
                    payload = self.server.handle_send(message, history, model=(model or "").strip() or None)
                    idempotency_cache[idem_key] = payload
                    if len(idempotency_cache) > _MAX_IDEMPOTENCY_ENTRIES:
                        idempotency_cache.pop(next(iter(idempotency_cache)))
                    self._ws_send_json({"type": "res", "id": req_id, "ok": True, "payload": payload})
                    continue

                self._ws_send_res_error(req_id, f"unknown method: {method}", details={"code": "METHOD_NOT_FOUND"})
        except (ValueError, ConnectionError) as exc:
            try:
                self.server.record_log("error", {"message": "ws connection error", "details": {"exception": exc.__class__.__name__}})
            except Exception:
                pass
            return
        finally:
            self.close_connection = True

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

    def _send_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ws_send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self._ws_send_frame(0x1, data)

    def _ws_send_res_error(self, req_id: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        error: dict[str, Any] = {"message": message}
        if details:
            error["details"] = details
        try:
            self.server.record_log("error", {"message": message, "details": details or {}})
        except Exception:
            pass
        self._ws_send_json({"type": "res", "id": req_id, "ok": False, "error": error})

    def _ws_send_frame(self, opcode: int, payload: bytes) -> None:
        header = _encode_ws_frame(opcode=opcode, payload=payload, mask=False)
        try:
            self.connection.sendall(header)
        except Exception:
            raise ConnectionError from None

    def _ws_recv_json(self) -> dict[str, Any] | None:
        while True:
            opcode, payload = self._ws_recv_frame()
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._ws_send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode != 0x1:
                raise ValueError("expected text frame")
            try:
                obj = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                raise ValueError("invalid json frame") from exc
            if not isinstance(obj, dict):
                raise ValueError("frame must be an object")
            return obj

    def _ws_recv_frame(self) -> tuple[int, bytes]:
        b1b2 = _read_exact(self.rfile, 2)
        b1 = b1b2[0]
        b2 = b1b2[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if not fin:
            raise ValueError("fragmented frames not supported")
        if length == 126:
            length = struct.unpack("!H", _read_exact(self.rfile, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _read_exact(self.rfile, 8))[0]
        if length > _MAX_WS_FRAME_BYTES:
            raise ValueError("frame too large")
        mask_key = _read_exact(self.rfile, 4) if masked else b""
        payload = _read_exact(self.rfile, length) if length else b""
        if masked and payload:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return opcode, payload


def _is_ws_upgrade(headers: Any) -> bool:
    upgrade = str(getattr(headers, "get", lambda _k, _d=None: None)("upgrade", "") or "").lower()
    connection = str(getattr(headers, "get", lambda _k, _d=None: None)("connection", "") or "").lower()
    key = getattr(headers, "get", lambda _k, _d=None: None)("sec-websocket-key")
    return upgrade == "websocket" and "upgrade" in connection and isinstance(key, str) and bool(key.strip())


def _encode_ws_frame(*, opcode: int, payload: bytes, mask: bool) -> bytes:
    fin_opcode = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", fin_opcode, (0x80 if mask else 0x00) | length)
        ext = b""
    elif length < (1 << 16):
        header = struct.pack("!BB", fin_opcode, (0x80 if mask else 0x00) | 126)
        ext = struct.pack("!H", length)
    else:
        header = struct.pack("!BB", fin_opcode, (0x80 if mask else 0x00) | 127)
        ext = struct.pack("!Q", length)
    if not mask:
        return header + ext + payload
    mask_key = secrets.token_bytes(4)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return header + ext + mask_key + masked


def _read_exact(stream: Any, n: int) -> bytes:
    if n <= 0:
        return b""
    data = stream.read(n)
    if not data or len(data) != n:
        raise ConnectionError("socket closed")
    return data


def _parse_req_frame(frame: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if frame.get("type") != "req":
        raise ValueError("frame.type must be 'req'")
    req_id = frame.get("id")
    if not isinstance(req_id, str) or not req_id.strip():
        raise ValueError("frame.id must be a non-empty string")
    method = frame.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("frame.method must be a non-empty string")
    params = frame.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("frame.params must be an object")
    return req_id.strip(), method.strip(), params


def _validate_connect_params(params: dict[str, Any], *, require_auth: bool) -> list[str]:
    errors: list[str] = []
    min_protocol = params.get("minProtocol")
    max_protocol = params.get("maxProtocol")
    if not isinstance(min_protocol, int):
        errors.append("minProtocol must be an int")
    if not isinstance(max_protocol, int):
        errors.append("maxProtocol must be an int")
    challenge = params.get("challenge")
    if not isinstance(challenge, dict):
        errors.append("challenge must be an object")
    else:
        nonce = challenge.get("nonce")
        if not isinstance(nonce, str) or not nonce.strip():
            errors.append("challenge.nonce must be a non-empty string")
    client = params.get("client")
    if not isinstance(client, dict):
        errors.append("client must be an object")
    else:
        for key in ("id", "version", "platform", "mode"):
            value = client.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"client.{key} must be a non-empty string")
    role = params.get("role")
    if not isinstance(role, str) or not role.strip():
        errors.append("role must be a non-empty string")
    auth = params.get("auth")
    if require_auth:
        if not isinstance(auth, dict):
            errors.append("auth must be an object")
        else:
            has_token = "token" in auth
            token = auth.get("token")
            has_credential = "credential" in auth or "mode" in auth
            mode = auth.get("mode")
            credential = auth.get("credential")
            token_ok = isinstance(token, str) and bool(token.strip())
            credential_ok = mode == "token" and isinstance(credential, str) and bool(credential.strip())
            if not (token_ok or credential_ok):
                if has_token:
                    errors.append("auth.token must be a non-empty string")
                if has_credential:
                    if mode != "token":
                        errors.append("auth.mode must be 'token'")
                    if not isinstance(credential, str) or not credential.strip():
                        errors.append("auth.credential must be a non-empty string")
                if not has_token and not has_credential:
                    errors.append("auth must be {mode:'token', credential:'...'} or {token:'...'}")
    scopes = params.get("scopes")
    if scopes is not None and not isinstance(scopes, list):
        errors.append("scopes must be a list")
    if "caps" in params:
        caps = params.get("caps")
        if not isinstance(caps, list):
            errors.append("caps must be a list")
        elif not all(isinstance(item, str) for item in caps):
            errors.append("caps entries must be strings")
    if "commands" in params:
        commands = params.get("commands")
        if not isinstance(commands, list):
            errors.append("commands must be a list")
        elif not all(isinstance(item, str) for item in commands):
            errors.append("commands entries must be strings")
    if "permissions" in params:
        permissions = params.get("permissions")
        if not isinstance(permissions, dict):
            errors.append("permissions must be an object")
        else:
            if not all(isinstance(key, str) for key in permissions.keys()):
                errors.append("permissions keys must be strings")
            if not all(isinstance(value, bool) for value in permissions.values()):
                errors.append("permissions values must be bools")
    if "locale" in params and not isinstance(params.get("locale"), str):
        errors.append("locale must be a string")
    if "userAgent" in params and not isinstance(params.get("userAgent"), str):
        errors.append("userAgent must be a string")
    if "device" in params and not isinstance(params.get("device"), dict):
        errors.append("device must be an object")
    return errors


def _is_loopback_host(host: str) -> bool:
    host = str(host or "").strip().lower()
    if not host:
        return True
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_send_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    message = params.get("message")
    if not isinstance(message, str) or not message.strip():
        errors.append("message must be a non-empty string")
    idempotency_key = params.get("idempotencyKey")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        errors.append("idempotencyKey must be a non-empty string")
    history = params.get("history")
    if history is not None and not isinstance(history, list):
        errors.append("history must be a list")
    model = params.get("model")
    if model is not None and not isinstance(model, str):
        errors.append("model must be a string")
    return errors


def _validate_logs_tail_params(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    limit = params.get("limit")
    if limit is not None and not isinstance(limit, int):
        errors.append("limit must be an int")
    if isinstance(limit, int) and limit <= 0:
        errors.append("limit must be a positive int")
    after = params.get("after")
    if after is not None and not isinstance(after, int):
        errors.append("after must be an int")
    if isinstance(after, int) and after < 0:
        errors.append("after must be a non-negative int")
    return errors


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
