from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time
import urllib.request
from pathlib import Path

import pytest

from gateway.server import GatewayConfig, GatewayServer
from llm.client import MockLLMClient

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


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


class _WSClient:
    def __init__(self, sock: socket.socket, initial: bytes) -> None:
        self.sock = sock
        self.buf = initial

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def send_json(self, obj: dict) -> None:
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        frame = _encode_ws_frame(opcode=0x1, payload=payload, mask=True)
        self.sock.sendall(frame)

    def recv_json(self, *, timeout_s: float = 2.0) -> dict:
        self.sock.settimeout(timeout_s)
        while True:
            opcode, payload = self._recv_frame()
            if opcode == 0x8:
                raise AssertionError("socket closed")
            if opcode == 0x9:
                self.sock.sendall(_encode_ws_frame(opcode=0xA, payload=payload, mask=True))
                continue
            if opcode == 0xA:
                continue
            assert opcode == 0x1
            return json.loads(payload.decode("utf-8"))

    def _recv_frame(self) -> tuple[int, bytes]:
        b1b2 = self._read_exact(2)
        b1 = b1b2[0]
        b2 = b1b2[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        assert fin is True
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask_key = self._read_exact(4) if masked else b""
        payload = self._read_exact(length) if length else b""
        if masked and payload:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError("socket closed")
            self.buf += chunk
        data = self.buf[:n]
        self.buf = self.buf[n:]
        return data


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
    mask_key = os.urandom(4)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return header + ext + mask_key + masked


def _ws_connect(host: str, port: int, *, path: str = "/gateway", timeout_s: float = 2.0) -> _WSClient:
    sock = socket.create_connection((host, port), timeout=timeout_s)
    sock.settimeout(timeout_s)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = "\r\n".join(
        [
            f"GET {path} HTTP/1.1",
            f"Host: {host}:{port}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {key}",
            "Sec-WebSocket-Version: 13",
            "",
            "",
        ]
    ).encode("utf-8")
    sock.sendall(req)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise AssertionError("handshake failed: socket closed")
        data += chunk
        if len(data) > 65536:
            raise AssertionError("handshake failed: headers too large")
    header_bytes, rest = data.split(b"\r\n\r\n", 1)
    header_text = header_bytes.decode("utf-8", errors="replace")
    lines = header_text.split("\r\n")
    assert lines and "101" in lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()
    accept = headers.get("sec-websocket-accept")
    expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("utf-8")).digest()).decode("ascii")
    assert accept == expected
    return _WSClient(sock, rest)


def test_gateway_ws_handshake_health_and_send(tmp_path: Path) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
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

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            assert challenge["type"] == "event"
            assert challenge["event"] == "connect.challenge"
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce
            assert isinstance(challenge.get("payload", {}).get("ts"), int)

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": nonce},
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "caps": ["pytest"],
                        "commands": ["health", "send"],
                        "permissions": {"operator.read": True},
                        "locale": "en-US",
                        "userAgent": "pytest",
                        "device": {"id": "pytest"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is True
            assert hello["payload"]["protocol"] == 1

            ws.send_json({"type": "req", "id": "2", "method": "methods.list", "params": {}})
            methods = ws.recv_json()
            assert methods["type"] == "res"
            assert methods["id"] == "2"
            assert methods["ok"] is True
            assert methods["payload"]["methods"] == ["health", "logs.tail", "methods.list", "send", "tools.catalog"]

            ws.send_json({"type": "req", "id": "3", "method": "tools.catalog", "params": {}})
            catalog = ws.recv_json()
            assert catalog["type"] == "res"
            assert catalog["id"] == "3"
            assert catalog["ok"] is True
            assert catalog["payload"]["tools"] == [
                {
                    "tool": "file",
                    "action": "read",
                    "risk_tier": "low",
                    "requires_approval": False,
                    "notes": "read-only workspace access",
                },
                {
                    "tool": "file",
                    "action": "write",
                    "risk_tier": "high",
                    "requires_approval": True,
                    "notes": "writes to workspace",
                },
                {
                    "tool": "grep",
                    "action": "search",
                    "risk_tier": "low",
                    "requires_approval": False,
                    "notes": "bounded repo search",
                },
                {
                    "tool": "patch",
                    "action": "apply",
                    "risk_tier": "high",
                    "requires_approval": True,
                    "notes": "applies patches to workspace",
                },
                {
                    "tool": "pytest",
                    "action": "run",
                    "risk_tier": "medium",
                    "requires_approval": False,
                    "notes": "executes local tests",
                },
                {
                    "tool": "web",
                    "action": "fetch",
                    "risk_tier": "medium",
                    "requires_approval": False,
                    "notes": "network access",
                },
                {
                    "tool": "web",
                    "action": "search",
                    "risk_tier": "medium",
                    "requires_approval": False,
                    "notes": "network access",
                },
            ]

            ws.send_json({"type": "req", "id": "4", "method": "health", "params": {}})
            health = ws.recv_json()
            assert health["type"] == "res"
            assert health["id"] == "4"
            assert health["ok"] is True
            assert health["payload"]["ok"] is True

            ws.send_json(
                {
                    "type": "req",
                    "id": "5",
                    "method": "send",
                    "params": {"idempotencyKey": "k1", "message": "hello", "history": []},
                }
            )
            sent = ws.recv_json(timeout_s=10.0)
            assert sent["type"] == "res"
            assert sent["id"] == "5"
            assert sent["ok"] is True
            assert sent["payload"]["reply"] == "hello from scripted llm"
            assert sent["payload"]["status"] == "SUCCESS"
        finally:
            ws.close()
    finally:
        server.stop()


def test_gateway_ws_rejects_challenge_mismatch(tmp_path: Path) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": f"{nonce}-bad"},
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is False
        finally:
            ws.close()
    finally:
        server.stop()


def test_gateway_ws_rejects_missing_auth_when_enabled(tmp_path: Path) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path, auth_token="secret")
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": nonce},
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is False
        finally:
            ws.close()
    finally:
        server.stop()

@pytest.mark.parametrize(
    "auth",
    [
        {"mode": "token", "credential": "secret"},
        {"token": "secret"},
    ],
)
def test_gateway_ws_allows_auth_when_enabled(tmp_path: Path, auth: dict) -> None:
    config = GatewayConfig(
        host="127.0.0.1",
        port=0,
        llm_provider="codex",
        workspace_dir=tmp_path,
        auth_token="secret",
    )
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": nonce},
                        "auth": auth,
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is True

            ws.send_json({"type": "req", "id": "2", "method": "health", "params": {}})
            health = ws.recv_json()
            assert health["type"] == "res"
            assert health["id"] == "2"
            assert health["ok"] is True
            assert health["payload"]["ok"] is True
        finally:
            ws.close()
    finally:
        server.stop()


def test_gateway_ws_rejects_invalid_auth_shape_when_enabled(tmp_path: Path) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path, auth_token="secret")
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": nonce},
                        "auth": {"token": 123},
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is False
        finally:
            ws.close()
    finally:
        server.stop()


def test_gateway_ws_rejects_connect_optional_param_type_mismatch(tmp_path: Path) -> None:
    config = GatewayConfig(host="127.0.0.1", port=0, llm_provider="codex", workspace_dir=tmp_path)
    try:
        server = GatewayServer(config)
    except PermissionError as exc:
        pytest.skip(f"socket operations not permitted in this environment: {exc}")
    server.start()
    try:
        _wait_for_healthz(server.url)

        ws = _ws_connect(server.host, server.port)
        try:
            challenge = ws.recv_json()
            nonce = challenge.get("payload", {}).get("nonce")
            assert isinstance(nonce, str)
            assert nonce

            ws.send_json(
                {
                    "type": "req",
                    "id": "1",
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "challenge": {"nonce": nonce},
                        "client": {"id": "pytest", "version": "0", "platform": "linux", "mode": "operator"},
                        "role": "operator",
                        "scopes": ["operator.read"],
                        "locale": 123,
                    },
                }
            )
            hello = ws.recv_json()
            assert hello["type"] == "res"
            assert hello["id"] == "1"
            assert hello["ok"] is False
        finally:
            ws.close()
    finally:
        server.stop()
