from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
from typing import Any, Callable
from urllib.parse import urlparse

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_HANDSHAKE_BYTES = 65536
_MAX_WS_FRAME_BYTES = 2_000_000


class GatewayHealthError(RuntimeError):
    pass


def call_gateway_rpc(
    url: str,
    method: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_ms: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + max(1, int(timeout_ms)) / 1000.0
    host, port, path = _parse_gateway_ws_url(url)
    ws = _ws_connect(host, port, path, deadline=deadline)
    try:
        _gateway_ws_handshake(ws, deadline=deadline, log=log)
        req_id = "2"
        ws.send_json(
            {"type": "req", "id": req_id, "method": method, "params": params or {}},
            deadline=deadline,
        )
        response = ws.recv_json(deadline=deadline)
        if response.get("type") != "res" or response.get("id") != req_id:
            raise GatewayHealthError("rpc call failed")
        return response
    finally:
        ws.close()


def check_gateway_health(
    url: str,
    *,
    timeout_ms: int,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    deadline = start + max(1, int(timeout_ms)) / 1000.0
    details: dict[str, Any] = {}
    ok = False
    error: str | None = None
    try:
        _run_gateway_health_check(url, deadline=deadline, details=details, log=log)
        ok = True
    except Exception as exc:
        error = _format_exception(exc)
        details.setdefault("exception", exc.__class__.__name__)
        ok = False
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return {
        "ok": ok,
        "url": url,
        "elapsed_ms": elapsed_ms,
        "error": error,
        "details": details if not ok else {},
    }


def _run_gateway_health_check(
    url: str,
    *,
    deadline: float,
    details: dict[str, Any],
    log: Callable[[str], None] | None,
) -> None:
    details["step"] = "parse_url"
    host, port, path = _parse_gateway_ws_url(url)

    details["step"] = "connect"
    ws = _ws_connect(host, port, path, deadline=deadline)
    try:
        _gateway_ws_handshake(ws, deadline=deadline, log=log, details=details)

        details["step"] = "send_health"
        ws.send_json({"type": "req", "id": "2", "method": "health", "params": {}}, deadline=deadline)
        details["step"] = "recv_health"
        health = ws.recv_json(deadline=deadline)
        if health.get("type") != "res" or health.get("id") != "2" or health.get("ok") is not True:
            raise GatewayHealthError("health request failed")
        payload = health.get("payload")
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise GatewayHealthError("gateway health not ok")
        _safe_log(log, "health ok")
    finally:
        ws.close()


class _WSClient:
    def __init__(self, sock: socket.socket, initial: bytes) -> None:
        self.sock = sock
        self.buf = initial

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass

    def send_json(self, obj: dict[str, Any], *, deadline: float) -> None:
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        frame = _encode_ws_frame(opcode=0x1, payload=payload, mask=True)
        _set_socket_timeout(self.sock, deadline)
        self.sock.sendall(frame)

    def recv_json(self, *, deadline: float) -> dict[str, Any]:
        while True:
            opcode, payload = self._recv_frame(deadline=deadline)
            if opcode == 0x8:
                raise GatewayHealthError("socket closed")
            if opcode == 0x9:
                self._send_pong(payload, deadline=deadline)
                continue
            if opcode == 0xA:
                continue
            if opcode != 0x1:
                raise GatewayHealthError("expected text frame")
            try:
                obj = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                raise GatewayHealthError("invalid json frame") from exc
            if not isinstance(obj, dict):
                raise GatewayHealthError("frame must be an object")
            return obj

    def _send_pong(self, payload: bytes, *, deadline: float) -> None:
        frame = _encode_ws_frame(opcode=0xA, payload=payload, mask=True)
        _set_socket_timeout(self.sock, deadline)
        self.sock.sendall(frame)

    def _recv_frame(self, *, deadline: float) -> tuple[int, bytes]:
        b1b2 = self._read_exact(2, deadline=deadline)
        b1 = b1b2[0]
        b2 = b1b2[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        if not fin:
            raise GatewayHealthError("fragmented frames not supported")
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2, deadline=deadline))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8, deadline=deadline))[0]
        if length > _MAX_WS_FRAME_BYTES:
            raise GatewayHealthError("frame too large")
        mask_key = self._read_exact(4, deadline=deadline) if masked else b""
        payload = self._read_exact(length, deadline=deadline) if length else b""
        if masked and payload:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _read_exact(self, n: int, *, deadline: float) -> bytes:
        while len(self.buf) < n:
            _set_socket_timeout(self.sock, deadline)
            chunk = self.sock.recv(4096)
            if not chunk:
                raise GatewayHealthError("socket closed")
            self.buf += chunk
            if len(self.buf) > _MAX_WS_FRAME_BYTES + 1024:
                raise GatewayHealthError("incoming buffer too large")
        data = self.buf[:n]
        self.buf = self.buf[n:]
        return data


def _ws_connect(host: str, port: int, path: str, *, deadline: float) -> _WSClient:
    remaining = _remaining_seconds(deadline)
    sock = socket.create_connection((host, port), timeout=remaining)
    sock.settimeout(remaining)
    try:
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
            _set_socket_timeout(sock, deadline)
            chunk = sock.recv(4096)
            if not chunk:
                raise GatewayHealthError("handshake failed: socket closed")
            data += chunk
            if len(data) > _MAX_HANDSHAKE_BYTES:
                raise GatewayHealthError("handshake failed: headers too large")
        header_bytes, rest = data.split(b"\r\n\r\n", 1)
        header_text = header_bytes.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        if not lines or "101" not in lines[0]:
            raise GatewayHealthError("handshake failed: expected 101 response")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        accept = headers.get("sec-websocket-accept")
        expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("utf-8")).digest()).decode("ascii")
        if accept != expected:
            raise GatewayHealthError("handshake failed: sec-websocket-accept mismatch")
        return _WSClient(sock, rest)
    except Exception:
        try:
            sock.close()
        except Exception:
            pass
        raise


def _parse_gateway_ws_url(url: str) -> tuple[str, int, str]:
    parsed = urlparse(url)
    if parsed.scheme != "ws":
        raise ValueError("url must use ws:// scheme")
    if not parsed.hostname:
        raise ValueError("url must include a hostname")
    if parsed.port is None:
        raise ValueError("url must include a port")
    path = parsed.path
    if not path or path == "/":
        path = "/gateway"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.hostname, int(parsed.port), path


def _gateway_ws_handshake(
    ws: _WSClient,
    *,
    deadline: float,
    log: Callable[[str], None] | None,
    details: dict[str, Any] | None = None,
) -> None:
    if details is not None:
        details["step"] = "recv_challenge"
    challenge = ws.recv_json(deadline=deadline)
    if challenge.get("type") != "event" or challenge.get("event") != "connect.challenge":
        raise GatewayHealthError("expected connect.challenge event")
    _safe_log(log, "received connect.challenge")

    if details is not None:
        details["step"] = "send_connect"
    ws.send_json(
        {
            "type": "req",
            "id": "1",
            "method": "connect",
            "params": {
                "minProtocol": 1,
                "maxProtocol": 1,
                "client": {
                    "id": "tokimon",
                    "version": "0",
                    "platform": sys.platform,
                    "mode": "operator",
                },
                "role": "operator",
                "scopes": ["operator.read"],
            },
        },
        deadline=deadline,
    )
    if details is not None:
        details["step"] = "recv_connect"
    hello = ws.recv_json(deadline=deadline)
    if hello.get("type") != "res" or hello.get("id") != "1" or hello.get("ok") is not True:
        raise GatewayHealthError("connect failed")
    _safe_log(log, "connect ok")


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


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("timeout")
    return remaining


def _set_socket_timeout(sock: socket.socket, deadline: float) -> None:
    sock.settimeout(_remaining_seconds(deadline))


def _format_exception(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"


def _safe_log(log: Callable[[str], None] | None, message: str) -> None:
    if log is None:
        return
    try:
        log(message)
    except Exception:
        return
