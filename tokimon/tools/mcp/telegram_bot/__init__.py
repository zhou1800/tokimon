#!/usr/bin/env python3
"""Send-only Telegram Bot API MCP server for Tokimon."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import mimetypes
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from tokimon.tools.mcp._protocol import response, error_response, json_text, tool_result, handle_request, serve_stdio
from tokimon.runtime import runtime_root as repository_runtime_root, source_root

WORKSPACE_ROOT = source_root()
RUNTIME_ROOT = repository_runtime_root(WORKSPACE_ROOT)
DEFAULT_SECRETS_PATH = RUNTIME_ROOT / "secrets.local.json"
DEFAULT_LOG_PATH = RUNTIME_ROOT / "tools" / "mcp" / "telegram-bot" / "sends.jsonl"
DEFAULT_API_BASE_URL = "https://api.telegram.org"
ALLOWED_DOCUMENT_ROOTS_ENV = "TOKIMON_TELEGRAM_ALLOWED_DOCUMENT_ROOTS"
MAX_MESSAGE_CHARS = 4096
MAX_DOCUMENT_CAPTION_CHARS = 1024
SERVER_NAME = "tokimon-telegram-bot"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSION = "2025-03-26"
LOGGER_NAME = "tokimon.tools.mcp.telegram_bot"


class TelegramBotError(RuntimeError):
    """Base error for Telegram bot tool failures."""


class TelegramConfigError(TelegramBotError):
    """Raised when local Telegram configuration is missing or invalid."""


class TelegramAPIError(TelegramBotError):
    """Raised when Telegram returns an API or HTTP failure."""

    def __init__(
        self,
        message: str,
        *,
        error_code: int | None = None,
        retry_after_sec: int | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after_sec = retry_after_sec
        self.payload = dict(payload or {})


@dataclass(frozen=True)
class TelegramSecrets:
    bot_token: str
    token_source: str
    secrets_path: Path
    allowed_chats: Mapping[str, str]
    default_chat_id: str | None = None
    default_chat: str | None = None
    mode: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatDestination:
    chat_id: str
    label: str
    source: str


@dataclass(frozen=True)
class DocumentUpload:
    name: str
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TelegramConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TelegramConfigError(f"Telegram secrets file must contain a JSON object: {path}")
    return payload


def _mode_details(path: Path) -> tuple[str | None, tuple[str, ...]]:
    if not path.exists() or os.name != "posix":
        return None, ()
    mode = stat.S_IMODE(path.stat().st_mode)
    mode_text = f"{mode:04o}"
    if mode != 0o600:
        return mode_text, (f"{path} should use file mode 0600; current mode is {mode_text}.",)
    return mode_text, ()


def _coerce_chat_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _extract_allowed_chats(raw_chats: object) -> dict[str, str]:
    if not isinstance(raw_chats, Mapping):
        return {}
    chats: dict[str, str] = {}
    for raw_name, raw_value in raw_chats.items():
        name = str(raw_name).strip()
        if not name:
            continue
        chat_id: str | None
        if isinstance(raw_value, Mapping):
            chat_id = _coerce_chat_id(
                raw_value.get("chat_id") or raw_value.get("chatId") or raw_value.get("id")
            )
        else:
            chat_id = _coerce_chat_id(raw_value)
        if chat_id:
            chats[name] = chat_id
    return chats


def load_secrets(
    path: Path = DEFAULT_SECRETS_PATH,
    *,
    env: Mapping[str, str] | None = None,
) -> TelegramSecrets:
    """Load local Telegram secrets without printing or returning raw file data."""

    effective_env = dict(os.environ if env is None else env)
    payload = _read_json_file(path)
    telegram = payload.get("telegram", {})
    if not isinstance(telegram, Mapping):
        raise TelegramConfigError("'telegram' in secrets file must be a JSON object.")

    file_token = (
        telegram.get("bot_token")
        or telegram.get("botToken")
        or payload.get("telegram_bot_token")
        or payload.get("telegramBotToken")
    )
    env_token = effective_env.get("TELEGRAM_BOT_TOKEN")
    token = str(env_token or file_token or "").strip()
    token_source = "env:TELEGRAM_BOT_TOKEN" if env_token else "file"
    if not token:
        raise TelegramConfigError(
            "Telegram bot token is not configured. Set TELEGRAM_BOT_TOKEN or .tokimon/secrets.local.json telegram.bot_token."
        )

    mode, warnings = _mode_details(path)
    allowed_chats = _extract_allowed_chats(telegram.get("allowed_chats") or telegram.get("chats"))
    default_chat_id = _coerce_chat_id(telegram.get("default_chat_id") or telegram.get("defaultChatId"))
    default_chat = str(telegram.get("default_chat") or telegram.get("defaultChat") or "").strip() or None

    return TelegramSecrets(
        bot_token=token,
        token_source=token_source,
        secrets_path=path,
        allowed_chats=allowed_chats,
        default_chat_id=default_chat_id,
        default_chat=default_chat,
        mode=mode,
        warnings=tuple(warnings),
    )


def resolve_destination(
    secrets: TelegramSecrets,
    *,
    chat: str | None = None,
    chat_id: str | int | None = None,
) -> ChatDestination:
    explicit_chat_id = _coerce_chat_id(chat_id)
    chat_name = (chat or "").strip() or None
    if explicit_chat_id and chat_name:
        raise TelegramConfigError("Use either --chat or --chat-id, not both.")
    if explicit_chat_id:
        return ChatDestination(chat_id=explicit_chat_id, label="explicit", source="chat_id")
    if chat_name:
        configured = secrets.allowed_chats.get(chat_name)
        if not configured:
            known = ", ".join(sorted(secrets.allowed_chats)) or "none"
            raise TelegramConfigError(f"Unknown Telegram chat name '{chat_name}'. Known chats: {known}.")
        return ChatDestination(chat_id=configured, label=chat_name, source="allowed_chats")
    if secrets.default_chat_id:
        return ChatDestination(chat_id=secrets.default_chat_id, label="default", source="default_chat_id")
    if secrets.default_chat:
        configured = secrets.allowed_chats.get(secrets.default_chat)
        if configured:
            return ChatDestination(chat_id=configured, label=secrets.default_chat, source="default_chat")
        raise TelegramConfigError(
            f"Default Telegram chat '{secrets.default_chat}' is not present in allowed_chats."
        )
    raise TelegramConfigError("No Telegram chat selected. Use --chat, --chat-id, or configure telegram.default_chat_id.")


UrlOpen = Callable[[urllib.request.Request, float], Any]


class TelegramBotClient:
    """Small Telegram Bot API client with injectable transport for tests."""

    def __init__(
        self,
        bot_token: str,
        *,
        api_base_url: str = DEFAULT_API_BASE_URL,
        timeout_seconds: float = 20.0,
        urlopen: UrlOpen | None = None,
    ) -> None:
        token = bot_token.strip()
        if not token:
            raise TelegramConfigError("Telegram bot token is empty.")
        self.bot_token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen or urllib.request.urlopen

    @classmethod
    def from_secrets(cls, secrets: TelegramSecrets, **kwargs: Any) -> "TelegramBotClient":
        return cls(secrets.bot_token, **kwargs)

    def get_me(self) -> Mapping[str, Any]:
        return self._request("getMe", {})

    def get_chat(self, chat_id: str | int) -> Mapping[str, Any]:
        return self._request("getChat", {"chat_id": str(chat_id)})

    def send_message(self, chat_id: str | int, text: str) -> Mapping[str, Any]:
        validate_message_text(text)
        return self._request("sendMessage", {"chat_id": str(chat_id), "text": text})

    def send_document(
        self,
        chat_id: str | int,
        document_path: Path | DocumentUpload,
        *,
        caption: str | None = None,
    ) -> Mapping[str, Any]:
        document = read_document(document_path)
        clean_caption = validate_document_caption(caption)
        params = {"chat_id": str(chat_id)}
        if clean_caption is not None:
            params["caption"] = clean_caption
        return self._multipart_request("sendDocument", params, file_field="document", document=document)

    def _request(self, method: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        url = f"{self.api_base_url}/bot{urllib.parse.quote(self.bot_token, safe='')}/{method}"
        data = urllib.parse.urlencode(params).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        return self._execute_request(request)

    def _multipart_request(
        self,
        method: str,
        params: Mapping[str, str],
        *,
        file_field: str,
        document: DocumentUpload,
    ) -> Mapping[str, Any]:
        url = f"{self.api_base_url}/bot{urllib.parse.quote(self.bot_token, safe='')}/{method}"
        content_type, body = build_multipart_body(params, file_field=file_field, file_path=document)
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", content_type)
        request.add_header("Content-Length", str(len(body)))
        return self._execute_request(request)

    def _execute_request(self, request: urllib.request.Request) -> Mapping[str, Any]:
        try:
            with self._urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError as read_exc:
                raise TelegramAPIError(
                    sanitize_for_logs(f"Telegram response read failed: {read_exc}", self.bot_token)
                ) from read_exc
            payload = _parse_response_payload(body)
            raise _api_error_from_payload(payload, fallback_code=exc.code, token=self.bot_token) from exc
        except urllib.error.URLError as exc:
            raise TelegramAPIError(sanitize_for_logs(f"Telegram request failed: {exc.reason}", self.bot_token)) from exc
        except (OSError, UnicodeError) as exc:
            raise TelegramAPIError(
                sanitize_for_logs(f"Telegram request failed: {exc}", self.bot_token)
            ) from exc

        payload = _parse_response_payload(raw_body)
        if payload.get("ok") is not True:
            raise _api_error_from_payload(payload, fallback_code=None, token=self.bot_token)
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise TelegramAPIError("Telegram response did not include an object result.", payload=payload)
        return result


def _parse_response_payload(raw_body: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise TelegramAPIError(f"Telegram response was not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TelegramAPIError("Telegram response was not a JSON object.")
    return payload


def _api_error_from_payload(
    payload: Mapping[str, Any],
    *,
    fallback_code: int | None,
    token: str,
) -> TelegramAPIError:
    error_code = payload.get("error_code")
    if not isinstance(error_code, int):
        error_code = fallback_code
    parameters = payload.get("parameters")
    retry_after: int | None = None
    if isinstance(parameters, Mapping) and isinstance(parameters.get("retry_after"), int):
        retry_after = parameters["retry_after"]
    description = str(payload.get("description") or "Telegram API request failed.")
    return TelegramAPIError(
        sanitize_for_logs(description, token),
        error_code=error_code,
        retry_after_sec=retry_after,
        payload=payload,
    )


def sanitize_for_logs(value: object, token: str | None = None) -> str:
    text = str(value)
    if token:
        text = text.replace(token, "[redacted-token]")
        text = text.replace(urllib.parse.quote(token, safe=""), "[redacted-token]")
    return text


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_message_text(text: str) -> None:
    if not text:
        raise TelegramConfigError("Telegram message text must not be empty.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise TelegramConfigError(
            f"Telegram message text is too long: {len(text)} chars, max {MAX_MESSAGE_CHARS}."
        )


def validate_document_caption(caption: str | None) -> str | None:
    if caption is None:
        return None
    clean_caption = caption.strip()
    if not clean_caption:
        return None
    if len(clean_caption) > MAX_DOCUMENT_CAPTION_CHARS:
        raise TelegramConfigError(
            f"Telegram document caption is too long: {len(clean_caption)} chars, max {MAX_DOCUMENT_CAPTION_CHARS}."
        )
    return clean_caption


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def configured_adopter_document_roots(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    raw_roots = (os.environ if env is None else env).get(ALLOWED_DOCUMENT_ROOTS_ENV, "")
    roots: list[Path] = []
    for raw_root in raw_roots.split(os.pathsep):
        if not raw_root.strip():
            continue
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            raise TelegramConfigError("Configured Telegram document roots must be absolute existing directories.")
        resolved = root.resolve(strict=False)
        if not resolved.is_dir():
            raise TelegramConfigError("Configured Telegram document roots must be absolute existing directories.")
        roots.append(resolved)
    return tuple(roots)


def validate_document_path(
    path: Path,
    *,
    enforce_local_boundary: bool = False,
    env: Mapping[str, str] | None = None,
) -> Path:
    try:
        expanded = path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise TelegramConfigError(f"Telegram document path could not be resolved: {path}") from exc
    if not expanded.exists():
        raise TelegramConfigError(f"Telegram document file does not exist: {expanded}")
    if not expanded.is_file():
        raise TelegramConfigError(f"Telegram document path is not a file: {expanded}")
    if not enforce_local_boundary:
        return expanded
    allowed_roots = (WORKSPACE_ROOT, RUNTIME_ROOT, *configured_adopter_document_roots(env))
    if not any(_path_within(expanded, root) for root in allowed_roots):
        raise TelegramConfigError(
            "Telegram document file must be under the Tokimon repository, .tokimon runtime root, "
            "or a configured adopter document root."
        )
    return expanded


def read_document(
    path: Path | DocumentUpload,
    *,
    enforce_local_boundary: bool = False,
) -> DocumentUpload:
    if isinstance(path, DocumentUpload):
        return path
    approved = validate_document_path(path, enforce_local_boundary=enforce_local_boundary)
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise TelegramConfigError("Telegram document reads require POSIX no-follow filesystem operations.")
    directory_fd = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(approved.anchor, flags)
        for component in approved.parts[1:-1]:
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(approved.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(file_fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise TelegramConfigError("Telegram document must be a regular file.")
            return DocumentUpload(path.name, handle.read())
    except OSError as exc:
        raise TelegramConfigError(f"Telegram document could not be opened safely: {approved}") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_multipart_body(
    params: Mapping[str, str],
    *,
    file_field: str,
    file_path: Path | DocumentUpload,
) -> tuple[str, bytes]:
    document = read_document(file_path)
    boundary = f"tokimon-telegram-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in params.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    filename = document.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    header_filename = filename.replace("\\", "%5C").replace('"', "%22").replace("\r", "%0D").replace("\n", "%0A")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{header_filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            document.content,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def append_jsonl(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def build_log_entry(
    *,
    destination: ChatDestination,
    text: str,
    ok: bool,
    correlation_id: str | None = None,
    message_id: int | None = None,
    error: TelegramBotError | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    timestamp = utc_now_iso()
    entry: dict[str, Any] = {
        "attempted_at": timestamp,
        "timestamp": timestamp,
        "event": "send",
        "ok": ok,
        "service": "tokimon.tools.mcp.telegram_bot",
        "runtime": "mcp.telegram-bot",
        "severity": "info" if ok else "error",
        "message": "Telegram message send succeeded." if ok else "Telegram message send failed.",
        "correlation_id": correlation_id or uuid.uuid4().hex,
        "chat_label": destination.label,
        "chat_source": destination.source,
        "text_chars": len(text),
        "text_sha256": text_sha256(text),
    }
    if message_id is not None:
        entry["message_id"] = message_id
    if error is not None:
        entry["error"] = sanitize_for_logs(error, token)
        if isinstance(error, TelegramAPIError):
            if error.error_code is not None:
                entry["error_code"] = error.error_code
            if error.retry_after_sec is not None:
                entry["retry_after_sec"] = error.retry_after_sec
    return entry


def send_message_and_log(
    client: TelegramBotClient,
    destination: ChatDestination,
    text: str,
    *,
    log_path: Path = DEFAULT_LOG_PATH,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    try:
        result = client.send_message(destination.chat_id, text)
    except TelegramBotError as exc:
        entry = build_log_entry(
            destination=destination,
            text=text,
            ok=False,
            correlation_id=correlation_id,
            error=exc,
            token=client.bot_token,
        )
        append_jsonl(log_path, entry)
        raise

    raw_message_id = result.get("message_id")
    message_id = raw_message_id if isinstance(raw_message_id, int) else None
    entry = build_log_entry(
        destination=destination,
        text=text,
        ok=True,
        correlation_id=correlation_id,
        message_id=message_id,
    )
    append_jsonl(log_path, entry)
    return entry


def build_document_log_entry(
    *,
    destination: ChatDestination,
    document_path: Path | DocumentUpload,
    ok: bool,
    caption: str | None = None,
    correlation_id: str | None = None,
    message_id: int | None = None,
    error: TelegramBotError | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    document = read_document(document_path)
    clean_caption = validate_document_caption(caption)
    timestamp = utc_now_iso()
    entry: dict[str, Any] = {
        "attempted_at": timestamp,
        "timestamp": timestamp,
        "event": "send_document",
        "ok": ok,
        "service": "tokimon.tools.mcp.telegram_bot",
        "runtime": "mcp.telegram-bot",
        "severity": "info" if ok else "error",
        "message": "Telegram document send succeeded." if ok else "Telegram document send failed.",
        "correlation_id": correlation_id or uuid.uuid4().hex,
        "chat_label": destination.label,
        "chat_source": destination.source,
        "document_name": document.name,
        "document_bytes": len(document.content),
        "document_sha256": document.sha256,
        "caption_chars": len(clean_caption or ""),
    }
    if clean_caption:
        entry["caption_sha256"] = text_sha256(clean_caption)
    if message_id is not None:
        entry["message_id"] = message_id
    if error is not None:
        entry["error"] = sanitize_for_logs(error, token)
        if isinstance(error, TelegramAPIError):
            if error.error_code is not None:
                entry["error_code"] = error.error_code
            if error.retry_after_sec is not None:
                entry["retry_after_sec"] = error.retry_after_sec
    return entry


def send_document_and_log(
    client: TelegramBotClient,
    destination: ChatDestination,
    document_path: Path | DocumentUpload,
    *,
    caption: str | None = None,
    log_path: Path = DEFAULT_LOG_PATH,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    document = read_document(document_path)
    clean_caption = validate_document_caption(caption)
    try:
        result = client.send_document(destination.chat_id, document, caption=clean_caption)
    except TelegramBotError as exc:
        entry = build_document_log_entry(
            destination=destination,
            document_path=document,
            caption=clean_caption,
            ok=False,
            correlation_id=correlation_id,
            error=exc,
            token=client.bot_token,
        )
        append_jsonl(log_path, entry)
        raise

    raw_message_id = result.get("message_id")
    message_id = raw_message_id if isinstance(raw_message_id, int) else None
    entry = build_document_log_entry(
        destination=destination,
        document_path=document,
        caption=clean_caption,
        ok=True,
        correlation_id=correlation_id,
        message_id=message_id,
    )
    append_jsonl(log_path, entry)
    return entry


def tail_logs(log_path: Path = DEFAULT_LOG_PATH, *, limit: int = 20) -> list[Mapping[str, Any]]:
    if limit < 1:
        raise TelegramConfigError("--limit must be at least 1.")
    if not log_path.exists():
        return []
    recent: deque[str] = deque(maxlen=limit)
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                recent.append(stripped)
    entries: list[Mapping[str, Any]] = []
    for line in recent:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"ok": False, "event": "malformed-log-line"}
        if isinstance(payload, Mapping):
            entries.append(payload)
    return entries


def _correlation_id(raw_value: str | None = None) -> str:
    return (raw_value or "").strip() or uuid.uuid4().hex


def _string_argument(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TelegramConfigError(f"{name} must be a string.")
    stripped = value.strip()
    return stripped or None


def _required_string_argument(arguments: Mapping[str, Any], name: str) -> str:
    value = _string_argument(arguments, name)
    if value is None:
        raise TelegramConfigError(f"{name} is required.")
    return value


def _bool_argument(arguments: Mapping[str, Any], name: str, *, default: bool = False) -> bool:
    value = arguments.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TelegramConfigError(f"{name} must be a boolean.")
    return value


def _int_argument(arguments: Mapping[str, Any], name: str, *, default: int) -> int:
    value = arguments.get(name)
    if value is None:
        return default
    if not isinstance(value, int):
        raise TelegramConfigError(f"{name} must be an integer.")
    return value


def _destination_payload(destination: ChatDestination) -> dict[str, str]:
    return {
        "chat_label": destination.label,
        "chat_id": destination.chat_id,
        "chat_source": destination.source,
    }


def telegram_health_payload(
    *,
    secrets_path: Path | None = None,
    live: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    resolved_secrets_path = secrets_path or DEFAULT_SECRETS_PATH
    payload: dict[str, Any] = {
        "ok": True,
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol_version": SUPPORTED_PROTOCOL_VERSION,
        },
        "secrets_path": str(resolved_secrets_path),
        "token_configured": False,
        "configuration_valid": True,
        "ready_for_send": False,
        "live_check": bool(live),
        "warnings": [],
        "allowed_chats": [],
        "default_chat_configured": False,
    }
    try:
        secrets = load_secrets(resolved_secrets_path, env=env)
    except TelegramConfigError as exc:
        payload["configuration_valid"] = "bot token is not configured" in str(exc).lower()
        payload["configuration_error"] = str(exc)
        return payload

    payload.update(
        {
            "token_configured": True,
            "token_source": secrets.token_source,
            "secrets_mode": secrets.mode,
            "warnings": list(secrets.warnings),
            "allowed_chats": sorted(secrets.allowed_chats),
            "default_chat_configured": bool(secrets.default_chat_id or secrets.default_chat),
            "ready_for_send": True,
        }
    )
    if live:
        try:
            live_result = TelegramBotClient.from_secrets(secrets).get_me()
        except TelegramBotError as exc:
            payload["ok"] = False
            payload["live_error"] = str(exc)
        else:
            payload["bot"] = {
                "id": live_result.get("id"),
                "username": live_result.get("username"),
                "first_name": live_result.get("first_name"),
            }
    return payload


def telegram_get_me_payload() -> dict[str, Any]:
    secrets = load_secrets(DEFAULT_SECRETS_PATH)
    return {"ok": True, "bot": dict(TelegramBotClient.from_secrets(secrets).get_me())}


def telegram_get_chat_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    secrets = load_secrets(DEFAULT_SECRETS_PATH)
    destination = resolve_destination(
        secrets,
        chat=_string_argument(arguments, "chat"),
        chat_id=_string_argument(arguments, "chat_id"),
    )
    result = TelegramBotClient.from_secrets(secrets).get_chat(destination.chat_id)
    return {"ok": True, "destination": _destination_payload(destination), "chat": dict(result)}


def telegram_send_message_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    secrets = load_secrets(DEFAULT_SECRETS_PATH)
    destination = resolve_destination(
        secrets,
        chat=_string_argument(arguments, "chat"),
        chat_id=_string_argument(arguments, "chat_id"),
    )
    text = _required_string_argument(arguments, "text")
    validate_message_text(text)
    correlation_id = _correlation_id(_string_argument(arguments, "correlation_id"))
    if _bool_argument(arguments, "dry_run", default=False):
        return {
            "ok": True,
            "dry_run": True,
            "correlation_id": correlation_id,
            "destination": _destination_payload(destination),
            "text_chars": len(text),
            "text_sha256": text_sha256(text),
        }
    entry = send_message_and_log(
        TelegramBotClient.from_secrets(secrets),
        destination,
        text,
        log_path=DEFAULT_LOG_PATH,
        correlation_id=correlation_id,
    )
    return {"ok": True, "send": entry, "log_path": str(DEFAULT_LOG_PATH)}


def telegram_send_document_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    secrets = load_secrets(DEFAULT_SECRETS_PATH)
    destination = resolve_destination(
        secrets,
        chat=_string_argument(arguments, "chat"),
        chat_id=_string_argument(arguments, "chat_id"),
    )
    document = read_document(
        Path(_required_string_argument(arguments, "file")),
        enforce_local_boundary=True,
    )
    caption = validate_document_caption(_string_argument(arguments, "caption"))
    correlation_id = _correlation_id(_string_argument(arguments, "correlation_id"))
    if _bool_argument(arguments, "dry_run", default=False):
        payload: dict[str, Any] = {
            "ok": True,
            "dry_run": True,
            "correlation_id": correlation_id,
            "destination": _destination_payload(destination),
            "document_name": document.name,
            "document_bytes": len(document.content),
            "document_sha256": document.sha256,
            "caption_chars": len(caption or ""),
        }
        if caption:
            payload["caption_sha256"] = text_sha256(caption)
        return payload
    entry = send_document_and_log(
        TelegramBotClient.from_secrets(secrets),
        destination,
        document,
        caption=caption,
        log_path=DEFAULT_LOG_PATH,
        correlation_id=correlation_id,
    )
    return {"ok": True, "send": entry, "log_path": str(DEFAULT_LOG_PATH)}


def telegram_log_tail_payload(arguments: Mapping[str, Any]) -> dict[str, Any]:
    limit = _int_argument(arguments, "limit", default=20)
    return {"ok": True, "log_path": str(DEFAULT_LOG_PATH), "entries": tail_logs(DEFAULT_LOG_PATH, limit=limit)}


def mcp_tools() -> list[dict[str, Any]]:
    destination_properties = {
        "chat": {"type": "string"},
        "chat_id": {"type": "string"},
    }
    return [
        {
            "name": "telegram_bot_health",
            "description": "Report Telegram Bot MCP server health and sanitized local configuration status.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"live": {"type": "boolean"}},
            },
        },
        {
            "name": "telegram_bot_get_me",
            "description": "Call Telegram getMe for the configured bot token.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "telegram_bot_get_chat",
            "description": "Resolve and validate a Telegram chat without reading messages.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": destination_properties,
            },
        },
        {
            "name": "telegram_bot_send_message",
            "description": "Send one plain text Telegram message or dry-run the send.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    **destination_properties,
                    "text": {"type": "string"},
                    "correlation_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
            },
        },
        {
            "name": "telegram_bot_send_document",
            "description": (
                "Send one Tokimon repository-owned, Tokimon runtime-owned, or configured adopter-owned "
                "document attachment or dry-run the send."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["file"],
                "properties": {
                    **destination_properties,
                    "file": {"type": "string"},
                    "caption": {"type": "string"},
                    "correlation_id": {"type": "string"},
                    "dry_run": {"type": "boolean"},
                },
            },
        },
        {
            "name": "telegram_bot_log_tail",
            "description": "Read recent sanitized Telegram send log entries.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"limit": {"type": "integer", "minimum": 1}},
            },
        },
    ]


class MCPServer:
    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        return handle_request(
            request, server_name=SERVER_NAME, server_version=SERVER_VERSION,
            protocol_version=SUPPORTED_PROTOCOL_VERSION,
            list_tools=mcp_tools, call_tool=self._call_tool,
        )


    def _call_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = str(params.get("name", ""))
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return tool_result({"error": "arguments must be an object"}, is_error=True)
        try:
            if name == "telegram_bot_health":
                return tool_result(telegram_health_payload(live=_bool_argument(arguments, "live")))
            if name == "telegram_bot_get_me":
                return tool_result(telegram_get_me_payload())
            if name == "telegram_bot_get_chat":
                return tool_result(telegram_get_chat_payload(arguments))
            if name == "telegram_bot_send_message":
                return tool_result(telegram_send_message_payload(arguments))
            if name == "telegram_bot_send_document":
                return tool_result(telegram_send_document_payload(arguments))
            if name == "telegram_bot_log_tail":
                return tool_result(telegram_log_tail_payload(arguments))
            return tool_result({"error": f"Unknown tool: {name}"}, is_error=True)
        except TelegramBotError as exc:
            payload: dict[str, Any] = {"error": str(exc)}
            if isinstance(exc, TelegramAPIError):
                if exc.error_code is not None:
                    payload["error_code"] = exc.error_code
                if exc.retry_after_sec is not None:
                    payload["retry_after_sec"] = exc.retry_after_sec
            return tool_result(payload, is_error=True)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": utc_now_iso(),
            "service": SERVER_NAME,
            "runtime": "stdio-mcp",
            "severity": record.levelname.lower(),
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "server"),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        method = getattr(record, "method", None)
        if method is not None:
            payload["method"] = method
        return json.dumps(payload, sort_keys=True)


def configure_runtime_logger(*, stream: TextIO = sys.stderr) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_runtime_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    request_id: Any = None,
    method: str | None = None,
    correlation_id: str | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "request_id": request_id,
            "method": method,
            "correlation_id": correlation_id or (str(request_id) if request_id is not None else "server"),
        },
    )


def serve(
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    server: MCPServer | None = None,
) -> int:
    return serve_stdio(
        input_stream=input_stream, output_stream=output_stream,
        server=server or MCPServer(), logger=configure_runtime_logger(), log_event=log_runtime_event,
        method_as_text=True, ensure_ascii_errors=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Telegram Bot MCP server.")
    parser.add_argument("--secrets", default=str(DEFAULT_SECRETS_PATH), help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the stdio MCP server")

    health_parser = subparsers.add_parser("health", help="Check Telegram MCP server startup and local configuration")
    health_parser.add_argument("--live", action="store_true", help="Run live Telegram getMe check when configured")
    health_parser.add_argument("--json", action="store_true")

    return parser


def _secrets_path_from_args(args: argparse.Namespace) -> Path:
    return Path(args.secrets).expanduser()


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _render_health_text(payload: Mapping[str, Any]) -> str:
    status = "ok" if payload.get("ok") is not False else "fail"
    lines = [
        f"Telegram bot health: {status}",
        f"token source: {payload.get('token_source')}",
        f"secrets file: {payload.get('secrets_path')}",
        f"allowed chats: {', '.join(payload.get('allowed_chats') or []) or 'none'}",
        f"default chat configured: {payload.get('default_chat_configured')}",
    ]
    configuration_error = payload.get("configuration_error")
    if configuration_error:
        lines.append(f"configuration error: {configuration_error}")
    live_error = payload.get("live_error")
    if live_error:
        lines.append(f"live error: {live_error}")
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        lines.extend(f"warning: {warning}" for warning in warnings)
    bot = payload.get("bot")
    if isinstance(bot, Mapping):
        lines.append(f"bot: @{bot.get('username') or 'unknown'}")
    return "\n".join(lines)


def _handle_health(args: argparse.Namespace) -> int:
    payload = telegram_health_payload(
        secrets_path=_secrets_path_from_args(args),
        live=args.live,
    )
    if args.json:
        _print_json(payload)
    else:
        print(_render_health_text(payload))
    return 0 if payload.get("ok") is not False else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            return serve()
        if args.command == "health":
            return _handle_health(args)
    except TelegramBotError as exc:
        payload = {"ok": False, "error": str(exc)}
        if isinstance(exc, TelegramAPIError):
            if exc.error_code is not None:
                payload["error_code"] = exc.error_code
            if exc.retry_after_sec is not None:
                payload["retry_after_sec"] = exc.retry_after_sec
        if getattr(args, "json", False):
            _print_json(payload)
        else:
            print(str(exc), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
