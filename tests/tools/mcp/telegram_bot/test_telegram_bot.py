from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tokimon.tools.mcp import telegram_bot as tb

EXAMPLE_CHAT_ID = "example-chat-id"
EXAMPLE_DEFAULT_CHAT_ID = "example-default-chat-id"


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class FakeUrlOpen:
    def __init__(self, *responses: dict | BaseException) -> None:
        self._responses = list(responses)
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request: object, *, timeout: float) -> FakeHTTPResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeHTTPResponse(response)


def write_secret(path: Path, payload: dict, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if os.name == "posix":
        path.chmod(mode)


class TelegramBotTests(unittest.TestCase):
    def test_load_secrets_supports_current_file_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(
                secrets_path,
                {
                    "telegram": {
                        "bot_token": "file-token",
                        "default_chat_id": EXAMPLE_CHAT_ID,
                    }
                },
            )

            secrets = tb.load_secrets(secrets_path, env={})

        self.assertEqual(secrets.bot_token, "file-token")
        self.assertEqual(secrets.token_source, "file")
        self.assertEqual(secrets.default_chat_id, EXAMPLE_CHAT_ID)
        self.assertEqual(secrets.allowed_chats, {})

    def test_load_secrets_supports_documented_path_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(secrets_path, {"telegram": {"bot_token": "file-token"}})

            secrets = tb.load_secrets(path=secrets_path, env={})

        self.assertEqual(secrets.bot_token, "file-token")
        self.assertEqual(secrets.secrets_path, secrets_path)

    def test_load_secrets_env_token_overrides_file_and_parses_allowed_chats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(
                secrets_path,
                {
                    "telegram": {
                        "bot_token": "file-token",
                        "allowed_chats": {
                            "ops": EXAMPLE_CHAT_ID,
                            "family": {"chat_id": 12345},
                        },
                    }
                },
            )

            secrets = tb.load_secrets(secrets_path, env={"TELEGRAM_BOT_TOKEN": "env-token"})

        self.assertEqual(secrets.bot_token, "env-token")
        self.assertEqual(secrets.token_source, "env:TELEGRAM_BOT_TOKEN")
        self.assertEqual(secrets.allowed_chats["ops"], EXAMPLE_CHAT_ID)
        self.assertEqual(secrets.allowed_chats["family"], "12345")

    def test_load_secrets_file_mode_warning_follows_host_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(secrets_path, {"telegram": {"bot_token": "file-token"}}, mode=0o644)

            secrets = tb.load_secrets(secrets_path, env={})

        if os.name == "posix":
            self.assertEqual(secrets.mode, "0644")
            self.assertIn("0600", secrets.warnings[0])
        else:
            self.assertIsNone(secrets.mode)
            self.assertEqual(secrets.warnings, ())

    def test_load_secrets_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(secrets_path, {"telegram": {}}, mode=0o600)

            with self.assertRaises(tb.TelegramConfigError):
                tb.load_secrets(secrets_path, env={})

    def test_resolve_destination_prefers_explicit_then_named_then_default(self) -> None:
        secrets = tb.TelegramSecrets(
            bot_token="token",
            token_source="file",
            secrets_path=Path("secrets"),
            allowed_chats={"ops": EXAMPLE_CHAT_ID},
            default_chat_id=EXAMPLE_DEFAULT_CHAT_ID,
        )

        explicit = tb.resolve_destination(secrets, chat_id="-1")
        named = tb.resolve_destination(secrets, chat="ops")
        default = tb.resolve_destination(secrets)

        self.assertEqual(explicit.chat_id, "-1")
        self.assertEqual(explicit.source, "chat_id")
        self.assertEqual(named.chat_id, EXAMPLE_CHAT_ID)
        self.assertEqual(named.label, "ops")
        self.assertEqual(default.chat_id, EXAMPLE_DEFAULT_CHAT_ID)
        self.assertEqual(default.source, "default_chat_id")

    def test_resolve_destination_rejects_unknown_chat(self) -> None:
        secrets = tb.TelegramSecrets(
            bot_token="token",
            token_source="file",
            secrets_path=Path("secrets"),
            allowed_chats={},
        )

        with self.assertRaises(tb.TelegramConfigError):
            tb.resolve_destination(secrets, chat="ops")

    def test_client_get_me_posts_to_telegram_api(self) -> None:
        transport = FakeUrlOpen({"ok": True, "result": {"id": 1, "username": "tokimon_clover_bot"}})
        client = tb.TelegramBotClient("token:secret", urlopen=transport, timeout_seconds=3.0)

        result = client.get_me()

        self.assertEqual(result["username"], "tokimon_clover_bot")
        self.assertEqual(transport.timeouts, [3.0])
        request = transport.requests[0]
        self.assertTrue(request.full_url.endswith("/getMe"))
        self.assertIn("bottoken%3Asecret", request.full_url)

    def test_send_message_and_log_writes_sanitized_success_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sends.jsonl"
            transport = FakeUrlOpen({"ok": True, "result": {"message_id": 6}})
            client = tb.TelegramBotClient("token-secret", urlopen=transport)
            destination = tb.ChatDestination(chat_id=EXAMPLE_CHAT_ID, label="ops", source="allowed_chats")
            text = "secret family message"

            entry = tb.send_message_and_log(
                client,
                destination,
                text,
                log_path=log_path,
                correlation_id="audit-correlation",
            )
            log_text = log_path.read_text(encoding="utf-8")

        self.assertTrue(entry["ok"])
        self.assertEqual(entry["message_id"], 6)
        self.assertEqual(entry["service"], "tokimon.tools.mcp.telegram_bot")
        self.assertEqual(entry["runtime"], "mcp.telegram-bot")
        self.assertEqual(entry["severity"], "info")
        self.assertEqual(entry["message"], "Telegram message send succeeded.")
        self.assertEqual(entry["correlation_id"], "audit-correlation")
        self.assertIn("timestamp", entry)
        self.assertNotIn("chat_id", entry)
        self.assertIn(hashlib.sha256(text.encode("utf-8")).hexdigest(), log_text)
        self.assertNotIn(EXAMPLE_CHAT_ID, log_text)
        self.assertNotIn("secret family message", log_text)
        self.assertNotIn("token-secret", log_text)

    def test_send_message_and_log_writes_sanitized_failure_with_retry_after(self) -> None:
        payload = {
            "ok": False,
            "error_code": 429,
            "description": "Too Many Requests: retry after 30 token-secret",
            "parameters": {"retry_after": 30},
        }
        error = urllib.error.HTTPError(
            url="https://api.telegram.org/bottoken-secret/sendMessage",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sends.jsonl"
            client = tb.TelegramBotClient("token-secret", urlopen=FakeUrlOpen(error))
            destination = tb.ChatDestination(chat_id=EXAMPLE_CHAT_ID, label="ops", source="allowed_chats")

            with self.assertRaises(tb.TelegramAPIError) as raised:
                tb.send_message_and_log(
                    client,
                    destination,
                    "hello",
                    log_path=log_path,
                    correlation_id="failure-correlation",
                )
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(raised.exception.retry_after_sec, 30)
        self.assertEqual(log_payload["severity"], "error")
        self.assertEqual(log_payload["message"], "Telegram message send failed.")
        self.assertEqual(log_payload["correlation_id"], "failure-correlation")
        self.assertEqual(log_payload["error_code"], 429)
        self.assertEqual(log_payload["retry_after_sec"], 30)
        self.assertNotIn("chat_id", log_payload)
        self.assertNotIn(EXAMPLE_CHAT_ID, json.dumps(log_payload))
        self.assertNotIn("token-secret", json.dumps(log_payload))
        self.assertNotIn("hello", json.dumps(log_payload))

    def test_url_error_redacts_encoded_credential_from_tool_and_send_log(self) -> None:
        token = "12345:FakeSecret"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secrets_path = root / ".tokimon" / "secrets.local.json"
            log_path = root / "sends.jsonl"
            write_secret(secrets_path, {"telegram": {"bot_token": token}})

            def fail(request: object, **_: object) -> None:
                raise urllib.error.URLError("unable to open " + request.full_url)

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(tb, "DEFAULT_LOG_PATH", log_path),
                mock.patch.object(tb.urllib.request, "urlopen", side_effect=fail),
            ):
                response = tb.MCPServer().handle({
                    "jsonrpc": "2.0", "id": "encoded-failure", "method": "tools/call",
                    "params": {"name": "telegram_bot_send_message", "arguments": {
                        "chat_id": EXAMPLE_CHAT_ID, "text": "hello",
                    }},
                })
            self.assertTrue(response["result"]["isError"])
            for value in (json.dumps(response), log_path.read_text()):
                self.assertNotIn(token, value)
                self.assertNotIn(urllib.parse.quote(token, safe=""), value)
                self.assertIn("[redacted-token]", value)

    def test_transport_and_response_timeouts_are_logged_tool_errors(self) -> None:
        token = "12345:FakeSecret"
        for failure_at in ("open", "read", "http_error_read"):
            with self.subTest(failure_at=failure_at), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                secrets_path = root / ".tokimon" / "secrets.local.json"
                log_path = root / "sends.jsonl"
                write_secret(secrets_path, {"telegram": {"bot_token": token}})
                timeout = TimeoutError("timed out " + urllib.parse.quote(token, safe=""))
                response = FakeHTTPResponse({})
                response.read = mock.Mock(side_effect=timeout)
                response.close = mock.Mock()
                transport = mock.Mock(return_value=response)
                if failure_at == "open":
                    transport.side_effect = timeout
                elif failure_at == "http_error_read":
                    transport.side_effect = urllib.error.HTTPError(
                        "https://example.invalid", 500, "failed", {}, response,
                    )
                with (
                    mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                    mock.patch.object(tb, "DEFAULT_LOG_PATH", log_path),
                    mock.patch.object(tb.urllib.request, "urlopen", transport),
                ):
                    result = tb.MCPServer().handle({
                        "jsonrpc": "2.0", "id": "timeout", "method": "tools/call",
                        "params": {"name": "telegram_bot_send_message", "arguments": {
                            "chat_id": EXAMPLE_CHAT_ID, "text": "hello",
                            "correlation_id": "timeout-correlation",
                        }},
                    })
                self.assertTrue(result["result"]["isError"])
                entry = json.loads(log_path.read_text())
                self.assertFalse(entry["ok"])
                self.assertEqual(entry["correlation_id"], "timeout-correlation")
                self.assertIn("timed out", entry["error"])
                self.assertNotIn("FakeSecret", json.dumps(result) + json.dumps(entry))

    def test_send_document_posts_multipart_and_logs_sanitized_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sends.jsonl"
            document_path = Path(tmpdir) / "full-log.md"
            document_bytes = b"# Full log\nsecret content\n"
            document_path.write_bytes(document_bytes)
            transport = FakeUrlOpen({"ok": True, "result": {"message_id": 9}})
            client = tb.TelegramBotClient("token-secret", urlopen=transport)
            destination = tb.ChatDestination(chat_id=EXAMPLE_CHAT_ID, label="ops", source="allowed_chats")

            entry = tb.send_document_and_log(
                client,
                destination,
                document_path,
                caption="NVDA winner: bullish",
                log_path=log_path,
                correlation_id="document-correlation",
            )
            request = transport.requests[0]
            request_body = request.data.decode("utf-8", errors="replace")
            log_payload = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(entry["event"], "send_document")
        self.assertEqual(entry["message_id"], 9)
        self.assertTrue(request.full_url.endswith("/sendDocument"))
        self.assertIn("multipart/form-data", request.headers["Content-type"])
        self.assertIn('name="chat_id"', request_body)
        self.assertIn('name="caption"', request_body)
        self.assertIn('name="document"; filename="full-log.md"', request_body)
        self.assertEqual(log_payload["document_name"], "full-log.md")
        self.assertEqual(log_payload["document_bytes"], len(document_bytes))
        self.assertEqual(log_payload["document_sha256"], hashlib.sha256(document_bytes).hexdigest())
        self.assertEqual(log_payload["severity"], "info")
        self.assertEqual(log_payload["message"], "Telegram document send succeeded.")
        self.assertEqual(log_payload["correlation_id"], "document-correlation")
        self.assertNotIn("chat_id", log_payload)
        self.assertNotIn(EXAMPLE_CHAT_ID, json.dumps(log_payload))
        self.assertNotIn("secret content", json.dumps(log_payload))
        self.assertNotIn("NVDA winner", json.dumps(log_payload))
        self.assertNotIn("token-secret", json.dumps(log_payload))

    def test_mcp_lists_tools_and_health_does_not_require_token(self) -> None:
        server = tb.MCPServer()

        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertIsNotNone(listed)
        tool_names = {tool["name"] for tool in listed["result"]["tools"]}
        self.assertIn("telegram_bot_health", tool_names)
        self.assertIn("telegram_bot_send_message", tool_names)
        self.assertIn("telegram_bot_send_document", tool_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            with mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "telegram_bot_health", "arguments": {}},
                    }
                )

        self.assertIsNotNone(response)
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["token_configured"])
        self.assertFalse(payload["ready_for_send"])

    def test_health_text_reports_live_failure(self) -> None:
        class FailingClient:
            def get_me(self) -> dict[str, object]:
                raise tb.TelegramAPIError("Telegram API unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(secrets_path, {"telegram": {"bot_token": "file-token"}})
            stdout = io.StringIO()

            with (
                mock.patch.object(tb.TelegramBotClient, "from_secrets", return_value=FailingClient()),
                mock.patch("sys.stdout", stdout),
            ):
                exit_code = tb.main(["--secrets", str(secrets_path), "health", "--live"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 1)
        self.assertIn("Telegram bot health: fail", output)
        self.assertIn("live error: Telegram API unavailable", output)

    def test_mcp_send_message_dry_run_uses_hash_without_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            server = tb.MCPServer()

            with mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_message",
                            "arguments": {
                                "chat": "ops",
                                "text": "secret family message",
                                "dry_run": True,
                                "correlation_id": "dry-run-correlation",
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertNotIn("isError", response["result"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["destination"]["chat_label"], "ops")
        self.assertEqual(payload["correlation_id"], "dry-run-correlation")
        self.assertEqual(payload["text_sha256"], hashlib.sha256(b"secret family message").hexdigest())
        self.assertNotIn("secret family message", json.dumps(payload))

    def test_mcp_get_chat_uses_fake_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(
                    tb.TelegramBotClient,
                    "from_secrets",
                    return_value=tb.TelegramBotClient(
                        "token-secret",
                        urlopen=FakeUrlOpen({"ok": True, "result": {"type": "group", "title": "Ops"}}),
                    ),
                ),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {"name": "telegram_bot_get_chat", "arguments": {"chat": "ops"}},
                    }
                )

        self.assertIsNotNone(response)
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["destination"]["chat_id"], EXAMPLE_CHAT_ID)
        self.assertEqual(payload["chat"]["title"], "Ops")

    def test_tail_logs_returns_recent_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "sends.jsonl"
            tb.append_jsonl(log_path, {"ok": True, "message_id": 1})
            tb.append_jsonl(log_path, {"ok": True, "message_id": 2})
            tb.append_jsonl(log_path, {"ok": True, "message_id": 3})

            entries = tb.tail_logs(log_path, limit=2)

        self.assertEqual([entry["message_id"] for entry in entries], [2, 3])

    def test_mcp_send_document_rejects_paths_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            document_path = Path(tmpdir) / "report.txt"
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            document_path.write_text("report body\n", encoding="utf-8")
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(tb, "WORKSPACE_ROOT", Path(tmpdir) / "source"),
                mock.patch.object(tb, "RUNTIME_ROOT", Path(tmpdir) / "runtime"),
                mock.patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: ""}),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_document",
                            "arguments": {
                                "chat": "ops",
                                "file": str(document_path),
                                "dry_run": True,
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("configured adopter document root", payload["error"])

    def test_mcp_send_document_allows_configured_adopter_root_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            secrets_path = tmp_root / ".tokimon" / "secrets.local.json"
            adopter_root = tmp_root / "adopter-runtime"
            document_path = adopter_root / "report.txt"
            adopter_root.mkdir()
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            document_path.write_text("private report body\n", encoding="utf-8")
            expected_hash = tb.file_sha256(document_path)
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: str(adopter_root) + os.pathsep}),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": "adopter-document-dry-run",
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_document",
                            "arguments": {
                                "chat": "ops",
                                "file": str(document_path),
                                "caption": "private caption",
                                "dry_run": True,
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertNotIn("isError", response["result"])
        payload = json.loads(response["result"]["content"][0]["text"])
        payload_text = json.dumps(payload)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["document_name"], document_path.name)
        self.assertEqual(payload["document_sha256"], expected_hash)
        self.assertNotIn(str(adopter_root), payload_text)
        self.assertNotIn("private report body", payload_text)
        self.assertNotIn("private caption", payload_text)

    def test_mcp_send_document_rejects_configured_root_prefix_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            secrets_path = tmp_root / ".tokimon" / "secrets.local.json"
            adopter_root = tmp_root / "adopter-runtime"
            sibling_root = tmp_root / "adopter-runtime-sibling"
            document_path = sibling_root / "report.txt"
            adopter_root.mkdir()
            sibling_root.mkdir()
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            document_path.write_text("private report body\n", encoding="utf-8")
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(tb, "WORKSPACE_ROOT", Path(tmpdir) / "source"),
                mock.patch.object(tb, "RUNTIME_ROOT", Path(tmpdir) / "runtime"),
                mock.patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: str(adopter_root)}),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": "prefix-sibling-document-dry-run",
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_document",
                            "arguments": {
                                "chat": "ops",
                                "file": str(document_path),
                                "dry_run": True,
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("configured adopter document root", payload["error"])

    def test_configured_adopter_document_roots_reject_relative_or_missing_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "report.txt"
            document_path.write_text("report body\n", encoding="utf-8")

            for configured_root in ("relative-root", str(Path(tmpdir) / "missing")):
                with self.subTest(configured_root=configured_root):
                    with self.assertRaisesRegex(tb.TelegramConfigError, "absolute existing directories"):
                        tb.validate_document_path(
                            document_path,
                            enforce_local_boundary=True,
                            env={tb.ALLOWED_DOCUMENT_ROOTS_ENV: configured_root},
                        )

    def test_mcp_send_document_allows_repo_and_runtime_paths_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            runtime_tmp = tb.RUNTIME_ROOT / "tmp" / "telegram-bot-tests"
            runtime_tmp.mkdir(parents=True, exist_ok=True)

            with (
                tempfile.TemporaryDirectory(prefix=".telegram-bot-repo-", dir=ROOT) as repo_tmpdir,
                tempfile.TemporaryDirectory(prefix="runtime-", dir=runtime_tmp) as runtime_tmpdir,
            ):
                documents = [
                    Path(repo_tmpdir) / "repo-report.txt",
                    Path(runtime_tmpdir) / "runtime-report.txt",
                ]
                for document_path in documents:
                    with self.subTest(document_path=document_path):
                        document_path.write_text("private report body\n", encoding="utf-8")
                        server = tb.MCPServer()

                        with mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path):
                            response = server.handle(
                                {
                                    "jsonrpc": "2.0",
                                    "id": "document-dry-run",
                                    "method": "tools/call",
                                    "params": {
                                        "name": "telegram_bot_send_document",
                                        "arguments": {
                                            "chat": "ops",
                                            "file": str(document_path),
                                            "caption": "private caption",
                                            "dry_run": True,
                                        },
                                    },
                                }
                            )

                    self.assertIsNotNone(response)
                    self.assertNotIn("isError", response["result"])
                    payload = json.loads(response["result"]["content"][0]["text"])
                    self.assertTrue(payload["dry_run"])
                    self.assertEqual(payload["document_name"], document_path.name)
                    self.assertEqual(payload["document_sha256"], tb.file_sha256(document_path))
                    self.assertNotIn("private report body", json.dumps(payload))
                    self.assertNotIn("private caption", json.dumps(payload))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is required")
    def test_mcp_send_document_rejects_symlink_escape(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            tempfile.TemporaryDirectory(prefix=".telegram-bot-repo-", dir=ROOT) as repo_tmpdir,
        ):
            secrets_path = Path(tmpdir) / ".tokimon" / "secrets.local.json"
            outside_document = Path(tmpdir) / "outside-report.txt"
            link_path = Path(repo_tmpdir) / "linked-outside-report.txt"
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            outside_document.write_text("outside body\n", encoding="utf-8")
            os.symlink(outside_document, link_path)
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(tb, "WORKSPACE_ROOT", Path(repo_tmpdir)),
                mock.patch.object(tb, "RUNTIME_ROOT", Path(tmpdir) / "runtime"),
                mock.patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: ""}),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_document",
                            "arguments": {
                                "chat": "ops",
                                "file": str(link_path),
                                "dry_run": True,
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("configured adopter document root", payload["error"])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is required")
    def test_mcp_send_document_rejects_configured_root_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            secrets_path = tmp_root / ".tokimon" / "secrets.local.json"
            adopter_root = tmp_root / "adopter-runtime"
            outside_document = tmp_root / "outside-report.txt"
            link_path = adopter_root / "linked-outside-report.txt"
            adopter_root.mkdir()
            write_secret(
                secrets_path,
                {"telegram": {"bot_token": "file-token", "allowed_chats": {"ops": EXAMPLE_CHAT_ID}}},
            )
            outside_document.write_text("outside body\n", encoding="utf-8")
            os.symlink(outside_document, link_path)
            server = tb.MCPServer()

            with (
                mock.patch.object(tb, "DEFAULT_SECRETS_PATH", secrets_path),
                mock.patch.object(tb, "WORKSPACE_ROOT", Path(tmpdir) / "source"),
                mock.patch.object(tb, "RUNTIME_ROOT", Path(tmpdir) / "runtime"),
                mock.patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: str(adopter_root)}),
            ):
                response = server.handle(
                    {
                        "jsonrpc": "2.0",
                        "id": "adopter-symlink-escape",
                        "method": "tools/call",
                        "params": {
                            "name": "telegram_bot_send_document",
                            "arguments": {
                                "chat": "ops",
                                "file": str(link_path),
                                "dry_run": True,
                            },
                        },
                    }
                )

        self.assertIsNotNone(response)
        self.assertTrue(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("configured adopter document root", payload["error"])

    def test_stdio_server_handles_ping(self) -> None:
        stdout = io.StringIO()

        exit_code = tb.serve(
            input_stream=io.StringIO('{"jsonrpc":"2.0","id":"ping-1","method":"ping"}\n'),
            output_stream=stdout,
            server=tb.MCPServer(),
        )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["result"], {})


if __name__ == "__main__":
    unittest.main()
