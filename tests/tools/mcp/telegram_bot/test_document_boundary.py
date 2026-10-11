from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import telegram_bot as tb
from tests.tools.mcp.telegram_bot.test_telegram_bot import FakeUrlOpen, write_secret


class TelegramDocumentBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.allowed = self.root / "adopter"
        self.allowed.mkdir()
        self.reports = self.allowed / "reports"
        self.reports.mkdir()
        self.document = self.reports / "report.txt"
        self.content = b"approved document\n"
        self.document.write_bytes(self.content)
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.canary = self.outside / "report.txt"
        self.canary.write_bytes(b"outside synthetic canary\n")
        secrets = self.root / "secrets.json"
        write_secret(secrets, {"telegram": {"bot_token": "synthetic-token", "default_chat_id": "example-chat"}})
        self.log = self.root / "sends.jsonl"
        self.transport = FakeUrlOpen({"ok": True, "result": {"message_id": 1}})
        self.client = tb.TelegramBotClient("synthetic-token", urlopen=self.transport)
        for name, value in (("WORKSPACE_ROOT", self.root / "source"),
                            ("RUNTIME_ROOT", self.root / "runtime"),
                            ("DEFAULT_SECRETS_PATH", secrets), ("DEFAULT_LOG_PATH", self.log)):
            patcher = patch.object(tb, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (patch.dict(os.environ, {tb.ALLOWED_DOCUMENT_ROOTS_ENV: str(self.allowed)}),
                        patch.object(tb.TelegramBotClient, "from_secrets", return_value=self.client)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def send(self, path=None, *, dry_run=False):
        response = tb.MCPServer().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": "telegram_bot_send_document",
                                                    "arguments": {"file": str(path or self.document), "dry_run": dry_run}}})
        return response["result"], json.loads(response["result"]["content"][0]["text"])

    def assert_approved(self, result, payload):
        self.assertFalse(result.get("isError"), payload)
        body = self.transport.requests[0].data
        self.assertIn(self.content, body)
        self.assertNotIn(self.canary.read_bytes(), body)
        record = payload["send"]
        self.assertEqual(record["document_bytes"], len(self.content))
        self.assertEqual(record["document_sha256"], hashlib.sha256(self.content).hexdigest())

    def test_approved_alias_cannot_be_redirected_after_validation(self):
        alias = self.allowed / "alias.txt"
        alias.symlink_to(self.document)
        validate = tb.validate_document_path

        def swap(path, **kwargs):
            approved = validate(path, **kwargs)
            alias.unlink()
            alias.symlink_to(self.canary)
            return approved

        with patch.object(tb, "validate_document_path", side_effect=swap):
            result, payload = self.send(alias)
        self.assert_approved(result, payload)

    def test_ancestor_link_introduced_before_open_is_rejected(self):
        validate = tb.validate_document_path

        def swap(path, **kwargs):
            approved = validate(path, **kwargs)
            self.reports.rename(self.allowed / "saved-reports")
            self.reports.symlink_to(self.outside, target_is_directory=True)
            return approved

        with patch.object(tb, "validate_document_path", side_effect=swap):
            result, payload = self.send()
        self.assertTrue(result["isError"], payload)
        self.assertEqual(self.transport.requests, [])
        self.assertFalse(self.log.exists())

    def test_replacing_open_file_or_directory_keeps_approved_descriptor(self):
        for target in ("report.txt", "reports"):
            with self.subTest(target=target):
                open_file = os.open
                changed = False

                def swap(name, flags, *args, **kwargs):
                    nonlocal changed
                    descriptor = open_file(name, flags, *args, **kwargs)
                    if name == target and not changed:
                        changed = True
                        if target == "report.txt":
                            self.document.rename(self.reports / "saved-report.txt")
                            self.document.symlink_to(self.canary)
                        else:
                            self.reports.rename(self.allowed / "saved-reports")
                            self.reports.symlink_to(self.outside, target_is_directory=True)
                    return descriptor

                with patch.object(os, "open", side_effect=swap):
                    result, payload = self.send()
                self.assertTrue(changed)
                self.assert_approved(result, payload)
                self.transport.requests.clear()
                self.transport._responses.append({"ok": True, "result": {"message_id": 2}})
                if target == "report.txt":
                    self.document.unlink()
                    (self.reports / "saved-report.txt").rename(self.document)
                else:
                    self.reports.unlink()
                    (self.allowed / "saved-reports").rename(self.reports)

    def test_upload_and_success_or_failure_log_use_identical_snapshot(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                self.document.write_bytes(self.content)

                def transport(request, *, timeout):
                    self.document.write_bytes(b"changed during external send\n")
                    if fails:
                        raise tb.TelegramAPIError("synthetic send failure")
                    return self.transport(request, timeout=timeout)

                with patch.object(self.client, "_urlopen", side_effect=transport):
                    result, payload = self.send()
                self.assertEqual(bool(result.get("isError")), fails)
                records = [json.loads(row) for row in self.log.read_text().splitlines()]
                self.assertEqual(records[-1]["document_sha256"], hashlib.sha256(self.content).hexdigest())
                self.assertEqual(records[-1]["document_bytes"], len(self.content))
                self.assertEqual(records[-1]["ok"], not fails)
                if not fails:
                    self.assert_approved(result, payload)

    def test_fifo_introduced_after_validation_is_rejected_without_send(self):
        validate = tb.validate_document_path

        def swap(path, **kwargs):
            approved = validate(path, **kwargs)
            self.document.unlink()
            os.mkfifo(self.document)
            return approved

        with patch.object(tb, "validate_document_path", side_effect=swap):
            result, payload = self.send()
        self.assertTrue(result["isError"], payload)
        self.assertEqual(self.transport.requests, [])

    def test_dry_run_reads_snapshot_without_sending_or_logging(self):
        result, payload = self.send(dry_run=True)
        self.assertFalse(result.get("isError"), payload)
        self.assertEqual(payload["document_sha256"], hashlib.sha256(self.content).hexdigest())
        self.assertEqual(self.transport.requests, [])
        self.assertFalse(self.log.exists())

    def test_filename_cannot_inject_multipart_headers(self):
        document = self.allowed / 'report"\r\nX-Synthetic: value.txt'
        document.write_bytes(self.content)
        result, payload = self.send(document)
        self.assert_approved(result, payload)
        body = self.transport.requests[0].data
        self.assertNotIn(b"\r\nX-Synthetic:", body)
        self.assertIn(b"report%22%0D%0AX-Synthetic: value.txt", body)
