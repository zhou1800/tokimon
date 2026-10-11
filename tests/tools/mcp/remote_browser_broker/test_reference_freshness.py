from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker.test_remote_browser_broker import (
    FakeLauncher,
    FakeNoVNCProxy,
)


class ScriptCDP:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            ["node", str(Path(__file__).parent / "fixtures/ref_dom.cjs")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )

    def __enter__(self) -> "ScriptCDP":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def call(self, method: str, params: dict | None = None) -> dict:
        self.process.stdin.write(json.dumps({"method": method, "params": params or {}}) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self) -> None:
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.process.stderr.close()


@unittest.skipUnless(shutil.which("node"), "Node.js is needed to execute the DOM fixture")
class ReferenceFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cdp = ScriptCDP()
        self.addCleanup(self.cdp.close)
        connection = patch.object(broker.CDPConnection, "for_session", return_value=self.cdp)
        connection.start()
        self.addCleanup(connection.stop)
        controller = broker.BrowserController()
        self.remote = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(runtime_root=self.root / "browser"),
            controller=controller,
            launcher=FakeLauncher(),
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(self.release)
        with patch.object(controller, "wait_until_ready"):
            self.remote.ensure("mcp-a")
        self.source = self.root / "attachment.txt"
        self.source.write_text("attachment", encoding="utf-8")

    def release(self) -> None:
        self.remote.shutdown_all_sessions()
        self.remote.release_shutdown_leases()

    def evaluate(self, expression: str) -> object:
        return broker.runtime_value(self.cdp.call(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True},
        ))

    def refs(self) -> list[str]:
        return [element["ref"] for element in self.remote.snapshot("mcp-a")["page"]["elements"]]

    def assert_stale_operations(self, refs: list[str]) -> None:
        for operation in (
            lambda: self.remote.click("mcp-a", ref=refs[0]),
            lambda: self.remote.type_text("mcp-a", ref=refs[1], text="private text"),
            lambda: self.remote.upload_file("mcp-a", ref=refs[2], path=str(self.source)),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(broker.BrowserBrokerError) as error:
                    operation()
                self.assertEqual(error.exception.code, broker.ERROR_STALE_REF)

    def test_unchanged_refs_support_click_type_and_upload_without_approval(self) -> None:
        refs = self.refs()
        self.remote.click("mcp-a", ref=refs[0])
        self.remote.type_text("mcp-a", ref=refs[1], text="hello")
        self.remote.upload_file("mcp-a", ref=refs[2], path=str(self.source))
        self.assertEqual(self.evaluate("document.elements[0].clicks"), 1)
        self.assertEqual(self.evaluate("document.elements[1].value"), "hello")
        self.assertEqual(self.evaluate("document.elements[2].files.length"), 1)
        self.assertEqual(self.evaluate("typeof globalThis.__tokimonSnapshot"), "undefined")

    def test_inserted_elements_cannot_receive_old_ref_actions(self) -> None:
        refs = self.refs()
        self.evaluate("document.elements.unshift(new Element('button', 'delete', 'Delete'))")
        self.assert_stale_operations(refs)
        self.assertEqual(self.evaluate("document.elements.map(el => el.clicks)"), [0, 0, 0, 0])
        self.assertEqual(self.evaluate("document.elements.map(el => el.value)"), ["", "", "", ""])
        self.assertEqual(self.evaluate("document.elements.map(el => el.files.length)"), [0, 0, 0, 0])
        fresh = self.refs()
        self.remote.click("mcp-a", ref=fresh[1])
        self.assertEqual(self.evaluate("document.elements[1].clicks"), 1)

    def test_identical_replacement_nodes_cannot_reuse_refs(self) -> None:
        refs = self.refs()
        self.evaluate("""document.elements = [
            new Element('button', 'continue', 'Continue'),
            new HTMLInputElement('search'), new HTMLInputElement('upload', 'file')
        ]; void 0""")
        self.assert_stale_operations(refs)

    def test_unrelated_dom_change_preserves_unchanged_target(self) -> None:
        refs = self.refs()
        self.evaluate("document.elements.push(new Element('button', 'other', 'Other'))")
        self.remote.click("mcp-a", ref=refs[0])
        self.assertEqual(self.evaluate("document.elements[0].clicks"), 1)

    def test_page_reload_invalidates_refs_even_with_identical_content(self) -> None:
        refs = self.refs()
        self.evaluate("reloadFixture()")
        self.assert_stale_operations(refs)

    def test_same_document_navigation_invalidates_refs(self) -> None:
        refs = self.refs()
        self.evaluate("location.href = 'https://example.com/other'")
        self.assert_stale_operations(refs)

    def test_changed_target_description_invalidates_ref(self) -> None:
        refs = self.refs()
        self.evaluate("document.elements[0].innerText = 'Delete'")
        with self.assertRaises(broker.BrowserBrokerError) as error:
            self.remote.click("mcp-a", ref=refs[0])
        self.assertEqual(error.exception.code, broker.ERROR_STALE_REF)
        self.assertEqual(self.evaluate("document.elements[0].clicks"), 0)

    def test_new_snapshot_invalidates_previous_refs(self) -> None:
        refs = self.refs()
        self.refs()
        self.assert_stale_operations(refs)


if __name__ == "__main__":
    unittest.main()
