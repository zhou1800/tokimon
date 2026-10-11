from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker.test_remote_browser_broker import FakeLauncher, FakeNoVNCProxy
from tests.tools.mcp.remote_browser_broker.test_reference_freshness import ScriptCDP

RUNTIME = broker.WORKSPACE_ROOT / ".tokimon/tmp/token-optimization-20260919"


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for synthetic DOM/CDP tests")
class BrowserArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=RUNTIME)
        cls.root = Path(cls.temporary.name)
        cls.binary = bytes(range(256)) * 800
        cls.downloads = {
            "/file.bin": ("application/octet-stream", cls.binary),
            "/chunked.bin": ("application/octet-stream", cls.binary),
            "/report.csv": ("text/csv; charset=utf-8", "name,value\n汉字,42\n".encode()),
            "/data.json": ("application/json", b'{"result":42}'),
            "/archive.zip": ("application/zip", b"PK\x03\x04opaque archive payload"),
            "/custom": ("application/x-fixture-format", b"arbitrary\x00format\xff"),
            "/untyped": (None, b"no content type"),
            "/empty.txt": ("text/plain", b""),
        }
        cls.html = "<main>Fixture</main>"
        cls.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.requests.append(self.path)
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/file.bin")
                    self.end_headers()
                    return
                if self.path == "/missing":
                    self.send_error(404)
                    return
                mime_type, data = cls.downloads.get(self.path, ("text/html", cls.html.encode()))
                self.send_response(200)
                if mime_type is not None:
                    self.send_header("Content-Type", mime_type)
                if self.path != "/chunked.bin":
                    self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_):
                pass

        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.http_thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.cdp = ScriptCDP.__new__(ScriptCDP)
        cls.cdp.process = subprocess.Popen(
            ["node", str(Path(__file__).parent / "fixtures/observation_dom.cjs")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=broker.WORKSPACE_ROOT,
        )
        cls.port = 1

    @classmethod
    def tearDownClass(cls):
        cls.cdp.close()
        cls.http.shutdown()
        cls.http.server_close()
        cls.http_thread.join()
        cls.temporary.cleanup()

    def setUp(self):
        connection = patch.object(broker.CDPConnection, "for_session", return_value=self.cdp)
        connection.start()
        self.addCleanup(connection.stop)
        self.controller = broker.BrowserController()
        self.remote = broker.RemoteBrowserBroker(
            config=broker.BrokerConfig(runtime_root=self.root / ("broker-" + self._testMethodName)),
            controller=self.controller, launcher=FakeLauncher(), novnc_proxy_factory=FakeNoVNCProxy)
        with patch.object(self.controller, "wait_until_ready"):
            self.remote.ensure("owner")
        self.session = self.remote._sessions_by_mcp["owner"]
        self.session.cdp_port = self.port
        self.server = broker.MCPServer(self.remote)
        self.url = f"http://127.0.0.1:{self.http.server_port}/"
        type(self).html = "<main id='content'>Fixture</main>"
        self.remote.navigate("owner", url=self.url)

    def tearDown(self):
        self.remote.shutdown_all_sessions()
        self.remote.release_shutdown_leases()

    def evaluate(self, script):
        return self.remote.evaluate("owner", expression=script)["result"]

    def load(self, html):
        self.evaluate("document.body.innerHTML = " + json.dumps(html))

    def test_backend_request_fetches_once_without_authentication_exchange(self):
        self.evaluate("""globalThis.savedFetch = fetch; globalThis.requestCalls = [];
            globalThis.fetch = async (url, options) => {
              requestCalls.push({url, credentials: options.credentials, headers: options.headers});
              let done = false;
              return {status: 200, statusText: 'OK', body: {getReader() {return {
                async read() {if (done) return {done: true}; done = true;
                  return {done: false, value: new TextEncoder().encode('response body')};},
                async cancel() {}
              };}}};
            };""")
        try:
            result = self.call("browser_backend_request", method="GET", path="/api/drafts", max_response_chars=4)
            calls = self.evaluate("requestCalls")
        finally:
            self.evaluate("globalThis.fetch = savedFetch")
        self.assertEqual(calls, [{"url": self.url + "api/drafts", "credentials": "include",
                                 "headers": {"Accept": "application/json"}}])
        self.assertEqual(result["response"]["status"], 200)
        self.assertEqual(result["response"]["text"], "resp")
        self.assertTrue(result["response"]["truncated"])

    def call(self, name, key="owner", expect_error=None, **arguments):
        response = self.server.handle({"jsonrpc":"2.0", "id":"fixture-请求", "method":"tools/call",
                                       "params":{"name":name, "arguments":arguments}}, mcp_session_key=key)
        result = json.loads(response["result"]["content"][0]["text"])
        if expect_error:
            self.assertTrue(response["result"].get("isError"), result)
            self.assertEqual(result["error_code"], expect_error)
        else:
            self.assertFalse(response["result"].get("isError"), result)
        return result

    def test_artifact_scope_and_pause(self):
        self.load("<main id='content'>Saved content</main><a href='/file.bin'>File</a><a href='/other'>Other</a>")
        for selector in ("#absent", "a"):
            self.call("browser_artifact", kind="html", scope_selector=selector,
                      expect_error="SCOPE_NOT_FOUND_OR_AMBIGUOUS")
        for arguments in ({"max_bytes": True}, {"max_bytes": 0}, {"scope_selector": False}, {"extra": 1}):
            self.call("browser_artifact", kind="html", expect_error=broker.ERROR_INVALID_ARGUMENT, **arguments)
        self.remote.pause_for_human("owner")
        self.call("browser_artifact", kind="html", scope_selector="#content")
        ref = self.call("browser_snapshot")["page"]["elements"][0]["ref"]
        self.call("browser_artifact", kind="download", ref=ref, expect_error=broker.ERROR_PAUSED)

    def test_artifact_sanitization_download_boundaries_and_paths(self):
        self.load("<main><h1>Hello</h1><script>const token='secret-script'</script><input value='secret-input'><textarea>secret-area</textarea><div hidden>secret-hidden</div><a href='/file.bin?token=secret-url' data-token='secret-attr'>Attachment</a></main>")
        result = self.call("browser_artifact", kind="html", scope_selector="main")
        artifact = result["artifact"]
        path = Path(artifact["path"])
        content = path.read_bytes()
        self.assertTrue(path.is_absolute())
        self.assertTrue(path.is_relative_to(self.remote.config.runtime_root))
        self.assertNotIn(b"secret", content)
        self.assertIn(b"Hello", content)
        self.assertEqual(artifact["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.call("browser_artifact", kind="html", max_bytes=1, expect_error="ARTIFACT_TOO_LARGE")
        for href, limit, expected in (("/file.bin", 1000000, None), ("/file.bin", 100, "ARTIFACT_TOO_LARGE"),
                                      ("/chunked.bin", 100, "ARTIFACT_TOO_LARGE"),
                                      ("/redirect", 1000000, "ARTIFACT_UNSUPPORTED"),
                                      ("https://example.org/foreign.bin", 1000000, "ARTIFACT_UNSUPPORTED"),
                                      ("/missing", 1000000, "ARTIFACT_UNSUPPORTED")):
            self.load(f"<a href='{href}'>Attachment</a>")
            ref = self.call("browser_snapshot")["page"]["elements"][0]["ref"]
            before = len(type(self).requests)
            result = self.call("browser_artifact", kind="download", ref=ref, max_bytes=limit, expect_error=expected)
            if not expected:
                self.assertEqual(Path(result["artifact"]["path"]).read_bytes(), self.binary)
                self.assertNotIn("base64", json.dumps(result))
            if href == "/redirect":
                self.assertEqual(type(self).requests[before:], ["/redirect"])
        self.call("browser_artifact", kind="download", ref="foreign", expect_error="STALE_REF")
        self.call("browser_artifact", kind="html", path="/tmp/escape", expect_error=broker.ERROR_INVALID_ARGUMENT)
        directory = self.remote._validated_session_root(self.session) / "artifacts"
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        directory.symlink_to(self.root)
        self.call("browser_artifact", kind="html", expect_error=broker.ERROR_BROKER)
        directory.unlink()

    def test_html_defaults_to_the_rendered_body_without_returning_its_content(self):
        self.load("<header>Rendered header</header><main><h1>Saved answer</h1>"
                  "<p>Report text</p></main><footer>Rendered footer</footer>"
                  "<div hidden>Not rendered</div>")
        result = self.call("browser_artifact", kind="html")
        content = Path(result["artifact"]["path"]).read_text()
        for text in ("Rendered header", "Saved answer", "Report text", "Rendered footer"):
            self.assertIn(text, content)
            self.assertNotIn(text, json.dumps(result))
        self.assertNotIn("Not rendered", content)
        scoped = self.call("browser_artifact", kind="html", scope_selector="main")
        scoped_content = Path(scoped["artifact"]["path"]).read_text()
        self.assertIn("Saved answer", scoped_content)
        self.assertNotIn("Rendered header", scoped_content)
        self.assertNotIn("Rendered footer", scoped_content)

    def test_hidden_styles_and_ancestors_are_not_artifact_evidence(self):
        self.load("<main>Visible<p style='visibility:hidden'>SYNTHETIC_HIDDEN_SECRET</p>"
                  "<div aria-hidden='true'><p id='inside'>SYNTHETIC_ANCESTOR_SECRET</p></div>"
                  "<span style='opacity:0'>SYNTHETIC_TRANSPARENT_SECRET</span></main>")
        result = self.call("browser_artifact", kind="html", scope_selector="main")
        self.assertNotIn("SECRET", Path(result["artifact"]["path"]).read_text())
        self.call("browser_artifact", kind="html", scope_selector="#inside", expect_error="SCOPE_NOT_VISIBLE")
        self.evaluate("document.querySelector('main').setAttribute('style','visibility:hidden')")
        self.call("browser_artifact", kind="html", scope_selector="main", expect_error="SCOPE_NOT_VISIBLE")

    def test_literal_html_selector_is_not_replaced_as_a_template_marker(self):
        self.load("<main id='SELECTOR'><p>Literal selector content</p></main>")
        result = self.call("browser_artifact", kind="html", scope_selector="#SELECTOR")
        self.assertIn("Literal selector content", Path(result["artifact"]["path"]).read_text())

    def test_download_preserves_all_file_types_and_reports_metadata(self):
        cases = {**self.downloads, "/": ("text/html", self.html.encode())}
        for href, (content_type, data) in cases.items():
            with self.subTest(href=href):
                self.load(f"<a href='{href}'>Download</a>")
                ref = self.call("browser_snapshot")["page"]["elements"][0]["ref"]
                result = self.call("browser_artifact", kind="download", ref=ref)
                artifact = result["artifact"]
                path = Path(artifact["path"])
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(artifact["size_bytes"], len(data))
                self.assertEqual(artifact["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(artifact["mime_type"], (content_type or "application/octet-stream").split(";")[0])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(set(artifact), {"path", "mime_type", "size_bytes", "sha256"})
                if href in {"/custom", "/untyped"}:
                    self.assertEqual(path.suffix, ".bin")
                elif href == "/report.csv":
                    self.assertEqual(path.suffix, ".csv")

    def test_artifact_quotas(self):
        self.load("<main>Content</main>")
        self.call("browser_artifact", kind="html", max_bytes=16777217, expect_error=broker.ERROR_INVALID_ARGUMENT)
        directory = self.remote._owned_child_directory(self.remote._validated_session_root(self.session), "artifacts")
        for i in range(100):
            (directory / f"fixture-{i}").write_bytes(b"synthetic")
        self.call("browser_artifact", kind="html", expect_error="ARTIFACT_QUOTA_EXCEEDED")
        self.remote.stop("owner")
        cleaned = self.remote.cleanup_session_artifacts("owner")
        self.assertEqual(cleaned["removed"]["artifacts"]["files"], 100)
        self.assertTrue(self.session.profile_dir.is_dir())

    def test_retained_runtime_and_wrong_editor_template_have_role_guidance(self):
        self.remote.stop("owner")
        conflict = self.call("browser_template_seal", template_name="fixture", expect_error=broker.ERROR_TEMPLATE_CONFLICT)
        self.assertEqual(conflict["current_state"], broker.STATE_STOPPED_RETAINED)
        self.assertEqual(conflict["next_actions"], ["browser_ensure"])
        with patch.object(self.controller, "wait_until_ready"):
            self.remote.create_template("editor", template_name="correct")
        conflict = self.call("browser_template_seal", key="editor", template_name="wrong", expect_error=broker.ERROR_TEMPLATE_CONFLICT)
        self.assertEqual(conflict["current_role"], broker.TEMPLATE_ROLE_EDITING)
        self.assertEqual(conflict["current_template_name"], "correct")
        self.assertEqual(conflict["next_actions"], ["browser_template_seal"])

    def test_template_conflict_explains_runtime_recovery(self):
        result = self.call("browser_template_seal", template_name="fixture", expect_error=broker.ERROR_TEMPLATE_CONFLICT)
        self.assertEqual(result["current_role"], "runtime")
        self.assertIn("browser_stop/browser_ensure", result["error"])
        self.assertEqual(self.session.state, broker.STATE_RUNNING)
