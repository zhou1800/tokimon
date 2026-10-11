from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tokimon.runtime import runtime_workspace_root
from tokimon.tools.mcp import remote_browser_broker as broker


START_SCRIPT = broker.IMAGE_CONTEXT_DIR / "start-session-browser.sh"
ROOT = Path(__file__).resolve().parents[4]
FAKE_PROCESS = r'''
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["STARTUP_TEST_ROOT"])
scenario = os.environ["STARTUP_TEST_SCENARIO"]
name = Path(sys.argv[0]).name
if name == "Xvfb":
    if scenario == "exited":
        sys.exit(7)
    time.sleep(60)
elif name == "xdpyinfo":
    if scenario == "hung":
        time.sleep(60)
    if scenario != "delayed":
        sys.exit(1)
    counter = root / "probes"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    if count < 3:
        sys.exit(1)
    (root / "ready").touch()
elif name == "curl":
    pass
else:
    ready = (root / "ready").exists()
    (root / (name + ".started")).write_text("ready" if ready else "too early")
    if not ready:
        sys.exit(8)
    time.sleep(0.2 if name == "google-chrome-stable" else 60)
'''


class BrowserImageStartupTests(unittest.TestCase):
    def run_startup(self, scenario: str) -> tuple[int, str, dict[str, str]]:
        with tempfile.TemporaryDirectory(
            prefix="browser-startup-test-", dir=runtime_workspace_root(ROOT),
        ) as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            executable = binaries / "process.py"
            executable.write_text(f"#!{sys.executable}\n" + FAKE_PROCESS, encoding="utf-8")
            executable.chmod(0o755)
            for name in ("Xvfb", "xdpyinfo", "openbox", "x11vnc", "websockify", "socat", "curl", "google-chrome-stable"):
                (binaries / name).symlink_to(executable)

            # Replace external paths only; execute the startup logic in a real shell.
            script = START_SCRIPT.read_text(encoding="utf-8")
            for original, target in {
                "/usr/bin/google-chrome-stable": binaries / "google-chrome-stable",
                "/tmp/openbox.log": root / "openbox.log",
                "/tmp/websockify.log": root / "websockify.log",
            }.items():
                script = script.replace(original, shlex.quote(str(target)))
            script_path = root / "start.sh"
            script_path.write_text(script, encoding="utf-8")
            clock = root / "clock.sh"
            # Advance the shell clock at its wait boundary; keep probe timeouts real.
            clock.write_text(
                'if [[ "$STARTUP_TEST_SCENARIO" == "unready" || '
                '"$STARTUP_TEST_SCENARIO" == "hung" ]]; then\n'
                '  sleep() { SECONDS=$((SECONDS + 60)); }\n'
                'fi\n',
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "BASH_ENV": str(clock),
                "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                "PROFILE_DIR": str(root / "profile"),
                "XDG_RUNTIME_DIR": str(root / "xdg"),
                "DISPLAY": ":99",
                "STARTUP_TEST_ROOT": str(root),
                "STARTUP_TEST_SCENARIO": scenario,
            }
            with (root / "output.log").open("w+") as output:
                process = subprocess.Popen(
                    ["bash", str(script_path)], env=env,
                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                )
                try:
                    code = process.wait(timeout=20)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                output.seek(0)
                diagnostics = output.read()
            clients = {path.stem: path.read_text() for path in root.glob("*.started")}
            return code, diagnostics, clients

    def test_delayed_display_is_ready_before_clients_start(self) -> None:
        code, diagnostics, clients = self.run_startup("delayed")

        self.assertEqual(code, 0, diagnostics)
        self.assertEqual(
            clients,
            dict.fromkeys(("openbox", "x11vnc", "websockify", "socat", "google-chrome-stable"), "ready"),
        )

    def test_xvfb_exit_fails_before_clients_start(self) -> None:
        code, diagnostics, clients = self.run_startup("exited")

        self.assertNotEqual(code, 0)
        self.assertIn("Xvfb exited before display", diagnostics)
        self.assertEqual(clients, {})

    def test_unready_display_times_out_before_clients_start(self) -> None:
        code, diagnostics, clients = self.run_startup("unready")

        self.assertNotEqual(code, 0)
        self.assertIn("did not become ready within 10 seconds", diagnostics)
        self.assertEqual(clients, {})

    def test_hung_display_probe_is_bounded(self) -> None:
        code, diagnostics, clients = self.run_startup("hung")

        self.assertNotEqual(code, 0)
        self.assertIn("did not become ready within 10 seconds", diagnostics)
        self.assertEqual(clients, {})
