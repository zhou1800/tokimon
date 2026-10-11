from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tokimon.tools.mcp.container_executor.deployment import download_tunnel_client


class DownloadTests(unittest.TestCase):
    def responses(self, *, valid_digest=True, machine="amd64", include_variants=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../../tunnel-client", b"verified-client")
            archive.writestr("../../never-extract", b"unrelated")
        archive = stream.getvalue()
        release = {"tag_name": "v0.1.0", "html_url": "https://github.com/openai/tunnel-client/releases/tag/v0.1.0", "assets": [
            {"name": f"tunnel-client-v0.1.0-linux-{machine}.zip", "browser_download_url": f"https://github.com/openai/tunnel-client/releases/download/v0.1.0/tunnel-client-v0.1.0-linux-{machine}.zip", "digest": "sha256:" + (hashlib.sha256(archive).hexdigest() if valid_digest else "0" * 64)}]}
        if include_variants:
            for name in (f"tunnel-client-runtime-v0.1.0-linux-{machine}.zip",
                         f"tunnel-client-runtime-cloudflared-v0.1.0-linux-{machine}.zip",
                         f"tunnel-client-v0.0.9-linux-{machine}.zip",
                         "tunnel-client-v0.1.0-linux-" + ("arm64" if machine == "amd64" else "amd64") + ".zip"):
                release["assets"].insert(0, {"name": name, "browser_download_url": "https://github.com/openai/tunnel-client/releases/download/v0.1.0/" + name, "digest": "sha256:" + "0" * 64})
        return [io.BytesIO(json.dumps(release).encode()), io.BytesIO(archive)]

    def test_verified_archive_writes_only_fixed_binary_and_receipt(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("platform.machine", return_value="x86_64"), mock.patch("urllib.request.urlopen", side_effect=self.responses()):
            payload = download_tunnel_client(workspace_root=Path(temporary))
            binary = Path(payload["client_binary"])
            self.assertEqual(binary.read_bytes(), b"verified-client")
            self.assertEqual(payload["binary_sha256"], hashlib.sha256(b"verified-client").hexdigest())
            self.assertEqual(sorted(p.name for p in binary.parent.iterdir()), ["receipt.json", "tunnel-client"])
            self.assertFalse(any(p.name == "never-extract" for p in Path(temporary).rglob("*")))

    def test_digest_mismatch_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch("platform.machine", return_value="x86_64"), mock.patch("urllib.request.urlopen", side_effect=self.responses(valid_digest=False)):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download_tunnel_client(workspace_root=Path(temporary))
            self.assertFalse((Path(temporary) / ".tokimon").exists())

    def test_mixed_release_variants_select_full_client_for_each_linux_architecture(self):
        for host_machine, asset_machine in (("x86_64", "amd64"), ("aarch64", "arm64")):
            with self.subTest(machine=host_machine), tempfile.TemporaryDirectory() as temporary:
                with mock.patch("platform.system", return_value="Linux"), mock.patch("platform.machine", return_value=host_machine), mock.patch("urllib.request.urlopen", side_effect=self.responses(machine=asset_machine, include_variants=True)) as fetch:
                    payload = download_tunnel_client(workspace_root=Path(temporary))
                self.assertEqual(Path(payload["client_binary"]).read_bytes(), b"verified-client")
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(fetch.call_args.args[0].full_url,
                                 f"https://github.com/openai/tunnel-client/releases/download/v0.1.0/tunnel-client-v0.1.0-linux-{asset_machine}.zip")

    def test_missing_or_duplicate_full_client_fails_before_archive_download(self):
        for duplicate in (False, True):
            responses = self.responses(include_variants=True)
            release = json.loads(responses[0].getvalue())
            full = next(a for a in release["assets"] if a["name"] == "tunnel-client-v0.1.0-linux-amd64.zip")
            if duplicate:
                release["assets"].append(full.copy())
            else:
                release["assets"].remove(full)
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as temporary:
                with mock.patch("platform.machine", return_value="x86_64"), mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(release).encode())) as fetch:
                    with self.assertRaisesRegex(ValueError, "no unique supported archive"):
                        download_tunnel_client(workspace_root=Path(temporary))
                    self.assertEqual(fetch.call_count, 1)
                self.assertFalse((Path(temporary) / ".tokimon").exists())

    def test_invalid_release_tag_fails_before_archive_download(self):
        responses = self.responses()
        release = json.loads(responses[0].getvalue())
        release["tag_name"] = "../../invalid"
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("platform.machine", return_value="x86_64"), mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(release).encode())) as fetch:
                with self.assertRaisesRegex(ValueError, "invalid official release tag"):
                    download_tunnel_client(workspace_root=Path(temporary))
                self.assertEqual(fetch.call_count, 1)
            self.assertFalse((Path(temporary) / ".tokimon").exists())
