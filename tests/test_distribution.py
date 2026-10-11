from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_support import workspace_data_files


ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_build_entrypoint_works_without_the_source_on_pythonpath(self) -> None:
        script = (
            "import json, pathlib, sys, types\n"
            "setuptools = types.ModuleType('setuptools')\n"
            "setuptools.setup = lambda **kwargs: print(json.dumps(kwargs['data_files']))\n"
            "sys.modules['setuptools'] = setuptools\n"
            "source = pathlib.Path(sys.argv[1])\n"
            "exec(compile(source.read_text(), str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT / "setup.py")], cwd=tmpdir, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        groups = dict(json.loads(result.stdout))
        self.assertIn("tokimon/runtime.py", groups["share/tokimon/tokimon"])
        self.assertIn("LICENSE", groups["share/tokimon"])

    def test_workspace_carries_configuration_skills_docs_and_complete_licenses(self) -> None:
        files = {path for _, group in workspace_data_files(ROOT) for path in group}
        expected = {
            ".gitignore", "AGENTS.md", "LICENSE", "THIRD_PARTY_NOTICES", "build_support.py", "setup.py",
            "tokimon/settings.example.json", "tokimon/runtime.py",
            "tokimon/ai_harness_interface/templates/plugin-manifest-overlay.json",
            "tokimon/ai_harness_interface/templates/claude-plugin-manifest-overlay.json",
            "tokimon/ai_harness_interface/templates/installed-routing-note.md",
            "tokimon/tools/mcp/container_executor/Dockerfile",
            "tokimon/tools/mcp/remote_browser_broker/image/Dockerfile",
            "tokimon/tools/mcp/remote_browser_broker/image/seccomp_profile.json",
            "tokimon/tools/mcp/remote_browser_broker/image/start-session-browser.sh",
            "docs/guidelines/repository-guidelines.md", "tests/test_runtime.py",
        }
        expected.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "licenses").glob("*.txt"))
        expected.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "skills").glob("*/skill.md"))
        self.assertTrue(expected.issubset(files), expected - files)
        self.assertFalse(any("chatgpt" in path.lower() for path in files))
        self.assertFalse(any(part in {".tokimon", ".venv", ".git", "__pycache__"} for path in files for part in Path(path).parts))

    def test_declared_assets_are_selected_without_private_runtime_or_undeclared_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in (".gitignore", "AGENTS.md", "README.md", "LICENSE", "THIRD_PARTY_NOTICES", "setup.py", "build_support.py", "MANIFEST.in"):
                (root / name).write_text("Fixture\n")
            (root / "pyproject.toml").write_text('[tool.setuptools.package-data]\ntokimon = ["templates/*.json"]\n')
            for relative in ("tokimon/__init__.py", "tokimon/templates/example.json", "tokimon/local-secret.txt", "tokimon/.tokimon/private.json", "tokimon/__pycache__/example.pyc", "skills/demo/skill.md", "docs/guide.md", "licenses/upstream.txt", "tests/test_demo.py", ".github/workflows/check.yml"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fixture": True}))

            groups = workspace_data_files(root)
            files = {path for _, group in groups for path in group}

            self.assertIn("tokimon/templates/example.json", files)
            self.assertIn("tokimon/__init__.py", files)
            self.assertNotIn("tokimon/local-secret.txt", files)
            self.assertNotIn("tokimon/.tokimon/private.json", files)
            self.assertNotIn("tokimon/__pycache__/example.pyc", files)
            self.assertTrue(all(destination.startswith("share/tokimon") for destination, _ in groups))

    def test_symlinked_distribution_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pyproject.toml").write_text('[tool.setuptools.package-data]\ntokimon = []\n')
            package = root / "tokimon"
            package.mkdir()
            external = root / "external.py"
            external.write_text("private = True\n")
            (package / "foreign.py").symlink_to(external)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                workspace_data_files(root)
