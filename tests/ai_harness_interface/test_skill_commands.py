from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tests.ai_harness_interface.test_generator import (
    ROOT,
    create_linked_checkout,
    seed_scaffold,
    write_settings,
)
from tokimon.ai_harness_interface import contract, generator, route, skills
from tokimon.memory.mem_palace.mem_palace import MemPalace
from tokimon.memory.mem_palace.project_context import resolve_project_memory_scope


class InstalledSkillCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_plugin_creator())
        self.enterContext(without_local_claude_cli())
        self.fixture = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.target = self.fixture / "adopting project"
        # A real Git boundary prevents the fixture from inheriting Tokimon's identity.
        git_dir = self.target / ".git"
        (git_dir / "objects").mkdir(parents=True)
        (git_dir / "refs").mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git_dir / "config").write_text(
            "[core]\nrepositoryformatversion = 0\nbare = false\n", encoding="utf-8"
        )
        shadow = self.target / "tokimon"
        shadow.mkdir()
        (shadow / "__init__.py").write_text(
            "raise RuntimeError('adopting-project package shadowed live Tokimon')\n",
            encoding="utf-8",
        )

    def run_command(self, command: str, *, pythonpath: str | None = None) -> str:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONSAFEPATH", None)
        env["PWD"] = str(self.target)
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=self.target,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.target / ".tokimon").exists())
        return result.stdout

    def test_generated_platform_commands_keep_project_and_global_memory(self) -> None:
        project = resolve_project_memory_scope(self.target, tokimon_root=ROOT)
        self.assertNotEqual(project.scope, "tokimon")
        self.assertEqual(project.identity_kind, "git-common-dir")
        database = self.fixture / "memory.db"
        with MemPalace(database) as palace:
            expected_ids = {
                palace.remember(
                    kind="fact", summary=f"routefixture {label}", scope=scope,
                    auto_extract_entities=False,
                ).id
                for label, scope in (("project", project.scope), ("global", "global"))
            }
            palace.remember(
                kind="fact", summary="routefixture unrelated", scope="other-project",
                auto_extract_entities=False,
            )
            palace.remember(
                kind="fact", summary="routefixture implementation", scope="tokimon",
                auto_extract_entities=False,
            )

        for platform in (contract.CODEX_PLATFORM, contract.CLAUDE_PLATFORM):
            with self.subTest(platform=platform.platform_id):
                output = self.fixture / platform.platform_id
                paths = generator.paths_for(
                    source_root=ROOT, output_root=output, target_platform=platform.platform_id,
                )
                output.mkdir()
                if platform.platform_id == "codex":
                    seed_scaffold(paths)
                settings = write_settings(output / "settings.local.json")
                generated = generator.generate_package(paths, settings_path=settings)
                for skill_id in generated["skill_ids"]:
                    text = (paths.package_root / "skills" / skill_id / "SKILL.md").read_text()
                    self.assertNotIn("{{tokimonCommand}}", text)
                    self.assertNotIn(".venv/bin/python -m tokimon", text)
                startup = (paths.package_root / "skills/using-tokimon/SKILL.md").read_text()
                memory = (paths.package_root / "skills/memory/SKILL.md").read_text()
                briefing = next(line for line in startup.splitlines() if "mem_palace briefing " in line)
                search = next(line for line in memory.splitlines() if "mem_palace search " in line)
                for command, name in ((briefing, "briefing"), (search, "search")):
                    command = command.replace(
                        f"mem_palace {name} ",
                        f"mem_palace --db {shlex.quote(str(database))} {name} ", 1,
                    ).replace("<keywords>", "routefixture")
                    payload = json.loads(self.run_command(command + " --json"))
                    if name == "briefing":
                        self.assertEqual(payload["memory_scope_selection"]["scopes"], [project.scope, "global"])
                        ids = {entry["id"] for entry in payload["memories"]}
                    else:
                        ids = {entry["id"] for entry in payload}
                    self.assertEqual(ids, expected_ids)

    def test_linked_source_quoted_paths_and_existing_imports_keep_caller_context(self) -> None:
        workspace = self.fixture / "workspace $(touch path-injected) ' space"
        primary, linked = create_linked_checkout(workspace)
        python = primary / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(ROOT / ".venv/bin/python")
        for checkout, label in ((primary, "primary"), (linked, "linked")):
            package = checkout / "tokimon"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "probe.py").write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "from extra_dependency import VALUE\n"
                f"print(json.dumps({{'selected': {label!r}, 'module': str(Path(__file__).resolve()), "
                "'interpreter': sys.executable, 'cwd': os.getcwd(), 'pwd': os.environ['PWD'], "
                "'pythonpath': os.environ['PYTHONPATH'], 'dependency': VALUE}))\n",
                encoding="utf-8",
            )
        dependency = self.fixture / "dependency $(touch env-injected) ' space"
        dependency.mkdir()
        (dependency / "extra_dependency.py").write_text("VALUE = 'retained'\n", encoding="utf-8")
        install_route = route.build_route(live_checkout_path=linked, runtime_root=primary / ".tokimon")
        text = skills.build_packaged_skill(
            skill_id="demo",
            source_text="---\nname: demo\ndescription: Probe.\n---\n\n{{tokimonCommand}} -m tokimon.probe\n",
            install_route=install_route,
        )
        command = next(line for line in text.splitlines() if " -m tokimon.probe" in line)
        payload = json.loads(self.run_command(command, pythonpath=str(dependency)))
        self.assertEqual(payload["selected"], "linked")
        self.assertEqual(payload["module"], str(linked / "tokimon/probe.py"))
        self.assertEqual(Path(payload["interpreter"]), python)
        self.assertEqual(payload["cwd"], str(self.target))
        self.assertEqual(payload["pwd"], str(self.target))
        self.assertEqual(payload["pythonpath"], os.pathsep.join((str(linked), str(dependency))))
        self.assertEqual(payload["dependency"], "retained")
        self.assertFalse((self.target / "path-injected").exists())
        self.assertFalse((self.target / "env-injected").exists())

    def test_package_validation_rejects_unresolved_command_prefix(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        text = skills.build_packaged_skill(
            skill_id="demo",
            source_text="---\nname: demo\ndescription: Demo.\n---\n\n{{tokimonCommand}} -m tokimon\n",
            install_route=install_route,
        )
        command = next(line for line in text.splitlines() if " -m tokimon" in line)
        broken = text.replace(command, "{{tokimonCommand}} -m tokimon")
        result = skills.validate_packaged_skill(skill_id="demo", text=broken, install_route=install_route)
        self.assertFalse(result["ok"])
        self.assertIn("unresolved tokimonCommand command placeholder", result["issues"])


if __name__ == "__main__":
    unittest.main()
