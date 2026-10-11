from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tests.ai_harness_interface.test_generator import copy_source_licenses, seed_scaffold, write_settings
from tokimon.ai_harness_interface import contract, generator
from tokimon.workflow.pair import peer_config, state


ROOT = Path(__file__).resolve().parents[2]


def package_snapshot(root: Path) -> dict[str, bytes | str]:
    snapshot: dict[str, bytes | str] = {}
    for path in (root, *root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[name] = "link:" + os.readlink(path)
        elif path.is_dir():
            snapshot[name] = "directory"
        elif path.is_file():
            snapshot[name] = path.read_bytes()
        else:
            snapshot[name] = f"special:{path.lstat().st_mode}"
    return snapshot


class PublicReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_claude_cli())
        self.enterContext(without_local_plugin_creator())

    def test_public_packages_distribute_licenses_without_a_website_skill(self) -> None:
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                settings = write_settings(Path(tmp) / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=Path(tmp) / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generated = generator.generate_package(paths, settings_path=settings)

                self.assertNotIn("chatgpt", generated["skill_ids"])
                self.assertFalse((paths.package_root / "skills/chatgpt").exists())
                self.assertEqual(json.loads(paths.manifest_path.read_text())["license"], "MIT")
                for relative_path in contract.LICENSE_FILE_PATHS:
                    self.assertEqual((paths.package_root / relative_path).read_bytes(), (ROOT / relative_path).read_bytes())
                self.assertTrue(generator.validate_generated_package(paths)["ok"])

    def test_packaged_skill_dependencies_and_command_entrypoints_work_from_another_project(self) -> None:
        command_pattern = re.compile(
            r"(?P<prefix>PYTHONPATH=[^\n`]*?) -m (?P<module>tokimon(?:\.[a-z_]+)+)\b"
        )
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                caller = root / "unrelated-project"
                caller.mkdir()
                (caller / "tokimon.py").write_text("raise RuntimeError('caller shadow module executed')\n")
                before = package_snapshot(caller)
                settings = write_settings(root / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generated = generator.generate_package(paths, settings_path=settings)
                command_prefixes: dict[str, str] = {}
                for skill_id in generated["skill_ids"]:
                    text = (paths.package_root / "skills" / skill_id / "SKILL.md").read_text(encoding="utf-8")
                    for dependency in re.findall(r"(?<![\w/])skills/([a-z0-9-]+)/skill\.md\b", text):
                        self.assertIn(dependency, generated["skill_ids"], f"{skill_id} depends on omitted {dependency}")
                        self.assertTrue((ROOT / "skills" / dependency / "skill.md").is_file())
                    for match in command_pattern.finditer(text):
                        command_prefixes[match["module"]] = match["prefix"]
                self.assertTrue(
                    {"tokimon.workflow.pair", "tokimon.workflow.learning", "tokimon.workflow.code_audit"}
                    <= command_prefixes.keys()
                )
                environment = os.environ.copy()
                environment.pop("PYTHONPATH", None)
                environment.pop("PYTHONHOME", None)
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                for module, prefix in sorted(command_prefixes.items()):
                    with self.subTest(module=module):
                        result = subprocess.run(
                            ["bash", "-c", prefix + ' -m "$1" --help', "skill-entrypoint", module],
                            cwd=caller, env=environment, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=10,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("usage:", result.stdout.lower())
                self.assertEqual(package_snapshot(caller), before)

    def test_package_validation_rejects_a_mismatched_manifest_license(self) -> None:
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                settings = write_settings(Path(tmp) / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=Path(tmp) / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generator.generate_package(paths, settings_path=settings)
                payload = json.loads(paths.manifest_path.read_text())
                payload["license"] = "a-different-license"
                paths.manifest_path.write_text(json.dumps(payload))
                with self.assertRaises(generator.PackageValidationError):
                    generator.validate_generated_package(paths)

    def test_package_validation_rejects_missing_or_altered_license_files(self) -> None:
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                settings = write_settings(Path(tmp) / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=Path(tmp) / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generator.generate_package(paths, settings_path=settings)
                for relative_path in contract.LICENSE_FILE_PATHS:
                    path = paths.package_root / relative_path
                    original = path.read_bytes()
                    for action in ("delete", "alter"):
                        with self.subTest(file=relative_path, action=action):
                            if action == "delete":
                                path.unlink()
                            else:
                                path.write_bytes(original + b"\nAltered notice\n")
                            with self.assertRaises(generator.PackageValidationError):
                                generator.validate_generated_package(paths)
                            path.write_bytes(original)
                self.assertTrue(generator.validate_generated_package(paths)["ok"])

    def test_invalid_source_license_cannot_replace_an_existing_package(self) -> None:
        for shape in ("missing", "empty", "symlink"):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "source"
                copy_source_licenses(root)
                (root / "tokimon").mkdir()
                (root / "tokimon/__init__.py").write_text("")
                skill = root / "skills/demo/skill.md"
                skill.parent.mkdir(parents=True)
                skill.write_text("---\nname: demo\ndescription: Demonstration skill.\n---\n\n# Demo\n")
                paths = generator.paths_for(source_root=root, output_root=Path(tmp) / "out")
                seed_scaffold(paths)
                before = {p.relative_to(paths.package_root): p.read_bytes() for p in paths.package_root.rglob("*") if p.is_file()}
                license_path = root / "LICENSE"
                license_path.unlink()
                if shape == "empty":
                    license_path.write_text("")
                elif shape == "symlink":
                    license_path.symlink_to(ROOT / "LICENSE")
                settings = write_settings(Path(tmp) / "settings.json", source_root=root)

                with self.assertRaisesRegex(generator.PackageValidationError, "source license file LICENSE"):
                    generator.generate_package(paths, settings_path=settings)
                after = {p.relative_to(paths.package_root): p.read_bytes() for p in paths.package_root.rglob("*") if p.is_file()}
                self.assertEqual(after, before)

    def test_regeneration_rejects_output_notice_symlinks_before_mutation(self) -> None:
        for platform in ("codex", "claude"):
            for relative_path in contract.LICENSE_FILE_PATHS:
                with self.subTest(platform=platform, file=relative_path), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    settings = write_settings(root / "settings.json")
                    paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                    if platform == "codex":
                        seed_scaffold(paths)
                    generator.generate_package(paths, settings_path=settings)
                    external = root / "external.txt"
                    external.write_bytes(b"Keep unrelated file unchanged.\n")
                    target = paths.package_root / relative_path
                    target.unlink()
                    target.symlink_to(external)
                    before = package_snapshot(paths.package_root)

                    with self.assertRaisesRegex(generator.PackageValidationError, "symbolic link"):
                        generator.generate_package(paths, settings_path=settings)

                    self.assertEqual(external.read_bytes(), b"Keep unrelated file unchanged.\n")
                    self.assertEqual(package_snapshot(paths.package_root), before)

    def test_regeneration_rejects_linked_output_directories_before_mutation(self) -> None:
        for platform in ("codex", "claude"):
            for relative_path in ("licenses", "."):
                with self.subTest(platform=platform, directory=relative_path), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    settings = write_settings(root / "settings.json")
                    paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                    if platform == "codex":
                        seed_scaffold(paths)
                    generator.generate_package(paths, settings_path=settings)
                    target = paths.package_root / relative_path
                    external = root / "external-directory"
                    target.rename(external)
                    target.symlink_to(external, target_is_directory=True)
                    sentinel = external / "keep.txt"
                    sentinel.write_bytes(b"Keep unrelated directory unchanged.\n")
                    before = package_snapshot(external)

                    with self.assertRaisesRegex(generator.PackageValidationError, "symbolic link"):
                        generator.generate_package(paths, settings_path=settings)

                    self.assertTrue(target.is_symlink())
                    self.assertEqual(package_snapshot(external), before)

    def test_regeneration_rejects_nonregular_notice_targets_before_mutation(self) -> None:
        for platform in ("codex", "claude"):
            for shape in ("directory", "fifo"):
                with self.subTest(platform=platform, shape=shape), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    settings = write_settings(root / "settings.json")
                    paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                    if platform == "codex":
                        seed_scaffold(paths)
                    generator.generate_package(paths, settings_path=settings)
                    target = paths.package_root / "LICENSE"
                    target.unlink()
                    if shape == "directory":
                        target.mkdir()
                    else:
                        os.mkfifo(target)
                    before = package_snapshot(paths.package_root)

                    with self.assertRaisesRegex(generator.PackageValidationError, "regular file"):
                        generator.generate_package(paths, settings_path=settings)

                    self.assertEqual(package_snapshot(paths.package_root), before)

    def test_atomic_notice_replacement_does_not_follow_a_late_symlink(self) -> None:
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = write_settings(root / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generator.generate_package(paths, settings_path=settings)
                external = root / "external.txt"
                external.write_bytes(b"Keep late link target unchanged.\n")
                replace = os.replace
                injected = False

                def replace_with_late_link(source, destination, **kwargs):
                    nonlocal injected
                    if destination == "LICENSE" and not injected:
                        parent = kwargs["dst_dir_fd"]
                        os.unlink(destination, dir_fd=parent)
                        os.symlink(str(external), destination, dir_fd=parent)
                        injected = True
                    return replace(source, destination, **kwargs)

                with mock.patch.object(os, "replace", side_effect=replace_with_late_link):
                    generator.generate_package(paths, settings_path=settings)

                self.assertTrue(injected)
                self.assertEqual(external.read_bytes(), b"Keep late link target unchanged.\n")
                self.assertEqual((paths.package_root / "LICENSE").read_bytes(), (ROOT / "LICENSE").read_bytes())
                self.assertTrue(generator.validate_generated_package(paths)["ok"])

    def test_regeneration_replaces_hardlinked_notices_without_changing_other_names(self) -> None:
        for platform in ("codex", "claude"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = write_settings(root / "settings.json")
                paths = generator.paths_for(source_root=ROOT, output_root=root / "out", target_platform=platform)
                if platform == "codex":
                    seed_scaffold(paths)
                generator.generate_package(paths, settings_path=settings)
                external = root / "external.txt"
                external.write_bytes(b"Keep the other hardlink unchanged.\n")
                target = paths.package_root / "LICENSE"
                target.unlink()
                os.link(external, target)

                generator.generate_package(paths, settings_path=settings)

                self.assertEqual(external.read_bytes(), b"Keep the other hardlink unchanged.\n")
                self.assertEqual(target.read_bytes(), (ROOT / "LICENSE").read_bytes())
                self.assertTrue(generator.validate_generated_package(paths)["ok"])

    def test_removed_peer_selection_is_rejected_without_changing_settings(self) -> None:
        for access in ("native-subagent", "local-cli"):
            with self.subTest(access=access), tempfile.TemporaryDirectory() as tmp:
                settings = write_settings(Path(tmp) / "settings.json")
                payload = json.loads(settings.read_text())
                payload["pair"].update(access=access, peer_b="chatgpt")
                settings.write_text(json.dumps(payload))
                before = settings.read_bytes()
                configured = subprocess.run(
                    [sys.executable, "-m", "tokimon.ai_harness_interface", "configure", "--yes", "--settings-path", str(settings), "--json"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                )
                self.assertEqual(configured.returncode, 1, configured.stderr)
                self.assertFalse(json.loads(configured.stdout)["ok"])
                self.assertEqual(settings.read_bytes(), before)
                with self.assertRaises(peer_config.PeerSelectionConfigError):
                    peer_config.select_peers(payload["pair"])

    def test_active_pair_state_rejects_a_removed_browser_peer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            initialized = state.init_run("Unsupported peer", state_root=tmp)
            state_path = Path(initialized["state_path"])
            before = state_path.read_bytes()
            peers, selection = peer_config.select_peers({"access": "native-subagent", "peer_b": "codex", "cwd": tmp})
            peers["b"].update(agent="chatgpt", capability="chatgpt-conversation")
            with self.assertRaisesRegex(ValueError, "peers.b.capability"):
                state.update_state(
                    initialized["run_id"], state_root=tmp, status="in_progress", peers=peers,
                    peer_selection_config=selection, task_behavior="read_only",
                )
            self.assertEqual(state_path.read_bytes(), before)
