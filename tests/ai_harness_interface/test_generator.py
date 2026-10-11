from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.ai_harness_interface.support import without_local_claude_cli, without_local_plugin_creator
from tokimon import runtime
from tokimon.ai_harness_interface import (
    claude_cli,
    configuration,
    contract,
    generator,
    mcp,
    plugin_manifest,
    route,
    skills,
)


def seed_scaffold(paths: generator.GenerationPaths) -> None:
    (paths.package_root / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (paths.package_root / "skills").mkdir(parents=True, exist_ok=True)
    paths.marketplace_path.parent.mkdir(parents=True, exist_ok=True)
    (paths.package_root / ".codex-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "tokimon",
                "version": "0.1.0",
                "description": "Seeded plugin",
                "author": {"name": "Local developer"},
                "skills": "./skills/",
                "mcpServers": "./.mcp.json",
                "interface": {
                    "displayName": "Tokimon",
                    "shortDescription": "Use Tokimon in Codex.",
                    "longDescription": "Tokimon adds a local Codex plugin scaffold.",
                    "developerName": "Local developer",
                    "category": "Productivity",
                    "capabilities": [],
                    "defaultPrompt": "Help me use Tokimon.",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (paths.package_root / ".mcp.json").write_text('{"mcpServers": {}}\n', encoding="utf-8")
    paths.marketplace_path.write_text(
        json.dumps(
            {
                "name": "tokimon-local",
                "interface": {"displayName": "Tokimon Local"},
                "plugins": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_settings(
    path: Path,
    *,
    source_root: Path = ROOT,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "codex": {"model": "gpt-test", "effort": "xhigh"},
                    "claude": {"model": "claude-test", "effort": "max"},
                },
                "ai_debate": {
                    "access": "local-cli",
                    "peer_b": "claude",
                },
                "pair": {
                    "access": "local-cli",
                    "peer_b": "claude",
                    "cwd": str(source_root),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def expected_package_files(source_root: Path = ROOT, *, manifest_dir: str = ".codex-plugin") -> list[str]:
    package_skill_ids = contract.package_skill_ids(source_root=source_root)
    metadata_files = (
        [
            f"skills/{skill_id}/agents/openai.yaml"
            for skill_id in package_skill_ids
            if skills.load_source_openai_metadata(source_root=source_root, skill_id=skill_id)
            is not None
        ]
        if manifest_dir == ".codex-plugin"
        else []
    )
    return sorted(
        [
            f"{manifest_dir}/plugin.json",
            ".mcp.json",
            *(f"skills/{skill_id}/SKILL.md" for skill_id in package_skill_ids),
            *metadata_files,
            *contract.LICENSE_FILE_PATHS,
        ]
    )


def copy_source_licenses(source_root: Path) -> None:
    for relative_path in contract.LICENSE_FILE_PATHS:
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative_path).read_bytes())


def generate_codex_fixture(tmp_root: Path) -> tuple[generator.GenerationPaths, route.InstallRoute]:
    paths = generator.paths_for(source_root=ROOT, output_root=tmp_root / "out")
    settings_path = write_settings(tmp_root / "settings.local.json")
    seed_scaffold(paths)
    generator.generate_package(paths, settings_path=settings_path)
    return paths, generator.install_route_for(paths)


def create_linked_checkout(tmp_root: Path) -> tuple[Path, Path]:
    primary = (tmp_root / "primary").resolve()
    linked = (tmp_root / "linked").resolve()
    git_dir = primary / ".git" / "worktrees" / "linked"
    git_dir.mkdir(parents=True)
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")
    skill_dir = linked / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.md").write_text("---\nname: demo\ndescription: Demo.\n---\n", encoding="utf-8")
    return primary, linked


class AiHarnessGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(without_local_plugin_creator())
        self.enterContext(mock.patch("tokimon.ai_harness_interface.installation_policy.which", return_value="/test/docker"))

    def test_default_output_root_is_local_runtime_generated_root(self) -> None:
        paths = generator.paths_for(source_root=ROOT)
        runtime_owner = runtime.runtime_workspace_root(ROOT)
        relative = paths.output_root.relative_to(runtime_owner).as_posix()
        ignored = subprocess.run(
            ["git", "check-ignore", relative],
            cwd=runtime_owner,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(relative, contract.CODEX_PLATFORM.generated_output_relative_path.as_posix())
        self.assertEqual(paths.package_root, paths.output_root / "plugins" / "tokimon")
        self.assertEqual(paths.marketplace_path, paths.output_root / ".agents" / "plugins" / "marketplace.json")
        self.assertEqual(ignored.returncode, 0, ignored.stderr)

    def test_generate_and_validate_closed_codex_package_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generated = generator.generate_package(paths, settings_path=settings_path)
            validation = generator.validate_generated_package(paths)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )

        self.assertTrue(generated["ok"])
        self.assertTrue(validation["ok"])
        self.assertEqual(files, expected_package_files())

    def test_generate_emits_supported_mcp_set_when_docker_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)

            generated = generator.generate_package(paths, settings_path=settings_path)
            payload = json.loads((paths.package_root / ".mcp.json").read_text(encoding="utf-8"))
            validation = generator.validate_generated_package(paths)

        expected_ids = [entry.server_id for entry in contract.MCP_ENTRIES]
        self.assertEqual(generated["mcp_server_ids"], expected_ids)
        self.assertEqual(set(payload["mcpServers"]), set(expected_ids))
        self.assertTrue(validation["ok"])

    def test_validation_rejects_missing_packaged_mcp_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)

            generator.generate_package(paths, settings_path=settings_path)
            payload = json.loads((paths.package_root / ".mcp.json").read_text(encoding="utf-8"))
            payload["mcpServers"].pop(contract.MCP_ENTRIES[0].server_id)
            (paths.package_root / ".mcp.json").write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                generator.PackageValidationError,
                "mcp-manifest",
            ):
                generator.validate_generated_package(paths)

    def test_generate_package_derives_skill_inventory_from_root_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output = Path(tmpdir) / "output"
            copy_source_licenses(root)
            (root / "tokimon").mkdir(parents=True)
            (root / "tokimon" / "__init__.py").write_text("", encoding="utf-8")
            skill_dir = root / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "skill.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n",
                encoding="utf-8",
            )
            paths = generator.paths_for(source_root=root, output_root=output)
            settings_path = write_settings(Path(tmpdir) / "settings.local.json", source_root=root)
            seed_scaffold(paths)

            generated = generator.generate_package(paths, settings_path=settings_path)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )

        self.assertEqual(generated["skill_ids"], ["demo"])
        self.assertIn("skills/demo/SKILL.md", files)
        self.assertNotIn("skills/tokimon/SKILL.md", files)
        self.assertNotIn("skills/code-audit/SKILL.md", files)

    def test_generated_skills_share_the_installed_routing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, install_route = generate_codex_fixture(Path(tmpdir))
            generated = {
                skill_id: (paths.package_root / "skills" / skill_id / "SKILL.md").read_text(
                    encoding="utf-8"
                )
                for skill_id in contract.package_skill_ids(source_root=ROOT)
            }

        for skill_id, text in generated.items():
            with self.subTest(skill_id=skill_id):
                self.assertEqual(skills.parse_skill(text).name, skill_id)
                self.assertIn("## Tokimon Routing", text)
                self.assertIn(str(install_route.liveCheckoutPath), text)
                self.assertIn(str(install_route.runtimeRoot), text)
                self.assertIn("Resolve Tokimon-owned command entrypoints", text)
                self.assertIn("Treat the current repository as the task subject", text)
                self.assertIn("When editing source files, use the current task checkout", text)
                self.assertNotIn("{{tokimonPython}}", text)

    def test_generated_code_audit_preserves_review_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            text = (paths.package_root / "skills" / "code-audit" / "SKILL.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("Do not judge code in isolation.", text)
        self.assertIn("one audit subject or small coherent subject group per subagent task", text)
        self.assertNotIn("pass-lock", text)
        self.assertNotIn("target recheck", text)

    def test_generated_code_audit_fix_preserves_remediation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            text = (
                paths.package_root / "skills" / "code-audit-fix" / "SKILL.md"
            ).read_text(encoding="utf-8")

        self.assertIn("Run a bounded remediation loop", text)
        self.assertIn("Freeze each target's subject and file set", text)
        self.assertIn("no new `fail` or `needs_human_review` rows", text)
        self.assertIn("Do not ask for human approval before ordinary scoped fixes", text)
        self.assertIn("Every edit hunk must be traceable to one selected finding", text)
        self.assertIn("use the newest `pass-locks.json`", text)
        self.assertIn("prior `pass` and `not_applicable` rows", text)
        self.assertIn("row lock must not discard the rest", text)

    def test_generated_document_audit_preserves_read_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            text = (
                paths.package_root / "skills" / "document-audit" / "SKILL.md"
            ).read_text(encoding="utf-8")

        self.assertIn("without editing audited documents", text)
        self.assertIn("Use `document-audit-fix`", text)
        self.assertIn("one document or small coherent document group per subagent task", text)

    def test_generated_document_audit_fix_preserves_remediation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            text = (
                paths.package_root / "skills" / "document-audit-fix" / "SKILL.md"
            ).read_text(encoding="utf-8")

        self.assertIn("Run a bounded remediation loop", text)
        self.assertIn("fresh re-audit", text)
        self.assertIn("Freeze each target's document list", text)
        self.assertIn("Proceed without human approval for ordinary scoped documentation fixes", text)
        self.assertIn("Every edit hunk must be traceable to one selected finding", text)
        self.assertIn("Do not re-audit or edit a clean or settled unit unless", text)
        self.assertIn("newest `clean-unit-locks.json`", text)
        self.assertIn("Pass the plan to `verify-reports` and `compile-reports`", text)
        self.assertIn("`active_unit_count` and `waiting_unit_count` are both zero", text)


    def test_product_development_packaged_skill_routes_continuity_to_tokimon_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            install_route = generator.install_route_for(paths)
            generator.generate_package(paths, settings_path=settings_path)
            product_development = (
                paths.package_root / "skills" / "product-development" / "SKILL.md"
            ).read_text(encoding="utf-8")

        self.assertIn("name: product-development", product_development)
        self.assertIn(str(install_route.runtimeRoot), product_development)
        self.assertIn("not resolve the paths below against that repository", product_development)
        self.assertIn("Do not add `.tokimon/` to that repository's `.gitignore`", product_development)

    def test_installed_routing_note_uses_markdown_template(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        template = (
            ROOT / "tokimon" / "ai_harness_interface" / "templates" / "installed-routing-note.md"
        ).read_text(encoding="utf-8")

        rendered = skills.build_installed_routing_note(install_route=install_route)

        self.assertIn("{{liveCheckoutPath}}", template)
        self.assertIn("{{runtimeRoot}}", template)
        self.assertIn("current task checkout", template)
        self.assertIn("## Tokimon Routing", rendered)
        self.assertIn(str(install_route.liveCheckoutPath), rendered)
        self.assertIn(str(install_route.runtimeRoot), rendered)
        self.assertIn("When editing source files, use the current task checkout", rendered)
        self.assertNotIn("{{liveCheckoutPath}}", rendered)
        self.assertNotIn("{{runtimeRoot}}", rendered)

    def test_plugin_manifest_overlay_uses_json_template(self) -> None:
        template_path = (
            ROOT / "tokimon" / "ai_harness_interface" / "templates" / "plugin-manifest-overlay.json"
        )
        overlay = json.loads(template_path.read_text(encoding="utf-8"))

        manifest = plugin_manifest.apply_manifest_overlay(
            {
                "name": "seed",
                "version": "seed-version",
                "interface": {"defaultPrompt": "seed prompt"},
            },
            version="9.9.9",
        )

        self.assertEqual(manifest["description"], overlay["description"])
        self.assertEqual(manifest["interface"]["longDescription"], overlay["interface"]["longDescription"])
        self.assertEqual(manifest["interface"]["defaultPrompt"], overlay["interface"]["defaultPrompt"])
        self.assertEqual(manifest["name"], contract.PACKAGE_ID)
        self.assertEqual(manifest["version"], "9.9.9")

    def test_codex_package_does_not_ship_hook_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )
            hooks_root_exists = (paths.package_root / "hooks").exists()

        self.assertFalse(hooks_root_exists)
        self.assertFalse(any(path.startswith("hooks/") for path in files))

    def test_manifest_hooks_field_remains_rejected_for_codex_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            manifest_path = paths.package_root / ".codex-plugin" / "plugin.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["hooks"] = "./hooks/session-start"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(generator.PackageValidationError, "plugin-manifest"):
                generator.validate_generated_package(paths)

    def test_manifest_template_drift_remains_rejected_for_codex_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            manifest_path = paths.package_root / ".codex-plugin" / "plugin.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["interface"]["defaultPrompt"] = "Use another assistant."
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(generator.PackageValidationError, "plugin-manifest"):
                generator.validate_generated_package(paths)

    def test_memory_activation_skill_is_packaged_and_validated_as_bounded_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            install_route = generator.install_route_for(paths)
            generator.generate_package(paths, settings_path=settings_path)
            text = (paths.package_root / "skills" / "memory" / "SKILL.md").read_text(
                encoding="utf-8"
            )

        result = skills.validate_packaged_skill(
            skill_id="memory",
            text=text,
            install_route=install_route,
        )

        self.assertTrue(result["ok"], result["issues"])
        self.assertNotIn("## Memory Authority", text)
        self.assertNotIn("Use when starting any conversation", text)
        self.assertIn("Keep activation to memory routing only", text)
        self.assertIn("Mem Palace remains authoritative", text)
        self.assertIn("Tokimon memory is degraded", text)
        self.assertIn("Do not write `.tokimon/memory/memory.db` directly", text)
        self.assertNotIn("TOKIMON_AI_HARNESS_CONFIG", text)

    def test_packaged_skill_validator_rejects_missing_routing_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            install_route = generator.install_route_for(paths)
            generator.generate_package(paths, settings_path=settings_path)
            text = (paths.package_root / "skills" / "code-audit" / "SKILL.md").read_text(
                encoding="utf-8"
            )

        self.assertTrue(
            skills.validate_packaged_skill(
                skill_id="code-audit",
                text=text,
                install_route=install_route,
            )["ok"]
        )
        cases = (
            text.replace("## Tokimon Routing", "## Missing Routing"),
            text.replace(str(install_route.liveCheckoutPath), "/missing/live"),
            text.replace(str(install_route.runtimeRoot), "/missing/runtime"),
            text.replace(
                "Resolve Tokimon-owned command entrypoints",
                "Use local command entrypoints",
            ),
            text.replace(
                "When editing source files, use the current task checkout",
                "Edit wherever command routing points",
            ),
        )
        for broken in cases:
            with self.subTest():
                self.assertFalse(
                    skills.validate_packaged_skill(
                        skill_id="code-audit",
                        text=broken,
                        install_route=install_route,
                    )["ok"]
                )

    def test_memory_activation_skill_preserves_bounded_memory_rules(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        text = skills.build_packaged_skill(
            skill_id="memory",
            source_text=skills.load_source_skill(source_root=ROOT, skill_id="memory"),
            install_route=install_route,
        )

        self.assertIn("Use Mem Palace for local durable memory and project continuity", text)
        self.assertIn('briefing --project-root "$PWD" --include-scope global --limit 8', text)
        self.assertIn(str(install_route.liveCheckoutPath / ".venv" / "bin" / "python"), text)
        self.assertIn("Treat Codex memory as supplemental", text)
        self.assertIn("Mem Palace remains authoritative", text)
        self.assertIn("unless this session or task already has a current briefing", text)
        self.assertIn("do not activate Tokimon tools, workflows, MCP servers", text)
        self.assertIn("Tokimon memory is degraded", text)
        self.assertFalse(
            skills.validate_packaged_skill(
                skill_id="memory",
                text=text.replace(
                    'briefing --project-root "$PWD" --include-scope global --limit 8',
                    "briefing --scope tokimon --limit 8",
                ),
                install_route=install_route,
            )["ok"]
        )

    def test_packaged_skill_validator_rejects_non_startup_conversation_trigger(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        text = skills.build_packaged_skill(
            skill_id="memory",
            source_text=skills.load_source_skill(source_root=ROOT, skill_id="memory"),
            install_route=install_route,
        )
        broken = text.replace(
            "description: Use for explicit Mem Palace search, durable memory writes, and deeper memory operations.",
            "description: Use when starting any conversation to enforce memory.",
        )

        result = skills.validate_packaged_skill(
            skill_id="memory",
            text=broken,
            install_route=install_route,
        )

        self.assertFalse(result["ok"])
        self.assertIn(
            "only the startup bootstrap skill may use conversation-start trigger wording",
            result["issues"],
        )

    def test_startup_skill_preserves_required_contract_text(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        text = skills.build_packaged_skill(
            skill_id="using-tokimon",
            source_text=skills.load_source_skill(source_root=ROOT, skill_id="using-tokimon"),
            install_route=install_route,
        )

        self.assertIn(
            "Before any substantive Tokimon-enabled work, you MUST complete the startup obligations below.",
            text,
        )
        self.assertIn("You do not have a choice: complete the startup obligations", text)
        self.assertIn("This is not optional.", text)
        self.assertIn("You cannot rationalize your way out of it.", text)
        self.assertIn("Treat the user's current repository and request as the task boundary.", text)
        self.assertIn("Load only the local routing and skill guidance the task needs.", text)
        self.assertIn("If a Tokimon skill applies, you MUST follow it exactly.", text)
        self.assertIn("You do not have a choice.", text)
        self.assertIn(
            'Instructions say WHAT, not HOW. "Add X" or "Fix Y" doesn\'t mean skip workflows.',
            text,
        )
        self.assertIn("Do not skip, reorder, soften, reinterpret, or substitute required steps.", text)
        self.assertIn("Do not claim compliance unless you actually followed", text)
        self.assertIn("If a required step is impossible, report the blocker", text)
        self.assertIn(
            "Run one bounded Mem Palace briefing before substantive Tokimon-enabled work unless this session or task already has a current briefing.",
            text,
        )
        self.assertNotIn("Use memory on demand", text)
        self.assertIn("Load user-specified docs first.", text)
        self.assertIn("choose the first suitable and available option", text)
        self.assertIn("`In-App Browser`", text)
        self.assertIn("`Connected Chrome`", text)
        self.assertIn("`mcp.remote-browser-broker`", text)
        self.assertLess(text.index("`In-App Browser`"), text.index("`Connected Chrome`"))
        self.assertLess(
            text.index("`Connected Chrome`"),
            text.index("`mcp.remote-browser-broker`"),
        )
        self.assertIn("Websites do not change this order", text)
        self.assertIn("not competing defaults", text)
        self.assertNotIn("IMPORTANT_TOKIMON_STARTUP", text)

    def test_packaged_skill_validator_allows_first_party_prose_changes(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        text = skills.build_packaged_skill(
            skill_id="using-tokimon",
            source_text=skills.load_source_skill(source_root=ROOT, skill_id="using-tokimon"),
            install_route=install_route,
        )
        changed = text.replace(
            "Before any substantive Tokimon-enabled work, you MUST complete the startup obligations below.",
            "Startup work must happen before substantive Tokimon-enabled work.",
        ).replace(
            'Instructions say WHAT, not HOW. "Add X" or "Fix Y" doesn\'t mean skip workflows.',
            "User goals do not replace required workflows.",
        )

        result = skills.validate_packaged_skill(
            skill_id="using-tokimon",
            text=changed,
            install_route=install_route,
        )

        self.assertTrue(result["ok"], result["issues"])

    def test_generated_package_validation_rejects_packaged_skill_missing_subject_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            skill_path = paths.package_root / "skills" / "learning" / "SKILL.md"
            text = skill_path.read_text(encoding="utf-8")
            subject_routing = "Treat the current repository as the task subject unless the user explicitly says otherwise."
            self.assertIn(subject_routing, text)
            skill_path.write_text(
                text.replace(subject_routing, "Treat Tokimon as the task subject."),
                encoding="utf-8",
            )

            with self.assertRaises(generator.PackageValidationError) as raised:
                generator.validate_generated_package(paths)

        self.assertIn("packaged-skills", str(raised.exception))

    def test_generated_package_validation_rejects_stale_packaged_skill_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            copy_source_licenses(root)
            (root / "tokimon").mkdir(parents=True)
            (root / "tokimon" / "__init__.py").write_text("", encoding="utf-8")
            source_skill_path = root / "skills" / "learning" / "skill.md"
            source_skill_path.parent.mkdir(parents=True)
            source_text = skills.load_source_skill(source_root=ROOT, skill_id="learning")
            source_skill_path.write_text(source_text, encoding="utf-8")
            paths = generator.paths_for(source_root=root, output_root=Path(tmpdir) / "out")
            settings_path = write_settings(Path(tmpdir) / "settings.local.json", source_root=root)
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)

            generated_text = (
                paths.package_root / "skills" / "learning" / "SKILL.md"
            ).read_text(encoding="utf-8")
            source_skill_path.write_text(
                source_text.rstrip() + "\n\nNew canonical guidance.\n",
                encoding="utf-8",
            )

            invariant_check = skills.validate_packaged_skill(
                skill_id="learning",
                text=generated_text,
                install_route=generator.install_route_for(paths),
            )
            self.assertTrue(invariant_check["ok"], invariant_check["issues"])

            with self.assertRaisesRegex(generator.PackageValidationError, "packaged-skills"):
                generator.validate_generated_package(paths)

    def test_generated_package_validation_reports_missing_packaged_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            (paths.package_root / "skills" / "learning" / "SKILL.md").unlink()

            with self.assertRaisesRegex(generator.PackageValidationError, "packaged-skills"):
                generator.validate_generated_package(paths)

    def test_generated_package_validation_reports_malformed_packaged_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            skill_path = paths.package_root / "skills" / "learning" / "SKILL.md"
            skill_path.write_text("malformed generated skill\n", encoding="utf-8")

            with self.assertRaisesRegex(generator.PackageValidationError, "packaged-skills"):
                generator.validate_generated_package(paths)

    def test_generated_package_validation_reports_non_utf8_packaged_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            skill_path = paths.package_root / "skills" / "learning" / "SKILL.md"
            skill_path.write_bytes(b"\xff")

            with self.assertRaisesRegex(generator.PackageValidationError, "packaged-skills"):
                generator.validate_generated_package(paths)

    def test_generated_package_validation_rejects_external_skill_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, _ = generate_codex_fixture(root)
            skill_path = paths.package_root / "skills" / "learning" / "SKILL.md"
            external = root / "external-skill.md"
            external.write_text(skill_path.read_text(encoding="utf-8"), encoding="utf-8")
            skill_path.unlink()
            skill_path.symlink_to(external)

            with self.assertRaisesRegex(
                generator.PackageValidationError,
                "exact-file-catalog",
            ):
                generator.validate_generated_package(paths)

            catalog = generator._check_exact_file_catalog(paths, policy=generator.load_installation_policy(paths))

        self.assertFalse(catalog["ok"])
        self.assertEqual(catalog["unsafe_links"], ["skills/learning/SKILL.md"])

    def test_generated_package_validation_reports_non_utf8_openai_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths, _ = generate_codex_fixture(Path(tmpdir))
            metadata_path = (
                paths.package_root
                / "skills"
                / "learning"
                / "agents"
                / "openai.yaml"
            )
            metadata_path.write_bytes(b"\xff")

            with self.assertRaisesRegex(
                generator.PackageValidationError,
                "packaged-skills",
            ):
                generator.validate_generated_package(paths)

    def test_regeneration_removes_stale_owned_skill_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            stale = paths.package_root / "skills" / "stale" / "SKILL.md"
            stale.parent.mkdir(parents=True)
            stale.write_text("---\nname: stale\ndescription: Stale.\n---\n", encoding="utf-8")
            stale_hook = paths.package_root / "hooks" / "session-start"
            stale_hook.parent.mkdir(parents=True)
            stale_hook.write_text("stale hook\n", encoding="utf-8")

            generator.generate_package(paths, settings_path=settings_path)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )
            hooks_root_exists = (paths.package_root / "hooks").exists()

        self.assertNotIn("skills/stale/SKILL.md", files)
        self.assertNotIn("hooks/session-start", files)
        self.assertFalse(hooks_root_exists)
        self.assertEqual(files, expected_package_files())

    def test_generated_package_outputs_are_not_tracked_source_files(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "plugins/tokimon", ".agents/plugins/marketplace.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(tracked.returncode, 0, tracked.stderr)
        self.assertEqual(tracked.stdout.strip(), "")

    def test_regeneration_uses_the_current_source_skill_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output = Path(tmpdir) / "output"
            copy_source_licenses(root)
            (root / "tokimon").mkdir(parents=True)
            (root / "tokimon" / "__init__.py").write_text("", encoding="utf-8")
            skill_dir = root / "skills" / "demo"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "skill.md"
            skill_path.write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\nSource body version one.\n",
                encoding="utf-8",
            )
            paths = generator.paths_for(source_root=root, output_root=output)
            settings_path = write_settings(Path(tmpdir) / "settings.local.json", source_root=root)
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)

            skill_path.write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\nSource body version two.\n",
                encoding="utf-8",
            )
            generator.generate_package(paths, settings_path=settings_path)
            generated = (paths.package_root / "skills" / "demo" / "SKILL.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("Source body version two.", generated)
        self.assertNotIn("Source body version one.", generated)

    def test_generated_mcp_manifest_routes_back_through_install_time_paths(self) -> None:
        paths = generator.paths_for(source_root=ROOT)
        install_route = generator.install_route_for(paths)
        manifest = mcp.build_mcp_manifest(
            install_route=install_route,
        )
        servers = manifest["mcpServers"]

        self.assertEqual(set(servers), {entry.server_id for entry in contract.MCP_ENTRIES})
        for entry in contract.MCP_ENTRIES:
            server = servers[entry.server_id]
            rendered = json.dumps(server)
            self.assertEqual(server["command"], str(install_route.liveCheckoutPath / ".venv" / "bin" / "python"))
            self.assertEqual(server["cwd"], str(install_route.liveCheckoutPath))
            self.assertEqual(server["args"], ["-m", entry.module, *entry.serve_args])
            self.assertEqual(server["args"][-1], "serve")
            self.assertNotIn("TOKIMON_AI_HARNESS_CONFIG", rendered)
            self.assertIn(entry.module, rendered)
            self.assertNotIn("/plugins/cache/", rendered)
        self.assertNotIn("serve-http", json.dumps(manifest))

    def test_mcp_health_distinguishes_route_from_operational_availability(self) -> None:
        install_route = generator.install_route_for(generator.paths_for(source_root=ROOT))
        unavailable = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout='{"ok": false, "dependency": "missing"}\n',
            stderr="",
        )
        with mock.patch.object(mcp.subprocess, "run", return_value=unavailable):
            checks = mcp.run_health_checks(install_route)

        self.assertEqual(len(checks), len(contract.MCP_ENTRIES))
        for check in checks:
            self.assertTrue(check["ok"])
            self.assertEqual(check["status"], "pass")
            self.assertEqual(check["returncode"], 1)
            self.assertFalse(check["parsed_stdout"]["ok"])

    def test_linked_checkout_uses_primary_python_and_linked_source_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary, linked = create_linked_checkout(Path(tmpdir))
            install_route = route.build_route(live_checkout_path=linked, runtime_root=primary / ".tokimon")
            python = primary / ".venv" / "bin" / "python"

            for platform in (contract.CODEX_PLATFORM, contract.CLAUDE_PLATFORM):
                with self.subTest(platform=platform.platform_id):
                    manifest = mcp.build_mcp_manifest(install_route=install_route, platform=platform)
                    for server in manifest["mcpServers"].values():
                        self.assertEqual(server["command"], str(python))
                        if platform.platform_id == "codex":
                            self.assertEqual(server["cwd"], str(linked))
                        else:
                            self.assertEqual(server["env"], {"PYTHONPATH": str(linked)})
            packaged = skills.build_packaged_skill(
                skill_id="demo",
                source_text="---\nname: demo\ndescription: Demo.\n---\n\n{{tokimonPython}} -m tokimon\n",
                install_route=install_route,
            )
            self.assertIn(f"{python} -m tokimon", packaged)
            self.assertNotIn(str(linked / ".venv"), packaged)
            completed = subprocess.CompletedProcess([], 0, stdout='{"ok": true}', stderr="")
            with mock.patch.object(mcp.subprocess, "run", return_value=completed) as runner:
                checks = mcp.run_health_checks(install_route)
            self.assertTrue(all(check["ok"] for check in checks))
            for call in runner.call_args_list:
                self.assertEqual(call.args[0][0], str(python))
                self.assertEqual(call.kwargs["cwd"], linked)

    def test_claude_plugin_timeout_preserves_partial_output_as_json_text(self) -> None:
        expired = subprocess.TimeoutExpired("claude", 120, output=b"partial \xff", stderr=b"error \xfe")
        with mock.patch.object(claude_cli, "which", return_value="/fake/claude"):
            with mock.patch.object(claude_cli.subprocess, "run", side_effect=expired):
                report = claude_cli.validate_plugin(validation_root=ROOT, required=True)

        emitted = json.loads(json.dumps(report))
        self.assertFalse(emitted["ok"])
        self.assertTrue(emitted["available"])
        self.assertFalse(emitted["skipped"])
        self.assertEqual(emitted["stdout"], "partial \ufffd")
        self.assertEqual(emitted["stderr"], "error \ufffd")

    def test_marketplace_output_uses_generated_local_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            payload = json.loads(paths.marketplace_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["name"], "tokimon-local")
        self.assertEqual(payload["plugins"][0]["name"], "tokimon")
        self.assertEqual(payload["plugins"][0]["source"]["path"], "./plugins/tokimon")
        self.assertEqual(payload["plugins"][0]["policy"]["installation"], "AVAILABLE")

    def test_source_skill_resource_dependency_scanner_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "skills" / "demo"
            (skill_dir / "scripts").mkdir(parents=True)
            (skill_dir / "scripts" / "helper.py").write_text("", encoding="utf-8")
            (skill_dir / "skill.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\nRun `scripts/helper.py`.\n",
                encoding="utf-8",
            )

            problems = skills.scan_unhandled_resource_dependencies(source_root=root, skill_id="demo")

        self.assertTrue(problems)
        self.assertIn("scripts/", problems[0])

    def test_agents_openai_metadata_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skill_dir = root / "skills" / "demo"
            (skill_dir / "agents").mkdir(parents=True)
            (skill_dir / "agents" / "openai.yaml").write_text("interface: {}\n", encoding="utf-8")
            (skill_dir / "skill.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n",
                encoding="utf-8",
            )

            problems = skills.scan_unhandled_resource_dependencies(source_root=root, skill_id="demo")

        self.assertEqual(problems, [])

    def test_generate_normalizes_invalid_source_metadata_as_package_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output = Path(tmpdir) / "output"
            skill_dir = root / "skills" / "demo"
            (root / "tokimon").mkdir(parents=True)
            (root / "tokimon" / "__init__.py").write_text("", encoding="utf-8")
            (skill_dir / "agents").mkdir(parents=True)
            (skill_dir / "skill.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n\n# Demo\n",
                encoding="utf-8",
            )
            (skill_dir / "agents" / "openai.yaml").write_bytes(b"\xff")
            paths = generator.paths_for(source_root=root, output_root=output)
            settings_path = write_settings(
                Path(tmpdir) / "settings.local.json",
                source_root=root,
            )
            seed_scaffold(paths)

            with self.assertRaisesRegex(
                generator.PackageValidationError,
                "source skill inventory is invalid",
            ):
                generator.generate_package(paths, settings_path=settings_path)

    def test_generate_fails_before_scaffold_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))

            with self.assertRaises(generator.PackageScaffoldError):
                generator.generate_package(paths)

    def test_generate_requires_configured_settings_after_scaffold_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            seed_scaffold(paths)

            with self.assertRaisesRegex(configuration.ConfigurationError, "configure"):
                generator.generate_package(paths, settings_path=Path(tmpdir) / "missing-settings.json")

    def test_default_claude_output_root_is_local_runtime_generated_root(self) -> None:
        paths = generator.paths_for(source_root=ROOT, target_platform="claude")

        self.assertEqual(paths.platform.platform_id, "claude")
        self.assertEqual(
            paths.output_root,
            runtime.runtime_workspace_root(ROOT) / ".tokimon" / "ai-harness-interface" / "claude",
        )
        self.assertEqual(paths.package_root, paths.output_root / "plugins" / "tokimon")
        self.assertEqual(paths.marketplace_path, paths.output_root / ".claude-plugin" / "marketplace.json")
        self.assertEqual(paths.manifest_path, paths.package_root / ".claude-plugin" / "plugin.json")

    def test_generate_and_validate_closed_claude_package_catalog_without_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")

            generated = generator.generate_package(paths, settings_path=settings_path)
            validation = generator.validate_generated_package(paths)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )

        self.assertTrue(generated["ok"])
        self.assertEqual(generated["target_platform"], "claude")
        self.assertTrue(validation["ok"])
        self.assertEqual(files, expected_package_files(manifest_dir=".claude-plugin"))

    def test_claude_manifest_is_tokimon_owned_and_omits_codex_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            generator.generate_package(paths, settings_path=settings_path)
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["name"], "tokimon")
        self.assertEqual(manifest["displayName"], "Tokimon")
        self.assertEqual(manifest["version"], contract.PACKAGE_VERSION)
        self.assertEqual(manifest["author"], {"name": "Tokimon"})
        self.assertEqual(manifest["mcpServers"], "./.mcp.json")
        self.assertIn("Claude Code plugin interface", manifest["description"])
        for field in ("hooks", "commands", "agents", "skills", "interface", "apps"):
            self.assertNotIn(field, manifest)

    def test_claude_validation_rejects_codex_only_interface_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            generator.generate_package(paths, settings_path=settings_path)
            manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
            manifest["interface"] = {"displayName": "Tokimon"}
            paths.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(generator.PackageValidationError, "plugin-manifest"):
                generator.validate_generated_package(paths)

    def test_claude_marketplace_uses_claude_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            generator.generate_package(paths, settings_path=settings_path)
            payload = json.loads(paths.marketplace_path.read_text(encoding="utf-8"))

        self.assertEqual(paths.marketplace_path, paths.output_root / ".claude-plugin" / "marketplace.json")
        self.assertEqual(payload["name"], "tokimon-local")
        self.assertEqual(payload["owner"], {"name": "Tokimon"})
        self.assertTrue(payload["description"].strip())
        self.assertEqual(len(payload["plugins"]), 1)
        self.assertEqual(payload["plugins"][0]["name"], "tokimon")
        self.assertEqual(payload["plugins"][0]["source"], "./plugins/tokimon")
        self.assertNotIn("interface", payload)
        self.assertNotIn("policy", payload["plugins"][0])

    def test_claude_mcp_manifest_routes_through_pythonpath_env_without_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            install_route = generator.install_route_for(paths)
            generator.generate_package(paths, settings_path=settings_path)
            payload = json.loads((paths.package_root / ".mcp.json").read_text(encoding="utf-8"))

        servers = payload["mcpServers"]
        self.assertEqual(set(servers), {entry.server_id for entry in contract.MCP_ENTRIES})
        for entry in contract.MCP_ENTRIES:
            server = servers[entry.server_id]
            self.assertEqual(server["command"], str(install_route.liveCheckoutPath / ".venv" / "bin" / "python"))
            self.assertEqual(server["args"], ["-m", entry.module, *entry.serve_args])
            self.assertEqual(server["env"], {"PYTHONPATH": str(install_route.liveCheckoutPath)})
            self.assertNotIn("cwd", server)
            self.assertNotIn("/plugins/cache/", json.dumps(server))

    def test_validation_rejects_tampered_mcp_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            generator.generate_package(paths, settings_path=settings_path)
            mcp_path = paths.package_root / ".mcp.json"
            pristine = mcp_path.read_text(encoding="utf-8")
            tampered_cases = (
                ("unexpected-cwd", lambda server: server.__setitem__("cwd", "/elsewhere")),
                ("wrong-command", lambda server: server.__setitem__("command", "/usr/bin/python3")),
            )

            for case_name, tamper in tampered_cases:
                with self.subTest(case=case_name):
                    payload = json.loads(pristine)
                    tamper(payload["mcpServers"]["tokimon-remote-browser-broker"])
                    mcp_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(generator.PackageValidationError, "mcp-manifest"):
                        generator.validate_generated_package(paths)
            mcp_path.write_text(pristine, encoding="utf-8")

    def test_cachebuster_is_rejected_for_claude_target_platform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")

            with self.assertRaisesRegex(generator.PackageValidationError, "cachebuster"):
                generator.generate_package(paths, cachebuster="20260709", settings_path=settings_path)

    def test_claude_regeneration_removes_stale_owned_skill_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, without_local_claude_cli():
            paths = generator.paths_for(
                source_root=ROOT,
                output_root=Path(tmpdir),
                target_platform="claude",
            )
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            stale = paths.package_root / "skills" / "stale" / "SKILL.md"
            stale.parent.mkdir(parents=True)
            stale.write_text("---\nname: stale\ndescription: Stale.\n---\n", encoding="utf-8")

            generator.generate_package(paths, settings_path=settings_path)
            files = sorted(
                path.relative_to(paths.package_root).as_posix()
                for path in paths.package_root.rglob("*")
                if path.is_file()
            )

        self.assertNotIn("skills/stale/SKILL.md", files)
        self.assertEqual(files, expected_package_files(manifest_dir=".claude-plugin"))

    def test_claude_validation_skips_when_claude_cli_unavailable_and_not_required(self) -> None:
        result = None
        with without_local_claude_cli():
            result = claude_cli.validate_plugin(validation_root=ROOT)

        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])
        self.assertFalse(result["available"])

    def test_claude_validation_fails_closed_when_claude_cli_required_but_unavailable(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                claude_cli.EXECUTABLE_ENV: "claude-cli-not-installed-for-tests",
                claude_cli.REQUIRE_ENV: "1",
            },
        ):
            result = claude_cli.validate_plugin(validation_root=ROOT)

        self.assertFalse(result["ok"])
        self.assertTrue(result["skipped"])

    def test_cachebuster_delegates_to_plugin_creator_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_settings(Path(tmpdir) / "settings.local.json")
            seed_scaffold(paths)

            with mock.patch.object(
                generator.plugin_creator,
                "update_cachebuster",
                return_value={"name": "plugin-creator-cachebuster", "ok": True},
            ) as update_cachebuster:
                generated = generator.generate_package(
                    paths,
                    cachebuster="20260604",
                    settings_path=settings_path,
                )

        self.assertTrue(generated["ok"])
        update_cachebuster.assert_called_once_with(
            package_root=paths.package_root,
            cachebuster="20260604",
            required=None,
        )


if __name__ == "__main__":
    unittest.main()
