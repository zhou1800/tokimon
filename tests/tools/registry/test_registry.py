from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokimon.tools import catalog, registry


ROOT = Path(__file__).resolve().parents[3]


ACTIVE_CAPABILITY_IDS = (
    "mcp.telegram-bot",
    "mcp.ai-debate",
    "mcp.remote-browser-broker",
    "mcp.remote-agent-broker",
    "mcp.container-executor",
)


def connector_inventory(capability_id: str, connector_id: str) -> str:
    return json.dumps(
        {
            "connectors": [
                {
                    "id": connector_id,
                    "capability_id": capability_id,
                    "source": "codex",
                    "status": "available",
                    "transport": "codex-mcp",
                    "tools": ["health", "run"],
                }
            ]
        }
    )


def no_connectors(**extra: str) -> dict[str, str]:
    return {"TOKIMON_MCP_CONNECTORS_JSON": "[]", **extra}


def seed_implementation(root: Path, capability: str) -> None:
    modules = {
        "telegram-bot": "telegram_bot",
        "ai-debate": "ai_debate",
        "remote-browser-broker": "remote_browser_broker",
        "remote-agent-broker": "remote_agent_broker",
        "container-executor": "container_executor",
    }
    implementation = root / "tokimon" / "tools" / "mcp" / modules[capability] / "__main__.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# implementation\n", encoding="utf-8")


def seed_ai_debate_settings(root: Path, *, peer_b: str = "claude") -> None:
    path = root / ".tokimon" / "settings.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "agents": {
                    "codex": {"model": "codex-model", "effort": "high"},
                    "claude": {"model": "claude-model", "effort": "max"},
                },
                "ai_debate": {"access": "local-cli", "peer_b": peer_b},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def seed_executable(root: Path, command: str) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    executable = bin_dir / command
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    return bin_dir


class ToolRegistryTests(unittest.TestCase):
    def test_container_executor_readiness_ignores_browser_docker_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed_implementation(root, "container-executor")
            binary_path = seed_executable(root, "docker")
            payload = registry.capability_map(workspace_root=root, env=no_connectors(PATH=str(binary_path), TOKIMON_RB_DOCKER="nonexistent-browser-docker"))
            executor = next(c for c in payload["capabilities"] if c["id"] == "mcp.container-executor")
            self.assertEqual(executor["status"], "available")

    def test_detect_host_profile_reports_gui_and_connectors(self) -> None:
        inventory = connector_inventory("mcp.remote-browser-broker", "browser")
        host = registry.detect_host_profile(
            platform_name="Darwin",
            host_id="mac-test",
            env={"TOKIMON_MCP_CONNECTORS_JSON": inventory},
        )

        self.assertTrue(host.has_gui)
        self.assertEqual(host.reachable_mcp_endpoints, ("browser",))

    def test_detect_host_profile_marks_headless_linux(self) -> None:
        host = registry.detect_host_profile(
            platform_name="Linux",
            host_id="linux-test",
            env=no_connectors(),
        )

        self.assertFalse(host.has_gui)
        self.assertNotIn("gui", host.local_tool_families)

    def test_load_mcp_connectors_reads_runtime_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / ".tokimon" / "tools" / "mcp" / "connectors.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                connector_inventory("mcp.telegram-bot", "telegram"),
                encoding="utf-8",
            )
            connectors = registry.load_mcp_connectors(workspace_root=root, env={})

        self.assertEqual(len(connectors), 1)
        self.assertEqual(connectors[0].capability_id, "mcp.telegram-bot")
        self.assertTrue(connectors[0].available)

    def test_load_mcp_connectors_wraps_inventory_read_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / ".tokimon" / "tools" / "mcp" / "connectors.json"
            path.mkdir(parents=True)
            with self.assertRaises(registry.MCPConnectorInventoryError) as caught:
                registry.load_mcp_connectors(workspace_root=root, env={})

        self.assertIn("Unable to read MCP connector inventory", str(caught.exception))
        self.assertIn(str(path), str(caught.exception))

    def test_load_mcp_connectors_wraps_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / ".tokimon" / "tools" / "mcp" / "connectors.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"\xff")
            with self.assertRaises(registry.MCPConnectorInventoryError) as caught:
                registry.load_mcp_connectors(workspace_root=root, env={})

        self.assertIn("Unable to read MCP connector inventory", str(caught.exception))
        self.assertIn(str(path), str(caught.exception))

    def test_load_mcp_connectors_ignores_unsupported_path_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            external = Path(tmpdir) / "external.json"
            external.write_text(
                connector_inventory("mcp.telegram-bot", "telegram"),
                encoding="utf-8",
            )
            connectors = registry.load_mcp_connectors(
                workspace_root=root,
                env={"TOKIMON_MCP_CONNECTORS_PATH": str(external)},
            )

        self.assertEqual(connectors, ())

    def test_load_mcp_connectors_rejects_invalid_json_with_source_diagnostic(self) -> None:
        with self.assertRaisesRegex(
            registry.MCPConnectorInventoryError,
            r"^Malformed MCP connector inventory in TOKIMON_MCP_CONNECTORS_JSON: "
            r"Expecting property name enclosed in double quotes\.$",
        ):
            registry.load_mcp_connectors(
                env={"TOKIMON_MCP_CONNECTORS_JSON": "{"},
            )

    def test_load_mcp_connectors_rejects_malformed_entry_fields(self) -> None:
        valid = {
            "id": "connector",
            "capability_id": "mcp.telegram-bot",
            "source": "codex",
            "status": "available",
            "transport": "codex-mcp",
            "tools": ["health"],
        }
        cases = (
            (
                {**valid, "id": 7},
                "field 'id' must be a nonempty string.",
            ),
            (
                {key: value for key, value in valid.items() if key != "capability_id"},
                "field 'capability_id' must be a nonempty string.",
            ),
            (
                {key: value for key, value in valid.items() if key != "source"},
                "field 'source' must be a nonempty string.",
            ),
            (
                {**valid, "status": "availble"},
                "field 'status' must be one of: available, unavailable.",
            ),
            (
                {**valid, "transport": []},
                "field 'transport' must be a nonempty string.",
            ),
            (
                {**valid, "tools": "health"},
                "field 'tools' must be a list of nonempty strings.",
            ),
            (
                {**valid, "tools": [""]},
                "field 'tools[1]' must be a nonempty string.",
            ),
        )
        for entry, suffix in cases:
            with self.subTest(suffix=suffix):
                raw = json.dumps({"connectors": [entry]})
                expected = (
                    "Malformed MCP connector inventory in TOKIMON_MCP_CONNECTORS_JSON: "
                    f"connector 1 {suffix}"
                )
                with self.assertRaisesRegex(
                    registry.MCPConnectorInventoryError,
                    f"^{re.escape(expected)}$",
                ):
                    registry.load_mcp_connectors(
                        env={"TOKIMON_MCP_CONNECTORS_JSON": raw},
                    )

    def test_telegram_local_availability_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "telegram-bot")
            missing = registry.find_capability(
                "mcp.telegram-bot",
                workspace_root=root,
                env=no_connectors(PATH=""),
            )
            available = registry.find_capability(
                "mcp.telegram-bot",
                workspace_root=root,
                env=no_connectors(PATH="", TELEGRAM_BOT_TOKEN="token"),
            )

        self.assertIsNotNone(missing)
        self.assertIsNotNone(available)
        assert missing is not None
        assert available is not None
        self.assertEqual(missing.status, "unavailable")
        self.assertEqual(available.status, "available")

    def test_telegram_token_can_come_from_local_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "telegram-bot")
            secrets = root / ".tokimon" / "secrets.local.json"
            secrets.parent.mkdir(parents=True)
            secrets.write_text('{"telegram":{"bot_token":"token"}}', encoding="utf-8")

            capability = registry.find_capability(
                "mcp.telegram-bot",
                workspace_root=root,
                env=no_connectors(PATH=""),
            )

        self.assertIsNotNone(capability)
        assert capability is not None
        self.assertEqual(capability.status, "available")

    def test_remote_browser_local_availability_requires_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "remote-browser-broker")
            missing = registry.find_capability(
                "mcp.remote-browser-broker",
                workspace_root=root,
                env=no_connectors(PATH=""),
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text("#!/bin/sh\n", encoding="utf-8")
            docker.chmod(0o755)
            available = registry.find_capability(
                "mcp.remote-browser-broker",
                workspace_root=root,
                env=no_connectors(PATH=str(bin_dir)),
            )

        self.assertIsNotNone(missing)
        self.assertIsNotNone(available)
        assert missing is not None
        assert available is not None
        self.assertEqual(missing.status, "unavailable")
        self.assertEqual(available.status, "available")

    def test_remote_browser_readiness_uses_runtime_docker_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "remote-browser-broker")
            bin_dir = seed_executable(root, "docker")
            seed_executable(root, "custom-docker")
            custom_docker = str(bin_dir / "custom-docker")
            cases = (
                (
                    no_connectors(PATH="", TOKIMON_RB_DOCKER=custom_docker),
                    "available",
                ),
                (
                    no_connectors(PATH=str(bin_dir), TOKIMON_RB_DOCKER="custom-docker"),
                    "available",
                ),
                (
                    no_connectors(
                        PATH=str(bin_dir), TOKIMON_RB_DOCKER=str(root / "missing")
                    ),
                    "unavailable",
                ),
                (
                    no_connectors(PATH=str(bin_dir), TOKIMON_RB_DOCKER=""),
                    "unavailable",
                ),
                (
                    no_connectors(PATH="", DOCKER_CLI=custom_docker),
                    "unavailable",
                ),
            )
            for env, expected_status in cases:
                with self.subTest(env=env):
                    capability = registry.find_capability(
                        "mcp.remote-browser-broker", workspace_root=root, env=env
                    )
                    self.assertIsNotNone(capability)
                    assert capability is not None
                    self.assertEqual(capability.status, expected_status)

    def test_ai_debate_local_availability_requires_settings_and_selected_clis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "ai-debate")
            bin_dir = seed_executable(root, "codex")
            missing_settings = registry.find_capability(
                "mcp.ai-debate",
                workspace_root=root,
                env=no_connectors(PATH=str(bin_dir)),
            )

            seed_ai_debate_settings(root, peer_b="codex")
            codex_only = registry.find_capability(
                "mcp.ai-debate",
                workspace_root=root,
                env=no_connectors(PATH=str(bin_dir)),
            )

            seed_ai_debate_settings(root, peer_b="claude")
            missing_claude = registry.find_capability(
                "mcp.ai-debate",
                workspace_root=root,
                env=no_connectors(PATH=str(bin_dir)),
            )
            seed_executable(root, "claude")
            with_claude = registry.find_capability(
                "mcp.ai-debate",
                workspace_root=root,
                env=no_connectors(PATH=str(bin_dir)),
            )

        for capability in (missing_settings, codex_only, missing_claude, with_claude):
            self.assertIsNotNone(capability)
        assert missing_settings is not None
        assert codex_only is not None
        assert missing_claude is not None
        assert with_claude is not None
        self.assertEqual(missing_settings.status, "unavailable")
        self.assertEqual(codex_only.status, "available")
        self.assertEqual(missing_claude.status, "unavailable")
        self.assertEqual(with_claude.status, "available")

    def test_ai_debate_local_availability_rejects_invalid_settings(self) -> None:
        cases = (
            "{",
            json.dumps({"ai_debate": [], "agents": {}}),
            json.dumps(
                {
                    "ai_debate": {"access": "remote", "peer_b": "codex"},
                    "agents": {
                        "codex": {"model": "codex-model", "effort": "high"},
                    },
                }
            ),
            json.dumps(
                {
                    "ai_debate": {"access": "local-cli", "peer_b": "unsupported"},
                    "agents": {
                        "codex": {"model": "codex-model", "effort": "high"},
                        "claude": {"model": "", "effort": "max"},
                    },
                }
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "ai-debate")
            bin_dir = seed_executable(root, "codex")
            seed_executable(root, "claude")
            settings_path = root / ".tokimon" / "settings.local.json"
            settings_path.parent.mkdir(parents=True, exist_ok=True)

            for payload in cases:
                with self.subTest(payload=payload):
                    settings_path.write_text(payload + "\n", encoding="utf-8")
                    capability = registry.find_capability(
                        "mcp.ai-debate",
                        workspace_root=root,
                        env=no_connectors(PATH=str(bin_dir)),
                    )

                    self.assertIsNotNone(capability)
                    assert capability is not None
                    self.assertEqual(capability.status, "unavailable")

    def test_remote_agent_local_availability_does_not_require_provider_clis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "remote-agent-broker")
            capability = registry.find_capability(
                "mcp.remote-agent-broker",
                workspace_root=root,
                env=no_connectors(PATH=""),
            )

        self.assertIsNotNone(capability)
        assert capability is not None
        self.assertEqual(capability.status, "available")
        self.assertNotIn("Codex", capability.description)

    def test_remote_agent_local_availability_requires_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            capability = registry.find_capability(
                "mcp.remote-agent-broker",
                workspace_root=Path(tmpdir),
                env=no_connectors(PATH=""),
            )

        self.assertIsNotNone(capability)
        assert capability is not None
        self.assertEqual(capability.status, "unavailable")

    def test_remote_agent_local_availability_does_not_require_contract_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "remote-agent-broker")
            entry = next(
                item
                for item in catalog.MCP_CATALOG
                if item.capability_id == "mcp.remote-agent-broker"
            )
            self.assertFalse((root / entry.docs).exists())

            capability = registry.find_capability(
                entry.capability_id,
                workspace_root=root,
                env=no_connectors(PATH=""),
            )

        self.assertIsNotNone(capability)
        assert capability is not None
        self.assertEqual(capability.status, "available")

    def test_capabilities_can_be_connector_backed(self) -> None:
        cases = (
            ("mcp.telegram-bot", "telegram"),
            ("mcp.ai-debate", "debate"),
            ("mcp.remote-browser-broker", "browser"),
            ("mcp.remote-agent-broker", "agent"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for capability_id, connector_id in cases:
                with self.subTest(capability_id=capability_id):
                    entry = next(
                        entry
                        for entry in catalog.MCP_CATALOG
                        if entry.capability_id == capability_id
                    )
                    capability = registry.find_capability(
                        capability_id,
                        workspace_root=root,
                        env={
                            "PATH": "",
                            "TOKIMON_MCP_CONNECTORS_JSON": connector_inventory(
                                capability_id,
                                connector_id,
                            ),
                        },
                    )
                    self.assertIsNotNone(capability)
                    assert capability is not None
                    self.assertEqual(capability.status, "available")
                    self.assertEqual(capability.implementation, f"codex-mcp:{connector_id}")
                    self.assertEqual(capability.name, entry.name)
                    self.assertEqual(capability.family, entry.kind)
                    self.assertEqual(capability.description, entry.connector.description)
                    self.assertEqual(capability.docs, entry.docs)
                    self.assertEqual(capability.artifact_policy, entry.connector.artifact_policy)
                    self.assertEqual(capability.failure_modes, entry.connector.failure_modes)

    def test_require_capability_returns_available_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "telegram-bot")
            env = no_connectors(PATH="", TELEGRAM_BOT_TOKEN="token")
            host = registry.detect_host_profile(
                platform_name="Darwin",
                host_id="mac-test",
                workspace_root=root,
                env=env,
            )

            capability = registry.require_capability(
                "mcp.telegram-bot",
                host,
                workspace_root=root,
                env=env,
            )

        self.assertEqual(capability.id, "mcp.telegram-bot")
        self.assertEqual(capability.status, "available")

    def test_require_capability_rejects_unavailable_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            seed_implementation(root, "telegram-bot")

            with self.assertRaises(registry.ToolCapabilityRequirementError) as raised:
                registry.require_capability(
                    "mcp.telegram-bot",
                    workspace_root=root,
                    env=no_connectors(PATH=""),
                )

        self.assertNotIsInstance(raised.exception, registry.UnknownToolCapabilityError)
        self.assertEqual(raised.exception.exit_code, 3)
        self.assertIsNotNone(raised.exception.capability)
        assert raised.exception.capability is not None
        self.assertEqual(raised.exception.capability.status, "unavailable")
        self.assertIn("mcp.telegram-bot [unavailable]", str(raised.exception))
        self.assertIn("Docs:", str(raised.exception))
        self.assertIn("Health check:", str(raised.exception))
        self.assertIn("Host requirements:", str(raised.exception))

    def test_require_capability_rejects_unknown_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(registry.UnknownToolCapabilityError) as raised:
                registry.require_capability(
                    "missing.tool",
                    workspace_root=Path(tmpdir),
                    env=no_connectors(PATH=""),
                )

        self.assertEqual(raised.exception.exit_code, 2)
        self.assertIsNone(raised.exception.capability)
        self.assertEqual(str(raised.exception), "Unknown tool capability: missing.tool")

    def test_shared_catalog_owns_complete_registry_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = registry.capability_map(
                workspace_root=Path(tmpdir),
                env=no_connectors(PATH=""),
            )

        expected_routes = (
            (
                "mcp.telegram-bot",
                "tokimon-telegram-bot",
                "tokimon.tools.mcp.telegram_bot",
                ("serve",),
                ("health", "--json"),
                True,
            ),
            (
                "mcp.ai-debate",
                "tokimon-ai-debate",
                "tokimon.tools.mcp.ai_debate",
                ("serve",),
                ("health", "--json"),
                True,
            ),
            (
                "mcp.remote-browser-broker",
                "tokimon-remote-browser-broker",
                "tokimon.tools.mcp.remote_browser_broker",
                ("serve",),
                ("health", "--json"),
                True,
            ),
            (
                "mcp.remote-agent-broker",
                "tokimon-remote-agent-broker",
                "tokimon.tools.mcp.remote_agent_broker",
                ("serve",),
                ("health", "--json"),
                False,
            ),
            (
                "mcp.container-executor",
                "tokimon-container-executor",
                "tokimon.tools.mcp.container_executor",
                ("serve", "--name", "default"),
                ("health", "--json"),
                False,
            ),
        )
        self.assertEqual(
            tuple(
                (
                    entry.capability_id,
                    entry.server_id,
                    entry.module,
                    entry.serve_args,
                    entry.health_args,
                    entry.package_included,
                )
                for entry in catalog.MCP_CATALOG
            ),
            expected_routes,
        )
        debate_entry = next(
            entry for entry in catalog.MCP_CATALOG if entry.capability_id == "mcp.ai-debate"
        )
        self.assertIn("by default", debate_entry.artifact_policy)
        self.assertIn("caller-selected output root", debate_entry.artifact_policy)
        self.assertIn("caller-selected output root", debate_entry.connector.artifact_policy)

        capabilities = payload["capabilities"]
        self.assertIsInstance(capabilities, list)
        assert isinstance(capabilities, list)
        self.assertEqual(
            tuple(capability["id"] for capability in capabilities),
            ACTIVE_CAPABILITY_IDS,
        )
        for entry, capability in zip(catalog.MCP_CATALOG, capabilities, strict=True):
            with self.subTest(capability_id=entry.capability_id):
                self.assertEqual(capability["name"], entry.name)
                self.assertEqual(capability["family"], entry.kind)
                self.assertEqual(capability["description"], entry.description)
                self.assertEqual(capability["implementation"], entry.implementation)
                self.assertEqual(capability["docs"], entry.docs)
                self.assertEqual(capability["commands"], list(entry.commands))
                self.assertEqual(
                    capability["host_requirements"],
                    list(entry.host_requirements),
                )
                self.assertEqual(capability["health_check"], entry.health_check)
                self.assertEqual(capability["artifact_policy"], entry.artifact_policy)
                self.assertEqual(capability["failure_modes"], list(entry.failure_modes))

    def test_catalog_c4_references_exist(self) -> None:
        missing = [
            f"{entry.capability_id}: {reference}"
            for entry in catalog.MCP_CATALOG
            for reference in entry.c4_references
            if not (ROOT / reference).is_file()
        ]

        self.assertEqual(missing, [])

    def test_registry_public_output_has_exact_active_capability_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            capabilities = registry.list_capabilities(
                workspace_root=root,
                env=no_connectors(PATH=""),
            )

        self.assertEqual(
            tuple(capability.id for capability in capabilities),
            ACTIVE_CAPABILITY_IDS,
        )

    def test_cli_json_reports_malformed_connector_inventory(self) -> None:
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"TOKIMON_MCP_CONNECTORS_JSON": "{"}, clear=True):
            with mock.patch("sys.stdout", stdout):
                exit_code = registry.main(["list", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, registry.MCPConnectorInventoryError.exit_code)
        self.assertFalse(payload["ok"])
        self.assertEqual(
            payload["error"],
            "Malformed MCP connector inventory in TOKIMON_MCP_CONNECTORS_JSON: "
            "Expecting property name enclosed in double quotes.",
        )

    def test_cli_json_reports_unreadable_connector_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / ".tokimon" / "tools" / "mcp" / "connectors.json"
            path.mkdir(parents=True)
            stdout = io.StringIO()
            with mock.patch.object(registry, "WORKSPACE_ROOT", root):
                with mock.patch.dict(os.environ, {}, clear=True):
                    with mock.patch("sys.stdout", stdout):
                        exit_code = registry.main(["list", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, registry.MCPConnectorInventoryError.exit_code)
        self.assertFalse(payload["ok"])
        self.assertIn("Unable to read MCP connector inventory", payload["error"])
        self.assertIn(str(path), payload["error"])

    def test_cli_show_unknown_capability_returns_json_error(self) -> None:
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {"TOKIMON_MCP_CONNECTORS_JSON": "[]"}, clear=True):
            with mock.patch("sys.stdout", stdout):
                exit_code = registry.main(["show", "missing.tool", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["capabilityId"], "missing.tool")

    def test_cli_require_available_capability_returns_json_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            inventory = connector_inventory("mcp.telegram-bot", "telegram")
            with mock.patch.object(registry, "WORKSPACE_ROOT", root):
                with mock.patch.dict(
                    os.environ,
                    {"TOKIMON_MCP_CONNECTORS_JSON": inventory},
                    clear=True,
                ):
                    with mock.patch("sys.stdout", stdout):
                        exit_code = registry.main(
                            ["require", "mcp.telegram-bot", "--json"]
                        )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["capability"]["id"], "mcp.telegram-bot")
        self.assertEqual(payload["capability"]["status"], "available")

    def test_cli_require_unavailable_capability_returns_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with mock.patch.object(registry, "WORKSPACE_ROOT", root):
                with mock.patch.dict(
                    os.environ,
                    no_connectors(PATH=""),
                    clear=True,
                ):
                    with mock.patch("sys.stdout", stdout):
                        exit_code = registry.main(
                            ["require", "mcp.telegram-bot", "--json"]
                        )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 3)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["capabilityId"], "mcp.telegram-bot")
        self.assertEqual(payload["capability"]["status"], "unavailable")
        self.assertIn("Health check:", payload["error"])

    def test_cli_require_unknown_capability_returns_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stdout = io.StringIO()
            with mock.patch.object(registry, "WORKSPACE_ROOT", root):
                with mock.patch.dict(
                    os.environ,
                    no_connectors(PATH=""),
                    clear=True,
                ):
                    with mock.patch("sys.stdout", stdout):
                        exit_code = registry.main(["require", "missing.tool", "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["capabilityId"], "missing.tool")
        self.assertNotIn("capability", payload)

    def test_cli_parser_exposes_inventory_and_requirement_commands(self) -> None:
        parser = registry.build_parser()

        self.assertEqual(parser.parse_args(["host", "--json"]).command, "host")
        self.assertEqual(parser.parse_args(["list", "--json"]).command, "list")
        show = parser.parse_args(["show", "mcp.telegram-bot", "--json"])
        self.assertEqual(show.command, "show")
        require = parser.parse_args(["require", "mcp.telegram-bot", "--json"])
        self.assertEqual(require.command, "require")
        self.assertEqual(require.capability_id, "mcp.telegram-bot")
        self.assertTrue(require.json)


if __name__ == "__main__":
    unittest.main()
