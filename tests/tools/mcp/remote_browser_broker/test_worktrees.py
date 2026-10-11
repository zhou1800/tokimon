from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokimon.tools.mcp import remote_browser_broker as broker
from tests.tools.mcp.remote_browser_broker.test_remote_browser_broker import (
    FakeController,
    FakeLauncher,
    FakeNoVNCProxy,
    owned_network_inspect_payload,
)


class BrowserWorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.primary = self.root / "primary"
        self.linked = self.root / "linked"
        self.linked.mkdir()
        metadata = self.primary / ".git" / "worktrees" / "linked"
        metadata.mkdir(parents=True)
        (self.linked / ".git").write_text(f"gitdir: {metadata}\n", encoding="utf-8")
        (metadata / "commondir").write_text("../..\n", encoding="utf-8")
        (metadata / "gitdir").write_text(f"{self.linked / '.git'}\n", encoding="utf-8")
        self.enterContext(patch.object(
            broker.subprocess, "run", side_effect=AssertionError("Unexpected external process")
        ))

    def config(self, checkout: Path, **env: str) -> broker.BrokerConfig:
        with patch.object(broker, "WORKSPACE_ROOT", checkout):
            return broker.BrokerConfig.from_env(env=env)

    def make_broker(
        self, config: broker.BrokerConfig, *, docker: bool = False
    ) -> broker.RemoteBrowserBroker:
        remote_broker = broker.RemoteBrowserBroker(
            config=config,
            launcher=broker.DockerSessionLauncher(config) if docker else FakeLauncher(),
            controller=FakeController(),
            novnc_proxy_factory=FakeNoVNCProxy,
        )
        self.addCleanup(remote_broker.release_shutdown_leases)
        self.addCleanup(remote_broker.begin_shutdown)
        return remote_broker

    def test_linked_checkout_keeps_primary_defaults_and_source_provenance(self) -> None:
        primary = self.config(self.primary)
        linked = self.config(self.linked)
        digest = hashlib.sha256(str(self.primary).encode("utf-8")).hexdigest()[:12]
        owner = broker.slug_component(broker.DEFAULT_OWNER_USER, fallback="user")
        previous_primary_id = f"{owner}-{digest}"[:64].rstrip("-._")

        self.assertEqual(primary.installation_id, previous_primary_id)
        self.assertEqual(primary.image_name, broker.default_image_name(previous_primary_id))
        self.assertEqual(primary.broker_port, broker.default_broker_port(previous_primary_id))
        self.assertEqual(linked.installation_id, primary.installation_id)
        self.assertEqual(linked.image_name, primary.image_name)
        self.assertEqual(linked.broker_port, primary.broker_port)
        self.assertEqual(linked.runtime_root, primary.runtime_root)
        self.assertEqual(linked.workspace_root, self.linked)
        self.assertEqual(primary.workspace_root, self.primary)

    def test_explicit_browser_configuration_survives_worktree_resolution(self) -> None:
        config = self.config(
            self.linked,
            TOKIMON_RB_INSTALLATION_ID="independent-browser",
            TOKIMON_RB_IMAGE="custom-browser:review",
            TOKIMON_RB_PORT="21800",
            TOKIMON_RB_RUNTIME_ROOT=str(self.root / "external-runtime"),
        )

        self.assertEqual(config.installation_id, "independent-browser")
        self.assertEqual(config.image_name, "custom-browser:review")
        self.assertEqual(config.broker_port, 21800)
        self.assertEqual(config.runtime_root, self.root / "external-runtime")
        self.assertEqual(config.workspace_root, self.linked)

    def test_distinct_repositories_users_and_installations_keep_resource_isolation(self) -> None:
        primary = self.config(self.primary)
        other_repo = self.config(self.root / "other-repository")
        with patch.object(broker, "DEFAULT_OWNER_USER", "another-browser-user"):
            other_user = self.config(self.primary)
        explicit = self.config(self.linked, TOKIMON_RB_INSTALLATION_ID="independent-browser")
        primary_launcher = broker.DockerSessionLauncher(primary)

        for other in (other_repo, other_user, explicit):
            with self.subTest(installation=other.installation_id):
                self.assertNotEqual(other.installation_id, primary.installation_id)
                self.assertNotEqual(other.image_name, primary.image_name)
                launcher = broker.DockerSessionLauncher(other)
                self.assertNotEqual(
                    launcher._network_name("session-a"), primary_launcher._network_name("session-a")
                )

    def test_linked_checkout_verifies_primary_container_and_network(self) -> None:
        for env in ({}, {"TOKIMON_RB_INSTALLATION_ID": "shared-explicit"}):
            with self.subTest(env=env):
                primary = self.config(self.primary, **env)
                linked = self.config(self.linked, **env)
                primary_launcher = broker.DockerSessionLauncher(primary)
                linked_launcher = broker.DockerSessionLauncher(linked)
                session = broker.BrowserSession(
                    id="session-a",
                    mcp_session_key="mcp-a",
                    profile_dir=primary.runtime_root / "sessions" / "session-a" / "profile",
                    cdp_port=31001,
                    novnc_port=41001,
                    novnc_backend_port=41002,
                    container_name=f"tokimon-rb-{primary.installation_id}-session-a",
                    container_id="a" * 64,
                    created_at="2026-01-01T00:00:00+00:00",
                    last_used_at="2026-01-01T00:00:00+00:00",
                )
                labels = primary_launcher._stable_ownership_labels(session.id)
                self.assertEqual(labels["tokimon.remote-browser-checkout"], str(self.primary))
                container = {
                    "Id": "a" * 64,
                    "Name": f"/{session.container_name}",
                    "Config": {"Labels": labels},
                    "Mounts": [{"Source": str(session.profile_dir), "Destination": "/profile"}],
                }
                with patch.object(broker.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, json.dumps([container]), ""
                )):
                    self.assertEqual(linked_launcher._owned_container_id(session), "a" * 64)

                previous_scope = json.dumps({
                    "checkout": str(self.primary),
                    "installation": primary.installation_id,
                    "owner": primary.owner_user,
                    "runtime_root": str(primary.runtime_root),
                    "session": session.id,
                }, sort_keys=True, separators=(",", ":"))
                previous_network = "tokimon-rbn-" + hashlib.sha256(
                    previous_scope.encode("utf-8")
                ).hexdigest()[:20]
                self.assertEqual(primary_launcher._network_name(session.id), previous_network)
                with patch.object(broker.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, owned_network_inspect_payload(primary_launcher, session.id), ""
                )) as inspect:
                    self.assertEqual(linked_launcher._inspect_owned_network(session.id), ("c" * 64, {}))
                self.assertEqual(inspect.call_args.args[0][-1], previous_network)

    def test_linked_checkout_recognizes_owned_orphan_container(self) -> None:
        primary = self.config(self.primary)
        linked_broker = self.make_broker(self.config(self.linked), docker=True)
        profile = primary.runtime_root / "sessions" / "session-a" / "profile"
        profile.mkdir(parents=True)
        container = {
            "id": "a" * 64,
            "name": f"tokimon-rb-{primary.installation_id}-session-a",
            "labels": broker.DockerSessionLauncher(primary)._stable_ownership_labels("session-a"),
            "mount_sources": [str(profile)],
        }
        with patch.object(broker, "docker_containers_with_label", return_value=([container], None)):
            linked_broker._refresh_orphaned_container_inventory()

        sessions = linked_broker.session_list()["sessions"]
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["state"], broker.STATE_ORPHANED)
        self.assertEqual(sessions[0]["id"], "session-a")
        self.assertTrue(profile.is_dir())

    def test_worktree_handoff_still_rejects_foreign_or_legacy_checkout_labels(self) -> None:
        config = self.config(self.linked)
        remote_broker = self.make_broker(config, docker=True)
        profile = config.runtime_root / "sessions" / "session-a" / "profile"
        profile.mkdir(parents=True)
        labels = broker.DockerSessionLauncher(self.config(self.primary))._stable_ownership_labels("session-a")

        for key, value in (
            ("tokimon.remote-browser-checkout", str(self.root / "foreign-repository")),
            ("tokimon.remote-browser-checkout", str(self.linked)),
            ("tokimon.remote-browser-owner-user", "foreign-user"),
            ("tokimon.remote-browser-installation", "foreign-installation"),
            ("tokimon.remote-browser-runtime-root", str(self.root / "foreign-runtime")),
        ):
            with self.subTest(label=key, value=value):
                foreign = {
                    "id": "a" * 64,
                    "name": f"tokimon-rb-{config.installation_id}-session-a",
                    "labels": {**labels, key: value},
                    "mount_sources": [str(profile)],
                }
                with patch.object(broker, "docker_containers_with_label", return_value=([foreign], None)):
                    with self.assertRaisesRegex(broker.BrowserBrokerError, "foreign or ambiguously labeled"):
                        remote_broker._refresh_orphaned_container_inventory()
                self.assertTrue(profile.is_dir())

    def test_linked_checkout_clones_primary_sealed_template(self) -> None:
        primary = self.make_broker(self.config(self.primary))
        primary.create_template("setup", template_name="shared")
        draft_profile = primary._session_or_error("setup").profile_dir
        (draft_profile / "fixture.txt").write_text("sealed profile", encoding="utf-8")
        sealed = primary.seal_template("setup", template_name="shared")
        sealed_profile = Path(sealed["template"]["profile_dir"])

        linked = self.make_broker(self.config(self.linked))
        linked.ensure("task", template_name="shared")
        cloned_profile = linked._session_or_error("task").profile_dir

        self.assertNotEqual(cloned_profile, sealed_profile)
        self.assertEqual((cloned_profile / "fixture.txt").read_text(encoding="utf-8"), "sealed profile")
        (cloned_profile / "fixture.txt").write_text("task changes", encoding="utf-8")
        self.assertEqual((sealed_profile / "fixture.txt").read_text(encoding="utf-8"), "sealed profile")
