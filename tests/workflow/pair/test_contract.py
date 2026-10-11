from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.pair import cli, contract


L2_DOC = ROOT / "docs" / "c4" / "level-2-containers" / "workflow" / "pair.md"
L3_DOC = ROOT / "docs" / "c4" / "level-3-components" / "workflow" / "pair-cli.md"
SKILL_DOC = ROOT / "skills" / "pair" / "skill.md"


class PairWorkflowContractTests(unittest.TestCase):
    def test_contract_declarations_match_pair_package(self) -> None:
        self.assertEqual(contract.PRODUCT_ID, "pair")
        self.assertEqual(contract.SKILL_ID, "pair")
        self.assertEqual(contract.WORKFLOW_ID, "pair")
        self.assertEqual(contract.PYTHON_PACKAGE, "tokimon.workflow.pair")
        self.assertEqual(contract.WORKFLOW_STATUS, "support")
        self.assertEqual(contract.CANONICAL_ENTRYPOINT, "tokimon.workflow.pair.cli:main")
        self.assertEqual(contract.OPERATOR_ENTRYPOINT, ".venv/bin/python -m tokimon.workflow.pair")
        self.assertEqual(contract.LOCAL_STATE_ROOT, Path(".tokimon") / "workflow" / "pair")
        self.assertEqual(contract.RUN_ROOT, Path(".tokimon") / "workflow" / "pair" / "runs")
        self.assertEqual(contract.SETTINGS_SECTION, "pair")
        self.assertEqual(
            contract.CODEX_CLAUDE_PEER_CAPABILITIES,
            ("native-subagent", "local-cli"),
        )
        self.assertEqual(contract.NATIVE_SUBAGENT_CAPABILITY, "native-subagent")
        self.assertEqual(contract.LOCAL_CLI_CAPABILITY, "local-cli")
        self.assertEqual(contract.CODEX_CLAUDE_PEER_AGENTS, ("codex", "claude"))
        self.assertEqual(contract.PEER_A_AGENT, "codex")
        self.assertEqual(contract.PEER_B_AGENTS, ("claude", "codex"))
        self.assertEqual(contract.REQUIRED_TOOL_CAPABILITIES, ())
        self.assertEqual(contract.COMMANDS, ("contract", "init", "select-peers", "status", "validate"))
        self.assertIn("paused_external_clarification", contract.VALID_STATUSES)
        self.assertEqual(contract.TERMINAL_OUTCOMES, ("consensus", "no_consensus"))
        self.assertIn("external_clarification", contract.ROUND_SIGNALS)
        self.assertEqual(contract.ROUND_NOTE_FILES, ("peer-a.md", "peer-b.md"))
        self.assertEqual(contract.CONSENSUS_ARTIFACT, "consensus.md")
        self.assertEqual(contract.NO_CONSENSUS_ARTIFACT, "no_consensus.md")
        self.assertIn("peer_a_evidence", contract.NO_CONSENSUS_FIELDS)
        self.assertIn("continuation_options", contract.NO_CONSENSUS_FIELDS)
        self.assertEqual(
            contract.REQUIRED_CONSENSUS_STATUS_FIELDS,
            (
                "result",
                "rounds_completed",
                "final_text_artifact",
                "final_text_sha256",
                "independent_notes",
                "exchange",
                "peer_acceptances",
            ),
        )
        self.assertIn("peer_a_artifact", contract.REQUIRED_CONSENSUS_INDEPENDENT_NOTE_FIELDS)
        self.assertIn("peer_a_response_artifact", contract.REQUIRED_CONSENSUS_EXCHANGE_FIELDS)
        self.assertIn("final_text_sha256", contract.REQUIRED_CONSENSUS_PEER_ACCEPTANCE_FIELDS)
        self.assertEqual(
            contract.REQUIRED_MUTABLE_WORK_FIELDS,
            (
                "writer_peer",
                "reviewer_peer",
                "write_round",
                "review_round",
                "writer_evidence_artifact",
                "reviewer_evidence_artifact",
                "reviewer_approved",
            ),
        )
        self.assertEqual(contract.WORKFLOW_READINESS["id"], "pair")
        self.assertEqual(contract.WORKFLOW_READINESS["entrypoint"], contract.OPERATOR_ENTRYPOINT)

    def test_runtime_defaults_use_contract_declarations(self) -> None:
        self.assertEqual(cli.required_tool_capabilities(), contract.REQUIRED_TOOL_CAPABILITIES)
        payload = cli._contract_payload()
        self.assertEqual(payload["workflow_id"], contract.WORKFLOW_ID)
        self.assertEqual(payload["workflow_readiness"], contract.WORKFLOW_READINESS)
        self.assertEqual(payload["required_tool_capabilities"], [])
        self.assertEqual(payload["commands"], list(contract.COMMANDS))
        self.assertEqual(payload["settings_path"], ".tokimon/settings.local.json")
        self.assertEqual(payload["settings_section"], contract.SETTINGS_SECTION)
        self.assertEqual(payload["codex_claude_peer_capabilities"], list(contract.CODEX_CLAUDE_PEER_CAPABILITIES))
        self.assertEqual(payload["codex_claude_peer_agents"], list(contract.CODEX_CLAUDE_PEER_AGENTS))
        self.assertEqual(payload["peer_a_agent"], contract.PEER_A_AGENT)
        self.assertEqual(payload["peer_b_agents"], list(contract.PEER_B_AGENTS))
        self.assertEqual(payload["round_note_files"], list(contract.ROUND_NOTE_FILES))
        self.assertEqual(payload["no_consensus_fields"], list(contract.NO_CONSENSUS_FIELDS))
        self.assertEqual(payload["consensus_status_fields"], list(contract.REQUIRED_CONSENSUS_STATUS_FIELDS))
        self.assertEqual(payload["mutable_work_fields"], list(contract.REQUIRED_MUTABLE_WORK_FIELDS))

    def test_docs_name_required_workflow_contract_surfaces(self) -> None:
        l2 = L2_DOC.read_text(encoding="utf-8")
        l3 = L3_DOC.read_text(encoding="utf-8")
        skill = SKILL_DOC.read_text(encoding="utf-8")
        combined = f"{l2}\n{l3}\n{skill}"

        for required_text in (
            "## Workflow Protocol",
            "## Workflow Package Declarations",
            "Workflow id:** `pair`",
            "Python package:** `tokimon/workflow/pair/`",
            "tokimon.workflow.pair.cli:main",
            ".venv/bin/python -m tokimon.workflow.pair",
            "tokimon/workflow/pair/contract.py",
            "WORKFLOW_READINESS",
            "Required tool capability:** none",
            "scope=workflow",
            "room=pair",
            ".tokimon/workflow/pair/runs/",
            ".tokimon/settings.local.json",
            "pair.access",
            "pair.peer_b",
            "pair.cwd",
            "native-subagent",
            "local-cli",
            "contract",
            "init",
            "select-peers",
            "status",
            "validate",
            "consensus",
            "no_consensus",
            "Auth Boundary",
            "tests/workflow/pair/",
            "The Mediator does not:",
            "draft, merge, rank, filter, or rewrite final findings or final text",
            "After each round, send Peer A's note to Peer B and Peer B's note to Peer A.",
            "That final text must be authored by a Peer or explicitly supplied by the human",
            "Validation is structural only.",
            "unresolved_issue",
            "peer_a_evidence",
            "continuation_options",
            "round_note_files",
            "artifact_contract",
            "consensus_status",
            "independent_notes",
            "peer_acceptances",
            "mutable_work",
            "final_text_sha256",
            "reviewer_approved",
        ):
            self.assertIn(required_text, combined)

        self.assertIn("terminal evidence required by `contract --json`", skill)
        self.assertIn("distinct writer/reviewer evidence", skill)


if __name__ == "__main__":
    unittest.main()
