from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokimon.ai_harness_interface import generator


def seed_ai_harness_scaffold(paths: generator.GenerationPaths) -> None:
    (paths.package_root / ".codex-plugin").mkdir(parents=True, exist_ok=True)
    (paths.package_root / "skills").mkdir(parents=True, exist_ok=True)
    paths.marketplace_path.parent.mkdir(parents=True, exist_ok=True)
    (paths.package_root / ".codex-plugin" / "plugin.json").write_text(
        (
            '{"name":"tokimon","version":"0.1.0","description":"Seeded plugin",'
            '"author":{"name":"Local developer"},"skills":"./skills/","mcpServers":"./.mcp.json",'
            '"interface":{"displayName":"Tokimon","shortDescription":"Use Tokimon in Codex.",'
            '"longDescription":"Tokimon adds a local Codex plugin scaffold.",'
            '"developerName":"Local developer","category":"Productivity","capabilities":[],'
            '"defaultPrompt":"Help me use Tokimon."}}\n'
        ),
        encoding="utf-8",
    )
    (paths.package_root / ".mcp.json").write_text('{"mcpServers": {}}\n', encoding="utf-8")
    paths.marketplace_path.write_text(
        '{"name":"tokimon-local","interface":{"displayName":"Tokimon Local"},"plugins":[]}\n',
        encoding="utf-8",
    )


def write_ai_harness_settings(path: Path) -> Path:
    path.write_text(
        (
            '{"agents":{"codex":{"model":"gpt-test","effort":"xhigh"},'
            '"claude":{"model":"claude-test","effort":"max"}},'
            '"ai_debate":{"access":"local-cli","peer_b":"claude"},'
            f'"pair":{{"access":"local-cli","peer_b":"claude","cwd":"{ROOT}"}}}}\n'
        ),
        encoding="utf-8",
    )
    return path


THIS_FILE = Path(__file__).resolve()
SEARCH_ROOTS = ("tokimon/memory", "tokimon/tools", "tokimon/workflow", "skills", "tests")
DISALLOWED_PATTERNS = (
    re.compile(r"\bBoo\b"),
    re.compile(r"\bboo\b"),
    re.compile(r"boo[_-]"),
    re.compile(r"\.boo\b"),
    re.compile(r"\bBOO_"),
    re.compile(r"\bAI_HARNESS_"),
    re.compile(r"\bai_harness[_-](?!interface\b)"),
    re.compile(r"\bai-harness[_-](?!interface\b)"),
)
FORBIDDEN_SKILL_C4_OPERATING_PATTERNS = (
    re.compile(r"docs/c4", re.IGNORECASE),
    re.compile(r"\bowning C4 docs\b", re.IGNORECASE),
    re.compile(
        r"\b(read|follow|load|open|use|see|consult|refer(?:ence)?|route)\b.*\bC4 docs\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bC4 docs\b.*\b(command discovery|exact command|arguments|json shape|safety details)",
        re.IGNORECASE,
    ),
    re.compile(r"\bC4 docs\b.*\b(normal activation|normal operation|normal use)", re.IGNORECASE),
)
ALLOWED_SKILL_C4_EVIDENCE_PATTERNS = (
    re.compile(r"\bwhen they are audit evidence\b", re.IGNORECASE),
    re.compile(r"\bC4 audits?\b.*docs/c4", re.IGNORECASE),
    re.compile(r"\bThis includes C4 docs\b", re.IGNORECASE),
    re.compile(r"\bTreat C4, code, and test mismatches as findings\b", re.IGNORECASE),
    re.compile(r"\bDo not force skills, prompts, or embedded payloads into C4 templates\b", re.IGNORECASE),
    re.compile(r"\bDo not load live C4 docs\b", re.IGNORECASE),
)


class TokimonPortExclusionTests(unittest.TestCase):
    def test_removed_chatgpt_conversation_operations_doc_has_no_callers(self) -> None:
        self.assertFalse((ROOT / "docs/operations/chatgpt-agent-conversation-operations.md").exists())

        hits: list[str] = []
        for root_name in ("AGENTS.md", "docs", "skills", "tokimon", "tests"):
            root_path = ROOT / root_name
            paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
            for path in paths:
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.resolve() == THIS_FILE:
                    continue
                text = path.read_text(encoding="utf-8")
                if (
                    "docs/operations/chatgpt-agent-conversation-operations.md" in text
                    or "ChatGPT Agent Conversation Operations" in text
                ):
                    hits.append(str(path.relative_to(ROOT)))

        self.assertEqual(hits, [])

    def test_ported_foundation_has_no_stale_boo_identifiers(self) -> None:
        hits: list[str] = []
        for root_name in SEARCH_ROOTS:
            root_path = ROOT / root_name
            self.assertTrue(root_path.is_dir(), f"scan root does not exist: {root_name}")
            for path in sorted(root_path.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                if path.resolve() == THIS_FILE:
                    continue
                text = path.read_text(encoding="utf-8")
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if any(pattern.search(line) for pattern in DISALLOWED_PATTERNS):
                        hits.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")

        self.assertEqual(hits, [])

    def test_ported_skills_do_not_route_to_excluded_agent_interface_surface(self) -> None:
        hits: list[str] = []
        for path in sorted((ROOT / "skills").rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if "agent-interface" in line or "agent interface" in line.lower():
                    hits.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")

        self.assertEqual(hits, [])

    def test_skill_entrypoints_do_not_use_c4_as_operating_manuals(self) -> None:
        hits: list[str] = []
        skill_roots = [ROOT / "skills"]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generator.paths_for(source_root=ROOT, output_root=Path(tmpdir))
            settings_path = write_ai_harness_settings(Path(tmpdir) / "settings.local.json")
            seed_ai_harness_scaffold(paths)
            generator.generate_package(paths, settings_path=settings_path)
            skill_roots.append(paths.package_root / "skills")

            for root in skill_roots:
                for path in sorted(root.rglob("*.md")):
                    if path.name not in {"skill.md", "SKILL.md"}:
                        continue
                    text = path.read_text(encoding="utf-8")
                    for line_number, line in enumerate(text.splitlines(), start=1):
                        if any(pattern.search(line) for pattern in ALLOWED_SKILL_C4_EVIDENCE_PATTERNS):
                            continue
                        if any(pattern.search(line) for pattern in FORBIDDEN_SKILL_C4_OPERATING_PATTERNS):
                            display_path = (
                                path.relative_to(ROOT)
                                if path.is_relative_to(ROOT)
                                else path.relative_to(paths.package_root)
                            )
                            hits.append(f"{display_path}:{line_number}:{line.strip()}")

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
