from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C4_ROOT = ROOT / "docs" / "c4"
REPOSITORY_GUIDELINES = ROOT / "docs" / "guidelines" / "repository-guidelines.md"

DISALLOWED_TARGET_STATUS_PATTERNS = (
    re.compile(r"Implementation status:", re.IGNORECASE),
    re.compile(r"planned target contract", re.IGNORECASE),
    re.compile(r"post-implementation", re.IGNORECASE),
    re.compile(r"referenced .* root exists", re.IGNORECASE),
    re.compile(r"\bplanned `[^`]+` root", re.IGNORECASE),
    re.compile(r"\bplanned package files\b", re.IGNORECASE),
    re.compile(r"\bplanned in this repository\b", re.IGNORECASE),
    re.compile(r"\bInstall and upgrade will\b", re.IGNORECASE),
    re.compile(r"\bActivation will\b", re.IGNORECASE),
    re.compile(r"\bafter implementation\b", re.IGNORECASE),
    re.compile(r"\bplanned v1\b", re.IGNORECASE),
    re.compile(r"\bafter the .* implementation exists\b", re.IGNORECASE),
)


class C4TargetLanguageTests(unittest.TestCase):
    def test_c4_docs_do_not_use_migration_status_wording(self) -> None:
        hits: list[str] = []
        for path in sorted(C4_ROOT.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if any(pattern.search(line) for pattern in DISALLOWED_TARGET_STATUS_PATTERNS):
                    hits.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")

        self.assertEqual(hits, [])

    def test_repository_routing_docs_do_not_use_migration_status_wording(self) -> None:
        hits: list[str] = []
        text = REPOSITORY_GUIDELINES.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in DISALLOWED_TARGET_STATUS_PATTERNS):
                hits.append(f"{REPOSITORY_GUIDELINES.relative_to(ROOT)}:{line_number}:{line.strip()}")

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
