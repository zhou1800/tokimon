from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import active_candidate
from tokimon.workflow.learning import validation


class LearningActiveExclusionTests(unittest.TestCase):
    def test_active_rejects_open_ended_repo_learning(self) -> None:
        packet = active_candidate()
        packet["trigger"] = "open_ended_repo_learning"

        result = validation.validate_candidate_packet(packet, expected_mode="active")

        self.assertFalse(result["ok"])
        self.assertIn("active trigger is excluded: open_ended_repo_learning", result["errors"])

    def test_active_requires_exact_source_spans(self) -> None:
        packet = active_candidate()
        packet["active_evidence"]["source_spans"] = []  # type: ignore[index]

        result = validation.validate_candidate_packet(packet, expected_mode="active")

        self.assertFalse(result["ok"])
        self.assertIn("active_evidence.source_spans must be a non-empty list", result["errors"])


if __name__ == "__main__":
    unittest.main()
