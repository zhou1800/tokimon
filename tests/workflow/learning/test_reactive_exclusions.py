from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import reactive_candidate
from tokimon.workflow.learning import validation


class LearningReactiveExclusionTests(unittest.TestCase):
    def test_reactive_rejects_product_decision_as_learning_failure(self) -> None:
        packet = reactive_candidate()
        packet["trigger"] = "product_decision"
        packet["reactive_evidence"]["failure_type"] = "product_decision"  # type: ignore[index]

        result = validation.validate_candidate_packet(packet, expected_mode="reactive")

        self.assertFalse(result["ok"])
        self.assertIn("reactive trigger is excluded: product_decision", result["errors"])
        self.assertIn("reactive failure_type is excluded: product_decision", result["errors"])


if __name__ == "__main__":
    unittest.main()
