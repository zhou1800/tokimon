from agents.retry import RetryGate, compute_call_signature
from flow_types import ProgressMetrics


def test_retry_blocks_identical_signature_without_progress():
    gate = RetryGate()
    signature = compute_call_signature(
        goal="goal",
        step_id="step",
        worker_type="Worker",
        key_inputs={"input": 1},
        strategy_id="strategy",
        retrieval_stage=1,
    )
    gate.record_signature(signature)
    prev = ProgressMetrics(failing_tests=2)
    new = ProgressMetrics(failing_tests=2)
    decision = gate.can_retry("task", signature, "fail", prev, new)
    assert decision.allow is False


def test_retry_allows_with_progress():
    gate = RetryGate()
    signature = compute_call_signature(
        goal="goal",
        step_id="step",
        worker_type="Worker",
        key_inputs={"input": 1},
        strategy_id="strategy",
        retrieval_stage=1,
    )
    gate.record_signature(signature)
    prev = ProgressMetrics(failing_tests=3)
    new = ProgressMetrics(failing_tests=2)
    decision = gate.can_retry("task", signature, "fail", prev, new)
    assert decision.allow is True
