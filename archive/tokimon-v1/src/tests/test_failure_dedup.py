from agents.retry import RetryGate, compute_call_signature
from flow_types import ProgressMetrics


def test_failure_signature_dedup_blocks_repeat():
    gate = RetryGate()
    signature = compute_call_signature(
        goal="goal",
        step_id="step",
        worker_type="Worker",
        key_inputs={"input": 1},
        strategy_id="strategy",
        retrieval_stage=1,
    )
    gate.record_failure("task", signature, "fail")
    prev = ProgressMetrics(failing_tests=1)
    new = ProgressMetrics(failing_tests=1)
    decision = gate.can_retry("task", signature, "fail", prev, new)
    assert decision.allow is False
