from __future__ import annotations

from agents.prompts import build_system_prompt


def test_worker_system_prompt_includes_instruction_precedence() -> None:
    prompt = build_system_prompt("TestTriager")
    assert "Instruction precedence (highest to lowest):" in prompt
    assert "Tokimon Constitution / immutable invariants" in prompt
    assert "Specs under `docs/`" in prompt


def test_worker_system_prompt_includes_tool_schema_compliance_gate() -> None:
    prompt = build_system_prompt("TestTriager")
    assert "Tool schema compliance" in prompt
    assert "do not guess missing required parameters" in prompt


def test_worker_system_prompt_includes_bounded_routing_rule() -> None:
    prompt = build_system_prompt("TestTriager")
    assert "Deterministic tool routing" in prompt
    assert "avoid broad scans" in prompt
    assert "parallelize reads only" in prompt


def test_worker_system_prompt_includes_json_only_output_contract() -> None:
    prompt = build_system_prompt("TestTriager")
    assert "Output contract:" in prompt
    assert "Reply with exactly one JSON object and nothing else" in prompt

