# Tokimon v2: Context

## System Purpose

Tokimon is a token monster that uses available tokens to improve itself. It accepts user directions about what to learn, spends idle time on self-improvement, and prepares itself before helping with tasks.

## Primary Actor

- User: feeds tokens, sets learning priorities, requests task help, inspects status

## External Systems

- Local filesystem: stores Tokimon state in `.tokimon/state.json`
- Test runner: validates that core product behavior still works
- Future benchmark harness: will evaluate whether Tokimon actually outperforms baseline agents

## Success Criteria

- Tokimon spends available tokens on improvement when idle
- Tokimon prioritizes directed skills over generic learning
- Tokimon improves task-relevant skills before answering when budget is available
- Tokimon quality claims are grounded in evaluation, not marketing prose

## Current Boundary

Tokimon v2 is currently a local Python CLI with a stateful learning engine. It is not yet a long-running autonomous service, multi-agent runtime, or benchmarked model platform.
