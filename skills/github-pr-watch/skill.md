---
name: github-pr-watch
description: Watch a GitHub pull request until merge readiness is green.
---

# GitHub PR Watch

Use GitHub connector tools or `gh` for PR state, comments, checks, and logs.
Use the `git` skill for every commit.
PR-watch commits are bounded automated commits; no human approval is required.

Do not merge unless the user explicitly asks for it.
Push only to the PR head branch.
Never force-push without explicit user approval.

## Inputs

Discover:
- PR identity, repository, head branch, and base branch;
- merge policy, required checks, and push permission;
- review and thread state.

## Merge-Ready

The PR is green only when:
- all required checks satisfy repository policy, including its rules for skipped or neutral results;
- no CI/CD job is pending, queued, running, failed, cancelled, timed out, or awaiting action;
- no unresolved blocking review thread or requested-changes review remains;
- the branch has no merge conflict and satisfies the repository's up-to-date requirement;
- all new comments since the last pass have an explicit disposition;
- the PR is not draft unless the user asked only for technical readiness.

## Watch Contract

Prefer GitHub events; use scheduled refresh only when events are unavailable or a timed retry is required.
On each relevant event, run one readiness pass, then wait rather than busy-poll.

During each pass:
- refresh PR state;
- handle actionable comments or failed checks;
- update the branch only by repository policy;
- verify focused changes locally;
- commit through the `git` skill;
- push and wait for checks.
Keep status updates concise and evidence-backed.

Persist enough state between passes to process only new evidence.

Merges, rebases, force-pushes, history rewrites, and base-branch or cross-branch mutations require explicit user authorization.
Keep unrelated local changes outside every commit and push.

## Review Dispositions

Treat human and AI-agent comments with the same evidence standard.
Every new or unresolved comment needs one visible disposition:
- **Fix now:** correct and in scope; implement, verify, and reply with the commit.
- **Defer:** valid but outside scope; record the owning TODO and explain.
- **Reject:** incorrect, obsolete, duplicate, or worse for the codebase; reply with evidence.
- **Clarify:** unsafe to decide; ask a focused question.

Do not resolve threads invisibly.

## CI and Branch Blockers

For failed checks, identify the failed workflow, job, step, command, commit SHA, and useful log excerpt.
Fix the root cause against owning docs, tests, and implementation contracts.
If blocked by external factors you cannot resolve, such as missing secrets, outages, hosted runner capacity, or permissions, report evidence.
Continue only while useful progress remains.

Resolve conflicts directly within the authorized PR head update when docs and tests make the correct result clear.
Ask only about ambiguous resolutions or operations outside that authorization.

## Completion Report

Finish with:
- PR state and URL;
- final commit SHA if changes were pushed;
- check summary;
- comment dispositions;
- verification run locally;
- remaining blockers or follow-up TODOs.
