---
title: "Helix"
summary: "Specification-led, evidence-aware development contract for keeping documents, C4, guidance, code, tests, and verification aligned."
read_when:
  - Before changing code, tests, CI, command contracts, or generated artifacts.
  - When a requested change may alter architecture, capability boundaries, workflow contracts, or operator behavior.
---

# Helix

This repository uses a Helix-inspired development method:

- Use `docs/guidelines/repository-guidelines.md` for top-level routing.
- Documents are specifications and design decisions that drive repository guidance, implementation, and interface evolution.
- Source code and tests are live implementation and verification evidence.

If C4 and code disagree, treat the disagreement as a finding.
Inspect the live code and tests, then make the smallest correct contract or implementation change.
The goal is to keep shared contracts, implementation, tests, and verification traceable to each other.

## Start Protocol

1. Use the already-loaded repository routing to identify the relevant C4 and guidance owners.
2. Read only the owning C4 or guidance docs needed for the change.
3. Explore source and tests as needed to understand the real implementation.

Avoid broad scans when the relevant C4 and source paths are already clear.
Do not refuse necessary source exploration.

## Change Protocol

For behavior changes, make the intended outcome and acceptance criteria explicit in the owning specification before implementation; verify the result against that section.

1. If the request changes architecture, behavior, command contracts, capability boundaries, or package surfaces, update the owning C4 doc.
   Do this before implementation.
2. If the request changes a reusable rule for future work, update the owning guideline.
   Do this before applying that rule elsewhere.
3. Change code/tests/package files to match the resolved contract.
4. Run focused verification, then broader verification when a shared component or contract changed.
5. Report doc deltas, code/test/package deltas, verification commands, and remaining risk.
