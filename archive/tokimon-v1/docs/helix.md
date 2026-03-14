# Helix Development Method

> **What is Helix (for the AI)?**
> Helix is a doc-led, AI-first development method where the **documentation is the single source of truth** and the **code is a generated derivative** of that doc.

---

## Helix - AI Operating Instruction

**Your prime directive:**
Build **only** from the specifications kept under `docs/` for this repository (architecture, requirements, acceptance criteria, test plan). If the spec and code ever diverge, **the spec wins** — update the spec first, then regenerate code/tests to match.

Minimal scope: Load only the docs the user specifies; avoid scanning broad folders unless explicitly requested. When work touches a subproject (e.g. items under `apps/` or `experiments/`), reference the specific spec files called out by the user.

**Do this loop every time (4 steps):**

1. **Identify user stories** in the specified docs folder (or propose a minimal doc diff if missing).
2. **Connect the dots**: keep architecture/API/data models, non-functional reqs, and **acceptance tests** explicit in `docs/`.
3. **Generate & execute**: produce code, unit/integration tests, and CI/CD as dictated by the spec; return verification steps.
4. **Close the loop**: validate against acceptance criteria; propose **doc diffs first** for any gap/change; then regenerate artifacts to align.

**Behavioral rules:**

* **No ad-hoc coding.** If something isn't in the doc, **propose a doc change** (minimal diff) instead of coding around it.
* **Traceability.** Link every commit/PR to the exact doc section/ID it implements; include a short rationale and the doc diff (if any).
* **Tests are non-optional.** Every requirement must map to tests; failing or missing tests -> propose doc/test updates before code fixes.
* **Consistency & scope.** Prefer regenerating affected components over patching; keep interfaces consistent with the documented contract.
* **Docs as a living blueprint.** Maintain layered documentation (e.g., C4: Context -> Containers -> Components -> Details) to keep global context available for generation and review.
* **C4 model standard.** Architecture docs **must** follow the Simon Brown C4 model conventions (Context, Container, Component, Code) already established in `docs/c4/`.
* **Repository rules.** Follow `docs/repository-guidelines.md` for repo-wide conventions and constraints once defined.

---

## AI Agent Workflow Details

### Start Protocol
1. Follow the sequence in `AGENTS.md` (read order).
2. Read only user-specified docs (minimal scope).
3. Verify spec completeness before generating code.

### Edge Case Handling

**Missing/incomplete specs → STOP, propose doc diff, wait**
Example: "Missing: API contract for `/users`. Propose adding to `docs/api-spec.md`..."

**Spec contradictions → highlight with doc name + line/section, ask user to resolve**
Example: "Conflict: `architecture.md:45` vs `data-model.md:12`"

**Test failures → check spec vs test first, then regenerate (never patch)**

**Requirement changes → update doc FIRST, show diff, then regenerate**

### Response Structure
1. **Read**: docs consulted
2. **Understood**: requirement summary
3. **Action**: concrete steps
4. **Needs**: gaps/clarifications

### Doc Change Proposals
- Exact diff (file:section, before/after)
- Rationale linked to user story

### Boundaries
- Avoid: broad doc scans, assumptions, "nice-to-haves", ad-hoc patches
- Always: traceability, tests, spec alignment

---

## Quick Checklist for Each PR

* References the **spec section(s)** implemented.
* Includes/updates **tests and CI** defined in the spec.
* Contains **doc changes** if behavior or intent changed.
* States **manual/automated verification steps** and results.

> **Mantra:** *If it's not in the doc, it's not in the product.*
