# Repository Guidelines

This document records the rules, conventions, and guardrails that govern work across this repository. It explains how contributors should structure, build, test, and safeguard changes in Tokimon.

## Purpose & Scope
- Helix is doc-led: files under `docs/` (plus component READMEs) are the single source of truth for requirements, architecture, and acceptance criteria. If the spec and code diverge, update the docs first and regenerate artifacts.
- Follow the read order in `AGENTS.md`, then load only the minimal set of docs the task references. Avoid scanning unrelated folders unless the spec explicitly instructs you to.
- Maintain traceability. Every change links back to its governing spec section, includes doc diffs when behavior shifts, and captures verification commands in task responses.
- Treat this guide as the canonical reference for repo-wide conventions, tooling entry points, and guardrails. Use the root `README.md` and component READMEs for deeper walkthroughs and project-specific commands.
- Keep operational command sequences up to date in the relevant README under a “Build, Test, and Development Commands” section when workflows change.

## Non-negotiables
- Always activate a local virtual environment before running Python or CLI commands from the repo root:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install the project in editable mode (from repo root):
  ```bash
  pip install -e src[dev]
  ```
- Chain commands as `source .venv/bin/activate && <command>` when executing one-offs.
- Do not bypass the doc-first workflow. Missing specs → propose a doc diff. Conflicts → call them out with file and line references before coding.
- Keep local infrastructure actions read-only. Deployments happen through CI/CD or designated maintainers only.
- AI assistants do not run `git commit`, `git push`, or tag operations unless the user explicitly directs it. Prepare changes; the user handles VCS writes.

## Coding Style & Naming Conventions
- Python adheres to PEP 8 with 4-space indentation, type hints where practical, and descriptive module names. Remove dead code, keep functions focused, and sort imports.
- Documentation stays in Markdown using ASCII punctuation. Lead with actionable guidance and cross-link to the authoritative spec in `docs/`.
- Configuration, manifests, and scripts live beside the tooling they control (e.g., keep app-specific configs within the corresponding subfolder). Promote shared settings into a single source instead of copying files across modules.

## Testing Guidelines
- Every documented requirement maps to automated tests. Update or add tests before modifying code so they codify the intended behavior.
- Prefer unit tests for Python modules; add integration or smoke tests only when the spec calls for live-system interaction.
- Run tests from repo root:
  ```bash
  pytest --maxfail=1 -c src/pyproject.toml src/tests
  ```
- Stub or record external system calls (APIs, databases, cloud services) unless the spec explicitly requires live requests. Document any credentials or fixtures needed for local runs and how they are sourced.
- Record deterministic reproduction commands in task responses, including relevant environment variables.

## Commit & Pull Request Guidelines
- Follow the Helix task template: Context, Change Summary, Artifacts, Verification, Doc Deltas, Risks & Rollback. Reference the exact doc section IDs or headings that govern the change.
- Commit messages are imperative and scoped (e.g., `feat(tokimon): add retry gate`, `docs(repo): clarify test commands`). Include AI tool attribution in the body when applicable.
- PRs link to their governing spec, paste relevant verification output, and attach artifacts (screenshots, `npm run build` logs, etc.) when changes touch UI or infrastructure.
- Keep mirrored or archived directories aligned with their upstream workflow if they are actively synced. Use this workspace for cross-repo documentation or tooling, not for direct commits against mirrored codebases.
- Large command outputs or generated artifacts stay out of version control; summarise results and store references in docs instead.

## Security & Infrastructure
- Never commit secrets, account IDs, or tokens. Use environment variables, `.env.example`, or secret managers instead. Remove any accidental credential immediately and rotate it.
- Treat all cloud CLIs (AWS, Azure, GCP, etc.) as read-only unless the spec authorises a deployment path. Preferred workflow: validate locally, deploy via CI/CD.
- Docker images ship from the provided Dockerfiles. Scan images before distribution and document published tags in `docs/`.
- Sanitize datasets used in experiments. No customer or sensitive data may live in this repo unless redaction steps are clearly documented in the module README.

## Maintenance
- Update this document as the command surface, tooling, or security posture evolves. Treat it as the source of truth for contributors and AI agents.
- When you add optional tooling (linting, type checks, CI jobs), document the install/run commands here and cross-link the governing spec section in `docs/`.
