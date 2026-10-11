# Tokimon

Tokimon provides repository-aware Agent skills, review workflows, local memory, and MCP tools for Codex and Claude Code.
Skills guide work in the active project; implementation and private runtime state remain in the configured Tokimon checkout.

## Start

Use Python 3.12 or later and Git.
Create the local environment from this repository:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m tokimon.ai_harness_interface configure --yes --json
```

Follow [the installation operation](docs/operations/install.md) to generate and install the Codex or Claude Code package.
Installation leaves unavailable optional capabilities disabled and does not install Docker, provider CLIs, or other external dependency software.

A wheel contains the same source workspace, skills, documentation, and notices.
Use it in a writable Python virtual environment; Tokimon resolves its workspace to that environment's `share/tokimon/` and stores local state in its `.tokimon/` directory.
Run the configuration and package-generation commands with that environment's Python.
Package generation requires POSIX filesystem operations, as provided by Linux and macOS.

## Capabilities

- Repository planning, implementation, review, and acceptance skills.
- Explicit Pair collaboration using native Codex peers or local Codex and Claude CLIs.
- Local Mem Palace project memory and audit artifacts.
- General browser, remote-agent, container-executor, AI-debate, and Telegram MCP tools.

Tokimon provides no dedicated ChatGPT website integration or automatic answer-collection workflow.
Container Executor's MCP and Tunnel connection workflow is a separate capability; users configure their own external accounts and access.

## Local Data And External Services

Keep `.tokimon/` and `.venv/` private.
They can contain local configuration, memory, tool state, credentials, and generated packages.
Share source files from Git, rather than an archive of the working directory.

Review permissions before mounting directories or sending content to any external provider or messaging service.
Only supply data you have permission to share, and follow the applicable account terms.
Tokimon is an independent project; provider names describe compatibility and do not imply endorsement.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
```

## License

Original Tokimon material is licensed under the [MIT License](LICENSE).
Retained third-party material uses its own licenses and attribution in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) and `licenses/`.
