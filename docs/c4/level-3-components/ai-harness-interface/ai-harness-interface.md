---
title: "AI Harness Interface - Component"
summary: "L3 contract for Codex and Claude Code package generation, route injection, handoff, and readiness reporting."
read_when:
  - Implementing or reviewing target-platform package files.
  - Designing package handoff guidance.
  - Checking whether logic belongs in the interface package or live project checkout.
---

# AI Harness Interface - Component

## Contents

- [Overview](#overview)
- [Components](#components)
- [Interfaces/Contracts](#interfacescontracts)
- [Data Ownership](#data-ownership)
- [Failure Modes](#failure-modes)
- [Acceptance Checks](#acceptance-checks)
- [References](#references)

## Overview

AI Harness Interface owns the Codex and Claude Code adapter package catalogs and Tokimon package content overlays.
It also owns handoff facts and post-install readiness reporting for target-platform install guidance.
Target-platform schemas and tooling own platform-specific package shape.
Account-level install procedures and mechanics stay with their platform owners.
Both adapters provide stable discovery and activation while keeping live Tokimon implementation and runtime state in the configured live checkout.

## Components

### Target-Platform Adapters

- **Target-platform contract registry:** Declares the supported target platforms with their manifest directory, marketplace path, and generated output root.
- **Platform tooling adapter:** Authors supported Codex compatibility structure and uses available legacy helpers for authoring or validation.
- **Claude CLI adapter:** Invokes the Claude Code CLI for read-only plugin-shape validation and read-only platform inspection.

### Generated Package Content

- **Package generator:** Writes Tokimon-owned content into the package selected through the target-platform contract.
  Codex uses supported compatibility structure; Claude Code uses a Tokimon-generated package.
- **Codex plugin manifest overlay:** Patches Tokimon-owned manifest values without owning Codex manifest shape.
- **Claude plugin manifest:** Writes the Tokimon-owned Claude Code plugin manifest within the published Claude Code plugin schema.
- **Packaged root skill directories:** Expose `skills/<skill-id>/` directories copied from the live skill inventory.
  Codex invocation metadata is included where declared.
- **MCP entries:** Materialize catalog-owned package entries and route them to live Tokimon modules.
- **Generated local marketplace:** Writes Tokimon local marketplace content in the selected target platform's marketplace shape.

### Setup and Route Data

- **Repository setup configuration:** Writes and validates `.tokimon/settings.local.json` before package generation.
- **Install-time route data:** Resolves the live checkout path and runtime root for package generation.

### Diagnostics and Readiness

- **Package handoff diagnostics:** Validate generated package output, route-back, cache freshness, and handoff facts.
- **Post-install readiness reporter:** Reports generated state, registration, operational readiness, configuration, blocking remaining work, and non-blocking capability setup.
  It does not mutate account-level platform state.

## Interfaces/Contracts

Canonical machine-readable contract: `.venv/bin/python -m tokimon.ai_harness_interface contract --json`.
This section adds generated-package semantics that the JSON contract cannot fully explain.

### Target-Platform Contract Registry

Supported target platforms are `codex` and `claude`.
Each target platform declares:

- platform id
- plugin manifest directory (`.codex-plugin` or `.claude-plugin`)
- generated marketplace path relative to the generated output root
- generated output root `.tokimon/ai-harness-interface/<platform-id>`, relative to the checkout root and resolved against the primary worktree that owns the runtime root
- account-level install guidance owner

`codex` is the default target platform when the operator does not select one.

### Account-Level Install Boundary

- **`AHI-INSTALL-001`:** Account-level installation stays outside AI Harness Interface.
  `docs/operations/install.md` owns the composed operator procedure.
  `plugin-creator` guidance owns the Codex-specific procedure.
  The Codex and Claude Code plugin CLIs perform their respective account-level mutations.
- **`AHI-INSTALL-002`:** AI Harness Interface never emits or runs account-level install or reinstall commands.
  It may inspect target-platform state through read-only commands.

### Codex Package File Catalog

The Codex v1 package file catalog is closed for each generation run.
The generated package output under `.tokimon/ai-harness-interface/codex/plugins/tokimon/` contains:

- `.codex-plugin/plugin.json`
- `.mcp.json`
- `skills/<skill-id>/SKILL.md` for every packaged root skill
- `skills/<skill-id>/agents/openai.yaml` for every canonical skill that declares Codex metadata

Installer, validator, generator, and diagnostic code is not copied into the generated package.
Future support files must be added explicitly to this closed catalog.

The root `skills/` directory owns the selectable repo skill inventory.
AI Harness Interface code must not add repo skills through code-only skill ids.
Generation selects root skill entrypoints through the installation policy; validation expects that recorded selection.

There is no account-level AI Harness Interface configuration file in v1.
Route data is generated from the checkout running package generation and baked into package source files.
`.app.json` is not a v1 surface unless an accepted contract adds it.
Assets are not v1 copied interface files unless an accepted contract adds exact asset paths to this catalog.

### Claude Package File Catalog

The Claude Code package file catalog is closed for each generation run.
The generated package output under `.tokimon/ai-harness-interface/claude/plugins/tokimon/` contains:

- `.claude-plugin/plugin.json`
- `.mcp.json`
- `skills/<skill-id>/SKILL.md` for every packaged root skill

The Claude Code adapter has no scaffold step.
`generate` writes the full package structure and marketplace file directly.
AI Harness Interface owns that content within the published Claude Code plugin schema.
Claude Code discovers `skills/<skill-id>/SKILL.md` directories at the plugin root without a manifest path declaration.
It loads `.mcp.json` through the manifest `mcpServers` path.
The packaged skill inventory, skill body contract, and routing invariants are the same as the Codex catalog.
Codex-only `agents/openai.yaml` files are not copied into the Claude Code package.
Cachebuster mechanics are a Codex `plugin-creator` concern and do not apply to the Claude Code adapter.
Claude Code refreshes installed plugins through its own marketplace and plugin update commands.

### Platform Tooling Adapter

Current `plugin-creator` guidance and the official OpenAI plugin format define Codex package shape.
The Codex adapter uses available legacy system-skill helpers and provides built-in compatibility scaffold and cachebuster authoring when those helpers are absent.
It never downloads or installs replacement helpers.
An available helper failure is reported rather than silently replaced.
Optional validation helpers report skipped checks; the operator validates Tokimon content and confirms target-host package acceptance through `AHI-INSTALL-001`.
`TOKIMON_REQUIRE_PLUGIN_CREATOR` or an explicitly required helper preserves fail-loud validation for operators who require legacy tooling.

Normal package generation must not run scaffold creation.
Codex generation may require a prior scaffold seed and may fail fast when structural files are missing.

### Claude CLI Adapter

The Claude Code adapter uses the `claude` CLI through a narrow read-only adapter.
The adapter owns executable resolution (default `claude`, overridable through `TOKIMON_CLAUDE_CLI`), the read-only command shapes, and the skip-when-unavailable semantics:

- `claude plugin validate <path>`: plugin and marketplace shape validation.
- `claude plugin list --json`: read-only installed-plugin inspection; post-install readiness runs this adapter-owned command through the readiness command runner.

Adapter commands are skipped and reported as skipped when the resolved executable is unavailable, unless the operator requires the CLI through `TOKIMON_REQUIRE_CLAUDE_CLI`.
The adapter must not run `claude plugin` install, uninstall, enable, disable, or marketplace mutation commands.

### Plugin Manifest Overlay

`.tokimon/ai-harness-interface/codex/plugins/tokimon/.codex-plugin/plugin.json` is seeded through built-in authoring or available legacy tooling, following current plugin guidance.
AI Harness Interface patches only Tokimon-owned values:

- `name`: `tokimon`
- `version`
- `description`
- `author.name`
- `skills`: `./skills/`
- `mcpServers`: `./.mcp.json`
- Tokimon `interface` presentation fields

OpenAI's plugin format defines manifest shape; Tokimon validates strict semver and its content invariants, with optional legacy helper validation.
Codex cachebuster metadata is authored through the available helper or the built-in compatibility path.
The manifest must not include unsupported fields such as `hooks`.
`apps` must stay out of the manifest unless `.app.json` exists and is in scope.

### Claude Plugin Manifest

`.tokimon/ai-harness-interface/claude/plugins/tokimon/.claude-plugin/plugin.json` is written by AI Harness Interface.
Tokimon-owned values:

- `name`: `tokimon`
- `displayName`: `Tokimon`
- `version`
- `description`
- `author.name`
- `mcpServers`: `./.mcp.json`

The Claude Code plugin schema owns accepted manifest shape; the Claude CLI adapter validates it when available or required.
The manifest must not include `hooks`, `commands`, `agents`, `apps`, or the Codex-only `interface` object.
Skills are exposed through the default `skills/` directory scan, so the manifest must not declare a `skills` path.

### Generated Local Marketplace

Each target platform has one generated local marketplace named `tokimon-local` with exactly one `tokimon` plugin entry sourced from `./plugins/tokimon`.

`.tokimon/ai-harness-interface/codex/.agents/plugins/marketplace.json` defines the generated Codex local marketplace.

Tokimon-owned fields:

- `name`: `tokimon-local`
- `interface.displayName`: `Tokimon Local`
- plugin entry `name`: `tokimon`
- plugin entry `source.source`: `local`
- plugin entry `source.path`: `./plugins/tokimon`
- plugin entry `policy.installation`: `AVAILABLE`
- plugin entry `policy.authentication`: `ON_INSTALL`
- plugin entry `category`: `Productivity`

The official Codex marketplace format defines entry shape; Tokimon validates its owned content and may use available legacy validators.

`.tokimon/ai-harness-interface/claude/.claude-plugin/marketplace.json` defines the generated Claude Code local marketplace.

Tokimon-owned fields:

- `name`: `tokimon-local`
- `owner.name`: `Tokimon`
- `description`
- plugin entry `name`: `tokimon`
- plugin entry `source`: `./plugins/tokimon`
- plugin entry `description`
- plugin entry `category`: `productivity`

The Claude Code marketplace schema owns the accepted entry shape; the Claude CLI adapter validates it when available or required.
AI Harness Interface reports each generated marketplace path and identity as handoff facts.
Account-level marketplace registration and plugin installation follow `AHI-INSTALL-001`.

### Install-Time Route Data

The generator resolves:

- `liveCheckoutPath`: absolute path to the Tokimon checkout running package generation
- `runtimeRoot`: absolute path to that checkout's Tokimon runtime root

The generator injects those paths into:

- packaged `SKILL.md` routing notes
- packaged `{{tokimonCommand}}` prefixes for Python module commands
- packaged `{{tokimonPython}}` command placeholders, resolved to the primary worktree's `.venv/bin/python` when the live checkout is linked
- generated `.mcp.json` server commands

`{{tokimonCommand}}` renders a complete POSIX shell command prefix.
It prepends `liveCheckoutPath` to `PYTHONPATH`, retains existing entries, and invokes the repository interpreter with `-P` so the caller's directory cannot shadow the configured source.
The generator shell-quotes each installed path; canonical skills use the prefix without surrounding quotes.
Installed skill Python module examples use this prefix and preserve the caller's working directory and project arguments.
`{{tokimonPython}}` remains an interpreter-path token for commands that already supply an explicit source import route.
Package validation rejects unresolved command placeholders.

The installed route does not choose the source-edit checkout for an active task.
Agents edit source files in the current task checkout unless the operator explicitly names another checkout.

The route data must not be written as an account-level AI Harness Interface config file.

### Repository Setup Configuration

`configure` writes repository-local Tokimon settings from `tokimon/settings.example.json`.
It preserves unrelated local settings.
It sets `pair.cwd` to the live checkout, keeps AI Debate on `local-cli`, defaults Pair to `native-subagent`, and applies operator choices for:

- `pair.access`
- `ai_debate.peer_b`
- `pair.peer_b`
- `installation.docker_features`: `auto` or `disabled`, with `auto` as the template default

Setup stores placement and execution policy, without provider model or effort settings.
`configure` removes legacy `agents.codex` and `agents.claude` settings while preserving unrelated settings.
`pair.access=local-cli` remains an explicit compatibility and rollback choice.

Operators must pass `--yes` to accept existing or template defaults, or pass explicit values for the choices they want to set.
`generate` must fail before writing package content when required local settings are missing or invalid.

### Installation Policy

Installation does not install, upgrade, or download external dependency software, provider CLIs, or container images.
Missing essential prerequisites block installation; unavailable optional capabilities are disabled and reported.
`installation.docker_features=auto` enables Docker capabilities only when the configured `TOKIMON_RB_DOCKER` command (default `docker`) resolves on `PATH`.
`disabled` keeps them off even on hosts with Docker.
Generation records the resolved selection in `installation-policy.json` at the generated output root, outside the plugin package.
Validation and readiness use this snapshot, so host changes do not silently change an existing package's membership.
The snapshot disables `mcp.remote-browser-broker` and the `container-executor` skill when Docker features are off.
The remaining package works with native browser surfaces; `using-tokimon` owns their selection order.
Legacy packages without a snapshot retain their complete catalog until regenerated.

### License Distribution

Generation reads `LICENSE`, `THIRD_PARTY_NOTICES`, and the four upstream license and notice files under `licenses/` before writing package content.
Both target-platform packages include byte-identical copies at the same relative paths.
Both generated manifests declare MIT for Tokimon's original material; third-party material retains its own licenses.
Missing, empty, unreadable, or symlinked source license files block generation.
Generation rejects symbolic links and nonregular output entries, including destination ancestors, before cleaning or writing an existing package.
File writes use atomic replacement through directory handles that do not follow symbolic links.
Replacing a hard-linked output file does not modify its other names.
Unavailable POSIX directory-relative operations block generation before output changes.
Validation rejects missing or altered packaged license files and a mismatched manifest license.
The public skill catalog contains no ChatGPT website automation skill.
Packaged skill dependencies resolve to included skills, and documented module launchers support `--help` from a subject checkout without creating runtime state there.

### Python Distribution

The wheel includes module-local configuration, templates, Dockerfiles, and browser image resources.
It also includes a complete source workspace under the environment's `share/tokimon/`, with Python implementation, skills, documentation, repository guidance, and license notices.
The source archive carries the same build inputs, so a wheel rebuilt from that archive preserves the workspace.
Build output excludes virtual environments, private runtime state, Git metadata, and Python bytecode.

An installed wheel resolves its source workspace from `sys.prefix/share/tokimon` and uses the invoking environment's Python for generated commands.
An ordinary or linked source checkout keeps its existing source and primary-worktree `.venv` routing.
Both modes keep settings and persistent state in the selected workspace's `.tokimon/`.
Wheel use requires a writable, user-owned Python environment; missing bundled workspace files fail with an actionable error.
Verification builds and unpacks artifacts in an isolated directory, without registering or updating account-level plugins.
Quality CI verifies both a directly built wheel and a wheel rebuilt from the source archive with the same isolated artifact checks.

### Packaged Root Skill Directories

AI Harness Interface copies canonical skill entrypoints, injects installed route data, copies Codex invocation metadata when declared, and validates package invariants.
The Skills and Memory containers and the canonical `using-tokimon` and `memory` entrypoints own activation and runtime behavior.

Each packaged root skill directory contains a copied `SKILL.md` entrypoint, not a pointer-only wrapper.
For Codex, it also contains a copied `agents/openai.yaml` when the canonical source skill declares invocation metadata or tool dependencies.
A declared tool dependency remains optional for the package as a whole but is required when that skill is invoked.
Each packaged root skill directory's `SKILL.md` must:

- have valid `name` and `description` frontmatter
- contain enough operating guidance to start the skill from an adopting repository
- include an installed routing note with the resolved `liveCheckoutPath` and `runtimeRoot`
- route Tokimon-owned commands under `liveCheckoutPath` and runtime artifacts under `runtimeRoot`
- treat the current repository as the task subject unless the operator says otherwise
- keep source edits in the current task checkout instead of inferring a source-edit checkout from `liveCheckoutPath` or `runtimeRoot`

Package validation ensures copied entrypoints preserve the activation and routing contracts owned by their canonical skills.
Every generated package-owned entry must be a regular file or directory inside the generated package tree; symbolic links, including links whose targets remain inside the output root, are rejected rather than followed during validation.

The generator discovers canonical repo skill entrypoints under `skills/<skill-name>/skill.md`.
It copies them to `skills/<skill-name>/SKILL.md`.
The generator must not author skill bodies from Python string templates.
The generator does not copy source skill `assets/`, `references/`, or `scripts/`.
Under `agents/`, it supports only a regular `openai.yaml` file and copies it only to Codex package output.
Changing `liveCheckoutPath` or `runtimeRoot` requires regenerating package output because those paths are baked into installed skill text and MCP commands.
After regeneration, follow the applicable procedure identified in `AHI-INSTALL-001`.

### MCP Entries

The shared tool catalog owns the complete capability, server-id, live-module, and package-membership mapping.
Its component contract is `docs/c4/level-3-components/tools/tool-registry.md`.
AI Harness Interface consumes its closed package set for both target platforms.

Generation writes the supported entries selected by the installation policy.
Docker-disabled entries are omitted from both target-platform packages and readiness checks.
Missing tokens or provider CLIs do not change the remaining package membership or fail generation.
Readiness reports those prerequisites as per-capability operational state.
The JSON contract exposes only `capability_id`, `server_id`, `module`, `serve_args`, and `health_args` for each package entry; registry-only metadata remains in the shared catalog.

Each `.mcp.json` server command must:

- use the source checkout `.venv/bin/python`, reusing the primary worktree interpreter when linked, or the owning Python environment for a wheel workspace
- execute the live module through `-m ... serve`
- avoid referencing the installed plugin cache

The Codex `.mcp.json` sets each server `cwd` to the live checkout so `-m` module resolution works from the checkout.
The Claude Code `.mcp.json` sets each server `env.PYTHONPATH` to the live checkout instead.
The Claude Code MCP schema documents `command`, `args`, and `env`, but not `cwd`.
Live modules resolve their workspace and runtime roots from source location rather than the working directory.

### CLI Commands

The AI Harness Interface CLI entrypoint is:

```bash
.venv/bin/python -m tokimon.ai_harness_interface <command>
```

Required commands:

- `bootstrap-codex`
- `configure`
- `contract --json`
- `generate`
- `validate`
- `diagnose`
- `post-install`

`contract`, `generate`, `validate`, `diagnose`, and `post-install` accept `--target-platform <codex|claude>` and default to `codex`.
`bootstrap-codex` is Codex-only because only the Codex adapter has a scaffold step.
`configure` is platform-neutral.
`generate --cachebuster` is Codex-only and must fail fast for `claude`.
The CLI obeys `AHI-INSTALL-002`.
`diagnose --json` reports handoff facts for target-platform install guidance without mutating account-level state.
`post-install` reports readiness and next steps after account-level installation.
`post-install --settings-path <path>` inspects the named repository-local settings file instead of the default `.tokimon/settings.local.json`.
It may run read-only target-platform inspection commands for the selected platform, but it must not mutate account-level plugin, MCP, or platform configuration.

### Generate, Validate, and Handoff

Codex bootstrap flow:

1. Use the available legacy scaffold helper or built-in supported compatibility authoring against `.tokimon/ai-harness-interface/codex/`.
2. Seed `.codex-plugin/plugin.json`, `.mcp.json`, `skills/`, and the generated local marketplace file.
3. Leave Tokimon package content generation to `generate`.

Generate flow:

1. For Codex, require the scaffolded structural files; the Claude Code adapter self-seeds missing package structure.
2. Validate repository-local setup under `.tokimon/settings.local.json`.
3. Resolve install-time route data from the live checkout path and runtime root.
4. For Codex, patch Tokimon manifest values in the scaffolded manifest; for Claude Code, write the Tokimon-owned manifest.
5. Write `.mcp.json` from the shared catalog's selected package set for the target platform.
6. Regenerate packaged root skill directories and copy Codex `agents/openai.yaml` metadata where declared.
7. Write Tokimon local marketplace content for the selected platform.
8. Record the resolved installation policy and report disabled capabilities without recommending dependency installation.

Validate flow:

1. Validate Tokimon package invariants for the selected platform.
2. For Codex, validate `plugin-creator` package shape and generated marketplace name through `plugin-creator` tooling when available or required.
3. For Claude Code, validate plugin and marketplace shape through the Claude CLI adapter when available or required.

Handoff flow:

1. Run `diagnose --json` after generation and validation.
2. Report target platform, generated package root, generated marketplace path, plugin id, marketplace name, `liveCheckoutPath`, and `runtimeRoot`.
3. Report the account-level install owner defined by `AHI-INSTALL-001`.
4. Honor the command boundary in `AHI-INSTALL-002`.

Post-install readiness flow:

1. Run `post-install` after account-level target-platform installation, preferably from a fresh target-platform thread when package interface files changed.
2. Report repository-local configuration status, including Pair choices.
3. Report generated package validity and install-time route data.
4. Report target-platform plugin and MCP server registration using read-only platform inspection when supported.
5. Report MCP route-back health separately from operational readiness.
6. Report sanitized blocking remaining work, non-blocking capability setup, and next steps.

For Pair native placement, post-install readiness reports configuration compatibility but does not claim that a model call or native subagent spawn was executed.
For local-CLI placement, it continues to report required provider executable discovery.
AI Debate health with unverified provider authentication counts as healthy route-back but not as verified operational readiness.

Codex platform inspection uses `codex plugin list` for plugin status and `codex mcp get <server-id>` for each packaged MCP server.
Claude Code platform inspection uses `claude plugin list --json` for plugin status.
It matches the `tokimon` entry from the `tokimon-local` marketplace with tolerant field parsing because the platform does not pin the JSON field names.
Claude Code per-server MCP registration derives from installed plugin presence; enablement derives from plugin enabled status.
Claude Code loads plugin `.mcp.json` servers when a plugin is enabled and does not document CLI inspection of plugin-scoped servers.
Route-back health checks stay platform-independent because they run the live modules directly.

`post-install` must distinguish route-back health from operational readiness.
A route-back check passes when the live health command executes and returns its structured health payload, even when that payload reports the capability unavailable.
For example, Telegram may remain unavailable for sends because no bot token is configured, while its package route remains healthy.
Missing external prerequisites produce non-blocking capability setup actions rather than failed installation readiness.
For Codex, a packaged server that is registered but disabled by platform policy also produces non-blocking enable guidance; missing registration remains blocking.
Skipped registration, route-back health, and operational checks set their summary `checked` field to `false` and render as `not checked`.

Live-only implementation or runtime-state changes do not require package regeneration.
They also do not require target-platform reinstall while package interface files are unchanged.
Adding, removing, or renaming any packaged root skill, including `memory`, is a package interface change.
It requires regeneration followed by the account-level procedure identified in `AHI-INSTALL-001`.

## Data Ownership

Owned data:

- Target-platform registry, package file catalogs, marketplace overlays, and install-time route fields.
- Repository setup, package generation, validation, diagnostic, handoff, and readiness contracts.

Consumed read-only:

- Canonical root skill inventory, shared MCP catalog, Tokimon settings template, and local settings.
- Target-platform schemas, tooling guidance, validation output, and read-only plugin or MCP registration status.

Derived or transformed:

- Generated package and marketplace output under `.tokimon/ai-harness-interface/<platform-id>/`.
- Repository-local settings updates under `.tokimon/settings.local.json`.
- Install-time route data baked into generated skills and MCP commands.
- Post-install readiness reports derived from repository settings, generated package validation, read-only target-platform inspection, and MCP health output.

Core Tokimon implementation, runtime state, provider credentials, and adopting repository state are not owned.

## Failure Modes

After correcting a route, configuration, package, or platform condition, the operator reruns the owning `configure`, `generate`, `validate`, `diagnose`, or `post-install` command.

Platform inspection and MCP health commands use bounded timeouts and are not retried automatically.
A timeout is reported as a failed check.

`generate` rewrites AI Harness Interface-owned package catalog files from the current source, route, configuration, version, and target-platform inputs.
After interruption, rerun `generate`; account-level state remains untouched.

- **Missing checkout or invalid install route:** Generation or diagnostics fail before using the invalid route.
  Recover by rerunning from an operator-confirmed live checkout.
- **Generated package or marketplace invalid:** Validation fails.
  Recover by fixing generated content or generator code.
- **Scaffold missing:** Codex generation fails before writing package content.
  Recover by running `bootstrap-codex`; Claude Code generation self-seeds.
- **Cachebuster requested for Claude Code:** Generation fails because cachebuster mechanics belong to Codex `plugin-creator`.
  Recover by rerunning without `--cachebuster`, then use Claude Code marketplace and plugin update commands.
- **Claude CLI unavailable:** Claude validation and platform inspection are skipped and reported as skipped.
  Recover by installing the Claude Code CLI or setting `TOKIMON_REQUIRE_CLAUDE_CLI` where the CLI exists.
- **Setup missing or invalid:** Generation fails, and `configure` does not prompt implicitly.
  Recover by running `configure --yes` or passing explicit settings flags, then rerun generation or `post-install` as appropriate.
- **Legacy `plugin-creator` helper unavailable:** Bootstrap and cachebuster use built-in compatibility authoring; optional helper validation reports skipped checks.
  Explicit legacy-tool requirements still fail and explain the missing helper.
- **Copied root skill invalid or stale:** Validation fails or activation exposes old package routes.
  Recover by fixing the canonical skill or route, regenerating, and following `AHI-INSTALL-001`; apply the Codex cachebuster when required.
- **Canonical skill or invocation metadata unreadable:** Generation and validation return a controlled package-validation failure, including for malformed UTF-8, empty metadata, unsafe links, or filesystem read errors.
  Recover by repairing the canonical skill entrypoint or `agents/openai.yaml`, then rerun generation.
- **MCP route-back fails:** Diagnostics report the failed server entry and command.
  Recover by fixing the generator, live checkout, or live module route, then regenerate when the package entry changes.
- **MCP dependency unavailable or capability not operational:** Readiness provides sanitized non-blocking setup guidance.
  Configure the missing dependency or secret and rerun `post-install`; package regeneration is unnecessary.
- **Docker absent or disabled:** Generation omits Docker-dependent surfaces and reports the policy decision.
  Readiness checks only selected surfaces and does not instruct an agent to install Docker.
- **Target-platform registration missing:** `post-install` reports the missing plugin or MCP server id.
  Recover through `AHI-INSTALL-001`, then start a fresh target-platform thread.

## Acceptance Checks

License and public-package verification:

```bash
.venv/bin/python -m unittest discover -s tests/ai_harness_interface -t .
```

Expected outcome: both platforms distribute the complete license and notice files, reject missing or altered notices, and omit website automation.
Pair setup accepts native Codex or local Codex/Claude peers and rejects removed provider selections without writing settings.

Codex package validation:

Run `bootstrap-codex` only when the scaffold is absent; on an existing output root, begin with `configure`.

```bash
.venv/bin/python -m tokimon.ai_harness_interface bootstrap-codex
.venv/bin/python -m tokimon.ai_harness_interface configure --yes
.venv/bin/python -m tokimon.ai_harness_interface generate
.venv/bin/python -m tokimon.ai_harness_interface validate
```

Claude Code package validation:

```bash
.venv/bin/python -m tokimon.ai_harness_interface configure --yes
.venv/bin/python -m tokimon.ai_harness_interface generate --target-platform claude
.venv/bin/python -m tokimon.ai_harness_interface validate --target-platform claude
```

Expected outcome: generated output contains only regular, package-local entries and matches the exact per-platform catalog and package-membership contract.
The skill inventory, copied Codex invocation metadata, package checks, and routing checks pass.

Installed skill command verification:

```bash
.venv/bin/python -m unittest discover -s tests/ai_harness_interface -t . -p 'test_skill_commands.py'
```

Expected outcome: generated Codex and Claude skill commands run from an independent adopting repository without an installed Tokimon distribution or ambient source route.
The checks retain existing `PYTHONPATH` imports, resolve the configured linked checkout through the primary interpreter, and handle spaces and shell metacharacters in paths.
Briefing and search include the adopting project's and global memories, exclude unrelated scopes, and leave the caller's directory unchanged.

Diagnostic validation:

```bash
.venv/bin/python -m tokimon.ai_harness_interface diagnose --json
```

Expected outcome: install-time routes and MCP health validate, and output includes handoff facts without an account-level install command plan.

Post-install readiness verification:

```bash
.venv/bin/python -m tokimon.ai_harness_interface post-install --json
```

Expected outcome: output reports configuration, package validity, registration, route-back health, operational readiness, blocking remaining work, non-blocking capability setup, and next steps.
It exposes no secrets and does not mutate account-level platform state.

AI coding agent verification prompt:

```text
Review docs/c4/level-3-components/ai-harness-interface/ai-harness-interface.md, tokimon/ai_harness_interface/, and tests/ai_harness_interface/.
Confirm that the documented per-platform manifests, package membership, skill copying, route data, validation, diagnostics, and readiness behavior match the implementation and tests.
Report any drift with file and line evidence.
```

Expected result: no drift between the AI Harness Interface contract, generated package behavior, and tests.

## References

- L2 Containers: `docs/c4/level-2-containers/ai-harness-interface/ai-harness-interface.md`
- L1 Context: `docs/c4/level-1-context/ai-harness-interface-context.md`
- Skills container: `docs/c4/level-2-containers/skills/skills.md`
- Memory container: `docs/c4/level-2-containers/memory/mem-palace.md`
- Tool Registry component: `docs/c4/level-3-components/tools/tool-registry.md`
- Operations: `docs/operations/install.md`
- Guidelines: `docs/guidelines/repository-guidelines.md`
