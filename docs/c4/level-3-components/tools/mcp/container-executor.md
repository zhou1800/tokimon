---
title: "Container Executor MCP - Components"
summary: "Owns deployment configuration, lifecycle commands, MCP tools, bounded job/file semantics, and acceptance checks."
read_when:
  - Implementing or verifying Container Executor public behavior.
---

# Container Executor MCP - Components

## Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Interfaces/Contracts](#interfacescontracts)
4. [Data Ownership](#data-ownership)
5. [Failure Modes](#failure-modes)
6. [Acceptance Checks](#acceptance-checks)
7. [References](#references)

## Overview

Container Executor turns authorized MCP requests into independent container-local commands and bounded file operations.
Owner configuration defines host access and resource limits, while immutable deployment revisions preserve a reviewable lifecycle.

## Components

- **Deployment manager:** Validates configuration and local Docker context; records revisions and exact resource ownership before lifecycle operations.
- **Directory controller:** Maps canonical caller directories to independent deployments, serializes the runtime registry and gateway publication, and reports actual service state.
- **Machine gateway:** Exposes a fixed workspace-aware MCP catalog through one stable Tunnel connection; forwards only authorized execution tools to recorded socket volumes.
- **Isolation bootstrap:** Reads only immutable image code and bounded policy, installs an atomic container-local nftables policy, drops privileges, and fails before socket publication when isolation cannot be proven.
- **Job runtime:** Starts process groups, records bounded output and terminal outcomes, preserves client-disconnect independence, and marks former active jobs interrupted on restart.
- **File boundary:** Uses descriptor-relative no-follow traversal for workspace and mount file operations; rejects special files and symlink traversal.
- **MCP transport:** Provides bounded JSON-line socket requests and a fixed in-executor connector for local health, maintenance and stdio; the gateway owns Tunnel supervision.

## Interfaces/Contracts

### MCP Protocol

The Executor and Gateway return their declared protocol version from `initialize`, including for unsupported, missing, or malformed client proposals.
Missing or non-string methods return `-32600`.
Messages with a string method and no `id` are notifications: they produce no reply and cannot invoke request-only methods, tools, or maintenance.
Explicit request ids, including `0`, an empty string, and `null`, keep their response correlation.
Worker socket and Gateway stdio responses escape unpaired Unicode surrogates as JSON escapes and preserve ordinary Unicode text.
Tool text and error messages also escape malformed characters before wrapping, so decoded display text is valid UTF-8.
Strict UTF-8 frames retain ids, errors, and tool content and do not interrupt following requests; response byte limits remain enforced.
JSON decoder errors return `-32700` within the existing frame limits; over-limit or incomplete frames close the transport.

### Operator Commands

Entrypoint: `.venv/bin/python -m tokimon.tools.mcp.container_executor`.
Commands emit JSON with `--json`; errors exit nonzero and include an actionable reason.

| Command | Contract |
| --- | --- |
| `init --name <name>` | Write default owner configuration without deploying or granting host mounts; refuse replacement. |
| `install --name <name> [--config <path>]` | Build a content-addressed image/revision, create labeled persistent volumes, start one Executor container, and verify MCP/privileges before recording the current revision. |
| `start --name <name>` | Start the installed revision with a freshly discovered host-network policy; refuse unrelated or identity-replaced resources. |
| `stop --name <name>` | Stop only recorded, label-verified container IDs. |
| `status --name <name>` | Report installation, container identities/status, current/previous revision, and separate tunnel configuration from remotely verified connectivity. |
| `doctor --name <name>` | Read-only prerequisite, resource, mount, MCP, bootstrap privilege and policy checks; no health result is proof of ChatGPT/dot acceptance. |
| `update --name <name> [--config <path>]` | Refuse active jobs, build/start a new revision, preserve volumes, and restore the old revision on failed startup. |
| `rollback --name <name>` | Refuse active jobs and restore the prior successful image/config with fresh host-network policy; preserve data. |
| `uninstall --name <name> [--purge-data]` | Remove only verified containers/networks; retain data and revision history by default; explicit purge removes exact owned volumes. Never remove remote account resources. |
| `serve --name <name>` | Relay stdio MCP through a fixed connector in an already-running, verified Executor; never run commands on the host. |
| `health --json` | Report read-only local prerequisites and implementation identity without starting a deployment. |
| `download-tunnel-client` | Fetch the latest official Linux release into the owned runtime, verify its published SHA256, and preserve a non-secret receipt. |

The downloader selects exactly `tunnel-client-<tag>-linux-<amd64|arm64>.zip` from the latest release after validating its tag.
Runtime variants, other platforms and other versions do not match; missing or duplicate full-client assets fail before archive download or publication, while SHA256 verification remains required.

### Owner Configuration

The schema version is `1`; unknown fields, wrong types, unsafe names, and invalid bounds fail before Docker mutation.
Names and mount aliases match `[a-z][a-z0-9-]{0,31}`.

```json
{
  "schema_version": 1,
  "name": "default",
  "uid": 1000,
  "gid": 1000,
  "mounts": [],
  "blocked_networks": [],
  "limits": {
    "cpus": 2,
    "memory_mb": 512,
    "pids": 128,
    "max_jobs": 1000,
    "concurrency": 4,
    "timeout_seconds": 900,
    "output_bytes": 1048576
  },
  "tunnel": null
}
```

Each mount has `source` (existing absolute nonsymlink directory), `alias`, and `read_only` (boolean), with optional `protect_sensitive` (boolean, default false).
Mount targets are derived rather than user-selected; aliases are unique.
Optional `protected_host_paths` contains at most 32 existing absolute regular files or directories to hide inside protected grants.
Protected grants retain broad/system-control roots, sources inside sensitive paths, and overlapping-grant refusals.
Only a safe ancestor of the runtime may bypass runtime-overlap rejection, after deriving credential-directory masks before Docker mutation.
Masks cover immediate children named `.ssh`, `.gnupg`, `.aws`, `.azure`, `.kube`, `.codex`, `.config`, `.docker`, `.tokimon`, `.secrets`, `.env` or beginning `.env.`, and declared protected descendants.
The active Tokimon runtime's `.tokimon` ancestor is exempt from the immediate-child mask; its existing Container Executor `secrets/` directory is masked at its exact relative path instead.
Directory services also declare the configured Tunnel key's containing directory as protected, including owner-selected key locations outside the runtime.
Neither the containing Tokimon checkout nor non-secret runtime data is masked just because it contains that directory; other nested sensitive names outside declared roots remain ordinary authorized project data.
Symlink, special-file, excessive-mask, and invalid-target cases fail closed; host targets are never created, chowned or chmodded.
Directories receive root-owned mode-000 readonly tmpfs overlays; regular files receive readonly empty mode-000 file overlays from owned runtime.
Masks are derived and revalidated at managed start, with at most 128 masks per grant.
New sensitive paths require stop/start before protection can be assumed; this is a snapshot rather than continuous secret discovery.
No caller-supplied entrypoint, container environment, Linux capability, Docker argument, or host-port override exists.
Worker configuration requires `tunnel: null`; a per-deployment tunnel is rejected before Docker mutation with instructions to use machine setup.
The machine profile's tunnel object has `tunnel_id`, `api_key_file`, and `client_binary`; it contains file references, never a key value.
The key must be a non-root owner-readable private regular file; the gateway uses that non-root operator identity.
Blocked networks accept canonical IPv4/IPv6 CIDRs; ordinary public IPv4 connectivity remains available without a port allowlist.
Container DNS uses public resolvers; the supported deployment target is local rootful Linux Docker, with IPv6 disabled.
UID/GID are positive integers; CPU accepts 0.25–64, memory 64–65536 MiB, PIDs 16–4096, job count 1–100000, concurrency 1–32, timeout 1–86400 seconds, and captured output 1024–16777216 bytes.

### MCP Tools

| Tool | Arguments and visible result |
| --- | --- |
| `executor_health` | No arguments; runtime identity, boot ID, UID/GID, capability masks, isolation policy, roots and limits. |
| `executor_start` | `command`, optional `cwd` (default `/workspace`) and `timeout_seconds`; returns a job ID and initial state. |
| `executor_status` | `job_id`; current state, timestamps, exit code, output byte counts and truncation. |
| `executor_read` | `job_id`, optional `stream` (`stdout`/`stderr`), `offset`, `max_bytes`; bounded text and next byte offset. |
| `executor_cancel` | `job_id`; idempotent terminal status or process-group cancellation request. |
| `executor_jobs` | Optional `limit` (1–200); bounded recent job metadata for continuation after reconnect. |
| `executor_list` | Optional `path` (default `/workspace`), `limit` (1–200); bounded directory entries and truncation. |
| `executor_file_read` | `path`, optional `offset`, `max_bytes` (1–1048576), `encoding` (`utf-8`/`base64`); bounded content and next offset. |
| `executor_file_write` | `path`, `content`, optional `encoding` and `overwrite` (default false); create or explicitly replace a regular file with at most 1048576 decoded bytes. |

Job states are `running`, `succeeded`, `failed`, `cancelled`, `timed_out`, and `interrupted`.
Nonzero command exit is a failed job rather than a transport error.
Cancellation and timeout signal the process group, escalate to kill after a bounded grace, and bound output drain even if a descendant retains a pipe.
Already terminal outcomes never change; job IDs do not authorize access beyond the current deployment.
Jobs persist across MCP disconnection; restart never silently re-executes them.
Configured job-count exhaustion fails explicitly; operators manage retained storage through authorized container commands or explicit uninstall purge.

File tools and job working directories accept only `/workspace` and declared `/mounts/<alias>` roots.
File traversal rejects symlinks and special files using no-follow directory/file descriptors, including concurrent path replacement.
Shell commands remain free to use files available to their ordinary container user; host access is bounded by Docker mounts, not a command-name allowlist.
Transport frames are bounded to 2 MiB, connections are bounded, and logs contain request/job correlation rather than command text, file contents, credentials, or environment dumps.

### Directory Commands and Machine Connection

The same entrypoint accepts `workspace <command>`; `--directory` defaults to the caller's current directory without changing to the Tokimon installation root.
Installed skill commands embed the trusted Tokimon import root and Python interpreter while preserving the caller's working directory.
The canonical absolute directory identifies `ws-<first 20 SHA256 hex digits>`; the registry also binds device/inode and refuses path replacement or hash collision.
Start mounts the caller-authorized directory read/write at `/mounts/project` using protected grants and the non-root operator UID/GID.
Directory services within one operator's selected Tokimon runtime share one Machine Gateway and Tunnel connection.

| Command | Contract |
| --- | --- |
| `workspace setup --tunnel-id <id> --key-file <path> [--client-binary <path>]` | Store validated private file references for one machine connection; reject blank or unchanged template keys before downloading the client or saving the profile; download and verify the official client when omitted; emit one-time ChatGPT connection instructions. Refuse profile replacement while a workspace is running. |
| `workspace start [--directory <path>] [--local]` | Start or reuse the exact directory deployment; missing machine profile drafts the private key template and returns setup instructions before Docker mutation. Explicit `--local` allows local MCP without a profile and never claims Tunnel readiness. Repeated start preserves jobs and resources; stop/start refreshes protection/network snapshots. |
| `workspace stop [--directory <path> | --workspace-id <id>]` | Stop only that directory's verified deployment, retain data and profile, and refresh routes; stopping the final workspace stops the gateway. |
| `workspace status [--directory <path> | --workspace-id <id>]` | Report directory, exact service identities, local readiness, gateway readiness and separately unverified account acceptance. |
| `workspace list [--all]` | List actual running managed directory services by default; include stopped entries with `--all`. Never imply this includes unrelated host containers. |
| `workspace remove [--directory <path> | --workspace-id <id>] [--purge-data]` | Retire only the selected owned deployment, retaining data by default; explicit purge removes exact owned volumes, never host directory data or account resources. |

One controller lock covers registry changes, workspace lifecycle and complete immutable route publication within that runtime.
Acquire the controller lock before the deployment-local lock.
Registry/profile/route reads are bounded regular-file no-follow reads; malformed or excessive state fails explicitly.
The registry holds at most 256 directories.
The gateway exposes `executor_workspaces` for discovery without `workspace_id`.
The nine tools above require an explicit `workspace_id` when called through the gateway; there is no mutable current workspace or fallback route.
Gateway defaults are `/mounts/project` for command working directory and directory listing.
Workspace discovery reports IDs, directory labels and roots; unknown/stopped routes return explicit errors and job IDs remain scoped to their chosen worker.
Only recorded readonly socket volumes enter the gateway; it has no host project mounts, Docker API, or remote lifecycle tools.
Gateway recreation updates routes while worker jobs continue; failed publication restores the previous healthy gateway and reports the new workspace's remote unavailability.
Setup saves references without launching Docker resources; the first start creates the gateway and one Executor, and each additional directory adds only one Executor.
When a ChatGPT-accessible start lacks machine setup, the controller creates `.tokimon/tools/mcp/container-executor/secrets/tunnel.key` only if absent, with mode `0600` and one `REPLACE_WITH_YOUR_OPENAI_TUNNEL_RUNTIME_KEY` marker.
The returned `setup.recommended_key_file` names the file and `setup.key_file_created` reports whether this call drafted it; the operator only replaces the marker locally and supplies the Tunnel ID.
Repeated attempts preserve existing file contents and permissions; symlink or special-file targets and symlink parents fail without modifying their targets.
Neither setup output nor errors expose key contents; blank or unchanged template keys block setup and configured start before download, profile write or Docker mutation.
The gateway must be running before creating or testing the ChatGPT Tunnel connection; local readiness still does not prove account acceptance.
Restored stale routes may fail explicitly for a stopped worker and never restart it or select another workspace.
`local_ready`, `gateway_ready`, `tunnel_ready`, and `remote_acceptance` remain separate; actual loopback Tunnel `/readyz` is transport evidence only.
One-time owner-side Tunnel creation/key provisioning and ChatGPT connection selection are external prerequisites; subsequent directories reuse the stable connection.
The packaged `container-executor` skill routes natural-language start/stop/list through this local controller, preserving the caller directory and the existing three closed MCP package entries.

### Deployment Identity and Revisions

The manager serializes lifecycle mutations with a deployment-local lock.
Every destructive operation checks the recorded Docker ID and the matching `tokimon.capability`, `tokimon.deployment`, and role labels.
Container creation records an intent with a unique creation token, owner, role, name and image before calling Docker.
An uncertain CLI outcome is reconciled only against that exact intent and token; the discovered ID is recorded before lifecycle changes.
Subsequent stop, replacement or uninstall also reconciles a pending intent, so a late Docker response cannot leave an untracked owned container or cause adoption of an unrelated name.
Image builds use a minimal source bundle; repository runtime, owner keys, and arbitrary workspace content never enter the build context.
Revision/image identity binds source, validated configuration and the optional Tunnel Client binary.
Owner configuration and generated compose-compatible manifests stay in the host runtime; an explicitly granted ancestor may expose this non-secret data, while the key directory remains inaccessible.
Changing host-network policy re-creates the executor through its verified bootstrap; restart never resumes without reapplying policy.
Image/config rollback does not restore workspace or bind-mount data.
Local health, maintenance and stdio invoke only the immutable connector through Docker exec.
It drops all capabilities, supplementary groups and root identity using the configured worker UID/GID, sets `NoNewPrivs=1`, and verifies that state before processing MCP data; no caller-supplied command runs during bootstrap.
Worker revisions record the executor-only layout so an old image cannot accidentally run its startup bootstrap as a connector.
Running paired deployments require explicit stop before migration. Directory stop/start retires their exact recorded containers and reinstalls one Executor while retaining volumes and jobs.
Named paired deployments use stop, uninstall without purge, then install. A rollback to the retired layout is refused before replacing a healthy Executor; old revisions remain historical evidence.

## Data Ownership

- Deployment configuration, revision manifests, successful current/previous identities and lifecycle journal are owned under `.tokimon/tools/mcp/container-executor/`.
- Machine profile, bounded directory registry, gateway identities/revisions and routing snapshots are owned beneath its `workspaces/` directory; profile contains private references, never key values.
- Docker volumes own `/workspace`, `/state`, and `/run/tokimon`; the gateway sees registered socket volumes read-only.
- Job metadata and bounded stdout/stderr are private deployment data; runtime logs contain metadata only.
- Explicit host mounts and tunnel key files remain owner data and are never chowned, copied to images, or removed by uninstall.
- The directory controller emits ChatGPT connection instructions during setup; remote Platform/ChatGPT resources remain external.

## Failure Modes

- Missing/nonlocal Docker, unavailable nftables, failed firewall/capability setup, or unsafe configuration fails closed before tools become accessible.
- A startup/update failure preserves diagnosis and restores the prior deployment when one exists; restoration failure remains explicit.
- Unexpected restart converts persisted running jobs to `interrupted`; callers may explicitly start a new job.
- Changed resource identity or missing/mismatched ownership labels blocks mutation instead of adopting or deleting a replacement.
- Mount permission failures are returned as tool errors; the service never broadens permissions or repairs host ownership.
- Missing/mismatched Tunnel credentials may leave local MCP healthy while remote access fails; account-side acceptance remains separate.
- Disk exhaustion, corrupted persisted state, or a command that damages its own runtime produces a visible failure; stopping/restarting does not repair user data.
- Failed gateway refresh preserves local jobs and produces a partial-failure result with bounded diagnosis; local readiness never manufactures remote acceptance.
- Removing a workspace without `--purge-data` unregisters it and retains its volumes.
  Later cleanup has no public selector for that removed workspace; an exact owned stopped Gateway may need to release its socket-volume reference before a named deployment purge.

## Acceptance Checks

```bash
.venv/bin/python -m unittest discover -s tests/tools/mcp/container_executor -t . -p 'test_*.py'
.venv/bin/python -m tokimon.tools.mcp.container_executor health --json
```

Expected: focused tests pass; health reports the local prerequisites without Docker mutation.
Shared protocol tests in `tests/tools/mcp/test_protocol.py` cover both service ports, version negotiation, silent notifications, explicit request ids, and maintenance admission.
Wire tests in `tests/tools/mcp/test_wire_encoding.py` cover strict UTF-8 frames, tool/error content, byte limits, and continued request handling.

```text
With no machine profile, invoke workspace start and verify it drafts a private single-line key template, reports its path and creation status, and performs no Docker operations. Repeat with an existing credential and verify bytes and mode remain unchanged. Reject symlink/special targets and symlink parents without changing their targets. Try setup with the unchanged marker or a blank file: it must fail before client download, profile save or Docker mutation, without printing key contents. Fill a private canary key and verify setup still saves references only. Explicit local start must not draft credentials.
```

Expected: first-time setup requires only filling the drafted key and providing the Tunnel ID; existing credentials remain intact and unfinished templates never reach a service.

```text
Install an isolated test deployment using only non-sensitive canary mounts. Through public MCP, verify command success/failure, reconnectable jobs, bounded output, timeout/cancel, file read/write, read-only denial, symlink denial, and restart interruption. Verify actual command-process UID/capability masks, inability to alter network rules or access the Docker API/unmounted host canary, public TCP and UDP connectivity, and denial of host/private destinations. Verify update/rollback preserve data, failed update restores the old deployment, and identity replacement blocks cleanup. Uninstall only test resources and preserve receipts. Test ChatGPT and dot separately before claiming either account connection works.
```

Expected: all executed checks have direct evidence, no unrelated resource changes occur, and unavailable account/client acceptance is named explicitly.

```text
Start two canary directories through workspace commands, repeat start, and verify explicit workspace-ID routing with different same-named files. Start a long job, refresh gateway routes and prove the job survives. Verify concurrent starts and stop/start keep a complete route snapshot. Stop one directory and prove the other remains healthy; stop the last and prove the gateway stops. From both a Tokimon-root grant and its parent grant, verify ordinary Tokimon source and non-secret runtime files are writable while the exact key directory, declared private file and other sensitive directory masks cannot be read, modified or renamed through commands or file tools; verify protected host contents/modes are unchanged. Verify a custom key directory is protected without hiding its parent project; preserve the configured key path. Reject unsafe roots and symlink/special masking targets before Docker mutation. Generate and validate both platform skill packages, preserving caller cwd. Remove only exact owned canaries.
```

Expected: each directory has exactly one Executor, the selected Tokimon runtime has at most one Gateway, and directory identity, scoped lifecycle, mask isolation, bounded state, gateway restoration and package routing have direct evidence; account invocation remains separately verified.
Inject a Docker create timeout after resource creation and verify the exact intended container is recorded and cleaned; a name reused without the creation token must block reconciliation without deletion.

## References

- L2 Containers: `docs/c4/level-2-containers/tools/mcp/container-executor.md`
- Skill: `skills/container-executor/skill.md`
- L1 Context: `docs/c4/level-1-context/tools-context.md`
