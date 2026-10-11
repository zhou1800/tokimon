---
name: container-executor
description: Start a ChatGPT-accessible Docker MCP for the current directory, stop that directory service, or list Tokimon managed running directory services.
---

# Container Executor

For a directory service, use the directory named in the request, or the caller's current working directory when the user requests it.
If no directory is specified for that service, ask which directory to mount before starting.
Credential setup alone does not authorize a mount.
Do not change to the Tokimon installation directory.
Starting a directory service grants ordinary files there read/write access in an isolated executor at `/mounts/project`.
Known sensitive children and the exact Tunnel key directory are protected by inaccessible readonly masks; a Tokimon checkout or parent grant keeps ordinary source and non-secret runtime files accessible.
Newly created sensitive paths require stop/start to refresh protection.
The supported host is local rootful Linux Docker with public IPv4 internet access for all ordinary protocols.
Executor UID/GID must already have the required host directory access; never change host ownership or permissions.
Stop/start also refreshes network policy after host addresses or routes change.

## Commands

These commands use Tokimon's trusted import root and Python while preserving the caller directory:

```bash
{{tokimonCommand}} -m tokimon.tools.mcp.container_executor workspace start --json
{{tokimonCommand}} -m tokimon.tools.mcp.container_executor workspace stop --json
{{tokimonCommand}} -m tokimon.tools.mcp.container_executor workspace list --json
```

Start is idempotent for the same canonical directory and preserves jobs.
Stop affects only that directory and retains data; stopping the final directory also stops the machine gateway.
List shows actual running Tokimon managed directory services; use `--all` to include stopped entries.
For a selected entry, `status`, `stop` and `remove` accept `--workspace-id <id>`; directory selection accepts `--directory <absolute-path>`.
Removal retains data unless the user requests `--purge-data`; never delete the host directory.
If the user explicitly asks about every host Docker container, inspect the host inventory read-only and identify Tokimon managed entries separately.

## First Connection

Directory services within one operator's selected Tokimon runtime share one Gateway, OpenAI Tunnel and ChatGPT custom MCP connection.
Setup saves configuration only. First start launches the shared Gateway and one Executor; later directories add only an Executor.
If start reports missing setup, it drafts the recommended private key file when absent and returns its path and `key_file_created` status.
Link that file for the owner to replace the marker locally; request only the Tunnel ID and confirmation that the file is filled, never key contents.
Existing files and permissions are preserved; offer an alternate private key-file path only when the owner already has one.
The owner creates/selects an OpenAI Tunnel associated with the intended ChatGPT workspace and a matching runtime key with Tunnels Read + Use.
The key must be a private regular file owned/readable by the non-root operator.
Use the recommended protected Tokimon runtime file `.tokimon/tools/mcp/container-executor/secrets/tunnel.key` with mode `0600`, or a private file outside project grants.
Preserve an existing key's configured location; confirm any proposed relocation with the owner before moving it.

```bash
{{tokimonCommand}} -m tokimon.tools.mcp.container_executor workspace setup --tunnel-id '<id>' --key-file '/absolute/private/key-file' --json
```

Setup downloads/verifies the official Tunnel Client unless `--client-binary <absolute-path>` is supplied.
Setup rejects a blank key file or one containing only the template marker before downloading the client or saving references.
After the directory is authorized, start its service and the shared Gateway.
Then follow the returned instructions to create a private ChatGPT MCP connection with Connection = Tunnel and that ID; subsequent starts reuse it.
Account actions require their own explicit user authorization and verified account context.
For a directory service, use `start --local` only when local operation is requested; never silently substitute it for a ChatGPT-accessible service.
Official Tunnel setup: https://developers.openai.com/api/docs/guides/secure-mcp-tunnels .

## Named Deployments

Use named deployment commands for explicit standalone local deployment or maintenance requests.
Reuse the trusted launcher above; select the recorded deployment name. Directory services use their workspace ID as that name.
`health --json` checks local prerequisites without starting a service.
For a new standalone deployment, `init --name <name>` returns its configuration path without granting host mounts.
Edit that configuration outside the executor; keep `mounts` empty for container-only work or grant narrow directories:

```json
{"source":"/absolute/input-directory","alias":"input","read_only":true}
```

Choose `read_only:false` only when the service may change those host files.
Add externally published host/NAT addresses to `blocked_networks`; worker configuration keeps `tunnel: null`.
`install --name <name>` starts and verifies the deployment; `doctor --name <name> --json` checks identity and isolation.
`serve --name <name>` connects stdio MCP to the running executor. Standalone installation does not register a ChatGPT Gateway route.
Docker restart policy keeps a running installation online after reboot; the executor reapplies its firewall before serving requests.
After configuration or source changes, use `update --name <name>`; use `rollback --name <name>` for the prior successful revision.
Both refuse active jobs; wait or cancel those jobs, or explicitly stop the deployment first.
Update and rollback preserve volume and mounted host data; image rollback does not restore file changes.

## Results and Recovery

Report directory, workspace ID, local readiness, gateway readiness and Tunnel readiness from command output.
Treat `remote_acceptance: unverified` honestly until actual ChatGPT tool results prove connection use; a running container or Tunnel `/readyz` is insufficient.
In ChatGPT, call `executor_workspaces` and include its selected `workspace_id` in every command, file and job operation.
Verify remote access with `executor_health` and a canary command against local deployment evidence; test dot separately.
Jobs remain in that workspace across connection loss; refreshing gateway routes does not stop workers.
Stopping a container clears remaining processes, including descendants outside a job process group.
Use `workspace status` for directory diagnosis; `doctor --name <name> --json` inspects its recorded deployment.
Removing a workspace without purge unregisters its ID; `workspace remove` cannot select that retained data again.
If an exact owned stopped Gateway still mounts the retained socket volume, release that reference before purging the volume.
For a retained named deployment record, `uninstall --name <name> --purge-data` remains available with exact ownership checks.
Uninstall retains data and revision history without `--purge-data`; it never deletes host directories or remote connectors, tunnels or API keys.
Persistent disk quotas and separate backups belong to the operator's filesystem.
If start reports an older paired service, require explicit stop before migration; subsequent start replaces its exact owned containers while retaining data.
For an older named paired deployment, stop, uninstall without purge, then install; move per-deployment Tunnel settings to workspace setup.
Retired paired revisions remain historical evidence and cannot be started or rolled back.
If a gateway refresh fails, report the partial failure and preserve local jobs; use status for diagnosis before retrying start.
Refuse unsafe mounts or ownership mismatches; never broaden Docker privileges, repair host permissions, expose key values, or stop unrelated resources.
