"""Authoritative static declarations for Tokimon MCP capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_HEALTH_ARGS = ("health", "--json")


def _python_command(module: str, args: tuple[str, ...]) -> str:
    return " ".join((".venv/bin/python", "-m", module, *args))


@dataclass(frozen=True)
class ConnectorMetadata:
    """Static presentation metadata used when a capability is connector-backed."""

    description: str
    artifact_policy: str
    failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class McpCatalogEntry:
    """Complete static metadata and package routing for one MCP capability."""

    capability_id: str
    server_id: str
    name: str
    kind: str
    module: str
    serve_args: tuple[str, ...]
    readiness_probe: str
    c4_references: tuple[str, ...]
    description: str
    command_args: tuple[tuple[str, ...], ...]
    command_notes: tuple[str, ...]
    host_requirements: tuple[str, ...]
    artifact_policy: str
    failure_modes: tuple[str, ...]
    connector: ConnectorMetadata
    health_args: tuple[str, ...] = DEFAULT_HEALTH_ARGS
    package_included: bool = False

    @property
    def implementation(self) -> str:
        return f"{self.module.replace('.', '/')}/"

    @property
    def implementation_entrypoint(self) -> Path:
        return Path(*self.module.split(".")) / "__main__.py"

    @property
    def docs(self) -> str:
        return self.c4_references[0]

    @property
    def commands(self) -> tuple[str, ...]:
        return (
            *(_python_command(self.module, args) for args in self.command_args),
            *self.command_notes,
        )

    @property
    def health_check(self) -> str:
        return _python_command(self.module, self.health_args)


MCP_CATALOG: tuple[McpCatalogEntry, ...] = (
    McpCatalogEntry(
        capability_id="mcp.telegram-bot",
        server_id="tokimon-telegram-bot",
        name="Telegram Bot MCP",
        kind="mcp",
        module="tokimon.tools.mcp.telegram_bot",
        serve_args=("serve",),
        readiness_probe="telegram-token",
        c4_references=(
            "docs/c4/level-3-components/tools/mcp/telegram-bot.md",
            "docs/c4/level-2-containers/tools/mcp/telegram-bot.md",
        ),
        description="Serve send-only Telegram text and document tools through stdio MCP.",
        command_args=(
            ("serve",),
            DEFAULT_HEALTH_ARGS,
        ),
        command_notes=(
            "MCP tools/list exposes telegram_bot_health, telegram_bot_get_me, "
            "telegram_bot_get_chat, telegram_bot_send_message, telegram_bot_send_document, "
            "and telegram_bot_log_tail.",
        ),
        host_requirements=(
            "Python 3.12 or compatible stdlib runtime",
            "Network access to Telegram Bot API",
            "Bot token in TELEGRAM_BOT_TOKEN or .tokimon/secrets.local.json telegram.bot_token",
            "Target chat id provided by MCP tool arguments, allowed_chats, or default_chat_id",
        ),
        artifact_policy=(
            "Reads local secrets from .tokimon/secrets.local.json; writes sanitized send logs "
            "under .tokimon/tools/mcp/telegram-bot/ without tokens or message bodies."
        ),
        failure_modes=(
            "Telegram bot token missing or invalid",
            "Target chat id missing, unknown, forbidden, or not joined by the bot",
            "Telegram API timeout or non-OK response",
            "Telegram rate limit returns retry_after without automatic retry",
            "Message exceeds Telegram text length limits",
            "Document path is missing, not a file, or rejected by Telegram",
        ),
        connector=ConnectorMetadata(
            description=(
                "Send Telegram text messages and document attachments through a configured "
                "MCP connector."
            ),
            artifact_policy=(
                "Connector-owned runtime; Tokimon-owned Telegram send logs belong under "
                ".tokimon/tools/mcp/telegram-bot/ when the local server is used."
            ),
            failure_modes=(
                "Connector becomes unavailable",
                "Connector tool list changes",
                "Provider-side MCP call fails or times out",
                "Telegram bot token or destination is unavailable in the connector runtime",
            ),
        ),
        package_included=True,
    ),
    McpCatalogEntry(
        capability_id="mcp.ai-debate",
        server_id="tokimon-ai-debate",
        name="AI Debate MCP",
        kind="mcp",
        module="tokimon.tools.mcp.ai_debate",
        serve_args=("serve",),
        readiness_probe="ai-debate",
        c4_references=(
            "docs/c4/level-3-components/tools/mcp/ai-debate.md",
            "docs/c4/level-2-containers/tools/mcp/ai-debate.md",
        ),
        description=(
            "Run persistent adversarial multi-round AI debates with opposing roles, "
            "explicit stop signals, and a final judge."
        ),
        command_args=(
            ("serve",),
            DEFAULT_HEALTH_ARGS,
            ("run", "--topic", "<text>", "--side-a", "<text>", "--side-b", "<text>"),
            ("resume", "<run_dir>"),
            ("status", "<run_dir>", "--json"),
            ("turn-start", "<run_dir>", "--json"),
            ("turn-status", "<run_dir>", "--json"),
            ("turn-finalize", "<run_dir>", "--json"),
            ("read-log", "<run_dir>"),
        ),
        command_notes=(
            "MCP tools/list exposes ai_debate_health, ai_debate_run, ai_debate_start, "
            "ai_debate_resume, ai_debate_resume_start, ai_debate_turn_start, "
            "ai_debate_turn_status, ai_debate_turn_finalize, ai_debate_status, and "
            "ai_debate_read_log.",
        ),
        host_requirements=(
            "Python 3.12 or compatible stdlib runtime",
            ".tokimon/settings.local.json configured with ai_debate and provider agent settings",
            "Codex CLI available through CODEX_CLI or PATH",
            "Selected side-B provider CLI available through CLAUDE_CODE_CLI or PATH when needed",
            "Provider CLIs already authenticated by the user",
        ),
        artifact_policy=(
            "Local-only debate state, transcripts, logs, and final judgments are written under "
            ".tokimon/tools/mcp/ai-debate/ by default, or under a caller-selected output root "
            "accepted by the runtime path boundary."
        ),
        failure_modes=(
            "Codex CLI missing or not authenticated",
            "Selected side-B provider CLI missing or not authenticated",
            "Required .tokimon/settings.local.json AI Debate or provider settings are missing",
            "Provider CLI idle timeout or non-zero exit",
            "Provider response is not valid structured JSON",
            "Run state or transcript missing during resume, status, or read operations",
        ),
        connector=ConnectorMetadata(
            description=(
                "Run persistent adversarial multi-round AI debates through a configured MCP "
                "connector."
            ),
            artifact_policy=(
                "Connector-owned runtime; when the local server is used, Tokimon-owned AI "
                "Debate state defaults to .tokimon/tools/mcp/ai-debate/ and may use a "
                "caller-selected output root accepted by the runtime path boundary."
            ),
            failure_modes=(
                "Connector becomes unavailable",
                "Connector tool list changes",
                "Provider-side MCP call fails or times out",
                "Run state or transcript is missing during resume, status, or read operations",
            ),
        ),
        package_included=True,
    ),
    McpCatalogEntry(
        capability_id="mcp.remote-browser-broker",
        server_id="tokimon-remote-browser-broker",
        name="Remote Browser Broker MCP",
        kind="mcp",
        module="tokimon.tools.mcp.remote_browser_broker",
        serve_args=("serve",),
        readiness_probe="docker",
        c4_references=(
            "docs/c4/level-3-components/tools/mcp/remote-browser-broker.md",
            "docs/c4/level-2-containers/tools/mcp/remote-browser-broker.md",
        ),
        description=(
            "Run an HTTP MCP broker that gives each MCP session one isolated visible browser "
            "runtime and can clone broker-local profile templates."
        ),
        command_args=(
            ("serve-http", "--host", "127.0.0.1"),
            DEFAULT_HEALTH_ARGS,
            ("build-image",),
        ),
        command_notes=(),
        host_requirements=(
            "Linux or WSL host capable of running Docker containers",
            "Docker CLI available through TOKIMON_RB_DOCKER or PATH",
            "Browser session ports available in the configured dynamic ranges",
            "Optional TOKIMON_RB_SSH_HOST set when clients need SSH tunnel guidance",
        ),
        artifact_policy=(
            "Local-only broker state, profiles, and session metadata are written under "
            ".tokimon/tools/mcp/remote-browser/."
        ),
        failure_modes=(
            "Docker missing or not permitted for the current user",
            "Docker image build fails",
            "No available port in the configured CDP or noVNC range",
            "MCP client does not preserve the mcp-session-id header",
            "Browser CDP or noVNC endpoint fails to become ready",
            "Browser session container exits unexpectedly",
        ),
        connector=ConnectorMetadata(
            description=(
                "Start and operate isolated visible browser runtimes and broker-local profile "
                "templates through a configured MCP connector."
            ),
            artifact_policy=(
                "Connector-owned runtime; Tokimon-owned browser broker state belongs under "
                ".tokimon/tools/mcp/remote-browser/ when the local broker is used."
            ),
            failure_modes=(
                "Connector becomes unavailable",
                "Connector tool list changes",
                "Provider-side MCP call fails or times out",
                "Browser session container fails after allocation",
            ),
        ),
        package_included=True,
    ),
    McpCatalogEntry(
        capability_id="mcp.remote-agent-broker",
        server_id="tokimon-remote-agent-broker",
        name="Remote Agent Broker MCP",
        kind="mcp",
        module="tokimon.tools.mcp.remote_agent_broker",
        serve_args=("serve",),
        readiness_probe="implementation",
        c4_references=(
            "docs/c4/level-3-components/tools/mcp/remote-agent-broker.md",
            "docs/c4/level-2-containers/tools/mcp/remote-agent-broker.md",
        ),
        description=(
            "Run an HTTP MCP broker for trusted remote shell, Copilot CLI, and Claude Code jobs."
        ),
        command_args=(
            ("serve-http", "--host", "127.0.0.1", "--port", "8938"),
            DEFAULT_HEALTH_ARGS,
            ("cleanup", "--ttl-days", "7"),
        ),
        command_notes=(),
        host_requirements=(
            "Trusted local or remote user account",
            "Optional Copilot CLI available through COPILOT_CLI or PATH for Copilot jobs",
            "Optional Claude Code CLI available through CLAUDE_CODE_CLI or PATH for Claude jobs",
            "Selected provider CLIs already authenticated by the user",
            "Optional TOKIMON_RA_SSH_HOST set when clients need SSH tunnel guidance",
        ),
        artifact_policy=(
            "Local-only job state, logs, transcripts, final outputs, and diffs are written under "
            ".tokimon/tools/mcp/remote-agent/."
        ),
        failure_modes=(
            "Copilot CLI missing or not authenticated",
            "Claude Code CLI missing or not authenticated",
            "Required .tokimon/settings.local.json Claude settings are missing for Claude jobs",
            "Broker loopback port is already in use",
            "Remote job process exits non-zero, times out, or is lost after broker restart",
            "Job artifact writes fail because runtime storage is unavailable",
        ),
        connector=ConnectorMetadata(
            description=(
                "Run trusted remote shell, Copilot CLI, and Claude Code jobs through a "
                "configured MCP connector."
            ),
            artifact_policy=(
                "Connector-owned runtime; Tokimon-owned remote agent broker state belongs under "
                ".tokimon/tools/mcp/remote-agent/ when the local broker is used."
            ),
            failure_modes=(
                "Connector becomes unavailable",
                "Connector tool list changes",
                "Provider-side MCP call fails or times out",
                "Remote job process exits non-zero, times out, or is lost after broker restart",
            ),
        ),
    ),
    McpCatalogEntry(
        capability_id="mcp.container-executor",
        server_id="tokimon-container-executor",
        name="Container Executor MCP",
        kind="mcp",
        module="tokimon.tools.mcp.container_executor",
        serve_args=("serve", "--name", "default"),
        readiness_probe="container-executor-docker",
        c4_references=(
            "docs/c4/level-3-components/tools/mcp/container-executor.md",
            "docs/c4/level-2-containers/tools/mcp/container-executor.md",
        ),
        description="Execute independent commands and bounded files in an explicitly mounted Docker workspace.",
        command_args=(("init", "--name", "default"), ("install", "--name", "default"), ("serve", "--name", "default"), DEFAULT_HEALTH_ARGS, ("doctor", "--name", "default", "--json")),
        command_notes=("Optional deployment; private ChatGPT Tunnel connection uses Container Executor workspace setup.",),
        host_requirements=("Local rootful Linux Docker", "Kernel nftables and capability revocation", "Explicit deployment configuration", "Optional official Tunnel Client and shared machine Gateway-only runtime key"),
        artifact_policy="Deployment revisions and journal under .tokimon/tools/mcp/container-executor/; bounded jobs in owned Docker volumes; host access only explicit mounts.",
        failure_modes=("Unsafe configuration or unsupported Docker context", "Isolation/bootstrap verification failure", "Resource identity mismatch", "Job timeout/interruption or storage failure", "Optional tunnel account permissions unavailable"),
        connector=ConnectorMetadata(
            description="Use independent command/file tools through an authorized Container Executor MCP connection.",
            artifact_policy="Executor-owned workspace/job data; connection setup returns private file references, never key values.",
            failure_modes=("Connector unavailable", "Container stopped or restarted", "Job output/state unavailable"),
        ),
    ),
)


def package_entries() -> tuple[McpCatalogEntry, ...]:
    """Return the ordered, closed MCP set included in generated packages."""

    return tuple(entry for entry in MCP_CATALOG if entry.package_included)
