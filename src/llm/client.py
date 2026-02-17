"""LLM client abstractions and mock adapters."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class LLMClient(Protocol):
    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class StubLLMClient:
    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Stub client not configured")


@dataclass
class MockLLMClient:
    script: list[dict[str, Any]]

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.script:
            return self.script.pop(0)
        return {
            "status": "PARTIAL",
            "summary": "mock response",
            "artifacts": [],
            "metrics": {"token_estimate": 0},
            "next_actions": [],
            "failure_signature": "mock-empty",
        }


class PlaceholderLLMClient:
    """Placeholder for plugging in a real client. Override send()."""

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Integrate your LLM client here.")


@dataclass(frozen=True)
class CodexCLISettings:
    cli_command: str = "codex"
    model: str | None = None
    profile: str | None = None
    sandbox: str = "read-only"
    ask_for_approval: str = "never"
    search: bool = False
    timeout_s: int = 900
    config: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_env() -> "CodexCLISettings":
        cli_command = os.environ.get("CODEX_CLI", "codex")
        model = os.environ.get("TOKIMON_CODEX_MODEL")
        profile = os.environ.get("TOKIMON_CODEX_PROFILE")
        sandbox = os.environ.get("TOKIMON_CODEX_SANDBOX", "read-only")
        ask_for_approval = os.environ.get("TOKIMON_CODEX_APPROVAL", "never")
        search_raw = os.environ.get("TOKIMON_CODEX_SEARCH", "")
        search = (search_raw or "").strip().lower() in {"1", "true", "yes", "on"}
        timeout_raw = os.environ.get("TOKIMON_CODEX_TIMEOUT_S", "900")
        try:
            timeout_s = int(timeout_raw)
        except ValueError:
            timeout_s = 900
        config = _load_json_env("TOKIMON_CODEX_CONFIG_JSON") or {}
        return CodexCLISettings(
            cli_command=cli_command,
            model=model,
            profile=profile,
            sandbox=sandbox,
            ask_for_approval=ask_for_approval,
            search=search,
            timeout_s=timeout_s,
            config=config,
        )


class CodexCLIClient:
    """LLMClient backed by Codex CLI (`codex exec`) structured output.

    This adapter shells out to Codex CLI and expects the agent's last message to be a JSON object.
    """

    def __init__(self, workspace_dir: Path, settings: CodexCLISettings | None = None) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.settings = settings or CodexCLISettings.from_env()

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Note: Codex CLI supports `--output-schema`, but its accepted JSON Schema subset
        # is stricter than standard Draft-07. For portability, we capture the last agent
        # message and parse JSON ourselves (mirrors the ai-agent-cli approach).
        tmp_root = _ensure_tmp_root(self.workspace_dir)
        env = os.environ.copy()
        if tmp_root is not None:
            env.update({"TMPDIR": str(tmp_root), "TEMP": str(tmp_root), "TMP": str(tmp_root)})
        env = _maybe_apply_codex_ripgrep_guard(env, tmp_root=tmp_root)
        env, delegation_depth = _mark_tokimon_delegated_env(env)
        prompt = _render_prompt(
            messages,
            tools=tools,
            preamble=_codex_cli_preamble(
                self.settings,
                self.workspace_dir,
                delegation_depth=delegation_depth,
            ),
        )

        tmp_kwargs: dict[str, Any] = {"prefix": "tokimon-codex-"}
        if tmp_root is not None:
            tmp_kwargs["dir"] = str(tmp_root)

        with tempfile.TemporaryDirectory(**tmp_kwargs) as tmpdir:
            tmp = Path(tmpdir)
            last_message_path = tmp / "last_message.txt"

            cmd = _build_codex_exec_command(
                self.settings,
                workspace_dir=self.workspace_dir,
                last_message_path=last_message_path,
            )
            try:
                completed = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    env=env,
                    cwd=str(self.workspace_dir),
                    timeout=self.settings.timeout_s,
                    check=False,
                )
            except FileNotFoundError as exc:
                return _llm_error(
                    "codex cli not found",
                    failure_signature="llm-codex-cli-missing",
                    details=str(exc),
                )
            except subprocess.TimeoutExpired:
                return _llm_error(
                    f"codex cli timed out after {self.settings.timeout_s}s",
                    failure_signature="llm-codex-timeout",
                )
            except Exception as exc:  # pragma: no cover
                return _llm_error(
                    "codex cli error",
                    failure_signature="llm-codex-exception",
                    details=str(exc),
                )

            raw_last = ""
            if last_message_path.exists():
                raw_last = last_message_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()

            if completed.returncode != 0 and not raw_last:
                details = _truncate(completed.stderr or completed.stdout, 2000)
                reason = _first_nonempty_line(details)
                summary = f"codex cli exited {completed.returncode}"
                if reason:
                    summary = f"{summary}: {reason}"
                return _llm_error(
                    summary,
                    failure_signature="llm-codex-nonzero-exit",
                    details=details,
                )

            candidate = raw_last or completed.stdout.strip()
            json_text = _extract_json_text(candidate)
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError as exc:
                return _llm_error(
                    f"codex returned invalid JSON: {exc}",
                    failure_signature="llm-codex-invalid-json",
                    details=_truncate(candidate, 2000),
                )
            if not isinstance(payload, dict):
                return _llm_error(
                    "codex returned non-object JSON",
                    failure_signature="llm-codex-non-object",
                    details=_truncate(json.dumps(payload), 2000),
                )
            return payload


def build_llm_client(provider: str, *, workspace_dir: Path) -> LLMClient:
    """Factory for LLMClient implementations.

    Supported providers:
    - mock: deterministic scripted client (default)
    - codex: Codex CLI-backed client
    """

    normalized = (provider or "").strip().lower()
    if normalized in {"", "mock"}:
        return MockLLMClient(script=[])
    if normalized in {"codex", "codex-cli"}:
        return CodexCLIClient(workspace_dir)
    raise ValueError(f"Unknown LLM provider: {provider}")


def _build_codex_exec_command(
    settings: CodexCLISettings,
    *,
    workspace_dir: Path,
    last_message_path: Path,
) -> list[str]:
    cmd = shlex.split(settings.cli_command) or ["codex"]
    if settings.search:
        cmd.append("--search")
    cmd.extend(["--sandbox", settings.sandbox])
    cmd.extend(["--ask-for-approval", settings.ask_for_approval])
    cmd.extend(["--cd", str(workspace_dir)])
    if settings.model:
        cmd.extend(["--model", settings.model])
    if settings.profile:
        cmd.extend(["--profile", settings.profile])
    for key, value in (settings.config or {}).items():
        cmd.extend(["--config", f"{key}={json.dumps(value)}"])

    cmd.extend(
        [
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            str(last_message_path),
            "-",
        ]
    )
    return cmd


def _render_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    preamble: str | None = None,
) -> str:
    lines: list[str] = []
    if preamble:
        lines.append(preamble.strip())
        lines.append("")

    lines.extend(
        [
            "You are the language model driving an agent workflow runner.",
            "",
            "Output contract:",
            "- Reply with exactly one JSON object and nothing else (no markdown).",
            "- If you need to call tools, reply with: {\"tool_calls\": [{\"tool\": \"...\", \"action\": \"...\", \"args\": {...}}]}",
            "- Tool call args must be a JSON object with keys matching the action parameter names.",
            "- Otherwise, reply with a final object that includes at least: {\"status\": \"SUCCESS|FAILURE|BLOCKED|PARTIAL\", \"summary\": \"...\"}",
            "",
        ]
    )

    if tools:
        lines.append("<tools>")
        lines.append("Request tools via tool_calls JSON; args keys must match the action parameter names.")
        for tool in sorted(tools, key=lambda t: str(t.get("name", ""))):
            name = str(tool.get("name", ""))
            actions = tool.get("actions", [])
            signatures = tool.get("signatures")
            if not name:
                continue
            if isinstance(signatures, dict) and isinstance(actions, list) and actions:
                lines.append(f"- {name}:")
                for action in sorted(str(a) for a in actions):
                    rendered = (
                        signatures.get(action)
                        if isinstance(signatures.get(action), str)
                        else None
                    )
                    lines.append(f"  - {rendered or action}")
                continue
            if isinstance(actions, list):
                actions_text = ", ".join(str(action) for action in actions)
            else:
                actions_text = str(actions)
            lines.append(f"- {name}: {actions_text}")
        lines.append("</tools>")
        lines.append("")

    lines.append("<conversation>")
    for msg in messages:
        role = str(msg.get("role", ""))
        content = msg.get("content", "")
        name = msg.get("name")
        prefix = role.upper() or "MESSAGE"
        if name:
            prefix = f"{prefix}({name})"
        lines.append(f"{prefix}: {content}")
    lines.append("</conversation>")
    lines.append("")
    return "\n".join(lines)


def _codex_cli_preamble(
    settings: CodexCLISettings,
    workspace_dir: Path,
    *,
    delegation_depth: int,
) -> str:
    shell = os.environ.get("SHELL", "")
    shell_name = Path(shell).name if shell else ""
    permissions = [
        "<permissions instructions>",
        f"sandbox_mode: {settings.sandbox}",
        f"approval_policy: {settings.ask_for_approval}",
        f"search_enabled: {settings.search}",
        "</permissions instructions>",
    ]
    env = [
        "<environment_context>",
        f"  <cwd>{workspace_dir}</cwd>",
        f"  <shell>{shell_name}</shell>",
        "</environment_context>",
    ]
    tokimon_context = [
        "<tokimon_context>",
        "  <delegated>true</delegated>",
        f"  <delegation_depth>{delegation_depth}</delegation_depth>",
        "</tokimon_context>",
    ]
    return "\n".join([*permissions, "", *env, "", *tokimon_context]).strip()


def _extract_json_text(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return "{}"
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    if candidate.startswith("```json"):
        candidate = candidate[len("```json") :].strip()
    if candidate.endswith("```"):
        candidate = candidate[: -len("```")].strip()
    return candidate


def _llm_error(
    summary: str,
    *,
    failure_signature: str,
    details: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "FAILURE",
        "summary": summary,
        "artifacts": [],
        "metrics": {},
        "next_actions": [],
        "failure_signature": failure_signature,
    }
    if details:
        payload["metrics"]["details"] = details
    return payload


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(truncated {len(text) - limit} chars)"


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _ensure_tmp_root(workspace_dir: Path) -> Path | None:
    try:
        tmp_root = (workspace_dir / ".tokimon-tmp").resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        return tmp_root
    except Exception:
        return None


_DEFAULT_CODEX_RIPGREP_MAX_COLUMNS = 4096

_CODEX_RIPGREP_GUARD_GLOBS: tuple[str, ...] = (
    "--glob=!**/runs/**",
    "--glob=!**/.tokimon-tmp/**",
    "--glob=!**/.venv/**",
    "--glob=!**/node_modules/**",
    "--glob=!**/dist/**",
    "--glob=!**/build/**",
    "--glob=!**/*.jsonl",
    "--glob=!**/*.ndjson",
)


def _parse_env_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_env_int(raw: str | None, *, default: int) -> int:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        return int(normalized)
    except ValueError:
        return default


def _mark_tokimon_delegated_env(env: dict[str, str]) -> tuple[dict[str, str], int]:
    prior_depth = _parse_env_int(env.get("TOKIMON_DELEGATION_DEPTH"), default=0)
    if prior_depth < 0:
        prior_depth = 0
    delegation_depth = prior_depth + 1
    env["TOKIMON_DELEGATED"] = "1"
    env["TOKIMON_DELEGATION_DEPTH"] = str(delegation_depth)
    return env, delegation_depth


def _maybe_apply_codex_ripgrep_guard(
    env: dict[str, str],
    *,
    tmp_root: Path | None,
) -> dict[str, str]:
    guard_enabled = _parse_env_bool(
        env.get("TOKIMON_CODEX_RIPGREP_GUARD"),
        default=True,
    )
    if not guard_enabled:
        return env
    if tmp_root is None:
        return env

    guard_path = tmp_root / "tokimon-codex.ripgreprc"
    original_env = env.copy()

    try:
        base_text = ""
        base_config_path = (env.get("RIPGREP_CONFIG_PATH") or "").strip()
        if base_config_path:
            base_path = Path(base_config_path)
            if base_path.is_file() and os.access(base_path, os.R_OK):
                base_text = base_path.read_text(encoding="utf-8", errors="replace")
                if base_text and not base_text.endswith("\n"):
                    base_text += "\n"

        max_columns = _parse_env_int(
            env.get("TOKIMON_CODEX_RIPGREP_MAX_COLUMNS"),
            default=_DEFAULT_CODEX_RIPGREP_MAX_COLUMNS,
        )
        if max_columns < 0:
            max_columns = _DEFAULT_CODEX_RIPGREP_MAX_COLUMNS

        guard_lines: list[str] = []
        if max_columns > 0:
            guard_lines.append(f"--max-columns={max_columns}")
            guard_lines.append("--max-columns-preview")
        guard_lines.extend(_CODEX_RIPGREP_GUARD_GLOBS)

        guard_text = "\n".join(guard_lines) + "\n"
        guard_path.write_text(base_text + guard_text, encoding="utf-8")

        env["RIPGREP_CONFIG_PATH"] = str(guard_path)
        return env
    except Exception:
        return original_env


def _load_json_env(var_name: str) -> dict[str, Any] | None:
    raw = os.environ.get(var_name)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
