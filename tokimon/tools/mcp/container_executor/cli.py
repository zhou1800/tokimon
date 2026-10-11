"""Container Executor operator command boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .deployment import Deployment, download_tunnel_client, health
from .workspaces import WorkspaceController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage independent Docker MCP execution.")
    commands = parser.add_subparsers(dest="command", required=True)
    workspace = commands.add_parser("workspace").add_subparsers(dest="workspace_command", required=True)
    for action in ("setup", "start", "stop", "status", "list", "remove"):
        command = workspace.add_parser(action)
        command.add_argument("--json", action="store_true")
        if action == "setup":
            command.add_argument("--tunnel-id", required=True)
            command.add_argument("--key-file", type=Path, required=True)
            command.add_argument("--client-binary", type=Path)
        elif action != "list":
            selector = command.add_mutually_exclusive_group()
            selector.add_argument("--directory", type=Path)
            if action != "start":
                selector.add_argument("--workspace-id")
        if action == "start":
            command.add_argument("--local", action="store_true")
        if action == "list":
            command.add_argument("--all", action="store_true")
        if action == "remove":
            command.add_argument("--purge-data", action="store_true")
    for name in ("init", "install", "start", "stop", "status", "doctor", "update", "rollback", "uninstall", "serve", "health", "download-tunnel-client"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true")
        if name not in {"health", "download-tunnel-client"}:
            command.add_argument("--name", default="default")
        if name in {"install", "update"}:
            command.add_argument("--config", type=Path)
        if name == "uninstall":
            command.add_argument("--purge-data", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "workspace":
            controller = WorkspaceController()
            action = args.workspace_command
            if action == "setup":
                payload = controller.setup(args.tunnel_id, args.key_file, args.client_binary)
            elif action == "start":
                payload = controller.start(args.directory, local=args.local)
            elif action == "list":
                payload = controller.list(include_stopped=args.all)
            else:
                options = {"identity": args.workspace_id}
                if action == "remove":
                    options["purge_data"] = args.purge_data
                payload = getattr(controller, action)(args.directory, **options)
        elif args.command == "health":
            payload = health()
        elif args.command == "download-tunnel-client":
            payload = download_tunnel_client()
        else:
            deployment = Deployment(args.name)
            if args.command == "serve":
                return deployment.serve()
            if args.command in {"install", "update"}:
                payload = getattr(deployment, args.command)(args.config)
            elif args.command == "uninstall":
                payload = deployment.uninstall(purge_data=args.purge_data)
            else:
                payload = getattr(deployment, args.command)()
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        payload = {"ok": False, "error": str(exc)}
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output, file=sys.stderr if args.command == "serve" else sys.stdout)
    return 0 if payload.get("ok") else 1
