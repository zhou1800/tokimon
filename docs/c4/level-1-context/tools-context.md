---
title: "Tools Layer - Context"
summary: "Tokimon's local and MCP tools, including ChatGPT access to local directories."
read_when:
  - Adding, changing, or using local or MCP-backed tools.
  - Deciding whether execution behavior belongs in tools, workflows, or operations docs.
---

# Tools Layer - Context

## Overview & Mission

Tokimon provides local tools and MCP servers for AI agents.
Container Executor runs an MCP server in Docker, mounting a directory you select so ChatGPT can execute commands and read or write its files.
You use local Tokimon to start, stop, or list directory services; ChatGPT connects through a private OpenAI Tunnel.

## Actors

### User

Chooses the task and grants access to directories and accounts.

### Codex Session

Calls available tools and handles local service-management requests.

### Workflow

Declares required tools; the registry checks their availability before execution.

### MCP Client

Calls configured MCP tools directly; ChatGPT uses the Tunnel for directory access.

## Dependencies

### External Systems

- **Telegram Bot API (telegram-bot-api):** Sends messages.
- **Local AI Provider CLIs (ai-provider-cli):** Run AI Debate and remote-agent turns.
- **Docker Engine (docker):** Runs browser and directory MCP containers.
- **Trusted Remote Hosts (remote-host):** Run selected shell and AI CLI jobs.

### Platform Services

- **OpenAI Secure MCP Tunnel (openai-tunnel):** Connects ChatGPT to the directory MCP.

### Repository Components

- **Tool Registry (tool-registry):** Lists tools and checks prerequisites.

## Scope Boundaries

### In Scope

- Provide tools, report readiness, and manage tool jobs and services.

### Out of Scope

- Workflow orchestration, business rules, and cross-machine synchronization.

## Traceability
