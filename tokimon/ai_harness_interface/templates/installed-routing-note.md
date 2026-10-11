## Tokimon Routing

Resolve Tokimon-owned command entrypoints under `{{liveCheckoutPath}}`; use `{{runtimeRoot}}` for runtime artifacts.
Treat the current repository as the task subject unless the user explicitly says otherwise.
When editing source files, use the current task checkout. Do not move source edits to `{{liveCheckoutPath}}` solely because command entrypoints or runtime artifacts resolve there.
