FROM python:3.12-slim@sha256:d67a7b66b989ad6b6d6b10d428dcc5e0bfc3e5f88906e67d490c4d3daac57047
RUN apt-get update && apt-get install -y --no-install-recommends nftables tini bash git openssh-client curl ca-certificates iproute2 iputils-ping && rm -rf /var/lib/apt/lists/*
ARG EXECUTOR_UID=1000
ARG EXECUTOR_GID=1000
RUN mkdir -p /workspace /state/jobs /run/tokimon /mounts /etc/tokimon && chown ${EXECUTOR_UID}:${EXECUTOR_GID} /workspace /state /state/jobs /run/tokimon
COPY tokimon /usr/local/lib/python3.12/site-packages/tokimon
COPY worker.json policy.json tunnel.json gateway.json /etc/tokimon/
COPY bin/ /usr/local/bin/
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/python", "-I", "-m", "tokimon.tools.mcp.container_executor.isolation"]
