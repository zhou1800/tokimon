"""Minimal immutable image bundles shared by worker and machine gateway lifecycles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import worker_config
from .jobs import atomic_json

SOURCE = Path(__file__).parent


def build_image(path: Path, config: dict, docker, *, gateway: dict | None = None) -> dict:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(json.dumps(gateway, sort_keys=True).encode())
    sources = {path.name: path.read_bytes() for path in SOURCE.iterdir() if path.suffix == ".py" or path.name == "Dockerfile"}
    for name, content in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + content)
    shared = {"_runtime.py": (SOURCE.parents[1] / "_runtime.py").read_bytes(),
              "_protocol.py": (SOURCE.parent / "_protocol.py").read_bytes()}
    for name, content in sorted(shared.items()):
        digest.update(name.encode() + b"\0" + content)
    binary = Path(config["tunnel"]["client_binary"]).read_bytes() if config["tunnel"] else None
    if binary:
        digest.update(binary)
    revision = digest.hexdigest()
    directory = path / "revisions" / revision
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "bundle"
    bundle.mkdir(exist_ok=True)
    package = bundle / "tokimon/tools/mcp/container_executor"
    package.mkdir(parents=True, exist_ok=True)
    for parent in (bundle / "tokimon", bundle / "tokimon/tools", bundle / "tokimon/tools/mcp"):
        (parent / "__init__.py").write_text("")
    for name, content in sources.items():
        (bundle / "Dockerfile" if name == "Dockerfile" else package / name).write_bytes(content)
    (bundle / "tokimon/tools/_runtime.py").write_bytes(shared["_runtime.py"])
    (bundle / "tokimon/tools/mcp/_protocol.py").write_bytes(shared["_protocol.py"])
    (bundle / "worker.json").write_text(json.dumps(worker_config(config)))
    (bundle / "policy.json").write_text(json.dumps({"blocked_networks": []}))
    (bundle / "tunnel.json").write_text(json.dumps({"tunnel_id": config["tunnel"]["tunnel_id"]} if config["tunnel"] else None))
    (bundle / "gateway.json").write_text(json.dumps(gateway))
    (bundle / "bin").mkdir(exist_ok=True)
    if binary:
        client = bundle / "bin/tunnel-client"
        client.write_bytes(binary)
        client.chmod(0o755)
    tag = f"tokimon-container-executor:{revision}"
    docker.run("build", "--tag", tag, "--build-arg", f"EXECUTOR_UID={config['uid']}", "--build-arg", f"EXECUTOR_GID={config['gid']}", str(bundle), timeout=1200)
    image = docker.inspect("image", tag)["Id"]
    result = {"revision": revision, "image": image, "config": config, "gateway": gateway}
    if gateway is None:
        result["worker_layout"] = 2
    atomic_json(directory / "revision.json", result)
    return result
