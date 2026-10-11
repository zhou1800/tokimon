"""Container-local network policy and irreversible bootstrap privilege revocation."""

from __future__ import annotations

import ctypes
import ipaddress
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from .logging import log_failure

NONPUBLIC = (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
    "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.88.99.0/24",
    "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24",
    "224.0.0.0/4", "240.0.0.0/4",
)
CAP_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")


def nft_policy(blocked: list[str]) -> str:
    networks = [ipaddress.ip_network(n, strict=True) for n in (*NONPUBLIC, *blocked)]
    ipv4 = sorted({str(n) for n in networks if n.version == 4})
    return '\n'.join([
        "table inet tokimon {",
        "chain output {", "type filter hook output priority -200; policy drop;",
        'oifname "lo" accept;',
        "meta nfproto ipv6 reject;",
        "ip daddr { " + ", ".join(ipv4) + " } reject;",
        "meta nfproto ipv4 accept;", "}",
        "chain input {", "type filter hook input priority -200; policy drop;",
        'iifname "lo" accept;', "ct state established,related accept;", "}",
        "chain forward {", "type filter hook forward priority -200; policy drop;", "}", "}",
    ]) + "\n"


def process_identity() -> dict:
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return {key: status[key].strip() for key in ("Uid", "Gid", "Groups", *CAP_FIELDS, "NoNewPrivs")}


def verify_unprivileged(identity: dict, uid: int, gid: int) -> None:
    if (identity["Uid"].split() != [str(uid)] * 4 or identity["Gid"].split() != [str(gid)] * 4
            or identity["Groups"] or any(int(identity[k], 16) for k in CAP_FIELDS)
            or identity["NoNewPrivs"] != "1"):
        raise RuntimeError("runtime identity/capability verification failed")


def prctl(option: int, arg: int = 0) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, arg, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "privilege revocation failed")


def no_new_privileges() -> None:
    prctl(38, 1)


def drop_privileges(config: dict) -> None:
    last = int(Path("/proc/sys/kernel/cap_last_cap").read_text())
    for cap in range(last + 1):
        prctl(24, cap)
    prctl(47, 4)
    os.setgroups([])
    os.setresgid(config["gid"], config["gid"], config["gid"])
    os.setresuid(config["uid"], config["uid"], config["uid"])
    no_new_privileges()
    verify_unprivileged(process_identity(), config["uid"], config["gid"])


def bootstrap() -> None:
    config = json.loads(Path("/etc/tokimon/worker.json").read_text())
    policy = json.loads(Path("/etc/tokimon/policy.json").read_text())
    subprocess.run(["nft", "-f", "-"], input=nft_policy(policy["blocked_networks"]), text=True,
                   check=True, capture_output=True, timeout=10)
    subprocess.run(["nft", "list", "table", "inet", "tokimon"], check=True, capture_output=True, timeout=10)
    drop_privileges(config)
    os.execv("/usr/bin/tini", ["tini", "-g", "--", "/usr/local/bin/python", "-I", "-m",
                              "tokimon.tools.mcp.container_executor.server"])


def connector(arguments: list[str]) -> None:
    if not arguments or arguments[0] not in {"request", "relay"}:
        raise ValueError("unknown fixed connector operation")
    config = json.loads(Path("/etc/tokimon/worker.json").read_text())
    drop_privileges(config)
    os.execv("/usr/local/bin/python", ["/usr/local/bin/python", "-I", "-m",
                                     "tokimon.tools.mcp.container_executor.transport", *arguments])


if __name__ == "__main__":
    correlation = uuid.uuid4().hex
    os.environ["TOKIMON_BOOT_ID"] = correlation
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] != "connect":
                raise ValueError("unknown isolation operation")
            connector(sys.argv[2:])
        else:
            bootstrap()
    except Exception as exc:
        log_failure("bootstrap", correlation, exc)
        raise SystemExit(1) from None
