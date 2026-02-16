from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def _clear_webtool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TOKIMON_WEB_ORG_ALLOWLIST",
        "TOKIMON_WEB_REQUEST_ALLOWLIST",
        "TOKIMON_WEB_DOMAIN_SECRETS_JSON",
    ):
        monkeypatch.delenv(var, raising=False)
