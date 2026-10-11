"""Install-time route data for generated AI Harness Interface packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokimon.runtime import python_executable


class RouteError(ValueError):
    """Raised when install-time route data is invalid."""


@dataclass(frozen=True)
class InstallRoute:
    liveCheckoutPath: Path
    runtimeRoot: Path

    @property
    def python_executable(self) -> Path:
        return python_executable(self.liveCheckoutPath)

    def to_json_object(self) -> dict[str, str]:
        return {
            "liveCheckoutPath": str(self.liveCheckoutPath),
            "runtimeRoot": str(self.runtimeRoot),
        }


def build_route(*, live_checkout_path: Path, runtime_root: Path) -> InstallRoute:
    return InstallRoute(
        liveCheckoutPath=live_checkout_path.expanduser().resolve(strict=False),
        runtimeRoot=runtime_root.expanduser().resolve(strict=False),
    )


def validate_route(
    install_route: InstallRoute,
    *,
    create_runtime_root: bool = False,
) -> None:
    _validate_live_checkout(install_route.liveCheckoutPath)
    runtime_root = install_route.runtimeRoot
    if runtime_root.exists() and not runtime_root.is_dir():
        raise RouteError(f"runtimeRoot must be a directory: {runtime_root}")
    if not runtime_root.exists() and create_runtime_root:
        runtime_root.mkdir(parents=True, exist_ok=True)


def _validate_live_checkout(path: Path) -> None:
    if not path.is_dir():
        raise RouteError(f"liveCheckoutPath does not exist: {path}")
    required = (path / "tokimon" / "__init__.py", path / "skills")
    missing = [str(candidate.relative_to(path)) for candidate in required if not candidate.exists()]
    if missing:
        raise RouteError(
            "liveCheckoutPath does not look like a Tokimon checkout; "
            f"missing: {', '.join(missing)}"
        )
