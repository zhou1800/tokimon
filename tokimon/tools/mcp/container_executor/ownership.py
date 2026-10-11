"""Exact Docker identity and role labels shared by local lifecycle owners."""

import re
import subprocess
import uuid

from . import CAPABILITY_ID
from .config import fields
from .state import StateStore

LABEL_CAPABILITY = "tokimon.capability"
LABEL_DEPLOYMENT = "tokimon.deployment"
LABEL_ROLE = "tokimon.role"
LABEL_CREATION = "tokimon.creation"


class DockerCommandError(RuntimeError):
    """Docker CLI completed with a nonzero exit status."""


class MissingResource(DockerCommandError):
    """Docker confirmed that the inspected identity does not exist."""


class ContainerCreation:
    """Persist and reconcile one container creation under the caller's lock."""

    def __init__(self, root, docker, owner_id, role):
        self.store = StateStore(root)
        self.docker, self.owner_id, self.role = docker, owner_id, role

    def _intent(self):
        intent = self.store.read("container-create.json")
        if intent is None:
            return None
        fields(intent, {"schema_version", "token", "owner_id", "role", "name", "image", "id", "pending"})
        if (intent["schema_version"] != 1 or intent["owner_id"] != self.owner_id or intent["role"] != self.role
                or not isinstance(intent["token"], str) or not re.fullmatch(r"[0-9a-f]{32}", intent["token"])
                or not isinstance(intent["name"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", intent["name"])
                or not isinstance(intent["image"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", intent["image"])
                or type(intent["pending"]) is not bool
                or intent["id"] is not None and (not isinstance(intent["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", intent["id"]))):
            raise ValueError("invalid container creation intent")
        return intent

    def resolve(self):
        intent = self._intent()
        if intent is None:
            return None
        try:
            item = self.docker.inspect("container", intent["id"] or intent["name"])
        except MissingResource:
            if intent["pending"]:
                raise RuntimeError("container creation is unresolved; retry after Docker settles; intent retained") from None
            return None
        identity = item.get("Id")
        actual_labels = item.get("Config", {}).get("Labels", {})
        if (not identity or intent["id"] is not None and identity != intent["id"]
                or item.get("Name") != "/" + intent["name"] or item.get("Image") != intent["image"]
                or actual_labels.get(LABEL_CREATION) != intent["token"]
                or any(actual_labels.get(k) != v for k, v in labels(self.owner_id, self.role).items())):
            raise RuntimeError("container creation identity/token mismatch; refusing adoption or mutation")
        if intent["pending"] or intent["id"] != identity:
            intent.update(id=identity, pending=False)
            self.store.write("container-create.json", intent)
        return identity

    def create(self, args, name, image):
        if self.resolve() is not None:
            raise RuntimeError("previous container creation is still present; remove it before replacement")
        intent = {"schema_version": 1, "token": uuid.uuid4().hex, "owner_id": self.owner_id, "role": self.role,
                  "name": name, "image": image, "id": None, "pending": True}
        self.store.write("container-create.json", intent)
        try:
            identity = self.docker.run(*args[:2], "--label", LABEL_CREATION + "=" + intent["token"], *args[2:], timeout=120)
        except (DockerCommandError, OSError, subprocess.SubprocessError) as failure:
            # Nonzero CLI exits and timeouts can both follow a committed create.
            try:
                return self.resolve()
            except RuntimeError as unresolved:
                raise unresolved from failure
        intent.update(id=identity, pending=False)
        self.store.write("container-create.json", intent)
        return self.resolve()

    def clear(self):
        self.store.write("container-create.json", None)


def labels(owner_id: str, role: str) -> dict:
    return {LABEL_CAPABILITY: CAPABILITY_ID, LABEL_DEPLOYMENT: owner_id, LABEL_ROLE: role}


def label_args(owner_id: str, role: str) -> list[str]:
    return [arg for key, value in labels(owner_id, role).items() for arg in ("--label", f"{key}={value}")]


def owned(docker, owner_id: str, kind: str, role: str, identity: str) -> dict:
    item = docker.inspect(kind, identity)
    actual = item.get("Name") if kind == "volume" else item.get("Id")
    actual_labels = item.get("Config", {}).get("Labels", {}) if kind == "container" else item.get("Labels", {})
    if actual != identity or any(actual_labels.get(k) != v for k, v in labels(owner_id, role).items()):
        raise RuntimeError(f"{role} {kind} ownership/identity mismatch; refusing mutation")
    return item
