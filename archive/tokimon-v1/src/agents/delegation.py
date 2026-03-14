"""Delegation graph and cycle detection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DelegationGraph:
    edges: dict[str, set[str]] = field(default_factory=dict)
    call_artifacts: dict[str, str] = field(default_factory=dict)
    subtree_signatures: set[str] = field(default_factory=set)

    def add_edge(self, parent: str, child: str) -> bool:
        self.edges.setdefault(parent, set()).add(child)
        return not self._has_cycle(parent)

    def _has_cycle(self, start: str) -> bool:
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in self.edges.get(node, set()):
                if dfs(child):
                    return True
            stack.remove(node)
            return False

        return dfs(start)

    def record_artifacts(self, call_signature: str, artifact_hash: str) -> bool:
        """Return True if repeated subtree detected."""
        subtree_sig = hashlib.sha256(f"{call_signature}:{artifact_hash}".encode()).hexdigest()
        if subtree_sig in self.subtree_signatures:
            return True
        self.subtree_signatures.add(subtree_sig)
        return False
