"""Registry of campaign executor adapters keyed by executor kind."""

from __future__ import annotations

import importlib.metadata as metadata
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from bashgym.campaigns.contracts import ActionAttempt, StageKind

ENTRY_POINT_GROUP = "bashgym.campaign_executors"


@runtime_checkable
class ExecutorAdapter(Protocol):
    kind: str
    allowed_stages: frozenset[StageKind]
    reuses_completed_results: bool

    def tick(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str: ...

    def reconcile(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str | None: ...

    def repair_allowed(self) -> bool:
        """Gate recovery repair, local-seal completion in reconcile, and expiry ownership."""
        ...


class ExecutorRegistry:
    """Immutable-after-freeze mapping from executor kind to adapter."""

    def __init__(self) -> None:
        self._adapters: dict[str, ExecutorAdapter] = {}
        self._frozen = False

    def register(self, adapter: ExecutorAdapter) -> None:
        if isinstance(adapter, type) or not isinstance(adapter, ExecutorAdapter):
            raise TypeError(
                f"executor adapter for kind {getattr(adapter, 'kind', '?')!r} does not implement ExecutorAdapter"
            )
        if self._frozen:
            raise RuntimeError("executor registry is frozen")
        if adapter.kind in self._adapters:
            raise ValueError(f"executor kind already registered: {adapter.kind}")
        self._adapters[adapter.kind] = adapter

    def freeze(self) -> None:
        self._frozen = True

    def get(self, kind: str) -> ExecutorAdapter:
        return self._adapters[kind]

    def is_registered(self, kind: object) -> bool:
        return kind in self._adapters

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def allowed_stages(self, kind: str) -> frozenset[StageKind]:
        return self._adapters[kind].allowed_stages


def discover_entry_points(group: str = ENTRY_POINT_GROUP) -> tuple[ExecutorAdapter, ...]:
    """Load third-party adapters; a missing group yields no adapters."""

    adapters: list[ExecutorAdapter] = []
    for entry_point in metadata.entry_points(group=group):
        loaded = entry_point.load()
        if isinstance(loaded, ExecutorAdapter) and not isinstance(loaded, type):
            adapter = loaded
        elif callable(loaded):
            adapter = loaded()
        else:
            adapter = loaded
        if not isinstance(adapter, ExecutorAdapter):
            raise TypeError(f"entry point {entry_point.name} is not an ExecutorAdapter")
        adapters.append(adapter)
    return tuple(adapters)


__all__ = ["ENTRY_POINT_GROUP", "ExecutorAdapter", "ExecutorRegistry", "discover_entry_points"]
