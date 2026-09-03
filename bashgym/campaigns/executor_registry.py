"""Registry of campaign executor adapters keyed by executor kind."""

from __future__ import annotations

import importlib.metadata as metadata
import logging
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

from bashgym.campaigns.contracts import ActionAttempt, Identifier, StageKind

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "bashgym.campaign_executors"
_IDENTIFIER = TypeAdapter(Identifier)


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
        try:
            _IDENTIFIER.validate_python(adapter.kind)
        except ValidationError as exc:
            raise ValueError(f"executor kind is not a valid identifier: {adapter.kind!r}") from exc
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
    """Load third-party adapters; a missing group yields no adapters.

    An entry point that fails to import, whose factory raises, or that does not
    implement ``ExecutorAdapter`` is skipped with a warning naming the entry
    point and the error. Its kind stays unregistered, so every campaign path
    that resolves that kind fails closed while the rest of the registry, and
    every campaign read that only needs the registered kinds, keeps working.
    """

    adapters: list[ExecutorAdapter] = []
    for entry_point in metadata.entry_points(group=group):
        try:
            adapter = _adapter_from_entry_point(entry_point)
        except Exception as exc:  # noqa: BLE001 - one plugin cannot break discovery
            logger.warning(
                "campaign executor entry point %s (%s) was skipped: %s: %s",
                getattr(entry_point, "name", "?"),
                getattr(entry_point, "value", "?"),
                type(exc).__name__,
                exc,
            )
            continue
        adapters.append(adapter)
    return tuple(adapters)


def _adapter_from_entry_point(entry_point: Any) -> ExecutorAdapter:
    """Resolve one entry point to an adapter instance, or raise for the caller."""

    loaded = entry_point.load()
    if isinstance(loaded, ExecutorAdapter) and not isinstance(loaded, type):
        adapter = loaded
    elif callable(loaded):
        adapter = loaded()
    else:
        adapter = loaded
    if not isinstance(adapter, ExecutorAdapter):
        raise TypeError(f"entry point {entry_point.name} is not an ExecutorAdapter")
    return adapter


__all__ = ["ENTRY_POINT_GROUP", "ExecutorAdapter", "ExecutorRegistry", "discover_entry_points"]
