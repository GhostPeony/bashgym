"""Built-in executor adapters that delegate to the campaign worker."""

from __future__ import annotations

import asyncio
import functools
from datetime import datetime
from typing import Any

from bashgym.campaigns.contracts import ActionAttempt, StageKind
from bashgym.campaigns.executor_registry import ExecutorRegistry, register_entry_points

_ALL_STAGES = frozenset(StageKind)
_REMOTE_STAGES = frozenset(
    {
        StageKind.DATA_BUILD,
        StageKind.CONTRACT_EVALUATION,
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
        StageKind.DEVELOPMENT_EVALUATION,
    }
)


class FakeExecutorAdapter:
    """Local deterministic executor used for wiring and lifecycle proofs."""

    kind = "fake"
    allowed_stages = _ALL_STAGES
    reuses_completed_results = False

    def tick(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str:
        return worker._fake_tick(attempt, now=now)

    def reconcile(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str | None:
        return None

    def repair_allowed(self) -> bool:
        return True


class SshRemoteExecutorAdapter:
    """Registered SSH compute executor whose results are reconciled remotely."""

    kind = "ssh_remote"
    allowed_stages = _REMOTE_STAGES
    reuses_completed_results = True

    def tick(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str:
        return asyncio.run(worker._remote_tick(attempt, now=now))

    def reconcile(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str | None:
        return worker._reconcile_remote(attempt, now=now)

    def repair_allowed(self) -> bool:
        return False


class DevelopmentEvaluationExecutorAdapter:
    """Local development evaluation executor for the development evaluation stage."""

    kind = "development_evaluation"
    allowed_stages = frozenset({StageKind.DEVELOPMENT_EVALUATION})
    reuses_completed_results = False

    def tick(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str:
        return worker._development_evaluation_tick(attempt, now=now)

    def reconcile(self, worker: Any, attempt: ActionAttempt, *, now: datetime) -> str | None:
        return None

    def repair_allowed(self) -> bool:
        return True


def build_default_registry() -> ExecutorRegistry:
    """Register the built-in adapters plus discovered entry points, then freeze.

    A built-in that fails to register is a repository defect and raises. A
    third-party entry point that fails to load or to register is skipped, so one
    unusable plugin cannot stop this process from reading campaigns.
    """

    registry = ExecutorRegistry()
    registry.register(FakeExecutorAdapter())
    registry.register(SshRemoteExecutorAdapter())
    registry.register(DevelopmentEvaluationExecutorAdapter())
    register_entry_points(registry)
    registry.freeze()
    return registry


@functools.lru_cache(maxsize=1)
def default_registry() -> ExecutorRegistry:
    """Return the process-wide registry of built-in and installed executor kinds."""

    return build_default_registry()


__all__ = [
    "DevelopmentEvaluationExecutorAdapter",
    "FakeExecutorAdapter",
    "SshRemoteExecutorAdapter",
    "build_default_registry",
    "default_registry",
]
