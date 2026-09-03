"""Executor registry behavior."""

from datetime import datetime, timezone

import pytest

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.executor_registry import ExecutorRegistry, discover_entry_points


class _Adapter:
    kind = "unit_test_executor"
    allowed_stages = frozenset({StageKind.DATA_BUILD})

    def tick(self, worker, attempt, *, now):
        return "unit_ticked"

    def reconcile(self, worker, attempt, *, now):
        return None

    def repair_allowed(self):
        return True


def test_register_get_and_kinds() -> None:
    registry = ExecutorRegistry()
    registry.register(_Adapter())

    assert registry.kinds() == ("unit_test_executor",)
    assert registry.is_registered("unit_test_executor")
    assert registry.allowed_stages("unit_test_executor") == frozenset({StageKind.DATA_BUILD})
    assert (
        registry.get("unit_test_executor").tick(None, None, now=datetime.now(timezone.utc))
        == "unit_ticked"
    )


def test_duplicate_and_unknown_kinds_fail() -> None:
    registry = ExecutorRegistry()
    registry.register(_Adapter())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_Adapter())
    with pytest.raises(KeyError):
        registry.get("missing")
    assert not registry.is_registered("missing")


def test_frozen_registry_rejects_registration() -> None:
    registry = ExecutorRegistry()
    registry.freeze()

    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(_Adapter())


def test_discover_entry_points_tolerates_no_plugins(monkeypatch) -> None:
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: ())

    assert discover_entry_points() == ()
