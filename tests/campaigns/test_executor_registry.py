"""Executor registry behavior."""

import logging
from datetime import datetime, timezone

import pytest

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.executor_adapters import (
    DevelopmentEvaluationExecutorAdapter,
    FakeExecutorAdapter,
    SshRemoteExecutorAdapter,
)
from bashgym.campaigns.executor_registry import ExecutorRegistry, discover_entry_points


class _Adapter:
    kind = "unit_test_executor"
    allowed_stages = frozenset({StageKind.DATA_BUILD})
    reuses_completed_results = False

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


def test_register_rejects_non_adapters() -> None:
    registry = ExecutorRegistry()

    plain_object = object()
    with pytest.raises(TypeError, match="does not implement ExecutorAdapter"):
        registry.register(plain_object)  # type: ignore

    with pytest.raises(TypeError, match="does not implement ExecutorAdapter"):
        registry.register(_Adapter)  # type: ignore


def test_register_rejects_an_adapter_without_a_declared_reuse_capability() -> None:
    class _Incomplete:
        kind = "incomplete_executor"
        allowed_stages = frozenset({StageKind.DATA_BUILD})

        def tick(self, worker, attempt, *, now):
            return "ticked"

        def reconcile(self, worker, attempt, *, now):
            return None

        def repair_allowed(self):
            return True

    registry = ExecutorRegistry()

    with pytest.raises(TypeError, match="does not implement ExecutorAdapter"):
        registry.register(_Incomplete())  # type: ignore[arg-type]


def test_freeze_preserves_and_protects_existing() -> None:
    registry = ExecutorRegistry()
    registry.register(_Adapter())

    assert registry.kinds() == ("unit_test_executor",)
    assert registry.is_registered("unit_test_executor")

    registry.freeze()

    assert registry.kinds() == ("unit_test_executor",)
    assert registry.is_registered("unit_test_executor")
    assert (
        registry.get("unit_test_executor").tick(None, None, now=datetime.now(timezone.utc))
        == "unit_ticked"
    )

    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(_Adapter())


def test_discover_entry_points_entry_point_variants(monkeypatch) -> None:
    import importlib.metadata as metadata
    from unittest.mock import Mock

    adapter_instance = _Adapter()
    adapter_class = _Adapter

    class NonAdapter:
        pass

    entry_point_instance = Mock()
    entry_point_instance.name = "instance_ep"
    entry_point_instance.load = Mock(return_value=adapter_instance)

    entry_point_class = Mock()
    entry_point_class.name = "class_ep"
    entry_point_class.load = Mock(return_value=adapter_class)

    entry_point_non_adapter = Mock()
    entry_point_non_adapter.name = "non_adapter_ep"
    entry_point_non_adapter.load = Mock(return_value=NonAdapter())

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: (entry_point_instance, entry_point_class),
    )

    adapters = discover_entry_points()
    assert len(adapters) == 2
    assert all(isinstance(a, _Adapter) for a in adapters)

    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: (entry_point_non_adapter,))

    assert discover_entry_points() == ()


def test_register_rejects_a_kind_that_is_not_an_identifier() -> None:
    class _SpacedKind(_Adapter):
        kind = "vendor gpu"

    class _EmptyKind(_Adapter):
        kind = ""

    registry = ExecutorRegistry()

    with pytest.raises(ValueError, match="vendor gpu"):
        registry.register(_SpacedKind())
    with pytest.raises(ValueError, match="executor kind"):
        registry.register(_EmptyKind())
    assert registry.kinds() == ()


def test_register_accepts_the_three_built_in_adapter_kinds() -> None:
    registry = ExecutorRegistry()

    for adapter in (
        FakeExecutorAdapter(),
        SshRemoteExecutorAdapter(),
        DevelopmentEvaluationExecutorAdapter(),
    ):
        registry.register(adapter)

    assert registry.kinds() == ("development_evaluation", "fake", "ssh_remote")


def test_discover_entry_points_skips_a_failing_entry_point(monkeypatch, caplog) -> None:
    import importlib.metadata as metadata
    from unittest.mock import Mock

    class _NonAdapter:
        pass

    def _failing_factory():
        raise RuntimeError("factory exploded")

    broken_load = Mock()
    broken_load.name = "broken_load"
    broken_load.value = "vendor.broken:adapter"
    broken_load.load = Mock(side_effect=ImportError("no module named vendor.broken"))

    broken_factory = Mock()
    broken_factory.name = "broken_factory"
    broken_factory.value = "vendor.factory:build"
    broken_factory.load = Mock(return_value=_failing_factory)

    non_adapter = Mock()
    non_adapter.name = "non_adapter"
    non_adapter.value = "vendor.plain:Thing"
    non_adapter.load = Mock(return_value=_NonAdapter())

    healthy = Mock()
    healthy.name = "healthy"
    healthy.value = "vendor.good:Adapter"
    healthy.load = Mock(return_value=_Adapter)

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: (broken_load, broken_factory, non_adapter, healthy),
    )

    with caplog.at_level(logging.WARNING, logger="bashgym.campaigns.executor_registry"):
        adapters = discover_entry_points()

    assert tuple(adapter.kind for adapter in adapters) == ("unit_test_executor",)
    assert "broken_load" in caplog.text
    assert "vendor.broken:adapter" in caplog.text
    assert "no module named vendor.broken" in caplog.text
    assert "broken_factory" in caplog.text
    assert "factory exploded" in caplog.text
    assert "non_adapter" in caplog.text
