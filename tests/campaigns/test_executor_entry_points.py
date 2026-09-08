"""Third-party executor adapters load through the entry-point group."""

import importlib.metadata as metadata
import logging

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.executor_adapters import build_default_registry, default_registry
from bashgym.campaigns.executor_registry import ENTRY_POINT_GROUP
from bashgym.campaigns.executors import fake_digest
from bashgym.campaigns.runtime import ActionSpec


class _PluginAdapter:
    kind = "plugin_executor"
    allowed_stages = frozenset({StageKind.DATA_BUILD})
    reuses_completed_results = False

    def tick(self, worker, attempt, *, now):
        return "plugin_ticked"

    def reconcile(self, worker, attempt, *, now):
        return None

    def repair_allowed(self):
        return True


class _EntryPoint:
    name = "plugin"

    def load(self):
        return _PluginAdapter


def test_entry_point_adapters_join_the_default_registry(monkeypatch) -> None:
    original = metadata.entry_points

    def fake_entry_points(**kwargs):
        if kwargs.get("group") == ENTRY_POINT_GROUP:
            return (_EntryPoint(),)
        return original(**kwargs)

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)

    registry = build_default_registry()

    assert "plugin_executor" in registry.kinds()
    assert registry.allowed_stages("plugin_executor") == frozenset({StageKind.DATA_BUILD})


class _InvalidKindAdapter(_PluginAdapter):
    kind = "vendor gpu"


class _DuplicateKindAdapter(_PluginAdapter):
    kind = "fake"


class _InvalidKindEntryPoint:
    name = "invalid_kind"
    value = "vendor.invalid:Adapter"

    def load(self):
        return _InvalidKindAdapter


class _DuplicateEntryPoint:
    name = "duplicate_kind"
    value = "vendor.duplicate:Adapter"

    def load(self):
        return _DuplicateKindAdapter


def _install_entry_points(monkeypatch, entry_points):
    original = metadata.entry_points

    def fake_entry_points(**kwargs):
        if kwargs.get("group") == ENTRY_POINT_GROUP:
            return tuple(entry_points)
        return original(**kwargs)

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)


def test_a_plugin_that_cannot_register_is_skipped_like_one_that_cannot_load(
    monkeypatch, caplog
) -> None:
    _install_entry_points(
        monkeypatch, (_InvalidKindEntryPoint(), _DuplicateEntryPoint(), _EntryPoint())
    )

    with caplog.at_level(logging.WARNING, logger="bashgym.campaigns.executor_registry"):
        registry = build_default_registry()

    assert registry.kinds() == (
        "development_evaluation",
        "fake",
        "plugin_executor",
        "ssh_remote",
    )
    assert "invalid_kind" in caplog.text
    assert "vendor.invalid:Adapter" in caplog.text
    assert "vendor gpu" in caplog.text
    assert "duplicate_kind" in caplog.text
    assert "already registered" in caplog.text


def test_campaign_reads_still_work_when_a_plugin_cannot_register(monkeypatch) -> None:
    _install_entry_points(
        monkeypatch, (_InvalidKindEntryPoint(), _DuplicateEntryPoint(), _EntryPoint())
    )
    default_registry.cache_clear()
    try:
        # available_executors in the evidence snapshot is default_registry().kinds().
        assert default_registry().kinds() == (
            "development_evaluation",
            "fake",
            "plugin_executor",
            "ssh_remote",
        )
        spec = ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract={},
            candidate_digest=fake_digest("candidate"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
        )
        assert spec.executor_kind == "fake"
    finally:
        default_registry.cache_clear()
