"""Third-party executor adapters load through the entry-point group."""

import importlib.metadata as metadata

from bashgym.campaigns.contracts import StageKind
from bashgym.campaigns.executor_adapters import build_default_registry
from bashgym.campaigns.executor_registry import ENTRY_POINT_GROUP


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
