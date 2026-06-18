"""Tests for the BashGym gold-trace Data Designer seed-reader plugin."""

import json

import pytest

from bashgym.factory.data_designer import DATA_DESIGNER_AVAILABLE


@pytest.mark.skipif(not DATA_DESIGNER_AVAILABLE, reason="data-designer not installed")
class TestGoldTracePlugin:
    def test_plugin_connector_valid(self):
        from data_designer.plugins import PluginType

        from bashgym.factory.dd_plugin.plugin import gold_trace_plugin

        assert gold_trace_plugin.plugin_type == PluginType.SEED_READER
        assert gold_trace_plugin.name == "bashgym_gold_trace"
        # Plugin validators resolved both classes.
        assert gold_trace_plugin.config_cls.__name__ == "GoldTraceSeedSource"
        assert gold_trace_plugin.impl_cls.__name__ == "GoldTraceSeedReader"

    def test_reader_extracts_seeds_and_skips_promptless(self, tmp_path):
        from bashgym.factory.dd_plugin.config import GoldTraceSeedSource
        from bashgym.factory.dd_plugin.impl import GoldTraceSeedReader
        from bashgym.factory.designer_pipelines import build_secret_resolver

        (tmp_path / "t1.json").write_text(
            json.dumps(
                {
                    "metadata": {"user_initial_prompt": "Fix the failing test in api.py"},
                    "trace": [
                        {"tool_name": "bash", "command": "pytest tests/a.py"},
                        {"tool_name": "edit", "command": "edit api.py"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        # No user prompt -> skipped during hydration.
        (tmp_path / "t2.json").write_text(
            json.dumps({"metadata": {}, "trace": []}), encoding="utf-8"
        )

        reader = GoldTraceSeedReader()
        reader.attach(GoldTraceSeedSource(path=str(tmp_path)), build_secret_resolver())
        assert set(reader.get_column_names()) == {
            "seed_task",
            "seed_tools",
            "seed_complexity",
            "seed_language",
            "seed_step_count",
            "source_path",
        }
        df = (
            reader.create_batch_reader(batch_size=10, index_range=None, shuffle=False)
            .read_next_batch()
            .to_pandas()
        )
        assert len(df) == 1
        row = df.iloc[0].to_dict()
        assert row["seed_task"] == "Fix the failing test in api.py"
        assert "bash" in row["seed_tools"] and "edit" in row["seed_tools"]
        assert row["seed_complexity"] == "simple"
        assert row["seed_step_count"] == 2

    def test_discovered_by_data_designer(self):
        """Entry-point discovery — requires `pip install -e .`; skipped otherwise."""
        from importlib.metadata import entry_points

        names = [e.name for e in entry_points(group="data_designer.plugins")]
        if "bashgym-gold-trace" not in names:
            pytest.skip("plugin entry point not registered (needs pip install -e .)")
        from data_designer.plugin_manager import PluginManager
        from data_designer.plugins.plugin import PluginType

        seed_plugins = PluginManager()._plugin_registry.get_plugins(PluginType.SEED_READER)
        assert "bashgym_gold_trace" in [p.name for p in seed_plugins]
