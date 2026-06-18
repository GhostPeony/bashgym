"""Plugin connector — discovered by Data Designer via the entry point in pyproject.toml."""

from __future__ import annotations

from data_designer.plugins import Plugin, PluginType

gold_trace_plugin = Plugin(
    config_qualified_name="bashgym.factory.dd_plugin.config.GoldTraceSeedSource",
    impl_qualified_name="bashgym.factory.dd_plugin.impl.GoldTraceSeedReader",
    plugin_type=PluginType.SEED_READER,
)
