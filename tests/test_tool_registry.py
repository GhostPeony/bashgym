# tests/test_tool_registry.py
import pytest
from bashgym.agent.tools import ToolRegistry, CORE_TOOLS, MEMORY_TOOLS


class TestToolRegistry:
    def test_core_tools_always_present(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        core_names = {t["name"] for t in CORE_TOOLS}
        built_names = {t["name"] for t in tools}
        assert core_names.issubset(built_names)

    def test_memory_tools_always_present(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        mem_names = {t["name"] for t in MEMORY_TOOLS}
        built_names = {t["name"] for t in tools}
        assert mem_names.issubset(built_names)

    def test_list_my_capabilities_present(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        assert any(t["name"] == "list_my_capabilities" for t in tools)

    def test_skill_tools_merged(self):
        skill_tools = [{"name": "hf_download_model", "description": "Download model", "input_schema": {"type": "object", "properties": {}}}]
        reg = ToolRegistry()
        tools = reg.build_tools(skill_tools=skill_tools)
        assert any(t["name"] == "hf_download_model" for t in tools)

    def test_no_duplicate_tools(self):
        dup_tools = [{"name": "import_traces", "description": "Duplicate", "input_schema": {"type": "object", "properties": {}}}]
        reg = ToolRegistry()
        tools = reg.build_tools(skill_tools=dup_tools)
        names = [t["name"] for t in tools]
        assert names.count("import_traces") == 1

    def test_capabilities_summary(self):
        reg = ToolRegistry()
        summary = reg.capabilities_summary()
        assert "Core Gym" in summary
        assert "Memory" in summary

    def test_list_capabilities_by_category(self):
        reg = ToolRegistry()
        result = reg.list_capabilities(category="memory")
        assert "remember_fact" in result
