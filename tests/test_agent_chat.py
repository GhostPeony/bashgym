import json
import pytest
import tempfile
from pathlib import Path
from bashgym.agent.memory import PeonyMemory
from bashgym.agent.skills.registry import SkillRegistry
from bashgym.agent.tools import ToolRegistry


class TestAgentIntegration:
    def test_memory_tools_execute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PeonyMemory(Path(tmpdir))
            result = mem.remember_fact("project", "Test fact")
            assert result["content"] == "Test fact"
            results = mem.recall_facts(category="project")
            assert len(results) == 1

    def test_skill_matching_produces_tools(self):
        reg = SkillRegistry()
        matches = reg.match("I want to train a model with grpo")
        tools = reg.get_tools(matches)
        tool_names = [t["name"] for t in tools]
        assert len(tool_names) > 0

    def test_tool_registry_builds_dynamic_list(self):
        skill_tools = [{"name": "hf_custom", "description": "Custom", "input_schema": {"type": "object", "properties": {}}}]
        reg = ToolRegistry()
        tools = reg.build_tools(skill_tools=skill_tools)
        names = [t["name"] for t in tools]
        assert "import_traces" in names
        assert "remember_fact" in names
        assert "list_my_capabilities" in names
        assert "hf_custom" in names

    def test_system_prompt_includes_all_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = PeonyMemory(Path(tmpdir))
            mem.update_profile("hf_username", "testuser")
            mem.remember_fact("project", "Test project")
            mem.save_episode("s1", "Did some testing")

            tool_reg = ToolRegistry()
            capabilities = tool_reg.capabilities_summary()
            memory_prompt = mem.build_memory_prompt()

            assert "YOUR CAPABILITIES" in capabilities
            assert "USER PROFILE" in memory_prompt
            assert "KNOWN FACTS" in memory_prompt
            assert "RECENT HISTORY" in memory_prompt
            assert "testuser" in memory_prompt
