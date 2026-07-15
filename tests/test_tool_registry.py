# tests/test_tool_registry.py
from bashgym.agent.tools import CORE_TOOLS, MEMORY_TOOLS, ToolRegistry


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

    def test_start_training_tool_accepts_canvas_provenance(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        training_tool = next(tool for tool in tools if tool["name"] == "start_training")
        props = training_tool["input_schema"]["properties"]

        assert {
            "dataset_path",
            "compute_target",
            "correlation_id",
            "tracking_context",
            "origin",
        }.issubset(props)
        assert {"panel_id", "terminal_id", "agent"}.issubset(props["origin"]["properties"])
        assert {
            "project_id",
            "experiment_id",
            "model_version_id",
            "dataset_version_id",
            "environment_id",
        }.issubset(props["tracking_context"]["required"])

    def test_start_training_tool_surfaces_all_direct_strategies_and_storage_policy(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        training_tool = next(tool for tool in tools if tool["name"] == "start_training")
        props = training_tool["input_schema"]["properties"]

        assert {
            "sft",
            "dpo",
            "grpo",
            "rlvr",
            "distillation",
            "session_distillation",
        } == set(props["strategy"]["enum"])
        config_props = props["config"]["properties"]
        assert {
            "checkpoint_limit",
            "artifact_retention",
            "auto_push_hf",
            "hf_private",
            "hf_upload_artifact",
        }.issubset(config_props)

    def test_data_designer_tool_accepts_runtime_and_canvas_inputs(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        designer_tool = next(tool for tool in tools if tool["name"] == "start_data_designer")
        props = designer_tool["input_schema"]["properties"]

        assert {
            "pipeline",
            "num_records",
            "seed_source",
            "model",
            "provider_endpoint",
            "origin",
        }.issubset(props)
        assert {"panel_id", "terminal_id", "agent"}.issubset(props["origin"]["properties"])

    def test_list_my_capabilities_present(self):
        reg = ToolRegistry()
        tools = reg.build_tools()
        assert any(t["name"] == "list_my_capabilities" for t in tools)

    def test_experiment_ledger_tools_are_agent_accessible(self):
        tools = ToolRegistry().build_tools()
        names = {tool["name"] for tool in tools}
        assert {
            "list_experiment_projects",
            "get_experiment_context",
            "get_experiment_run",
        } <= names

    def test_skill_tools_merged(self):
        skill_tools = [
            {
                "name": "hf_download_model",
                "description": "Download model",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        reg = ToolRegistry()
        tools = reg.build_tools(skill_tools=skill_tools)
        assert any(t["name"] == "hf_download_model" for t in tools)

    def test_no_duplicate_tools(self):
        dup_tools = [
            {
                "name": "import_traces",
                "description": "Duplicate",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
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
