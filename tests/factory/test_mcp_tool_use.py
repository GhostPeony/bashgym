"""Tests for the BashGym sandbox MCP server and the mcp_tool_use pipeline."""

import pytest

from bashgym.factory.data_designer import DATA_DESIGNER_AVAILABLE, PipelineConfig
from bashgym.factory.designer_pipelines import HAS_MCP
from bashgym.mcp.sandbox_server import LocalWorkspace, build_server

# =========================================================================
# LocalWorkspace (Docker-free backend) — cross-platform, no API
# =========================================================================


class TestLocalWorkspace:
    def test_write_read_edit(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            ws.write_file("app.py", "print('hi')\n")
            assert "print('hi')" in ws.read_file("app.py")
            ws.edit_file("app.py", "hi", "hello")
            assert "print('hello')" in ws.read_file("app.py")
        finally:
            ws.close()

    def test_edit_missing_old_string(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            ws.write_file("a.txt", "abc")
            assert "not found" in ws.edit_file("a.txt", "zzz", "y")
        finally:
            ws.close()

    def test_grep_and_list(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            ws.write_file("a.txt", "needle here\nother")
            assert "needle" in ws.grep("needle")
            assert "a.txt" in ws.list_files()
            assert "no matches" in ws.grep("zzz-not-present")
        finally:
            ws.close()

    def test_bash_runs(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            assert "sandbox-ok" in ws.bash("echo sandbox-ok")
        finally:
            ws.close()

    def test_dangerous_command_blocked(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            assert "BLOCKED" in ws.bash("rm -rf /")
        finally:
            ws.close()

    def test_path_escape_blocked(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            with pytest.raises(ValueError, match="escapes workspace"):
                ws.read_file("../../etc/passwd")
        finally:
            ws.close()

    def test_read_missing_file(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            assert "no such file" in ws.read_file("nope.txt")
        finally:
            ws.close()


class TestServerBuild:
    def test_build_server(self):
        ws = LocalWorkspace(timeout_sec=20)
        try:
            server = build_server(workspace=ws)
            assert server.name == "bashgym-sandbox"
        finally:
            ws.close()


# =========================================================================
# DD wiring (provider / tool config / pipeline)
# =========================================================================


@pytest.mark.skipif(
    not (DATA_DESIGNER_AVAILABLE and HAS_MCP),
    reason="data-designer with MCP support not installed",
)
class TestMcpWiring:
    def test_providers_only_when_enabled(self):
        from bashgym.factory.designer_pipelines import build_mcp_providers

        assert build_mcp_providers(PipelineConfig()) == []
        provs = build_mcp_providers(PipelineConfig(enable_tools=True))
        assert len(provs) == 1
        assert provs[0].name == "bashgym-sandbox"
        # subprocess must be able to import bashgym
        assert "PYTHONPATH" in provs[0].env

    def test_tool_config(self):
        from bashgym.factory.designer_pipelines import build_sandbox_tool_config

        tc = build_sandbox_tool_config(PipelineConfig(mcp_tool_alias="sandbox"))
        assert tc.tool_alias == "sandbox"
        assert "bash" in tc.allow_tools and "write_file" in tc.allow_tools

    def test_pipeline_registered_and_builds(self):
        from bashgym.factory.designer_pipelines import PIPELINES

        assert "mcp_tool_use" in PIPELINES
        builder = PIPELINES["mcp_tool_use"](PipelineConfig(pipeline="mcp_tool_use"))
        assert builder is not None

    def test_tool_pipeline_enables_tools_at_construction(self):
        # Order-independent: set in __post_init__ so .designer attaches the MCP
        # provider regardless of access order.
        assert PipelineConfig(pipeline="mcp_tool_use").enable_tools is True
        assert PipelineConfig(pipeline="coding_agent_sft").enable_tools is False
