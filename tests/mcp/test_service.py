"""Service-level compatibility diagnostics for MCP snapshots."""

from bashgym.mcp.contracts import McpProfile, StreamableHttpTransport
from bashgym.mcp.persistence import McpRepository
from bashgym.mcp.service import McpWorkbenchService


def test_snapshot_records_claude_projection_and_reserved_name_warnings(tmp_path):
    service = McpWorkbenchService(McpRepository(tmp_path / "mcp.sqlite3"))
    profile = McpProfile(
        profile_id="profile-1",
        workspace_id="workspace-a",
        label="Compatibility fixture",
        transport=StreamableHttpTransport(url="https://example.test/mcp"),
    )

    snapshot = service._snapshot_from_connection(
        profile,
        {
            "initialization": {
                "protocolVersion": "2025-11-25",
                "serverInfo": {"name": "workspace", "version": "1.0"},
                "instructions": "x" * 2049,
                "capabilities": {"tools": {}},
            },
            "inventory": {
                "tools": [
                    {
                        "name": "projected_tool",
                        "description": "y" * 2049,
                        "inputSchema": {"oneOf": [{"type": "object"}]},
                        "_meta": {
                            "anthropic/alwaysLoad": True,
                            "anthropic/maxResultSizeChars": 100_000,
                        },
                    }
                ]
            },
        },
    )

    assert "claude_server_instructions_over_2kb" in snapshot.schema_warnings
    assert "claude_reserved_server_name:workspace" in snapshot.schema_warnings
    assert "claude_description_over_2kb:projected_tool" in snapshot.schema_warnings
    assert "claude_root_schema_projection:projected_tool" in snapshot.schema_warnings
    assert snapshot.tools[0]["_meta"]["anthropic/alwaysLoad"] is True
