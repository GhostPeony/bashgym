"""Claude MCP config compatibility preview tests."""

from bashgym.mcp.claude_compat import preview_claude_mcp_config


def test_import_preview_normalizes_http_alias_and_secret_references():
    [candidate] = preview_claude_mcp_config(
        {
            "mcpServers": {
                "github": {
                    "type": "streamable-http",
                    "url": "https://example.test/mcp",
                    "headers": {"Authorization": "Bearer ${GITHUB_MCP_TOKEN}"},
                    "alwaysLoad": True,
                }
            }
        }
    )

    assert candidate.supported is True
    assert candidate.profile_input == {
        "label": "github",
        "transport": "streamable_http",
        "enabled": True,
        "remote": {
            "url": "https://example.test/mcp",
            "header_secret_refs": {"Authorization": "GITHUB_MCP_TOKEN"},
            "allow_private_network": False,
            "auth_mode": "headers",
            "oauth_scopes": [],
        },
    }
    assert candidate.preserved_fields == {"alwaysLoad": True}


def test_import_preview_blocks_raw_values_without_echoing_them():
    [candidate] = preview_claude_mcp_config(
        {
            "mcpServers": {
                "unsafe": {
                    "command": "python",
                    "env": {"API_TOKEN": "sk-live-do-not-echo"},
                }
            }
        }
    )

    assert candidate.supported is False
    assert candidate.profile_input is None
    assert "sk-live-do-not-echo" not in candidate.model_dump_json()
    assert any(issue.code == "raw_env_value" for issue in candidate.issues)


def test_import_preview_preserves_unsupported_shape_without_secret_values():
    [candidate] = preview_claude_mcp_config(
        {
            "mcpServers": {
                "events": {
                    "type": "ws",
                    "url": "wss://example.test/mcp",
                    "headers": {"Authorization": "secret"},
                }
            }
        }
    )

    assert candidate.supported is False
    assert candidate.preserved_fields["transport"] == "ws"
    assert candidate.preserved_fields["field_names"] == ["headers", "type", "url"]
    assert "secret" not in candidate.model_dump_json()
