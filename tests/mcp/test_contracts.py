"""Contract tests for the SDK-independent MCP Workbench boundary."""

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from bashgym.mcp.contracts import (
    McpCapabilitySnapshot,
    McpOperation,
    McpProfile,
    McpProfileRevision,
    McpStdioLaunchApproval,
    McpTransport,
    OperationKind,
    OperationState,
    StdioTransport,
    StreamableHttpTransport,
)
from bashgym.mcp.operations import (
    InvalidOperationTransitionError,
    is_terminal_operation_state,
    transitioned_operation,
)


def test_transport_union_normalizes_streamable_http_and_stdio():
    adapter = TypeAdapter(McpTransport)

    remote = adapter.validate_python(
        {
            "type": "streamable_http",
            "url": "https://mcp.example.test/api",
            "header_secret_refs": {"Authorization": "MCP_EXAMPLE_AUTH"},
        }
    )
    local = adapter.validate_python(
        {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "example.server"],
            "env_secret_refs": {"EXAMPLE_TOKEN": "MCP_EXAMPLE_TOKEN"},
        }
    )

    assert isinstance(remote, StreamableHttpTransport)
    assert remote.allow_private_network is False
    assert isinstance(local, StdioTransport)
    assert local.args == ["-m", "example.server"]


def test_remote_oauth_config_is_explicit_and_secret_free():
    remote = StreamableHttpTransport(
        url="https://mcp.example.test/api",
        auth_mode="oauth",
        oauth_scopes=["read", "write"],
        oauth_callback_port=8765,
        oauth_client_id="bashgym-client",
        oauth_client_secret_ref="MCP_OAUTH_CLIENT_SECRET",
    )
    assert remote.auth_mode == "oauth"
    assert remote.oauth_scopes == ["read", "write"]
    assert "secret-value" not in remote.model_dump_json()

    with pytest.raises(ValidationError, match="cannot also configure"):
        StreamableHttpTransport(
            url="https://mcp.example.test/api",
            auth_mode="oauth",
            header_secret_refs={"Authorization": "MCP_TOKEN"},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "type": "streamable_http",
            "url": "https://user:secret@example.test/mcp",
        },
        {
            "type": "streamable_http",
            "url": "https://example.test/mcp",
            "api_key": "raw-secret",
        },
        {
            "type": "streamable_http",
            "url": "https://example.test/mcp",
            "header_secret_refs": {"Authorization": "Bearer raw-secret"},
        },
        {
            "type": "streamable_http",
            "url": "https://example.test/mcp?access_token=raw-secret",
        },
        {
            "type": "stdio",
            "command": "python\nmalicious",
        },
        {
            "type": "stdio",
            "command": "python",
            "env": {"TOKEN": "raw-secret"},
        },
        {
            "type": "stdio",
            "command": "python",
            "args": ["--token", "raw-secret"],
        },
    ],
)
def test_transport_contract_rejects_inline_credentials_and_raw_secret_fields(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(McpTransport).validate_python(payload)


def test_stdio_explicit_cwd_is_consistent():
    with pytest.raises(ValidationError, match="requires cwd"):
        StdioTransport(command="python", cwd_policy="explicit")
    with pytest.raises(ValidationError, match="only with the explicit"):
        StdioTransport(command="python", cwd="C:/workspace")
    assert StdioTransport(command="python", cwd_policy="isolated").cwd is None


def test_profile_and_revision_require_matching_workspace_identity():
    profile = McpProfile(
        profile_id="profile-1",
        workspace_id="workspace-a",
        label="Example",
        transport=StdioTransport(command="python"),
    )

    revision = McpProfileRevision(
        workspace_id="workspace-a",
        profile_id="profile-1",
        revision=1,
        profile=profile,
    )
    assert revision.profile.workspace_id == "workspace-a"

    with pytest.raises(ValidationError, match="identity must match"):
        McpProfileRevision(
            workspace_id="workspace-b",
            profile_id="profile-1",
            revision=1,
            profile=profile,
        )


def test_snapshot_hash_covers_contract_but_not_capture_identity():
    contract = {
        "negotiated_protocol_version": "2025-11-25",
        "server_info": {"name": "fixture", "version": "1.0"},
        "capabilities": {"tools": {"listChanged": True}},
        "tools": [{"name": "read", "description": "Read a fixture"}],
    }
    first = McpCapabilitySnapshot(
        snapshot_id="snapshot-1",
        workspace_id="workspace-a",
        profile_id="profile-1",
        profile_revision=1,
        captured_at=datetime(2026, 7, 10, 10, tzinfo=UTC),
        **contract,
    )
    second = McpCapabilitySnapshot(
        snapshot_id="snapshot-2",
        workspace_id="workspace-b",
        profile_id="profile-9",
        profile_revision=8,
        captured_at=datetime(2026, 7, 11, 10, tzinfo=UTC),
        **contract,
    )

    assert first.contract_hash == second.contract_hash
    assert len(first.contract_hash) == 64

    changed = McpCapabilitySnapshot(
        snapshot_id="snapshot-3",
        workspace_id="workspace-a",
        profile_id="profile-1",
        profile_revision=1,
        **{**contract, "tools": [{"name": "read", "description": "Better description"}]},
    )
    assert changed.contract_hash != first.contract_hash


def test_snapshot_rejects_a_forged_contract_hash():
    with pytest.raises(ValidationError, match="contract_hash does not match"):
        McpCapabilitySnapshot(
            snapshot_id="snapshot-1",
            workspace_id="workspace-a",
            profile_id="profile-1",
            profile_revision=1,
            negotiated_protocol_version="2025-11-25",
            contract_hash="0" * 64,
        )


def test_operation_lifecycle_includes_unknown_upstream_cancellation():
    operation = McpOperation(
        operation_id="operation-1",
        correlation_id="correlation-1",
        workspace_id="workspace-a",
        kind=OperationKind.MANUAL_CALL,
    )
    running = transitioned_operation(operation, OperationState.RUNNING)
    cancelled = transitioned_operation(running, OperationState.CANCELLED_UPSTREAM_UNKNOWN)

    assert running.revision == 2
    assert running.started_at is not None
    assert cancelled.revision == 3
    assert cancelled.completed_at is not None
    assert is_terminal_operation_state(cancelled.state)

    with pytest.raises(InvalidOperationTransitionError):
        transitioned_operation(cancelled, OperationState.RUNNING)


def test_operation_result_must_be_sanitized():
    operation = McpOperation(
        operation_id="operation-result",
        correlation_id="correlation-result",
        workspace_id="workspace-a",
        kind=OperationKind.CONNECT,
        result={
            "session_id": "session-1",
            "authorization": "[redacted]",
            "token_count": 42,
            "access_tokens": ["diagnostic-term"],
        },
    )
    assert operation.result == {
        "session_id": "session-1",
        "authorization": "[redacted]",
        "token_count": 42,
        "access_tokens": ["diagnostic-term"],
    }

    with pytest.raises(ValidationError, match="unredacted secret"):
        McpOperation(
            operation_id="operation-secret",
            correlation_id="correlation-secret",
            workspace_id="workspace-a",
            kind=OperationKind.CONNECT,
            result={"api_key": "raw-secret"},
        )

    with pytest.raises(ValidationError, match="unredacted secret"):
        transitioned_operation(
            operation,
            OperationState.RUNNING,
            result={"access_token": "raw-secret"},
        )


def test_stdio_launch_approval_hashes_launch_material_and_keeps_only_env_names():
    transport = StdioTransport(
        command="python",
        args=["-m", "fixture.server"],
        cwd_policy="isolated",
        env_secret_refs={"FIXTURE_TOKEN": "MCP_FIXTURE_TOKEN"},
        sandbox_policy="required",
    )
    approval = McpStdioLaunchApproval.for_transport(
        workspace_id="workspace-a",
        profile_id="profile-stdio",
        profile_revision=1,
        executable_fingerprint="a" * 64,
        transport=transport,
    )

    serialized = approval.model_dump_json()
    assert approval.fingerprint.env_names == ("FIXTURE_TOKEN",)
    assert len(approval.launch_fingerprint) == 64
    assert "MCP_FIXTURE_TOKEN" not in serialized
    assert "fixture.server" not in serialized


def test_managed_operation_kinds_cover_disconnect_and_quick_test():
    assert OperationKind.DISCONNECT.value == "disconnect"
    assert OperationKind.QUICK_TEST.value == "quick_test"
