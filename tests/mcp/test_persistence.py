"""SQLite ownership, revision, idempotency, and recovery tests for MCP."""

import sqlite3

import pytest

from bashgym.mcp.contracts import (
    McpCapabilitySnapshot,
    McpOperation,
    McpProfile,
    McpSession,
    McpStdioLaunchApproval,
    OperationKind,
    OperationState,
    SessionState,
    StdioTransport,
    StreamableHttpTransport,
)
from bashgym.mcp.persistence import (
    McpRepository,
    RecordNotFoundError,
    RevisionConflictError,
)


def profile(workspace_id: str = "workspace-a", profile_id: str = "profile-1") -> McpProfile:
    return McpProfile(
        profile_id=profile_id,
        workspace_id=workspace_id,
        label="Fixture MCP",
        transport=StreamableHttpTransport(
            url="https://mcp.example.test/api",
            header_secret_refs={"Authorization": "MCP_FIXTURE_AUTH"},
        ),
    )


def snapshot(
    workspace_id: str = "workspace-a",
    profile_id: str = "profile-1",
    snapshot_id: str = "snapshot-1",
) -> McpCapabilitySnapshot:
    return McpCapabilitySnapshot(
        snapshot_id=snapshot_id,
        workspace_id=workspace_id,
        profile_id=profile_id,
        profile_revision=1,
        negotiated_protocol_version="2025-11-25",
        server_info={"name": "fixture", "version": "1.0"},
        capabilities={"tools": {}},
        tools=[{"name": "read_fixture", "inputSchema": {"type": "object"}}],
    )


@pytest.fixture
def repository(tmp_path):
    value = McpRepository(tmp_path / "state" / "mcp.sqlite3")
    value.initialize()
    return value


def test_initialize_applies_numbered_migrations_and_enables_sqlite_guards(repository):
    assert repository.schema_versions() == [1, 2, 3, 4]
    assert repository.journal_mode() == "wal"
    assert repository.foreign_keys_enabled() is True

    with sqlite3.connect(repository.db_path) as connection:
        owned_tables = {
            "mcp_profiles",
            "mcp_profile_revisions",
            "mcp_snapshots",
            "mcp_sessions",
            "mcp_operations",
            "mcp_stdio_launch_approvals",
        }
        for table in owned_tables:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert "workspace_id" in columns


def test_profile_update_is_optimistic_and_revisions_are_immutable(repository):
    original = repository.create_profile(profile())
    candidate = original.model_copy(update={"label": "Renamed fixture"})
    updated = repository.update_profile(candidate, expected_revision=1)

    assert updated.revision == 2
    assert repository.get_profile_revision("workspace-a", "profile-1", 1).profile.label == (
        "Fixture MCP"
    )
    assert repository.get_profile_revision("workspace-a", "profile-1", 2).profile.label == (
        "Renamed fixture"
    )

    with pytest.raises(RevisionConflictError) as conflict:
        repository.update_profile(candidate, expected_revision=1)
    assert conflict.value.current == 2


def test_cross_workspace_lookups_are_indistinguishable_from_missing(repository):
    repository.create_profile(profile())

    with pytest.raises(RecordNotFoundError) as wrong_workspace:
        repository.get_profile("workspace-b", "profile-1")
    with pytest.raises(RecordNotFoundError) as missing:
        repository.get_profile("workspace-a", "does-not-exist")

    assert str(wrong_workspace.value) == str(missing.value) == "profile not found"
    assert repository.list_profiles("workspace-b") == []


def test_tombstone_preserves_history_and_hides_current_profile(repository):
    repository.create_profile(profile())
    deleted = repository.tombstone_profile("workspace-a", "profile-1", expected_revision=1)

    assert deleted.deleted_at is not None
    assert deleted.enabled is False
    assert deleted.revision == 2
    with pytest.raises(RecordNotFoundError):
        repository.get_profile("workspace-a", "profile-1")
    assert (
        repository.get_profile("workspace-a", "profile-1", include_deleted=True).deleted_at
        is not None
    )
    assert repository.get_profile_revision("workspace-a", "profile-1", 1).profile.enabled


def test_snapshot_and_session_are_pinned_to_profile_revision(repository):
    repository.create_profile(profile())
    saved_snapshot = repository.save_snapshot(snapshot())
    session = repository.create_session(
        McpSession(
            session_id="session-1",
            workspace_id="workspace-a",
            profile_id="profile-1",
            profile_revision=1,
            snapshot_id=saved_snapshot.snapshot_id,
            state=SessionState.CONNECTED,
        )
    )

    assert repository.latest_snapshot("workspace-a", "profile-1").contract_hash == (
        saved_snapshot.contract_hash
    )
    assert repository.get_session("workspace-a", "session-1") == session
    with pytest.raises(RecordNotFoundError, match="session not found"):
        repository.get_session("workspace-b", "session-1")


def test_operation_idempotency_is_scoped_to_workspace_and_kind(repository):
    repository.create_profile(profile())
    first = repository.create_operation(
        McpOperation(
            operation_id="operation-1",
            correlation_id="correlation-1",
            workspace_id="workspace-a",
            profile_id="profile-1",
            kind=OperationKind.CONNECT,
            idempotency_key="request-1",
        )
    )
    duplicate = repository.create_operation(
        McpOperation(
            operation_id="operation-2",
            correlation_id="correlation-2",
            workspace_id="workspace-a",
            profile_id="profile-1",
            kind=OperationKind.CONNECT,
            idempotency_key="request-1",
        )
    )
    other_kind = repository.create_operation(
        McpOperation(
            operation_id="operation-3",
            correlation_id="correlation-3",
            workspace_id="workspace-a",
            profile_id="profile-1",
            kind=OperationKind.REFRESH,
            idempotency_key="request-1",
        )
    )

    assert duplicate.operation_id == first.operation_id
    assert other_kind.operation_id == "operation-3"


def test_operation_state_updates_use_revisions_and_support_unknown_cancel(repository):
    operation = repository.create_operation(
        McpOperation(
            operation_id="operation-1",
            correlation_id="correlation-1",
            workspace_id="workspace-a",
            kind=OperationKind.MANUAL_CALL,
        )
    )
    running = repository.update_operation_state(
        "workspace-a",
        operation.operation_id,
        OperationState.RUNNING,
        expected_revision=1,
    )
    cancelled = repository.update_operation_state(
        "workspace-a",
        operation.operation_id,
        OperationState.CANCELLED_UPSTREAM_UNKNOWN,
        expected_revision=running.revision,
        result={"session_id": "session-1", "status": "unknown"},
    )

    assert cancelled.state == OperationState.CANCELLED_UPSTREAM_UNKNOWN
    assert repository.get_operation("workspace-a", operation.operation_id).result == {
        "session_id": "session-1",
        "status": "unknown",
    }
    with pytest.raises(RevisionConflictError):
        repository.update_operation_state(
            "workspace-a",
            operation.operation_id,
            OperationState.FAILED,
            expected_revision=1,
        )


def test_operation_update_rejects_an_unsanitized_result(repository):
    operation = repository.create_operation(
        McpOperation(
            operation_id="operation-secret-result",
            correlation_id="correlation-secret-result",
            workspace_id="workspace-a",
            kind=OperationKind.CONNECT,
        )
    )

    with pytest.raises(ValueError, match="unredacted secret"):
        repository.update_operation_state(
            "workspace-a",
            operation.operation_id,
            OperationState.RUNNING,
            expected_revision=operation.revision,
            result={"refresh_token": "raw-secret"},
        )


def test_new_repository_recovers_lost_operations_sessions_and_snapshots(tmp_path):
    db_path = tmp_path / "mcp.sqlite3"
    before = McpRepository(db_path)
    before.initialize()
    before.create_profile(profile())
    before.save_snapshot(snapshot())
    before.create_session(
        McpSession(
            session_id="session-1",
            workspace_id="workspace-a",
            profile_id="profile-1",
            profile_revision=1,
            snapshot_id="snapshot-1",
            state=SessionState.CONNECTED,
        )
    )
    before.create_operation(
        McpOperation(
            operation_id="running-operation",
            correlation_id="correlation-running",
            workspace_id="workspace-a",
            profile_id="profile-1",
            session_id="session-1",
            kind=OperationKind.MANUAL_CALL,
            state=OperationState.RUNNING,
        )
    )
    before.create_operation(
        McpOperation(
            operation_id="queued-operation",
            correlation_id="correlation-queued",
            workspace_id="workspace-a",
            profile_id="profile-1",
            kind=OperationKind.REFRESH,
            state=OperationState.QUEUED,
        )
    )

    after = McpRepository(db_path)
    recovery = after.initialize()

    assert recovery.operations_interrupted == 1
    assert recovery.sessions_disconnected == 1
    assert recovery.snapshots_marked_stale == 1
    recovered_operation = after.get_operation("workspace-a", "running-operation")
    assert recovered_operation.state == OperationState.INTERRUPTED
    assert recovered_operation.error_code == "backend_restarted"
    assert after.get_operation("workspace-a", "queued-operation").state == OperationState.QUEUED
    recovered_session = after.get_session("workspace-a", "session-1")
    assert recovered_session.state == SessionState.DISCONNECTED
    assert recovered_session.stale is True
    assert after.get_snapshot("workspace-a", "snapshot-1").stale is True


def test_profile_serialization_contains_refs_but_never_inline_secret(repository):
    repository.create_profile(profile())
    database_bytes = repository.db_path.read_bytes()

    assert b"MCP_FIXTURE_AUTH" in database_bytes
    assert b"raw-secret-value" not in database_bytes


def test_stdio_profile_persists_argv_without_shell_expansion(repository):
    saved = repository.create_profile(
        McpProfile(
            profile_id="stdio-profile",
            workspace_id="workspace-a",
            label="Local fixture",
            transport=StdioTransport(
                command="python",
                args=["-m", "bashgym.mcp.sandbox_server"],
                env_secret_refs={"FIXTURE_TOKEN": "MCP_FIXTURE_TOKEN"},
            ),
        )
    )

    loaded = repository.get_profile("workspace-a", saved.profile_id)
    assert isinstance(loaded.transport, StdioTransport)
    assert loaded.transport.command == "python"
    assert loaded.transport.args == ["-m", "bashgym.mcp.sandbox_server"]


def test_stdio_launch_approval_is_revision_and_workspace_scoped(repository):
    transport = StdioTransport(
        command="python",
        args=["-m", "fixture.server"],
        cwd_policy="isolated",
        env_secret_refs={"FIXTURE_TOKEN": "MCP_FIXTURE_TOKEN"},
        sandbox_policy="required",
    )
    repository.create_profile(
        McpProfile(
            profile_id="stdio-profile",
            workspace_id="workspace-a",
            label="Local fixture",
            transport=transport,
        )
    )
    approval = McpStdioLaunchApproval.for_transport(
        workspace_id="workspace-a",
        profile_id="stdio-profile",
        profile_revision=1,
        executable_fingerprint="b" * 64,
        transport=transport,
    )

    repository.save_stdio_launch_approval(approval)
    assert repository.get_stdio_launch_approval("workspace-a", "stdio-profile", 1) == approval
    with pytest.raises(RecordNotFoundError, match="stdio launch approval not found"):
        repository.get_stdio_launch_approval("workspace-b", "stdio-profile", 1)

    with sqlite3.connect(repository.db_path) as connection:
        fingerprint_json = connection.execute(
            "SELECT fingerprint_json FROM mcp_stdio_launch_approvals"
        ).fetchone()[0]
    assert "MCP_FIXTURE_TOKEN" not in fingerprint_json
    assert "fixture.server" not in fingerprint_json


def test_session_snapshot_refresh_and_filtered_listing(repository):
    repository.create_profile(profile())
    repository.save_snapshot(snapshot(snapshot_id="snapshot-1"))
    repository.save_snapshot(snapshot(snapshot_id="snapshot-2"))
    session = repository.create_session(
        McpSession(
            session_id="session-1",
            workspace_id="workspace-a",
            profile_id="profile-1",
            profile_revision=1,
            snapshot_id="snapshot-1",
            state=SessionState.CONNECTED,
        )
    )

    refreshed = repository.update_session_snapshot(
        "workspace-a",
        session.session_id,
        "snapshot-2",
        expected_revision=session.revision,
    )

    assert refreshed.snapshot_id == "snapshot-2"
    assert refreshed.revision == 2
    assert repository.list_sessions("workspace-a", profile_id="profile-1", live_only=True) == [
        refreshed
    ]
    assert repository.list_sessions("workspace-b") == []
