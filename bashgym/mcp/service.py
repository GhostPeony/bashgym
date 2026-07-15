"""Application service for the MCP Workbench inspection milestone.

The service joins durable workspace-scoped records to the task-owned official
SDK runtime. API routes stay thin and never receive secret values or SDK types.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from bashgym.mcp.client_runtime import (
    McpClientRuntime,
    McpOAuthError,
    McpRuntimeError,
    SessionClosedError,
    SessionNotFoundError,
)
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
from bashgym.mcp.oauth import (
    CredentialOAuthStorage,
    OAuthRuntimeConfig,
    oauth_status,
    oauth_storage_namespace,
)
from bashgym.mcp.operations import InvalidOperationTransitionError
from bashgym.mcp.persistence import (
    McpRepository,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.mcp.policy import McpPolicyError, SecretResolutionError, prepare_stdio_launch
from bashgym.secrets import get_secret

logger = logging.getLogger(__name__)


class McpServiceError(RuntimeError):
    """A stable, secret-safe service error suitable for the API boundary."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        self.code = code
        self.safe_message = message
        self.retryable = retryable
        super().__init__(message)


class ApprovalRequiredError(McpServiceError):
    def __init__(self, message: str = "Explicit approval is required before this action."):
        super().__init__("approval_required", message)


OperationRunner = Callable[[McpOperation], Awaitable[dict[str, Any] | None]]


def _record_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _tool_policy(tool: dict[str, Any], *, bundled_reference: bool) -> str:
    if bundled_reference:
        return "allow"
    annotations = tool.get("annotations")
    if isinstance(annotations, dict) and annotations.get("destructiveHint") is True:
        return "ask"
    return "ask"


def _normalize_tools(
    tools: list[dict[str, Any]], *, bundled_reference: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    names: set[str] = set()
    for item in tools:
        name = str(item.get("name") or "").strip()
        if not name:
            warnings.append("tool_missing_name")
            continue
        if name in names:
            warnings.append(f"duplicate_tool_name:{name}")
        names.add(name)
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            warnings.append(f"weak_description:{name}")
        elif len(description) > 2048:
            warnings.append(f"claude_description_over_2kb:{name}")
        input_schema = item.get("inputSchema", item.get("input_schema", {}))
        if not isinstance(input_schema, dict):
            warnings.append(f"invalid_input_schema:{name}")
            input_schema = {}
        elif any(keyword in input_schema for keyword in ("anyOf", "oneOf", "allOf")):
            warnings.append(f"claude_root_schema_projection:{name}")
        output_schema = item.get("outputSchema", item.get("output_schema"))
        if output_schema is not None and not isinstance(output_schema, dict):
            warnings.append(f"invalid_output_schema:{name}")
            output_schema = None
        normalized.append(
            {
                "name": name,
                "title": item.get("title"),
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "annotations": item.get("annotations") or {},
                "_meta": item.get("_meta") or {},
                "policy": _tool_policy(item, bundled_reference=bundled_reference),
            }
        )
    return normalized, warnings


class McpWorkbenchService:
    """Coordinate profiles, live SDK sessions, snapshots, and operations."""

    def __init__(
        self,
        repository: McpRepository,
        *,
        runtime: McpClientRuntime | None = None,
        workspace_root: str | Path | None = None,
        secret_resolver: Callable[[str], str | None] = get_secret,
    ) -> None:
        self.repository = repository
        self.runtime = runtime or McpClientRuntime()
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.secret_resolver = secret_resolver
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._operation_session_ids: dict[str, str] = {}
        self._active_profile_sessions: dict[tuple[str, str], str] = {}
        self._isolated_workspaces: dict[str, tempfile.TemporaryDirectory[str]] = {}

    def initialize(self):
        return self.repository.initialize()

    def list_profiles(self, workspace_id: str) -> list[McpProfile]:
        return self.repository.list_profiles(workspace_id)

    def get_profile(self, workspace_id: str, profile_id: str) -> McpProfile:
        return self.repository.get_profile(workspace_id, profile_id)

    def active_session_id(self, workspace_id: str, profile_id: str) -> str | None:
        return self._active_profile_sessions.get((workspace_id, profile_id))

    def oauth_status(self, workspace_id: str, profile_id: str) -> dict[str, Any]:
        profile = self.repository.get_profile(workspace_id, profile_id)
        if not isinstance(profile.transport, StreamableHttpTransport):
            raise McpServiceError(
                "invalid_profile", "OAuth is available only for hosted HTTP MCPs."
            )
        storage = CredentialOAuthStorage(
            oauth_storage_namespace(workspace_id, profile_id, profile.transport.url)
        )
        return {
            "profile_id": profile_id,
            "auth_mode": profile.transport.auth_mode,
            "interactive_oauth": profile.transport.auth_mode in {"auto", "oauth"},
            **oauth_status(storage),
        }

    async def logout_oauth(self, workspace_id: str, profile_id: str) -> dict[str, Any]:
        profile = self.repository.get_profile(workspace_id, profile_id)
        if not isinstance(profile.transport, StreamableHttpTransport):
            raise McpServiceError(
                "invalid_profile", "OAuth is available only for hosted HTTP MCPs."
            )
        session_id = self.active_session_id(workspace_id, profile_id)
        if session_id:
            try:
                await self.runtime.abort(session_id)
            except (SessionClosedError, SessionNotFoundError):
                pass
            try:
                self.repository.mark_session_lost(workspace_id, session_id)
            except RecordNotFoundError:
                pass
            self._active_profile_sessions.pop((workspace_id, profile_id), None)
        storage = CredentialOAuthStorage(
            oauth_storage_namespace(workspace_id, profile_id, profile.transport.url)
        )
        cleared = await asyncio.to_thread(storage.clear)
        return {"profile_id": profile_id, "authenticated": False, "cleared": cleared}

    def create_profile(self, profile: McpProfile) -> McpProfile:
        return self.repository.create_profile(profile)

    def update_profile(self, profile: McpProfile, *, expected_revision: int) -> McpProfile:
        return self.repository.update_profile(profile, expected_revision=expected_revision)

    def delete_profile(
        self, workspace_id: str, profile_id: str, *, expected_revision: int
    ) -> McpProfile:
        return self.repository.tombstone_profile(
            workspace_id, profile_id, expected_revision=expected_revision
        )

    def get_snapshot(self, workspace_id: str, profile_id: str) -> McpCapabilitySnapshot:
        return self.repository.latest_snapshot(workspace_id, profile_id)

    def get_session(self, workspace_id: str, session_id: str) -> McpSession:
        return self.repository.get_session(workspace_id, session_id)

    def get_operation(self, workspace_id: str, operation_id: str) -> McpOperation:
        return self.repository.get_operation(workspace_id, operation_id)

    def preview_stdio_launch(
        self, workspace_id: str, profile_id: str, profile_revision: int
    ) -> dict[str, Any]:
        profile = self.repository.get_profile_revision(
            workspace_id, profile_id, profile_revision
        ).profile
        if not isinstance(profile.transport, StdioTransport):
            raise McpServiceError("invalid_profile", "This profile does not use stdio.")
        launch = prepare_stdio_launch(profile.transport.command, profile.transport.args)
        approval = McpStdioLaunchApproval.for_transport(
            workspace_id=workspace_id,
            profile_id=profile_id,
            profile_revision=profile_revision,
            executable_fingerprint=launch.fingerprint.sha256,
            transport=profile.transport,
        )
        return {
            "profile_id": profile_id,
            "profile_revision": profile_revision,
            "command": launch.command,
            "args": list(launch.args),
            "cwd_policy": profile.transport.cwd_policy,
            "env_names": sorted(profile.transport.env_secret_refs),
            "sandbox_policy": profile.transport.sandbox_policy,
            "executable": launch.fingerprint.to_dict(),
            "launch_fingerprint": approval.launch_fingerprint,
        }

    def approve_stdio_launch(
        self,
        workspace_id: str,
        profile_id: str,
        profile_revision: int,
        *,
        executable_sha256: str,
        launch_fingerprint: str,
    ) -> McpStdioLaunchApproval:
        profile = self.repository.get_profile_revision(
            workspace_id, profile_id, profile_revision
        ).profile
        if not isinstance(profile.transport, StdioTransport):
            raise McpServiceError("invalid_profile", "This profile does not use stdio.")
        launch = prepare_stdio_launch(profile.transport.command, profile.transport.args)
        approval = McpStdioLaunchApproval.for_transport(
            workspace_id=workspace_id,
            profile_id=profile_id,
            profile_revision=profile_revision,
            executable_fingerprint=launch.fingerprint.sha256,
            transport=profile.transport,
        )
        if (
            executable_sha256 != approval.executable_fingerprint
            or launch_fingerprint != approval.launch_fingerprint
        ):
            raise McpServiceError(
                "launch_changed",
                "The executable or launch configuration changed; preview it again before approval.",
            )
        return self.repository.save_stdio_launch_approval(approval)

    def _new_operation(
        self,
        workspace_id: str,
        kind: OperationKind,
        *,
        profile_id: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[McpOperation, bool]:
        proposed = McpOperation(
            operation_id=_record_id("mcp_operation"),
            correlation_id=_record_id("mcp_correlation"),
            workspace_id=workspace_id,
            kind=kind,
            profile_id=profile_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        saved = self.repository.create_operation(proposed)
        return saved, saved.operation_id == proposed.operation_id

    def _schedule(
        self,
        operation: McpOperation,
        created: bool,
        runner: OperationRunner,
    ) -> McpOperation:
        if not created:
            return operation
        task = asyncio.create_task(
            self._run_operation(operation, runner),
            name=f"bashgym-{operation.operation_id}",
        )
        self._tasks[operation.operation_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(operation.operation_id, None))
        return operation

    def _advance(
        self,
        operation: McpOperation,
        state: OperationState,
        *,
        error_code: str | None = None,
        safe_message: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> McpOperation:
        current = self.repository.get_operation(operation.workspace_id, operation.operation_id)
        return self.repository.update_operation_state(
            operation.workspace_id,
            operation.operation_id,
            state,
            expected_revision=current.revision,
            error_code=error_code,
            safe_message=safe_message,
            result=result,
        )

    async def _run_operation(self, operation: McpOperation, runner: OperationRunner) -> None:
        try:
            running = self._advance(operation, OperationState.RUNNING)
            result = await runner(running)
            self._advance(running, OperationState.COMPLETED, result=result)
        except asyncio.CancelledError:
            try:
                current = self.repository.get_operation(
                    operation.workspace_id, operation.operation_id
                )
                if current.state not in {
                    OperationState.CANCELLED,
                    OperationState.COMPLETED,
                    OperationState.FAILED,
                }:
                    self._advance(current, OperationState.CANCELLED)
            except Exception:
                logger.debug("Could not persist cancelled MCP operation %s", operation.operation_id)
            raise
        except (McpServiceError, McpPolicyError, McpRuntimeError) as exc:
            code, message = self._safe_error(exc)
            try:
                self._advance(
                    operation, OperationState.FAILED, error_code=code, safe_message=message
                )
            except Exception:
                logger.debug("Could not persist failed MCP operation %s", operation.operation_id)
        except Exception as exc:
            logger.error(
                "MCP operation %s failed (%s)",
                operation.operation_id,
                exc.__class__.__name__,
            )
            try:
                self._advance(
                    operation,
                    OperationState.FAILED,
                    error_code="upstream_error",
                    safe_message="The MCP operation failed without exposing upstream details.",
                )
            except Exception:
                logger.debug("Could not persist failed MCP operation %s", operation.operation_id)

    @staticmethod
    def _safe_error(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, McpServiceError):
            return exc.code, exc.safe_message
        if isinstance(exc, McpOAuthError):
            return exc.code, exc.safe_message
        if isinstance(exc, SecretResolutionError):
            return "auth_required", "A configured MCP secret reference is unavailable."
        if isinstance(exc, McpPolicyError):
            return "policy_denied", str(exc)
        if isinstance(exc, SessionNotFoundError):
            return "session_not_found", "The live MCP session is unavailable."
        if isinstance(exc, SessionClosedError):
            return "session_closed", "The MCP session closed before the operation completed."
        return "upstream_error", "The MCP server returned an error."

    def start_connect(
        self,
        workspace_id: str,
        profile_id: str,
        *,
        profile_revision: int,
        idempotency_key: str | None = None,
    ) -> McpOperation:
        profile = self.repository.get_profile(workspace_id, profile_id)
        if profile.revision != profile_revision:
            raise RevisionConflictError("profile", profile_revision, profile.revision)
        operation, created = self._new_operation(
            workspace_id,
            OperationKind.CONNECT,
            profile_id=profile_id,
            idempotency_key=idempotency_key,
        )
        return self._schedule(
            operation,
            created,
            lambda active: self._connect(active, profile),
        )

    async def _connect(self, operation: McpOperation, profile: McpProfile) -> dict[str, Any]:
        session_id = _record_id("mcp_session")
        self._operation_session_ids[operation.operation_id] = session_id
        isolated: tempfile.TemporaryDirectory[str] | None = None
        try:
            if isinstance(profile.transport, StreamableHttpTransport):
                authorization_header = any(
                    name.lower() == "authorization" for name in profile.transport.header_secret_refs
                )
                use_oauth = profile.transport.auth_mode == "oauth" or (
                    profile.transport.auth_mode == "auto" and not authorization_header
                )
                oauth_storage: CredentialOAuthStorage | None = None
                oauth_config: OAuthRuntimeConfig | None = None
                if use_oauth:
                    oauth_storage = CredentialOAuthStorage(
                        oauth_storage_namespace(
                            profile.workspace_id,
                            profile.profile_id,
                            profile.transport.url,
                        )
                    )
                    client_secret: str | None = None
                    if profile.transport.oauth_client_secret_ref:
                        client_secret = self.secret_resolver(
                            profile.transport.oauth_client_secret_ref
                        )
                        if not client_secret:
                            raise SecretResolutionError(
                                "OAuth client secret reference is unavailable"
                            )
                    oauth_config = OAuthRuntimeConfig(
                        storage=oauth_storage,
                        scopes=tuple(profile.transport.oauth_scopes),
                        callback_port=profile.transport.oauth_callback_port,
                        client_id=profile.transport.oauth_client_id,
                        client_secret=client_secret,
                    )
                connection = await self.runtime.connect_http(
                    session_id,
                    profile.transport.url,
                    secret_header_refs=profile.transport.header_secret_refs,
                    resolve_secret=self.secret_resolver,
                    allow_private_network=profile.transport.allow_private_network,
                    oauth_config=oauth_config,
                )
                connection["oauthAuthenticated"] = bool(
                    oauth_storage and oauth_storage.has_tokens()
                )
            else:
                approval = self.repository.get_stdio_launch_approval(
                    profile.workspace_id, profile.profile_id, profile.revision
                )
                launch = prepare_stdio_launch(profile.transport.command, profile.transport.args)
                expected = McpStdioLaunchApproval.for_transport(
                    workspace_id=profile.workspace_id,
                    profile_id=profile.profile_id,
                    profile_revision=profile.revision,
                    executable_fingerprint=launch.fingerprint.sha256,
                    transport=profile.transport,
                )
                if (
                    approval.executable_fingerprint != expected.executable_fingerprint
                    or approval.fingerprint != expected.fingerprint
                    or approval.launch_fingerprint != expected.launch_fingerprint
                ):
                    raise ApprovalRequiredError(
                        "The stdio launch changed and must be approved again."
                    )
                if profile.transport.sandbox_policy == "required":
                    raise McpServiceError(
                        "sandbox_unavailable",
                        "Required stdio sandbox enforcement is not available in this build.",
                    )
                cwd: str | None
                if profile.transport.cwd_policy == "workspace":
                    cwd = str(self.workspace_root)
                elif profile.transport.cwd_policy == "isolated":
                    isolated = tempfile.TemporaryDirectory(prefix="bashgym-mcp-")
                    cwd = isolated.name
                else:
                    explicit_cwd = (
                        Path(profile.transport.cwd or "").expanduser().resolve(strict=True)
                    )
                    if not explicit_cwd.is_dir():
                        raise McpServiceError(
                            "invalid_profile",
                            "The explicit stdio working directory is not a directory.",
                        )
                    cwd = str(explicit_cwd)
                connection = await self.runtime.connect_stdio(
                    session_id,
                    profile.transport.command,
                    profile.transport.args,
                    cwd=cwd,
                    secret_env_refs=profile.transport.env_secret_refs,
                    resolve_secret=self.secret_resolver,
                    expected_fingerprint=launch.fingerprint,
                )

            snapshot = self._snapshot_from_connection(profile, connection)
            try:
                previous = self.repository.latest_snapshot(profile.workspace_id, profile.profile_id)
            except RecordNotFoundError:
                previous = None
            if previous is not None and snapshot.contract_hash != previous.contract_hash:
                snapshot = snapshot.model_copy(update={"drifted": True})
            self.repository.save_snapshot(snapshot)
            session = McpSession(
                session_id=session_id,
                workspace_id=profile.workspace_id,
                profile_id=profile.profile_id,
                profile_revision=profile.revision,
                snapshot_id=snapshot.snapshot_id,
                state=SessionState.CONNECTED,
            )
            self.repository.create_session(session)
            self._active_profile_sessions[(profile.workspace_id, profile.profile_id)] = session_id
            if isolated is not None:
                self._isolated_workspaces[session_id] = isolated
                isolated = None
            return {
                "session_id": session_id,
                "profile_revision": profile.revision,
                "snapshot_id": snapshot.snapshot_id,
                "negotiated_protocol_version": snapshot.negotiated_protocol_version,
                "oauth_authenticated": bool(connection.get("oauthAuthenticated")),
            }
        except Exception:
            try:
                await self.runtime.abort(session_id)
            except Exception:
                pass
            raise
        finally:
            if isolated is not None:
                isolated.cleanup()
            self._operation_session_ids.pop(operation.operation_id, None)

    def _snapshot_from_connection(
        self, profile: McpProfile, connection: dict[str, Any]
    ) -> McpCapabilitySnapshot:
        initialization = connection.get("initialization") or {}
        inventory = connection.get("inventory") or {}
        server_info = initialization.get("serverInfo") or initialization.get("server_info") or {}
        bundled = server_info.get("name") == "bashgym-mcp-reference"
        tools, warnings = _normalize_tools(
            list(inventory.get("tools") or []), bundled_reference=bundled
        )
        warnings.extend(str(item) for item in inventory.get("warnings") or [])
        instructions = initialization.get("instructions")
        if isinstance(instructions, str) and len(instructions) > 2048:
            warnings.append("claude_server_instructions_over_2kb")
        if server_info.get("name") == "workspace":
            warnings.append("claude_reserved_server_name:workspace")
        return McpCapabilitySnapshot(
            snapshot_id=_record_id("mcp_snapshot"),
            workspace_id=profile.workspace_id,
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            negotiated_protocol_version=str(
                initialization.get("protocolVersion")
                or initialization.get("protocol_version")
                or "unknown"
            ),
            server_info=server_info,
            instructions=instructions,
            capabilities=initialization.get("capabilities") or {},
            tools=tools,
            resources=list(inventory.get("resources") or []),
            resource_templates=list(inventory.get("resourceTemplates") or []),
            prompts=list(inventory.get("prompts") or []),
            schema_warnings=warnings,
            upstream_version=server_info.get("version"),
        )

    def start_refresh(
        self,
        workspace_id: str,
        profile_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> McpOperation:
        self.repository.get_profile(workspace_id, profile_id)
        operation, created = self._new_operation(
            workspace_id,
            OperationKind.REFRESH,
            profile_id=profile_id,
            idempotency_key=idempotency_key,
        )
        return self._schedule(operation, created, lambda active: self._refresh(active, profile_id))

    async def _refresh(self, operation: McpOperation, profile_id: str) -> dict[str, Any]:
        session_id = self.active_session_id(operation.workspace_id, profile_id)
        if not session_id:
            raise McpServiceError("session_not_found", "Connect this MCP before refreshing it.")
        session = self.repository.get_session(operation.workspace_id, session_id)
        profile = self.repository.get_profile_revision(
            operation.workspace_id, profile_id, session.profile_revision
        ).profile
        inventory = await self.runtime.refresh(session_id)
        actor_initialization = await self.runtime.initialization(session_id)
        snapshot = self._snapshot_from_connection(
            profile,
            {"initialization": actor_initialization, "inventory": inventory},
        )
        previous = self.repository.get_snapshot(operation.workspace_id, str(session.snapshot_id))
        if snapshot.contract_hash != previous.contract_hash:
            snapshot = snapshot.model_copy(update={"drifted": True})
        self.repository.save_snapshot(snapshot)
        self.repository.update_session_snapshot(
            operation.workspace_id,
            session_id,
            snapshot.snapshot_id,
            expected_revision=session.revision,
        )
        return {"session_id": session_id, "snapshot_id": snapshot.snapshot_id}

    def start_quick_test(
        self, workspace_id: str, profile_id: str, *, idempotency_key: str | None = None
    ) -> McpOperation:
        operation, created = self._new_operation(
            workspace_id,
            OperationKind.QUICK_TEST,
            profile_id=profile_id,
            idempotency_key=idempotency_key,
        )
        return self._schedule(
            operation, created, lambda active: self._quick_test(active, profile_id)
        )

    async def _quick_test(self, operation: McpOperation, profile_id: str) -> dict[str, Any]:
        refreshed = await self._refresh(operation, profile_id)
        snapshot = self.repository.get_snapshot(
            operation.workspace_id, str(refreshed["snapshot_id"])
        )
        return {
            **refreshed,
            "ok": True,
            "tool_count": len(snapshot.tools),
            "resource_count": len(snapshot.resources),
            "prompt_count": len(snapshot.prompts),
            "schema_warning_count": len(snapshot.schema_warnings),
            "contract_hash": snapshot.contract_hash,
        }

    def start_manual_call(
        self,
        workspace_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        approved: bool,
        typed_confirmation: str | None = None,
        timeout_seconds: float = 30,
        max_result_bytes: int = 1024 * 1024,
        idempotency_key: str | None = None,
    ) -> McpOperation:
        session = self.repository.get_session(workspace_id, session_id)
        snapshot = self.repository.get_snapshot(workspace_id, str(session.snapshot_id))
        tool = next((item for item in snapshot.tools if item.get("name") == tool_name), None)
        if tool is None:
            raise McpServiceError(
                "tool_not_found", "The selected tool is not in this session snapshot."
            )
        annotations = tool.get("annotations") if isinstance(tool, dict) else {}
        destructive = isinstance(annotations, dict) and annotations.get("destructiveHint") is True
        if tool.get("policy") != "allow" and not approved:
            raise ApprovalRequiredError()
        if destructive and typed_confirmation != tool_name:
            raise ApprovalRequiredError(
                "Type the exact tool name to approve this destructive call."
            )
        operation, created = self._new_operation(
            workspace_id,
            OperationKind.MANUAL_CALL,
            profile_id=session.profile_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        return self._schedule(
            operation,
            created,
            lambda active: self._manual_call(
                active,
                session_id,
                tool_name,
                arguments,
                timeout_seconds,
                max_result_bytes,
            ),
        )

    async def _manual_call(
        self,
        operation: McpOperation,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        timeout_seconds: float,
        max_result_bytes: int,
    ) -> dict[str, Any]:
        self._operation_session_ids[operation.operation_id] = session_id
        try:
            result = await self.runtime.call_tool(
                session_id,
                tool_name,
                arguments,
                timeout_seconds=timeout_seconds,
                max_result_bytes=max_result_bytes,
            )
            return {"tool_name": tool_name, "tool_result": result}
        finally:
            self._operation_session_ids.pop(operation.operation_id, None)

    def start_disconnect(
        self,
        workspace_id: str,
        session_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> McpOperation:
        session = self.repository.get_session(workspace_id, session_id)
        operation, created = self._new_operation(
            workspace_id,
            OperationKind.DISCONNECT,
            profile_id=session.profile_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
        )
        return self._schedule(
            operation, created, lambda active: self._disconnect(active, session_id)
        )

    async def _disconnect(self, operation: McpOperation, session_id: str) -> dict[str, Any]:
        session = self.repository.get_session(operation.workspace_id, session_id)
        disconnecting = self.repository.update_session_state(
            operation.workspace_id,
            session_id,
            SessionState.DISCONNECTING,
            expected_revision=session.revision,
        )
        try:
            await self.runtime.close(session_id)
        except SessionNotFoundError:
            pass
        disconnected = self.repository.update_session_state(
            operation.workspace_id,
            session_id,
            SessionState.DISCONNECTED,
            expected_revision=disconnecting.revision,
        )
        self._active_profile_sessions.pop((operation.workspace_id, disconnected.profile_id), None)
        isolated = self._isolated_workspaces.pop(session_id, None)
        if isolated is not None:
            isolated.cleanup()
        return {"session_id": session_id, "disconnected": True}

    def start_self_test(
        self, workspace_id: str, *, idempotency_key: str | None = None
    ) -> McpOperation:
        operation, created = self._new_operation(
            workspace_id, OperationKind.SELF_TEST, idempotency_key=idempotency_key
        )
        return self._schedule(operation, created, self._self_test)

    async def _self_test(self, operation: McpOperation) -> dict[str, Any]:
        session_id = _record_id("mcp_selftest")
        try:
            connection = await self.runtime.connect_stdio(
                session_id,
                sys.executable,
                ["-m", "bashgym.mcp.reference_server", "--transport", "stdio"],
                cwd=str(self.workspace_root),
            )
            result = await self.runtime.call_tool(
                session_id,
                "read_fixture",
                {"name": "alpha"},
                timeout_seconds=10,
                max_result_bytes=64 * 1024,
            )
            return {
                "ok": True,
                "protocol_version": (connection.get("initialization") or {}).get("protocolVersion"),
                "tool_count": len((connection.get("inventory") or {}).get("tools") or []),
                "read_fixture": result,
            }
        finally:
            try:
                await self.runtime.close(session_id)
            except (SessionClosedError, SessionNotFoundError):
                pass

    async def cancel_operation(self, workspace_id: str, operation_id: str) -> McpOperation:
        operation = self.repository.get_operation(workspace_id, operation_id)
        if operation.state in {
            OperationState.COMPLETED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.CANCELLED_UPSTREAM_UNKNOWN,
            OperationState.INTERRUPTED,
        }:
            return operation
        try:
            operation = self.repository.update_operation_state(
                workspace_id,
                operation_id,
                OperationState.CANCELLED,
                expected_revision=operation.revision,
            )
        except (RevisionConflictError, InvalidOperationTransitionError):
            operation = self.repository.get_operation(workspace_id, operation_id)
            if operation.state in {
                OperationState.COMPLETED,
                OperationState.FAILED,
                OperationState.CANCELLED,
                OperationState.CANCELLED_UPSTREAM_UNKNOWN,
                OperationState.INTERRUPTED,
            }:
                return operation
        session_id = self._operation_session_ids.get(operation_id)
        if session_id:
            try:
                await self.runtime.abort(session_id)
            except (SessionClosedError, SessionNotFoundError):
                pass
            try:
                lost = self.repository.mark_session_lost(workspace_id, session_id)
                self._active_profile_sessions.pop((workspace_id, lost.profile_id), None)
                isolated = self._isolated_workspaces.pop(session_id, None)
                if isolated is not None:
                    isolated.cleanup()
            except RecordNotFoundError:
                pass
        task = self._tasks.get(operation_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return self.repository.get_operation(workspace_id, operation_id)

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        await self.runtime.aclose()
        for isolated in self._isolated_workspaces.values():
            isolated.cleanup()
        self._isolated_workspaces.clear()
        self._active_profile_sessions.clear()


__all__ = ["ApprovalRequiredError", "McpServiceError", "McpWorkbenchService"]
