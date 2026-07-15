"""Desktop MCP Workbench profile, inspection, and manual-call routes."""

from __future__ import annotations

from typing import Any, Literal, Never
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bashgym.config import get_bashgym_dir
from bashgym.mcp.claude_compat import preview_claude_mcp_config
from bashgym.mcp.contracts import (
    McpCapabilitySnapshot,
    McpOperation,
    McpProfile,
    StdioTransport,
    StreamableHttpTransport,
)
from bashgym.mcp.persistence import (
    McpRepository,
    ProfileInUseError,
    RecordAlreadyExistsError,
    RecordNotFoundError,
    RevisionConflictError,
)
from bashgym.mcp.policy import McpPolicyError
from bashgym.mcp.service import ApprovalRequiredError, McpServiceError, McpWorkbenchService

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RemoteProfileInput(ApiModel):
    url: str = Field(min_length=1, max_length=2048)
    header_secret_refs: dict[str, str] = Field(default_factory=dict)
    allow_private_network: bool = False
    auth_mode: Literal["auto", "oauth", "headers", "none"] = "auto"
    oauth_scopes: list[str] = Field(default_factory=list, max_length=64)
    oauth_callback_port: int | None = Field(default=None, ge=1024, le=65535)
    oauth_client_id: str | None = Field(default=None, max_length=512)
    oauth_client_secret_ref: str | None = Field(default=None, max_length=160)


class StdioProfileInput(ApiModel):
    command: str = Field(min_length=1, max_length=2048)
    args: list[str] = Field(default_factory=list, max_length=256)
    cwd_policy: Literal["workspace", "isolated", "explicit"] = "workspace"
    cwd: str | None = Field(default=None, max_length=4096)
    env_secret_refs: dict[str, str] = Field(default_factory=dict)
    sandbox_policy: Literal["required", "preferred", "disabled"] = "preferred"


class ProfileInput(ApiModel):
    workspace_id: str
    label: str = Field(min_length=1, max_length=160)
    transport: Literal["streamable_http", "stdio"]
    remote: RemoteProfileInput | None = None
    stdio: StdioProfileInput | None = None
    enabled: bool = True
    catalog_source: str | None = Field(default=None, max_length=160)
    policy_id: str | None = None

    @model_validator(mode="after")
    def validate_transport_payload(self) -> ProfileInput:
        if self.transport == "streamable_http":
            if self.remote is None or self.stdio is not None:
                raise ValueError("Streamable HTTP profiles require remote config only")
        elif self.stdio is None or self.remote is not None:
            raise ValueError("stdio profiles require stdio config only")
        return self

    def transport_contract(self) -> StreamableHttpTransport | StdioTransport:
        if self.transport == "streamable_http":
            assert self.remote is not None
            return StreamableHttpTransport(**self.remote.model_dump())
        assert self.stdio is not None
        return StdioTransport(**self.stdio.model_dump())


class ProfileUpdateInput(ProfileInput):
    expected_revision: int = Field(ge=1)


class ProfileDeleteInput(ApiModel):
    workspace_id: str
    expected_revision: int = Field(ge=1)


class ConnectInput(ApiModel):
    workspace_id: str
    profile_revision: int = Field(ge=1)
    idempotency_key: str | None = Field(default=None, max_length=160)


class ManagedProfileInput(ApiModel):
    workspace_id: str
    idempotency_key: str | None = Field(default=None, max_length=160)


class ManagedSessionInput(ApiModel):
    workspace_id: str
    idempotency_key: str | None = Field(default=None, max_length=160)


class ManualCallInput(ApiModel):
    workspace_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    approved: bool = False
    typed_confirmation: str | None = Field(default=None, max_length=160)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_result_bytes: int = Field(default=1024 * 1024, ge=1024, le=8 * 1024 * 1024)
    idempotency_key: str | None = Field(default=None, max_length=160)


class WorkspaceInput(ApiModel):
    workspace_id: str
    idempotency_key: str | None = Field(default=None, max_length=160)


class StdioApprovalInput(ApiModel):
    workspace_id: str
    profile_revision: int = Field(ge=1)
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ClaudeConfigPreviewInput(ApiModel):
    workspace_id: str
    config: dict[str, Any]
    source_scope: Literal["local", "project", "user"] = "project"


def _service(request: Request) -> McpWorkbenchService:
    current = getattr(request.app.state, "mcp_workbench", None)
    if isinstance(current, McpWorkbenchService):
        return current
    repository = McpRepository(get_bashgym_dir() / "mcp" / "mcp.sqlite3")
    observer = getattr(request.app.state, "runtime_observer", None)
    workspace_root = getattr(observer, "workspace_root", None)
    current = McpWorkbenchService(repository, workspace_root=workspace_root)
    current.initialize()
    request.app.state.mcp_workbench = current
    return current


def _raise_api(exc: Exception) -> Never:
    if isinstance(exc, RecordNotFoundError):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": str(exc)})
    if isinstance(exc, RevisionConflictError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "revision_conflict",
                "message": str(exc),
                "expected": exc.expected,
                "current": exc.current,
            },
        )
    if isinstance(exc, (RecordAlreadyExistsError, ProfileInUseError)):
        raise HTTPException(status_code=409, detail={"code": "conflict", "message": str(exc)})
    if isinstance(exc, ApprovalRequiredError):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.safe_message, "retryable": False},
        )
    if isinstance(exc, McpServiceError):
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.safe_message, "retryable": exc.retryable},
        )
    if isinstance(exc, McpPolicyError):
        raise HTTPException(
            status_code=400,
            detail={"code": "policy_denied", "message": str(exc), "retryable": False},
        )
    raise exc


def _profile_payload(service: McpWorkbenchService, profile: McpProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "workspace_id": profile.workspace_id,
        "profile_revision": profile.revision,
        "label": profile.label,
        "transport": profile.transport.type,
        "remote": None,
        "stdio": None,
        "enabled": profile.enabled,
        "catalog_source": profile.catalog_source,
        "policy_id": profile.policy_id,
        "active_session_id": service.active_session_id(profile.workspace_id, profile.profile_id),
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
    }
    if isinstance(profile.transport, StreamableHttpTransport):
        payload["remote"] = profile.transport.model_dump(mode="json", exclude={"type"})
    else:
        payload["stdio"] = profile.transport.model_dump(mode="json", exclude={"type"})
    return payload


def _snapshot_payload(snapshot: McpCapabilitySnapshot) -> dict[str, Any]:
    return snapshot.model_dump(mode="json")


_OPERATION_STATUS = {
    "queued": "queued",
    "running": "running",
    "waiting_for_approval": "awaiting_approval",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "cancelled_upstream_unknown": "cancelled",
    "interrupted": "interrupted",
}


def _operation_payload(operation: McpOperation, *, accepted: bool = False) -> dict[str, Any]:
    payload = {
        "operation_id": operation.operation_id,
        "correlation_id": operation.correlation_id,
        "kind": operation.kind.value,
        "status": _OPERATION_STATUS[operation.state.value],
        "phase": operation.state.value,
        "error": operation.safe_message,
        "error_code": operation.error_code,
        "result": operation.result,
        "created_at": operation.created_at.isoformat(),
        "updated_at": operation.updated_at.isoformat(),
    }
    if accepted:
        payload.update(
            {
                "status_url": f"/api/mcp/operations/{operation.operation_id}",
                "cancel_url": f"/api/mcp/operations/{operation.operation_id}/cancel",
            }
        )
    return payload


@router.get("/profiles")
async def list_profiles(request: Request, workspace_id: str = Query(...)):
    service = _service(request)
    try:
        return [_profile_payload(service, item) for item in service.list_profiles(workspace_id)]
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles", status_code=status.HTTP_201_CREATED)
async def create_profile(request: Request, body: ProfileInput):
    service = _service(request)
    try:
        profile = McpProfile(
            profile_id=f"mcp_profile_{uuid4().hex}",
            workspace_id=body.workspace_id,
            label=body.label,
            transport=body.transport_contract(),
            enabled=body.enabled,
            catalog_source=body.catalog_source,
            policy_id=body.policy_id,
        )
        return _profile_payload(service, service.create_profile(profile))
    except Exception as exc:
        _raise_api(exc)


@router.put("/profiles/{profile_id}")
async def update_profile(request: Request, profile_id: str, body: ProfileUpdateInput):
    service = _service(request)
    try:
        current = service.get_profile(body.workspace_id, profile_id)
        proposed = current.model_copy(
            update={
                "label": body.label,
                "transport": body.transport_contract(),
                "enabled": body.enabled,
                "catalog_source": body.catalog_source,
                "policy_id": body.policy_id,
            }
        )
        updated = service.update_profile(proposed, expected_revision=body.expected_revision)
        return _profile_payload(service, updated)
    except Exception as exc:
        _raise_api(exc)


@router.delete("/profiles/{profile_id}")
async def delete_profile(request: Request, profile_id: str, body: ProfileDeleteInput):
    service = _service(request)
    try:
        return _profile_payload(
            service,
            service.delete_profile(
                body.workspace_id, profile_id, expected_revision=body.expected_revision
            ),
        )
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles/{profile_id}/connect", status_code=status.HTTP_202_ACCEPTED)
async def connect_profile(request: Request, profile_id: str, body: ConnectInput):
    service = _service(request)
    try:
        operation = service.start_connect(
            body.workspace_id,
            profile_id,
            profile_revision=body.profile_revision,
            idempotency_key=body.idempotency_key,
        )
        return _operation_payload(operation, accepted=True)
    except Exception as exc:
        _raise_api(exc)


@router.get("/profiles/{profile_id}/snapshot")
async def get_profile_snapshot(request: Request, profile_id: str, workspace_id: str = Query(...)):
    try:
        return _snapshot_payload(_service(request).get_snapshot(workspace_id, profile_id))
    except Exception as exc:
        _raise_api(exc)


@router.get("/profiles/{profile_id}/oauth/status")
async def get_oauth_status(request: Request, profile_id: str, workspace_id: str = Query(...)):
    try:
        return _service(request).oauth_status(workspace_id, profile_id)
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles/{profile_id}/oauth/logout")
async def logout_oauth(request: Request, profile_id: str, body: WorkspaceInput):
    try:
        return await _service(request).logout_oauth(body.workspace_id, profile_id)
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles/{profile_id}/refresh", status_code=status.HTTP_202_ACCEPTED)
async def refresh_profile(request: Request, profile_id: str, body: ManagedProfileInput):
    try:
        return _operation_payload(
            _service(request).start_refresh(
                body.workspace_id, profile_id, idempotency_key=body.idempotency_key
            ),
            accepted=True,
        )
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles/{profile_id}/quick-test", status_code=status.HTTP_202_ACCEPTED)
async def quick_test_profile(request: Request, profile_id: str, body: ManagedProfileInput):
    try:
        return _operation_payload(
            _service(request).start_quick_test(
                body.workspace_id, profile_id, idempotency_key=body.idempotency_key
            ),
            accepted=True,
        )
    except Exception as exc:
        _raise_api(exc)


@router.get("/profiles/{profile_id}/stdio/preview")
async def preview_stdio(
    request: Request,
    profile_id: str,
    workspace_id: str = Query(...),
    profile_revision: int = Query(..., ge=1),
):
    try:
        return _service(request).preview_stdio_launch(workspace_id, profile_id, profile_revision)
    except Exception as exc:
        _raise_api(exc)


@router.post("/profiles/{profile_id}/stdio/approve")
async def approve_stdio(request: Request, profile_id: str, body: StdioApprovalInput):
    try:
        return (
            _service(request)
            .approve_stdio_launch(
                body.workspace_id,
                profile_id,
                body.profile_revision,
                executable_sha256=body.executable_sha256,
                launch_fingerprint=body.launch_fingerprint,
            )
            .model_dump(mode="json")
        )
    except Exception as exc:
        _raise_api(exc)


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str, workspace_id: str = Query(...)):
    try:
        return _service(request).get_session(workspace_id, session_id).model_dump(mode="json")
    except Exception as exc:
        _raise_api(exc)


@router.post("/sessions/{session_id}/disconnect", status_code=status.HTTP_202_ACCEPTED)
async def disconnect_session(request: Request, session_id: str, body: ManagedSessionInput):
    try:
        return _operation_payload(
            _service(request).start_disconnect(
                body.workspace_id, session_id, idempotency_key=body.idempotency_key
            ),
            accepted=True,
        )
    except Exception as exc:
        _raise_api(exc)


@router.post(
    "/sessions/{session_id}/tools/{tool_name}/call",
    status_code=status.HTTP_202_ACCEPTED,
)
async def call_tool(request: Request, session_id: str, tool_name: str, body: ManualCallInput):
    try:
        return _operation_payload(
            _service(request).start_manual_call(
                body.workspace_id,
                session_id,
                tool_name,
                body.arguments,
                approved=body.approved,
                typed_confirmation=body.typed_confirmation,
                timeout_seconds=body.timeout_seconds,
                max_result_bytes=body.max_result_bytes,
                idempotency_key=body.idempotency_key,
            ),
            accepted=True,
        )
    except Exception as exc:
        _raise_api(exc)


@router.get("/operations/{operation_id}")
async def get_operation(request: Request, operation_id: str, workspace_id: str = Query(...)):
    try:
        return _operation_payload(_service(request).get_operation(workspace_id, operation_id))
    except Exception as exc:
        _raise_api(exc)


@router.post("/operations/{operation_id}/cancel")
async def cancel_operation(request: Request, operation_id: str, body: WorkspaceInput):
    try:
        return _operation_payload(
            await _service(request).cancel_operation(body.workspace_id, operation_id)
        )
    except Exception as exc:
        _raise_api(exc)


@router.post("/self-test", status_code=status.HTTP_202_ACCEPTED)
async def start_self_test(request: Request, body: WorkspaceInput):
    try:
        return _operation_payload(
            _service(request).start_self_test(
                body.workspace_id, idempotency_key=body.idempotency_key
            ),
            accepted=True,
        )
    except Exception as exc:
        _raise_api(exc)


@router.post("/imports/claude/preview")
async def preview_claude_config(body: ClaudeConfigPreviewInput):
    _ = body.workspace_id  # Workspace ownership is applied only when a candidate is saved.
    try:
        return [
            candidate.model_dump(mode="json")
            for candidate in preview_claude_mcp_config(body.config, source_scope=body.source_scope)
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_claude_mcp_config", "message": str(exc)},
        ) from exc
