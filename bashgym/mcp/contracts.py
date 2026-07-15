"""Stable, SDK-independent contracts for the MCP Workbench.

These models are the boundary between MCP SDK objects, persistence, and the
HTTP API.  In particular, connector profiles contain credential *references*
only.  Secret values belong to BashGym's credential store and are resolved by
the runtime immediately before a connection is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
WorkspaceId = Identifier
SecretRefName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=2, max_length=160, pattern=r"^[A-Z][A-Z0-9_]*$"
    ),
]
HexFingerprint = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]

_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_ARGUMENT = re.compile(
    r"^--?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|bearer[-_]?token|token|password|secret)(?:=|$)",
    re.IGNORECASE,
)
_SECRET_VALUE_PREFIXES = ("sk-", "ghp_", "hf_", "xoxb-", "xoxp-", "bearer ")


class ContractModel(BaseModel):
    """Base configuration shared by all public MCP contracts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenContractModel(BaseModel):
    """Base for immutable evidence records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StreamableHttpTransport(ContractModel):
    """A remote Streamable HTTP endpoint with credential reference names."""

    type: Literal["streamable_http"] = "streamable_http"
    url: str = Field(min_length=1, max_length=2048)
    header_secret_refs: dict[str, SecretRefName] = Field(default_factory=dict)
    allow_private_network: bool = False
    auth_mode: Literal["auto", "oauth", "headers", "none"] = "auto"
    oauth_scopes: list[str] = Field(default_factory=list, max_length=64)
    oauth_callback_port: int | None = Field(default=None, ge=1024, le=65535)
    oauth_client_id: str | None = Field(default=None, max_length=512)
    oauth_client_secret_ref: SecretRefName | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Streamable HTTP URL must use http or https and include a host")
        if parsed.username or parsed.password:
            raise ValueError("Credentials must be secret references, not URL user info")
        if parsed.fragment:
            raise ValueError("Streamable HTTP URL must not include a fragment")
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in {
                "api_key",
                "access_token",
                "refresh_token",
                "token",
                "password",
                "secret",
            } or item.lower().startswith(_SECRET_VALUE_PREFIXES):
                raise ValueError("URL credentials must use header secret references")
        return normalized

    @field_validator("header_secret_refs")
    @classmethod
    def validate_header_names(cls, value: dict[str, SecretRefName]) -> dict[str, SecretRefName]:
        if any(not _HEADER_NAME.fullmatch(name) for name in value):
            raise ValueError("Invalid HTTP header name")
        return value

    @field_validator("oauth_scopes")
    @classmethod
    def validate_oauth_scopes(cls, value: list[str]) -> list[str]:
        normalized = [scope.strip() for scope in value]
        if any(
            not scope or any(character.isspace() for character in scope) for scope in normalized
        ):
            raise ValueError("OAuth scopes must be non-empty values without whitespace")
        if len(set(normalized)) != len(normalized):
            raise ValueError("OAuth scopes must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_auth_configuration(self) -> StreamableHttpTransport:
        authorization_header = any(
            name.lower() == "authorization" for name in self.header_secret_refs
        )
        if self.auth_mode == "oauth" and authorization_header:
            raise ValueError("OAuth mode cannot also configure an Authorization header")
        if self.auth_mode == "none" and self.header_secret_refs:
            raise ValueError("No-auth mode cannot configure credential headers")
        if self.oauth_client_secret_ref and not self.oauth_client_id:
            raise ValueError("OAuth client secret reference requires a client ID")
        if self.auth_mode in {"headers", "none"} and any(
            (
                self.oauth_scopes,
                self.oauth_callback_port,
                self.oauth_client_id,
                self.oauth_client_secret_ref,
            )
        ):
            raise ValueError("OAuth options require auto or oauth auth mode")
        return self


class StdioTransport(ContractModel):
    """A local MCP process described as argv, never as a shell command."""

    type: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1, max_length=2048)
    args: list[str] = Field(default_factory=list, max_length=256)
    cwd_policy: Literal["workspace", "isolated", "explicit"] = "workspace"
    cwd: str | None = Field(default=None, max_length=4096)
    env_secret_refs: dict[str, SecretRefName] = Field(default_factory=dict)
    sandbox_policy: Literal["required", "preferred", "disabled"] = "preferred"

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("Command must be one executable path or name")
        return value.strip()

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        if any("\x00" in item for item in value):
            raise ValueError("Arguments must not contain NUL bytes")
        if any(
            _SECRET_ARGUMENT.match(item) or item.lower().startswith(_SECRET_VALUE_PREFIXES)
            for item in value
        ):
            raise ValueError("Stdio credentials must use environment secret references")
        return value

    @field_validator("env_secret_refs")
    @classmethod
    def validate_env_names(cls, value: dict[str, SecretRefName]) -> dict[str, SecretRefName]:
        if any(not _ENV_NAME.fullmatch(name) for name in value):
            raise ValueError("Invalid environment variable name")
        return value

    @model_validator(mode="after")
    def validate_cwd_policy(self) -> StdioTransport:
        if self.cwd_policy == "explicit" and not self.cwd:
            raise ValueError("Explicit cwd policy requires cwd")
        if self.cwd_policy != "explicit" and self.cwd is not None:
            raise ValueError("cwd is accepted only with the explicit cwd policy")
        return self


McpTransport = Annotated[
    StreamableHttpTransport | StdioTransport,
    Field(discriminator="type"),
]


class McpProfile(ContractModel):
    """Mutable connector configuration with secret-free transport metadata."""

    profile_id: Identifier
    workspace_id: WorkspaceId
    label: str = Field(min_length=1, max_length=160)
    transport: McpTransport
    enabled: bool = True
    catalog_source: str | None = Field(default=None, max_length=160)
    policy_id: Identifier | None = None
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class McpProfileRevision(FrozenContractModel):
    """An immutable historical representation of one profile revision."""

    workspace_id: WorkspaceId
    profile_id: Identifier
    revision: int = Field(ge=1)
    profile: McpProfile
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_identity(self) -> McpProfileRevision:
        if (
            self.profile.workspace_id != self.workspace_id
            or self.profile.profile_id != self.profile_id
            or self.profile.revision != self.revision
        ):
            raise ValueError("Profile revision identity must match its embedded profile")
        return self


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value using one deterministic representation."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class McpStdioLaunchFingerprint(FrozenContractModel):
    """Secret-free fingerprint inputs for one approved stdio launch."""

    argv_hash: HexFingerprint
    cwd_hash: HexFingerprint
    env_names: tuple[str, ...] = ()
    sandbox_policy: Literal["required", "preferred", "disabled"]

    @field_validator("env_names")
    @classmethod
    def validate_env_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("env_names must be sorted and unique")
        if any(not _ENV_NAME.fullmatch(name) for name in value):
            raise ValueError("Invalid environment variable name")
        return value


class McpStdioLaunchApproval(FrozenContractModel):
    """Durable approval for one exact profile revision and launch fingerprint."""

    workspace_id: WorkspaceId
    profile_id: Identifier
    profile_revision: int = Field(ge=1)
    executable_fingerprint: HexFingerprint
    fingerprint: McpStdioLaunchFingerprint
    launch_fingerprint: HexFingerprint = ""
    approved_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verify_launch_fingerprint(self) -> McpStdioLaunchApproval:
        expected = canonical_hash(self.fingerprint.model_dump(mode="json"))
        if self.launch_fingerprint and self.launch_fingerprint != expected:
            raise ValueError("launch_fingerprint does not match fingerprint metadata")
        if not self.launch_fingerprint:
            object.__setattr__(self, "launch_fingerprint", expected)
        return self

    @classmethod
    def for_transport(
        cls,
        *,
        workspace_id: str,
        profile_id: str,
        profile_revision: int,
        executable_fingerprint: str,
        transport: StdioTransport,
    ) -> McpStdioLaunchApproval:
        """Build an approval without retaining argv, cwd, or secret reference values."""

        fingerprint = McpStdioLaunchFingerprint(
            argv_hash=canonical_hash([transport.command, *transport.args]),
            cwd_hash=canonical_hash({"cwd_policy": transport.cwd_policy, "cwd": transport.cwd}),
            env_names=tuple(sorted(transport.env_secret_refs)),
            sandbox_policy=transport.sandbox_policy,
        )
        return cls(
            workspace_id=workspace_id,
            profile_id=profile_id,
            profile_revision=profile_revision,
            executable_fingerprint=executable_fingerprint,
            fingerprint=fingerprint,
        )


class McpCapabilitySnapshot(FrozenContractModel):
    """Immutable, normalized capability inventory captured from one profile revision."""

    snapshot_id: Identifier
    workspace_id: WorkspaceId
    profile_id: Identifier
    profile_revision: int = Field(ge=1)
    captured_at: datetime = Field(default_factory=utc_now)
    negotiated_protocol_version: str = Field(min_length=1, max_length=64)
    server_info: dict[str, Any] = Field(default_factory=dict)
    instructions: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    resources: list[dict[str, Any]] = Field(default_factory=list)
    resource_templates: list[dict[str, Any]] = Field(default_factory=list)
    prompts: list[dict[str, Any]] = Field(default_factory=list)
    schema_warnings: list[str] = Field(default_factory=list)
    upstream_version: str | None = Field(default=None, max_length=160)
    contract_hash: str = ""
    stale: bool = False
    drifted: bool = False

    def contract_payload(self) -> dict[str, Any]:
        """Return only fields that define the advertised MCP contract."""

        return {
            "negotiated_protocol_version": self.negotiated_protocol_version,
            "server_info": self.server_info,
            "instructions": self.instructions,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "resources": self.resources,
            "resource_templates": self.resource_templates,
            "prompts": self.prompts,
            "schema_warnings": self.schema_warnings,
            "upstream_version": self.upstream_version,
        }

    @model_validator(mode="after")
    def verify_contract_hash(self) -> McpCapabilitySnapshot:
        expected = canonical_hash(self.contract_payload())
        if self.contract_hash and self.contract_hash != expected:
            raise ValueError("contract_hash does not match the normalized contract")
        if not self.contract_hash:
            object.__setattr__(self, "contract_hash", expected)
        return self


class SessionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class McpSession(ContractModel):
    """Persisted metadata for a live or historical upstream MCP connection."""

    session_id: Identifier
    workspace_id: WorkspaceId
    profile_id: Identifier
    profile_revision: int = Field(ge=1)
    snapshot_id: Identifier | None = None
    state: SessionState = SessionState.CONNECTING
    stale: bool = False
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    disconnected_at: datetime | None = None


class OperationKind(str, Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    REFRESH = "refresh"
    QUICK_TEST = "quick_test"
    MANUAL_CALL = "manual_call"
    SELF_TEST = "self_test"
    EVAL = "eval"


class OperationState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLED_UPSTREAM_UNKNOWN = "cancelled_upstream_unknown"
    INTERRUPTED = "interrupted"


class McpOperation(ContractModel):
    """Durable lifecycle record for one managed Workbench action."""

    operation_id: Identifier
    correlation_id: Identifier
    workspace_id: WorkspaceId
    kind: OperationKind
    state: OperationState = OperationState.QUEUED
    profile_id: Identifier | None = None
    session_id: Identifier | None = None
    idempotency_key: Identifier | None = None
    retry_of: Identifier | None = None
    revision: int = Field(default=1, ge=1)
    error_code: str | None = Field(default=None, max_length=160)
    safe_message: str | None = Field(default=None, max_length=2000)
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @field_validator("result")
    @classmethod
    def validate_sanitized_result(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject common unredacted credential shapes from durable operation results."""

        forbidden_keys = {
            "api_key",
            "apikey",
            "authorization",
            "bearer_token",
            "client_secret",
            "cookie",
            "id_token",
            "password",
            "passwd",
            "refresh_token",
            "access_token",
            "secret",
            "set_cookie",
        }
        forbidden_suffixes = (
            "_api_key",
            "_password",
            "_secret",
            "_bearer_token",
            "_access_token",
            "_refresh_token",
        )

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    lowered_key = str(key).lower().replace("-", "_")
                    if lowered_key in forbidden_keys or lowered_key.endswith(forbidden_suffixes):
                        if nested != "[redacted]":
                            raise ValueError("Operation result contains an unredacted secret field")
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)
            elif isinstance(item, str) and item.lower().startswith(_SECRET_VALUE_PREFIXES):
                raise ValueError("Operation result contains a secret-shaped value")

        if value is not None:
            visit(value)
        return value


class RecoverySummary(FrozenContractModel):
    """Counts of records made safe during process restart recovery."""

    operations_interrupted: int = Field(ge=0)
    sessions_disconnected: int = Field(ge=0)
    snapshots_marked_stale: int = Field(ge=0)
