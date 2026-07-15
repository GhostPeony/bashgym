"""Tolerant, secret-safe Claude ``.mcp.json`` import preview helpers.

Import is deliberately a preview step. Unsupported transports and fields remain
visible as issues instead of being silently discarded or executed.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaudeImportIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["info", "warning", "blocked"]
    code: str
    message: str
    field: str | None = None


class ClaudeImportCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_name: str
    supported: bool
    source_scope: Literal["local", "project", "user"]
    profile_input: dict[str, Any] | None = None
    issues: list[ClaudeImportIssue] = Field(default_factory=list)
    preserved_fields: dict[str, Any] = Field(default_factory=dict)


_EXACT_ENV = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_BEARER_ENV = re.compile(r"^Bearer\s+\$\{([A-Z][A-Z0-9_]*)\}$", re.IGNORECASE)


def _issue(
    severity: Literal["info", "warning", "blocked"],
    code: str,
    message: str,
    field: str | None = None,
) -> ClaudeImportIssue:
    return ClaudeImportIssue(severity=severity, code=code, message=message, field=field)


def _secret_reference(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    exact = _EXACT_ENV.fullmatch(value.strip())
    if exact:
        return exact.group(1), None
    bearer = _BEARER_ENV.fullmatch(value.strip())
    if bearer:
        return bearer.group(1), "Bearer prefix must be included in the resolved secret value."
    return None, None


def preview_claude_mcp_config(
    config: dict[str, Any],
    *,
    source_scope: Literal["local", "project", "user"] = "project",
) -> list[ClaudeImportCandidate]:
    """Normalize supported Claude servers without resolving or retaining secrets."""

    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError('Claude MCP config must contain an object named "mcpServers"')

    candidates: list[ClaudeImportCandidate] = []
    for raw_name, raw_server in servers.items():
        name = str(raw_name).strip()
        issues: list[ClaudeImportIssue] = []
        if not name or not isinstance(raw_server, dict):
            candidates.append(
                ClaudeImportCandidate(
                    server_name=name or "unnamed",
                    supported=False,
                    source_scope=source_scope,
                    issues=[
                        _issue("blocked", "invalid_server", "Server entries must be named objects.")
                    ],
                )
            )
            continue

        server = dict(raw_server)
        server_type = server.get("type")
        if server_type == "streamable-http":
            server_type = "http"
            issues.append(
                _issue("info", "transport_alias", "Normalized streamable-http to HTTP.", "type")
            )
        if server_type is None:
            server_type = "stdio" if "command" in server else None

        preserved: dict[str, Any] = {}
        if isinstance(server.get("alwaysLoad"), bool):
            preserved["alwaysLoad"] = server["alwaysLoad"]
        if isinstance(server.get("timeout"), (int, float)) and not isinstance(
            server.get("timeout"), bool
        ):
            preserved["timeout"] = server["timeout"]
        if "alwaysLoad" in server:
            issues.append(
                _issue(
                    "warning",
                    "claude_always_load",
                    "Claude alwaysLoad is preserved for eval projection but does not change Workbench discovery.",
                    "alwaysLoad",
                )
            )
        if "timeout" in server:
            issues.append(
                _issue(
                    "warning",
                    "claude_timeout",
                    "Claude's timeout is preserved; configure BashGym runtime budgets explicitly.",
                    "timeout",
                )
            )

        profile_input: dict[str, Any] | None = None
        supported = True
        if server_type == "stdio":
            command = server.get("command")
            args = server.get("args", [])
            if (
                not isinstance(command, str)
                or not command.strip()
                or not isinstance(args, list)
                or not all(isinstance(item, str) for item in args)
            ):
                supported = False
                issues.append(
                    _issue("blocked", "invalid_stdio", "stdio requires command and string args.")
                )
            if (
                isinstance(command, str)
                and "${" in command
                or any("${" in item for item in args if isinstance(item, str))
            ):
                supported = False
                issues.append(
                    _issue(
                        "blocked",
                        "inline_expansion",
                        "Command/argument environment expansion must be resolved into an explicit approved launch.",
                    )
                )
            env_refs: dict[str, str] = {}
            env = server.get("env", {})
            if not isinstance(env, dict):
                supported = False
                issues.append(
                    _issue("blocked", "invalid_env", "stdio env must be an object.", "env")
                )
            else:
                for env_name, raw_value in env.items():
                    reference, note = _secret_reference(raw_value)
                    if reference is None:
                        supported = False
                        issues.append(
                            _issue(
                                "blocked",
                                "raw_env_value",
                                f"{env_name} must be migrated to an opaque secret reference.",
                                f"env.{env_name}",
                            )
                        )
                    else:
                        env_refs[str(env_name)] = reference
                        if note:
                            issues.append(
                                _issue("warning", "secret_value_prefix", note, f"env.{env_name}")
                            )
            if supported:
                profile_input = {
                    "label": name,
                    "transport": "stdio",
                    "enabled": True,
                    "stdio": {
                        "command": command.strip(),
                        "args": args,
                        "cwd_policy": "workspace",
                        "env_secret_refs": env_refs,
                        "sandbox_policy": "preferred",
                    },
                }
        elif server_type == "http":
            url = server.get("url")
            if not isinstance(url, str) or not url.strip() or "${" in url:
                supported = False
                issues.append(
                    _issue(
                        "blocked",
                        "invalid_remote_url",
                        "HTTP imports require one explicit URL.",
                        "url",
                    )
                )
            header_refs: dict[str, str] = {}
            headers = server.get("headers", {})
            if not isinstance(headers, dict):
                supported = False
                issues.append(
                    _issue("blocked", "invalid_headers", "headers must be an object.", "headers")
                )
            else:
                for header_name, raw_value in headers.items():
                    reference, note = _secret_reference(raw_value)
                    if reference is None:
                        supported = False
                        issues.append(
                            _issue(
                                "blocked",
                                "raw_header_value",
                                f"{header_name} must be migrated to an opaque secret reference.",
                                f"headers.{header_name}",
                            )
                        )
                    else:
                        header_refs[str(header_name)] = reference
                        if note:
                            issues.append(
                                _issue(
                                    "warning", "secret_value_prefix", note, f"headers.{header_name}"
                                )
                            )
            for field in ("oauth", "headersHelper", "authServerMetadataUrl"):
                if field in server:
                    value = server[field]
                    preserved[field] = {
                        "present": True,
                        "field_names": sorted(value) if isinstance(value, dict) else [],
                    }
                    supported = False
                    issues.append(
                        _issue(
                            "blocked",
                            f"unsupported_{field.lower()}",
                            f"{field} is preserved for migration but is not executed in M1.",
                            field,
                        )
                    )
            if supported:
                profile_input = {
                    "label": name,
                    "transport": "streamable_http",
                    "enabled": True,
                    "remote": {
                        "url": url.strip(),
                        "header_secret_refs": header_refs,
                        "allow_private_network": False,
                        "auth_mode": (
                            "headers"
                            if any(name.lower() == "authorization" for name in header_refs)
                            else "auto"
                        ),
                        "oauth_scopes": [],
                    },
                }
        else:
            supported = False
            preserved.update(
                {
                    "transport": server_type or "unknown",
                    "field_names": sorted(str(key) for key in server),
                }
            )
            issues.append(
                _issue(
                    "blocked",
                    "unsupported_transport",
                    f"Transport {server_type or 'unknown'} is inspection-only in M1; use HTTP or stdio.",
                    "type",
                )
            )

        candidates.append(
            ClaudeImportCandidate(
                server_name=name,
                supported=supported,
                source_scope=source_scope,
                profile_input=profile_input,
                issues=issues,
                preserved_fields=preserved,
            )
        )
    return candidates


__all__ = ["ClaudeImportCandidate", "ClaudeImportIssue", "preview_claude_mcp_config"]
