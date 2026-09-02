"""Launch-scoped MCP bridge for durable BashGym experiment campaigns."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from bashgym.api_base import normalize_api_base
from bashgym.campaigns.client import CampaignApiClient, CampaignClientError
from bashgym.campaigns.visibility import (
    project_public_campaign_artifact,
    project_public_campaign_event,
)
from bashgym.mcp.policy import validate_secret_ref_name


def _default_api_base() -> str:
    configured = (
        os.environ.get("BASHGYM_API_BASE")
        or os.environ.get("BASHGYM_API_URL")
        or "http://127.0.0.1:8003/api"
    )
    return normalize_api_base(configured)


DEFAULT_CAMPAIGN_API_BASE = _default_api_base()
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_MAX_NESTED_ENTRIES = 100
_MAX_NESTED_DEPTH = 8
_MAX_STRING_CHARS = 4096
_MAX_SERIALIZED_PAYLOAD_BYTES = 65_536
_TRUNCATION_MARKER = "[truncated]"

CampaignId = Annotated[
    str,
    Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
AttemptId = CampaignId
ActionId = CampaignId
StudyId = CampaignId
ProposalId = CampaignId
SourceId = CampaignId
MetricName = CampaignId
MetricSource = Annotated[str, Field(min_length=1, max_length=240)]
ExpectedVersion = Annotated[int, Field(ge=1)]
EventCursor = Annotated[int, Field(ge=0)]
ResearchWaitSeconds = Annotated[int, Field(ge=1, le=55)]
ArtifactCursor = Annotated[
    str,
    Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
MetricCursor = Annotated[int, Field(ge=-1)]
ListLimit = Annotated[int, Field(ge=1, le=100)]
EventLimit = Annotated[int, Field(ge=1, le=200)]
MetricLimit = Annotated[int, Field(ge=1, le=1000)]
CancelReason = Annotated[str, Field(min_length=1, max_length=2000)]
Reason = CancelReason
ManifestRevisionNumber = Annotated[int, Field(ge=1)]
Priority = Annotated[int, Field(ge=0, le=100)]
ExportFormat = Literal["markdown", "json", "csv", "png", "docx", "pdf"]
ResearchRole = Literal["baseline", "candidate", "diagnostic"]
ResearchFamilyDisposition = Literal["supported", "exhausted", "inconclusive"]
ResearchCategory = Literal["research", "github", "pdf"]
ResearchLimit = Annotated[int, Field(ge=1, le=10)]
HexDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SetupSessionId = Annotated[str, Field(pattern=r"^setupsess_[0-9a-f]{32}$")]
CampaignStatusFilter = Literal[
    "draft",
    "validating",
    "ready",
    "active",
    "paused",
    "awaiting_authority",
    "cancelling",
    "completed",
    "exhausted",
    "failed",
    "cancelled",
]
CampaignKindFilter = Literal["embedding_retrieval", "general"]
_SERVER_OWNED_PROPOSAL_FIELDS = frozenset(
    {
        "workspace_id",
        "workspace",
        "campaign_id",
        "expected_version",
        "role",
        "autoresearch_role",
        "parent_proposal_id",
        "planner_actor_id",
        "creation_sequence",
        "status",
        "credential_ref",
        "credential",
        "actor",
        "agent",
        "profile",
        "autonomy_profile",
        "capabilities",
        "authorization",
    }
)


class CampaignRequestClient(Protocol):
    """Narrow injectable boundary used by the MCP tool handlers."""

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any: ...


def _bounded_string(value: str) -> str:
    if len(value) <= _MAX_STRING_CHARS:
        return value
    prefix_length = _MAX_STRING_CHARS - len(_TRUNCATION_MARKER) - 1
    return f"{value[:prefix_length]} {_TRUNCATION_MARKER}"


def _bounded_nested_value(value: Any, *, depth: int) -> Any:
    if depth > _MAX_NESTED_DEPTH:
        return _TRUNCATION_MARKER
    if isinstance(value, str):
        return _bounded_string(value)
    if isinstance(value, list):
        if len(value) >= _MAX_NESTED_ENTRIES:
            return [
                *(
                    _bounded_nested_value(item, depth=depth + 1)
                    for item in value[: _MAX_NESTED_ENTRIES - 1]
                ),
                _TRUNCATION_MARKER,
            ]
        return [_bounded_nested_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        items = list(value.items())
        entry_limit = (
            _MAX_NESTED_ENTRIES - 1 if len(items) >= _MAX_NESTED_ENTRIES else _MAX_NESTED_ENTRIES
        )
        bounded = {
            _bounded_string(str(key)): _bounded_nested_value(item, depth=depth + 1)
            for key, item in items[:entry_limit]
        }
        if len(items) >= _MAX_NESTED_ENTRIES:
            bounded["_truncated"] = f"{len(items) - len(bounded)} entries omitted"
        return bounded
    return value


def _bounded_value(value: Any) -> Any:
    """Return planner-safe JSON with per-field and total payload limits."""

    bounded = _bounded_nested_value(value, depth=0)
    if _serialized_size(bounded) > _MAX_SERIALIZED_PAYLOAD_BYTES:
        return _payload_truncation()
    return bounded


def _serialized_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _payload_truncation() -> dict[str, str]:
    return {"_truncated": (f"payload exceeded {_MAX_SERIALIZED_PAYLOAD_BYTES} serialized bytes")}


def _bounded_named_result(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "ok": True,
        name: _bounded_nested_value(dict(value), depth=0),
    }
    if _serialized_size(result) > _MAX_SERIALIZED_PAYLOAD_BYTES:
        result[name] = _payload_truncation()
    return result


def _bounded_mutation_result(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "ok": True,
        **_bounded_nested_value(dict(value), depth=0),
    }
    if _serialized_size(result) > _MAX_SERIALIZED_PAYLOAD_BYTES:
        return {"ok": True, **_payload_truncation()}
    return result


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("campaign API returned a non-object response")
    return value


def _invalid_response() -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": "campaign_response_invalid",
            "message": "The BashGym campaign API returned an invalid response.",
            "retryable": False,
            "status_code": 502,
        },
    }


def _invalid_request(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": "campaign_request_invalid",
            "message": message,
            "retryable": False,
            "status_code": 400,
        },
    }


def _bounded_collection(
    response: Any,
    *,
    key: str,
    limit: int,
) -> tuple[dict[str, Any], list[Any], bool]:
    payload = _mapping(response)
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"campaign API response is missing {key}")
    bounded_value = _bounded_value(values[:limit])
    if isinstance(bounded_value, list):
        bounded = bounded_value
        payload_truncated = False
    else:
        bounded = [bounded_value]
        payload_truncated = True
    return payload, bounded, len(values) > limit or payload_truncated


def _mutation_headers(
    *,
    agent: str,
    workspace_id: str,
    campaign_id: str,
    operation: str,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    canonical = json.dumps(
        {
            "workspace_id": workspace_id,
            "campaign_id": campaign_id,
            "operation": operation,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    request_digest = hashlib.sha256(canonical.encode()).hexdigest()[:24]
    agent_digest = hashlib.sha256(agent.encode()).hexdigest()[:12]
    return {
        "Idempotency-Key": f"campaign-mcp-{operation}-{request_digest}",
        "X-Correlation-ID": f"campaign-mcp-{agent_digest}-{operation}",
    }


def build_server(
    *,
    workspace_id: str,
    credential_ref: str,
    agent: str,
    api_base: str = DEFAULT_CAMPAIGN_API_BASE,
    client: CampaignRequestClient | None = None,
) -> FastMCP:
    """Build a server whose authority and workspace cannot be changed by tools."""

    if not _IDENTIFIER.fullmatch(workspace_id):
        raise ValueError("workspace_id must be a simple campaign identifier")
    if not _IDENTIFIER.fullmatch(agent):
        raise ValueError("agent must be a simple campaign identifier")
    validate_secret_ref_name(credential_ref)
    request_client = client or CampaignApiClient(
        api_base=api_base,
        credential_ref=credential_ref,
    )

    server = FastMCP(
        "bashgym-campaigns",
        instructions=(
            f"Operate BashGym campaigns only in launch-bound workspace {workspace_id!r}. "
            "Authority is derived from the launch-bound credential reference by the "
            "campaign API. Never ask tools to change workspace, actor, autonomy profile, "
            "or capabilities. Read persisted evidence before changing campaign state."
        ),
        json_response=True,
        log_level="ERROR",
    )

    async def request(
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            kwargs = {"query": query, "payload": payload, "headers": headers}
            if timeout is not None:
                kwargs["timeout"] = timeout
            result = await asyncio.to_thread(
                request_client.request_json,
                method,
                path,
                **kwargs,
            )
            return {"ok": True, "data": result}
        except CampaignClientError as exc:
            return {"ok": False, "error": exc.as_dict()}
        except (TypeError, ValueError):
            return _invalid_response()
        except Exception:
            return {
                "ok": False,
                "error": {
                    "code": "campaign_api_unavailable",
                    "message": "The BashGym campaign API is unavailable.",
                    "retryable": True,
                    "status_code": None,
                },
            }

    async def mutate(
        operation: str,
        campaign_id: str,
        path: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        strict_body = {"workspace_id": workspace_id, **dict(body)}
        result = await request(
            "POST",
            path,
            payload=strict_body,
            headers=_mutation_headers(
                agent=agent,
                workspace_id=workspace_id,
                campaign_id=campaign_id,
                operation=operation,
                payload=strict_body,
            ),
        )
        if not result["ok"]:
            return result
        try:
            mutation = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_mutation_result(mutation)

    async def transition(
        operation: str,
        campaign_id: str,
        expected_version: int,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"expected_version": expected_version}
        if reason is not None:
            body["stop_reason"] = reason
        return await mutate(
            operation,
            campaign_id,
            f"/campaigns/{campaign_id}/{operation}",
            body,
        )

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    state_change = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
    cancellation = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    )

    @server.tool(structured_output=True, annotations=read_only)
    async def research_context(
        campaign_id: CampaignId,
        proposal_id: ProposalId,
        query: Annotated[str, Field(min_length=1, max_length=1000)],
        categories: list[ResearchCategory],
        limit: ResearchLimit = 5,
    ) -> dict[str, Any]:
        """Collect bounded paper/repository citations for one proposed experiment."""

        if not categories or len(categories) != len(set(categories)):
            return _invalid_request("Research categories must be non-empty and unique.")
        result = await request(
            "POST",
            "/research/context",
            payload={
                "workspace_id": workspace_id,
                "campaign_id": campaign_id,
                "proposal_id": proposal_id,
                "query": query,
                "categories": categories,
                "limit": limit,
            },
        )
        if not result["ok"]:
            return result
        try:
            payload = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_named_result("context", payload.get("context", payload))

    @server.tool(structured_output=True, annotations=read_only)
    async def research_prepare(
        session_id: SetupSessionId | None = None,
    ) -> dict[str, Any]:
        """Discover or resume guided campaign setup without starting compute."""

        query = {"workspace_id": workspace_id}
        if session_id is not None:
            query["session_id"] = session_id
        result = await request("GET", "/campaigns/setup/context", query=query)
        if not result["ok"]:
            return result
        try:
            context = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_named_result("context", context)

    @server.tool(structured_output=True, annotations=read_only)
    async def research_state(campaign_id: CampaignId) -> dict[str, Any]:
        """Read the compact durable experiment state for native agent goals and visuals."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/research-state",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            research = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_named_result("research", research)

    @server.tool(structured_output=True, annotations=read_only)
    async def research_failures(campaign_id: CampaignId) -> dict[str, Any]:
        """Compare bounded evaluator-authored failure categories for the latest result."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/research-failures",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            failures = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_named_result("failures", failures)

    @server.tool(structured_output=True, annotations=read_only)
    async def research_wait(
        campaign_id: CampaignId,
        after_cursor: EventCursor = 0,
        timeout_seconds: ResearchWaitSeconds = 30,
    ) -> dict[str, Any]:
        """Wait briefly for progress, then return the latest durable agent state."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/research-wait",
            query={
                "workspace_id": workspace_id,
                "after_cursor": after_cursor,
                "timeout_seconds": timeout_seconds,
            },
            timeout=timeout_seconds + 5,
        )
        if not result["ok"]:
            return result
        try:
            wait = _mapping(result["data"])
        except ValueError:
            return _invalid_response()
        return _bounded_named_result("wait", wait)

    @server.tool(structured_output=True, annotations=state_change)
    async def research_start(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Start a ready research campaign after the operator explicitly authorizes it."""

        return await transition("start", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=state_change)
    async def research_submit_iteration(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        role: ResearchRole,
        proposal: dict[str, Any],
        parent_proposal_id: ProposalId | None = None,
    ) -> dict[str, Any]:
        """Submit a baseline, model candidate, or agent-designed diagnostic."""

        if _SERVER_OWNED_PROPOSAL_FIELDS.intersection(proposal):
            return _invalid_request("Proposal contains server-owned fields.")
        if role in {"candidate", "diagnostic"} and parent_proposal_id is None:
            return _invalid_request(f"{role.title()} iteration requires a parent proposal.")
        if role == "baseline" and parent_proposal_id is not None:
            return _invalid_request("Baseline iteration cannot have a parent proposal.")

        body = {"expected_version": expected_version, **proposal}
        if parent_proposal_id is not None:
            body["parent_proposal_id"] = parent_proposal_id
        suffix = {
            "baseline": "baseline",
            "candidate": "candidates",
            "diagnostic": "diagnostics",
        }[role]
        return await mutate(
            f"autoresearch-{role}",
            campaign_id,
            f"/campaigns/{campaign_id}/autoresearch/{suffix}",
            body,
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def research_conclude_hypothesis_family(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        hypothesis_family_id: ProposalId,
        disposition: ResearchFamilyDisposition,
        summary: Annotated[str, Field(min_length=1, max_length=2000)],
        follow_up_family_id: ProposalId | None = None,
        follow_up_hypothesis: Annotated[str, Field(min_length=1, max_length=4000)] | None = None,
    ) -> dict[str, Any]:
        """Conclude a completed hypothesis family and optionally record the next idea."""

        if (follow_up_family_id is None) != (follow_up_hypothesis is None):
            return _invalid_request("Follow-up family and hypothesis must be provided together.")
        body: dict[str, Any] = {
            "expected_version": expected_version,
            "disposition": disposition,
            "summary": summary,
        }
        if follow_up_family_id is not None:
            body["follow_up_family_id"] = follow_up_family_id
            body["follow_up_hypothesis"] = follow_up_hypothesis
        return await mutate(
            "autoresearch-family-conclude",
            campaign_id,
            (
                f"/campaigns/{campaign_id}/autoresearch/hypothesis-families/"
                f"{hypothesis_family_id}/conclude"
            ),
            body,
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def research_report(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        formats: list[ExportFormat],
    ) -> dict[str, Any]:
        """Request bounded campaign evidence exports for review and presentation."""

        if not formats or len(formats) != len(set(formats)):
            return _invalid_request("Report formats must be non-empty and unique.")
        return await mutate(
            "export",
            campaign_id,
            f"/campaigns/{campaign_id}/export",
            {"expected_version": expected_version, "formats": formats},
        )

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_list(
        status: CampaignStatusFilter | None = None,
        kind: CampaignKindFilter | None = None,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List bounded campaign summaries in the launch-bound workspace."""

        result = await request(
            "GET",
            "/campaigns",
            query={"workspace_id": workspace_id, "status": status, "kind": kind},
        )
        if not result["ok"]:
            return result
        try:
            _payload, campaigns, truncated = _bounded_collection(
                result["data"], key="campaigns", limit=limit
            )
        except ValueError:
            return _invalid_response()
        return {
            "ok": True,
            "campaigns": campaigns,
            "count": len(campaigns),
            "truncated": truncated,
        }

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_inspect(campaign_id: CampaignId) -> dict[str, Any]:
        """Inspect one campaign aggregate in the launch-bound workspace."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            campaign = _bounded_value(_mapping(result["data"]))
        except ValueError:
            return _invalid_response()
        return {"ok": True, "campaign": campaign}

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_manifest(
        campaign_id: CampaignId,
        revision: ManifestRevisionNumber,
    ) -> dict[str, Any]:
        """Read one immutable manifest revision for a campaign."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/manifest/{revision}",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            manifest = _bounded_value(_mapping(result["data"]))
        except ValueError:
            return _invalid_response()
        return {"ok": True, "manifest_revision": manifest}

    async def read_collection(
        campaign_id: str,
        suffix: str,
        key: str,
        limit: int,
    ) -> dict[str, Any]:
        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/{suffix}",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            _payload, values, truncated = _bounded_collection(result["data"], key=key, limit=limit)
        except ValueError:
            return _invalid_response()
        return {"ok": True, key: values, "count": len(values), "truncated": truncated}

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_evidence(campaign_id: CampaignId) -> dict[str, Any]:
        """Read bounded planner-safe evidence; protected rows and raw logs stay sealed."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/evidence",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            evidence = _bounded_value(_mapping(result["data"]))
        except ValueError:
            return _invalid_response()
        return {"ok": True, "evidence": evidence}

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_artifacts(
        campaign_id: CampaignId,
        after_cursor: ArtifactCursor | None = None,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List one bounded page of safe artifact identities and seal metadata."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/artifacts",
            query={
                "workspace_id": workspace_id,
                **({"after_cursor": after_cursor} if after_cursor else {}),
                "limit": limit,
            },
        )
        if not result["ok"]:
            return result
        try:
            _payload, values, truncated = _bounded_collection(
                result["data"], key="artifacts", limit=limit
            )
            next_cursor = _payload.get("next_cursor")
            has_more = _payload.get("has_more")
            if next_cursor is not None and (
                not isinstance(next_cursor, str) or not _IDENTIFIER.fullmatch(next_cursor)
            ):
                raise ValueError("invalid artifact continuation cursor")
            if not isinstance(has_more, bool) or has_more != (next_cursor is not None):
                raise ValueError("invalid artifact pagination state")
            artifacts = [
                project_public_campaign_artifact(item).model_dump(mode="json")
                for item in values
                if isinstance(item, dict)
            ]
            if len(artifacts) != len(values):
                raise ValueError("invalid public artifact collection")
        except (KeyError, TypeError, ValueError):
            return _invalid_response()
        return {
            "ok": True,
            "artifacts": artifacts,
            "count": len(artifacts),
            "next_cursor": next_cursor,
            "has_more": has_more,
            "truncated": truncated or has_more,
        }

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_proposals(
        campaign_id: CampaignId,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List bounded study proposals and persisted validation outcomes."""

        return await read_collection(campaign_id, "proposals", "proposals", limit)

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_studies(
        campaign_id: CampaignId,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List persisted studies, lifecycle states, and immutable stage cursors."""

        return await read_collection(campaign_id, "studies", "studies", limit)

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_study(
        campaign_id: CampaignId,
        study_id: StudyId,
    ) -> dict[str, Any]:
        """Read one persisted study and its current immutable stage cursor."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/studies/{study_id}",
            query={"workspace_id": workspace_id},
        )
        if not result["ok"]:
            return result
        try:
            study = _bounded_value(_mapping(result["data"]))
        except ValueError:
            return _invalid_response()
        return {"ok": True, "study": study}

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_attempts(
        campaign_id: CampaignId,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List bounded execution attempts without sealed-result locations."""

        return await read_collection(campaign_id, "attempts", "attempts", limit)

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_comparisons(
        campaign_id: CampaignId,
        limit: ListLimit = 50,
    ) -> dict[str, Any]:
        """List bounded persisted development comparisons for a campaign."""

        return await read_collection(campaign_id, "comparisons", "comparisons", limit)

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_events(
        campaign_id: CampaignId,
        after_cursor: EventCursor = 0,
        limit: EventLimit = 100,
    ) -> dict[str, Any]:
        """Read a bounded persisted event page for restart-safe reconciliation."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/events",
            query={
                "workspace_id": workspace_id,
                "after_cursor": after_cursor,
                "limit": limit,
            },
        )
        if not result["ok"]:
            return result
        try:
            payload, items, truncated = _bounded_collection(
                result["data"], key="items", limit=limit
            )
            safe_items = []
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("cursor"), int)
                    or not isinstance(item.get("event"), dict)
                ):
                    raise ValueError("invalid public event page item")
                safe_items.append(
                    {
                        "cursor": item["cursor"],
                        "event": project_public_campaign_event(item["event"]).model_dump(
                            mode="json", exclude_none=True
                        ),
                    }
                )
        except (KeyError, TypeError, ValueError):
            return _invalid_response()
        return {
            "ok": True,
            "items": safe_items,
            "next_cursor": payload.get("next_cursor", after_cursor),
            "truncated": truncated,
        }

    @server.tool(structured_output=True, annotations=read_only)
    async def campaign_metrics(
        campaign_id: CampaignId,
        attempt_id: AttemptId,
        metric_name: MetricName,
        source: MetricSource,
        after_step: MetricCursor = -1,
        limit: MetricLimit = 500,
    ) -> dict[str, Any]:
        """Read one bounded persisted metric series from an exact source."""

        result = await request(
            "GET",
            f"/campaigns/{campaign_id}/attempts/{attempt_id}/metrics",
            query={
                "workspace_id": workspace_id,
                "metric_name": metric_name,
                "source": source,
                "after_step": after_step,
                "limit": limit,
            },
        )
        if not result["ok"]:
            return result
        try:
            payload, values, truncated = _bounded_collection(
                result["data"], key="values", limit=limit
            )
        except ValueError:
            return _invalid_response()
        return {
            "ok": True,
            "metric_name": payload.get("metric_name", metric_name),
            "source": payload.get("source", source),
            "values": values,
            "next_after_step": payload.get("next_after_step", after_step),
            "truncated": truncated,
        }

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_create_from_template(
        campaign_id: CampaignId,
        title: Annotated[str, Field(min_length=1, max_length=240)],
        template_id: CampaignId,
        stop_rules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create from an approved template; AutoResearch requires explicit stop rules."""

        payload: dict[str, Any] = {
            "campaign_id": campaign_id,
            "title": title,
            "template_id": template_id,
        }
        if stop_rules is not None:
            payload["stop_rules"] = stop_rules
        return await mutate(
            "create-template",
            campaign_id,
            "/campaigns/from-template",
            payload,
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_create(
        campaign_id: CampaignId,
        title: Annotated[str, Field(min_length=1, max_length=240)],
        kind: CampaignKindFilter,
        objective: Annotated[str, Field(min_length=1, max_length=4000)],
        target_model: dict[str, Any],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a draft from a typed manifest when server authority permits it."""

        return await mutate(
            "create",
            campaign_id,
            "/campaigns",
            {
                "campaign_id": campaign_id,
                "title": title,
                "kind": kind,
                "objective": objective,
                "target_model": target_model,
                "manifest": manifest,
            },
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_revise(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        manifest: dict[str, Any],
        reason: Reason,
    ) -> dict[str, Any]:
        """Create an immutable manifest revision with an auditable reason."""

        return await mutate(
            "revise",
            campaign_id,
            f"/campaigns/{campaign_id}/manifest/revise",
            {
                "expected_version": expected_version,
                "manifest": manifest,
                "reason": reason.strip(),
            },
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_propose_study(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        proposal_id: ProposalId,
        hypothesis: Annotated[str, Field(min_length=1, max_length=4000)],
        study_family: CampaignId,
        primary_variable: Annotated[str, Field(min_length=1, max_length=1000)],
        expected_outcome: Annotated[str, Field(min_length=1, max_length=2000)],
        falsification_criterion: Annotated[str, Field(min_length=1, max_length=2000)],
        estimated_cost: Annotated[float, Field(ge=0)],
        dataset_recipe: dict[str, Any],
        training_recipe: dict[str, Any],
        evaluation_recipe: dict[str, Any],
        stage_plan: dict[str, Any],
        rationale: Annotated[str, Field(min_length=1, max_length=4000)],
        evidence_references: list[str] | None = None,
        controlled_variables: list[str] | None = None,
        priority: Priority = 50,
        prerequisite_study_ids: list[str] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit falsifiable study evidence for deterministic server validation."""

        proposal = {
            "proposal_id": proposal_id,
            "hypothesis": hypothesis,
            "evidence_references": evidence_references or [],
            "study_family": study_family,
            "primary_variable": primary_variable,
            "controlled_variables": controlled_variables or [],
            "expected_outcome": expected_outcome,
            "falsification_criterion": falsification_criterion,
            "estimated_cost": estimated_cost,
            "priority": priority,
            "prerequisite_study_ids": prerequisite_study_ids or [],
            "dataset_recipe": dataset_recipe,
            "training_recipe": training_recipe,
            "evaluation_recipe": evaluation_recipe,
            "required_capabilities": required_capabilities or [],
            "stage_plan": stage_plan,
            "rationale": rationale,
        }
        return await mutate(
            "propose",
            campaign_id,
            f"/campaigns/{campaign_id}/proposals",
            {"expected_version": expected_version, **proposal},
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_withdraw_proposal(
        campaign_id: CampaignId,
        proposal_id: ProposalId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Withdraw a still-submitted proposal without deleting its audit record."""

        return await mutate(
            "withdraw-proposal",
            campaign_id,
            f"/campaigns/{campaign_id}/proposals/{proposal_id}/withdraw",
            {"expected_version": expected_version},
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_prepare_code_lineage(
        campaign_id: CampaignId,
        proposal_id: ProposalId,
    ) -> dict[str, Any]:
        """Prepare the capability-gated private worktree for a code hypothesis."""

        return await mutate(
            "prepare-code-lineage",
            campaign_id,
            f"/campaigns/{campaign_id}/proposals/{proposal_id}/code-lineage/prepare",
            {},
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_capture_code_lineage(
        campaign_id: CampaignId,
        proposal_id: ProposalId,
    ) -> dict[str, Any]:
        """Capture one approved code commit after editing the prepared worktree."""

        return await mutate(
            "capture-code-lineage",
            campaign_id,
            f"/campaigns/{campaign_id}/proposals/{proposal_id}/code-lineage/capture",
            {},
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_start(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Start a ready campaign when the launch credential authorizes it."""

        return await transition("start", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_advance(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Persist one controller wakeup; the server remains responsible for scheduling."""

        return await transition("advance", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_pause(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Pause new campaign scheduling without discarding persisted evidence."""

        return await transition("pause", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_resume(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Resume a paused campaign when the launch credential authorizes it."""

        return await transition("resume", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=cancellation)
    async def campaign_cancel(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        reason: CancelReason,
    ) -> dict[str, Any]:
        """Request terminal campaign cancellation with an auditable reason."""

        return await transition("cancel", campaign_id, expected_version, reason=reason.strip())

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_conclude(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        reason: Reason,
    ) -> dict[str, Any]:
        """Conclude without promotion after server-side running-action and evidence checks."""

        return await transition("conclude", campaign_id, expected_version, reason=reason.strip())

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_retry(
        campaign_id: CampaignId,
        action_id: ActionId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Request one retry that remains bounded by persisted policy and input hashes."""

        return await mutate(
            "retry",
            campaign_id,
            f"/campaigns/{campaign_id}/actions/{action_id}/retry",
            {"expected_version": expected_version},
        )

    @server.tool(structured_output=True, annotations=cancellation)
    async def campaign_abandon_study(
        campaign_id: CampaignId,
        study_id: StudyId,
        expected_version: ExpectedVersion,
        reason: Reason,
    ) -> dict[str, Any]:
        """Abandon a nonterminal study while preserving all collected evidence."""

        return await mutate(
            "abandon-study",
            campaign_id,
            f"/campaigns/{campaign_id}/studies/{study_id}/abandon",
            {"expected_version": expected_version, "reason": reason.strip()},
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_amend_budget(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        resource: CampaignId,
        delta: float,
        reason: Reason,
    ) -> dict[str, Any]:
        """Append an authorized budget adjustment; protected epochs remain unchanged."""

        return await mutate(
            "amend-budget",
            campaign_id,
            f"/campaigns/{campaign_id}/budget/amend",
            {
                "expected_version": expected_version,
                "resource": resource,
                "delta": delta,
                "reason": reason.strip(),
            },
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_approve_source(
        campaign_id: CampaignId,
        source_id: SourceId,
        expected_version: ExpectedVersion,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Record typed provenance, license, privacy, and contamination evidence."""

        return await mutate(
            "approve-source",
            campaign_id,
            f"/campaigns/{campaign_id}/sources/{source_id}/approve",
            {"expected_version": expected_version, "evidence": evidence},
        )

    @server.tool(structured_output=True, annotations=cancellation)
    async def campaign_force_stop(
        campaign_id: CampaignId,
        action_id: ActionId,
        expected_version: ExpectedVersion,
        expected_remote_process_identity: dict[str, Any],
        reason: Reason,
        confirmed: Literal[True],
    ) -> dict[str, Any]:
        """Stop only an exact persisted remote identity; raw PIDs and commands are rejected."""

        return await mutate(
            "force-stop",
            campaign_id,
            f"/campaigns/{campaign_id}/actions/{action_id}/force-stop",
            {
                "expected_version": expected_version,
                "expected_remote_process_identity": expected_remote_process_identity,
                "confirmed": confirmed,
                "reason": reason.strip(),
            },
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_protected_lease(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
    ) -> dict[str, Any]:
        """Request the one-time server-mediated protected evaluation lease."""

        return await transition("protected-lease", campaign_id, expected_version)

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_protected_result(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        protected_epoch_id: CampaignId,
        candidate_digest: HexDigest,
        passed: bool,
        metrics: dict[str, float],
        artifact_sha256: HexDigest,
    ) -> dict[str, Any]:
        """Record a bounded result for the exact candidate-locked protected epoch."""

        return await mutate(
            "protected-result",
            campaign_id,
            f"/campaigns/{campaign_id}/protected-result",
            {
                "expected_version": expected_version,
                "result": {
                    "protected_epoch_id": protected_epoch_id,
                    "candidate_digest": candidate_digest,
                    "passed": passed,
                    "metrics": metrics,
                    "artifact_sha256": artifact_sha256,
                },
            },
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_promote(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        override_reason: Reason | None = None,
    ) -> dict[str, Any]:
        """Compute a champion decision; overrides remain separately capability-gated."""

        body: dict[str, Any] = {"expected_version": expected_version}
        if override_reason is not None:
            body["override_reason"] = override_reason.strip()
        return await mutate(
            "promote",
            campaign_id,
            f"/campaigns/{campaign_id}/promotion",
            body,
        )

    @server.tool(structured_output=True, annotations=state_change)
    async def campaign_export(
        campaign_id: CampaignId,
        expected_version: ExpectedVersion,
        formats: list[ExportFormat],
    ) -> dict[str, Any]:
        """Request reproducible bounded evidence exports as a campaign action."""

        if not formats or len(formats) != len(set(formats)):
            return _invalid_response()
        return await mutate(
            "export",
            campaign_id,
            f"/campaigns/{campaign_id}/export",
            {"expected_version": expected_version, "formats": formats},
        )

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--credential-ref", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--api-base", default=DEFAULT_CAMPAIGN_API_BASE)
    args = parser.parse_args(argv)
    build_server(
        workspace_id=args.workspace_id,
        credential_ref=args.credential_ref,
        agent=args.agent,
        api_base=args.api_base,
    ).run(transport="stdio")


if __name__ == "__main__":
    main()


__all__ = ["CampaignRequestClient", "build_server", "main"]
