"""
Observability API Routes

Provides REST API endpoints for:
- Profiler trace data
- Guardrail events and statistics
- Settings management for instrumentation
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


# Pydantic models for API
class TraceSpanResponse(BaseModel):
    """Response model for a trace span."""
    span_id: str
    name: str
    kind: str
    duration_ms: float
    status: str
    attributes: Dict[str, Any] = {}


class TraceSummaryResponse(BaseModel):
    """Response model for a trace summary."""
    trace_id: str
    name: str
    duration_ms: float
    total_spans: int
    llm_calls: Dict[str, Any] = {}
    tool_calls: Dict[str, Any] = {}
    bottlenecks: List[Dict[str, Any]] = []


class TraceListResponse(BaseModel):
    """Response model for trace list."""
    traces: List[TraceSummaryResponse]
    total: int


class GuardrailEventResponse(BaseModel):
    """Response model for a guardrail event."""
    timestamp: str
    check_type: str
    location: str
    action_taken: str
    confidence: float
    model_source: Optional[str] = None
    original_content: str
    modified_content: Optional[str] = None
    details: Dict[str, Any] = {}


class GuardrailEventsResponse(BaseModel):
    """Response model for guardrail events list."""
    events: List[GuardrailEventResponse]
    total: int


class GuardrailStatsResponse(BaseModel):
    """Response model for guardrail statistics."""
    total_events: int
    by_action: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    block_rate: float = 0.0


class MetricsResponse(BaseModel):
    """Response model for aggregated metrics."""
    profiler: Dict[str, Any] = {}
    guardrails: GuardrailStatsResponse


class SettingsUpdateRequest(BaseModel):
    """Request model for updating settings."""
    enabled: Optional[bool] = None
    pii_filtering: Optional[bool] = None
    injection_detection: Optional[bool] = None
    code_safety: Optional[bool] = None


class SettingsResponse(BaseModel):
    """Response model for current settings."""
    guardrails: Dict[str, Any] = {}
    profiler: Dict[str, Any] = {}


# Create router
router = APIRouter(prefix="/api/observability", tags=["observability"])


def get_instrumentation():
    """Get the global instrumentation instance."""
    try:
        from bashgym.core import get_instrumentation as get_inst
        return get_inst()
    except ImportError:
        return None


@router.get("/traces", response_model=TraceListResponse)
async def list_traces(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    List recent profiler traces.

    Returns summary information for recent traces.
    """
    inst = get_instrumentation()
    if not inst or not inst.profiler_enabled:
        return TraceListResponse(traces=[], total=0)

    all_traces = inst._profiler.get_all_traces()
    total = len(all_traces)

    # Apply pagination
    traces = all_traces[offset:offset + limit]

    summaries = []
    for trace in traces:
        summary = TraceSummaryResponse(
            trace_id=trace.trace_id,
            name=trace.name,
            duration_ms=trace.duration_ms,
            total_spans=len(trace.spans),
            llm_calls={
                "count": trace.total_llm_calls,
                "total_tokens": trace.total_tokens
            },
            tool_calls={
                "count": trace.total_tool_calls
            }
        )
        summaries.append(summary)

    return TraceListResponse(traces=summaries, total=total)


@router.get("/traces/{trace_id}", response_model=TraceSummaryResponse)
async def get_trace(trace_id: str):
    """
    Get details for a specific trace.

    Returns full trace information with all spans.
    """
    inst = get_instrumentation()
    if not inst or not inst.profiler_enabled:
        raise HTTPException(status_code=404, detail="Profiler not enabled")

    if trace_id not in inst._profiler.traces:
        raise HTTPException(status_code=404, detail="Trace not found")

    trace = inst._profiler.traces[trace_id]
    summary = inst._profiler.get_trace_summary(trace_id)

    return TraceSummaryResponse(
        trace_id=trace.trace_id,
        name=trace.name,
        duration_ms=trace.duration_ms,
        total_spans=len(trace.spans),
        llm_calls=summary.get("llm_calls", {}),
        tool_calls=summary.get("tool_calls", {}),
        bottlenecks=summary.get("bottlenecks", [])
    )


@router.get("/guardrails/events", response_model=GuardrailEventsResponse)
async def list_guardrail_events(
    action: Optional[str] = Query(None, description="Filter by action (block, warn, modify)"),
    check_type: Optional[str] = Query(None, description="Filter by type (injection, code_safety, pii)"),
    model_source: Optional[str] = Query(None, description="Filter by source (student, teacher)"),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    List recent guardrail events.

    Filter by action, type, or model source.
    """
    inst = get_instrumentation()
    if not inst or not inst.guardrails_enabled:
        return GuardrailEventsResponse(events=[], total=0)

    events = inst.get_guardrail_events(
        action=action,
        check_type=check_type,
        model_source=model_source,
        limit=limit
    )

    event_responses = [
        GuardrailEventResponse(
            timestamp=e.timestamp.isoformat(),
            check_type=e.check_type.value,
            location=e.location,
            action_taken=e.action_taken.value,
            confidence=e.confidence,
            model_source=e.model_source,
            original_content=e.original_content[:500],
            modified_content=e.modified_content[:500] if e.modified_content else None,
            details=e.details
        )
        for e in events
    ]

    return GuardrailEventsResponse(events=event_responses, total=len(events))


@router.get("/guardrails/stats", response_model=GuardrailStatsResponse)
async def get_guardrail_stats():
    """
    Get aggregated guardrail statistics.

    Returns counts by action type, check type, and source.
    """
    inst = get_instrumentation()
    if not inst or not inst.guardrails_enabled:
        return GuardrailStatsResponse(total_events=0)

    stats = inst.get_stats()

    return GuardrailStatsResponse(
        total_events=stats.get("total_events", 0),
        by_action=stats.get("by_action", {}),
        by_type=stats.get("by_type", {}),
        by_source=stats.get("by_source", {}),
        block_rate=stats.get("block_rate", 0.0)
    )


@router.get("/guardrails/dpo-negatives")
async def get_dpo_negatives(limit: int = Query(100, ge=1, le=1000)):
    """
    Get blocked events formatted for DPO negative examples.

    Returns student model blocked responses suitable for DPO training.
    """
    inst = get_instrumentation()
    if not inst:
        return {"negatives": [], "total": 0}

    negatives = inst.get_blocked_events_for_dpo(limit=limit)
    return {"negatives": negatives, "total": len(negatives)}


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get aggregated metrics for both profiler and guardrails.

    Returns combined metrics suitable for dashboard display.
    """
    inst = get_instrumentation()

    profiler_metrics = {}
    if inst and inst.profiler_enabled:
        all_traces = inst._profiler.get_all_traces()
        if all_traces:
            total_duration = sum(t.duration_ms for t in all_traces)
            total_tokens = sum(t.total_tokens for t in all_traces)
            profiler_metrics = {
                "total_traces": len(all_traces),
                "avg_duration_ms": total_duration / len(all_traces),
                "total_tokens": total_tokens,
                "avg_tokens_per_trace": total_tokens / len(all_traces) if all_traces else 0
            }

    guardrail_stats = GuardrailStatsResponse(total_events=0)
    if inst and inst.guardrails_enabled:
        stats = inst.get_stats()
        guardrail_stats = GuardrailStatsResponse(
            total_events=stats.get("total_events", 0),
            by_action=stats.get("by_action", {}),
            by_type=stats.get("by_type", {}),
            by_source=stats.get("by_source", {}),
            block_rate=stats.get("block_rate", 0.0)
        )

    return MetricsResponse(
        profiler=profiler_metrics,
        guardrails=guardrail_stats
    )


@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """
    Get current instrumentation settings.
    """
    inst = get_instrumentation()
    if not inst:
        return SettingsResponse()

    return SettingsResponse(
        guardrails={
            "enabled": inst._guardrails_settings.enabled,
            "pii_filtering": inst._guardrails_settings.pii_filtering,
            "injection_detection": inst._guardrails_settings.injection_detection,
            "code_safety": inst._guardrails_settings.code_safety,
        },
        profiler={
            "enabled": inst._profiler_settings.enabled,
            "profile_tokens": inst._profiler_settings.profile_tokens,
            "profile_latency": inst._profiler_settings.profile_latency,
        }
    )


@router.post("/settings/guardrails")
async def update_guardrails_settings(request: SettingsUpdateRequest):
    """
    Update guardrails settings at runtime.
    """
    inst = get_instrumentation()
    if not inst:
        raise HTTPException(status_code=503, detail="Instrumentation not available")

    updates = {}
    if request.enabled is not None:
        updates["enabled"] = request.enabled
    if request.pii_filtering is not None:
        updates["pii_filtering"] = request.pii_filtering
    if request.injection_detection is not None:
        updates["injection_detection"] = request.injection_detection
    if request.code_safety is not None:
        updates["code_safety"] = request.code_safety

    inst.update_settings(guardrails=updates)

    return {"status": "ok", "updated": updates}


@router.post("/settings/profiler")
async def update_profiler_settings(request: SettingsUpdateRequest):
    """
    Update profiler settings at runtime.
    """
    inst = get_instrumentation()
    if not inst:
        raise HTTPException(status_code=503, detail="Instrumentation not available")

    updates = {}
    if request.enabled is not None:
        updates["enabled"] = request.enabled

    inst.update_settings(profiler=updates)

    return {"status": "ok", "updated": updates}
