"""
Instrumentation Module - Unified Guardrails + Profiler Integration

Provides a single entry point for all instrumentation needs:
- Guardrails: Safety checks before execution
- Profiler: Performance tracking around execution
- Events: Guardrail events for training signals

Usage:
    from bashgym.core import get_instrumentation

    inst = get_instrumentation()

    # Check command safety + profile execution
    async with inst.instrument_command("rm -rf /tmp/test") as ctx:
        if ctx.allowed:
            result = execute_command(ctx.content)
            ctx.set_result(success=True, output=result)

    # Check and filter PII from text
    filtered_text = await inst.filter_pii("Contact john@example.com")

    # Get recent guardrail events (for DPO training)
    events = inst.get_guardrail_events(action="block")
"""

import asyncio
import inspect
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import json

from ..config import get_settings, GuardrailsSettings, ObservabilitySettings
from ..judge.guardrails import (
    NemoGuard,
    GuardrailsConfig,
    GuardrailAction,
    GuardrailType,
    CheckResult,
)
from ..observability.profiler import (
    AgentProfiler,
    ProfilerConfig,
    SpanKind,
    TraceSpan,
)


@dataclass
class GuardrailEvent:
    """Record of a guardrail check for training signals and audit."""

    timestamp: datetime
    check_type: GuardrailType
    location: str  # e.g., "gym.execute_bash", "import.user_prompt"
    action_taken: GuardrailAction
    original_content: str  # Truncated for storage
    modified_content: Optional[str] = None  # If PII was redacted
    confidence: float = 1.0
    model_source: Optional[str] = None  # "teacher" or "student"
    trace_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "check_type": self.check_type.value,
            "location": self.location,
            "action_taken": self.action_taken.value,
            "original_content": self.original_content[:500],  # Truncate
            "modified_content": self.modified_content[:500] if self.modified_content else None,
            "confidence": self.confidence,
            "model_source": self.model_source,
            "trace_id": self.trace_id,
            "details": self.details,
        }


@dataclass
class InstrumentationContext:
    """Context for an instrumented operation."""

    allowed: bool
    content: str  # Original or modified content
    action: GuardrailAction
    span: Optional[TraceSpan] = None
    check_result: Optional[CheckResult] = None
    _instrumentation: Optional["Instrumentation"] = None
    _location: str = ""

    def set_result(
        self,
        success: bool,
        output: str = "",
        tokens: int = 0,
        **attributes
    ):
        """Set the result of the operation for profiling."""
        if self.span:
            self.span.set_attribute("success", success)
            self.span.set_attribute("output_preview", output[:100])
            if tokens:
                self.span.set_tokens(input_tokens=0, output_tokens=tokens)
            for k, v in attributes.items():
                self.span.set_attribute(k, v)


class Instrumentation:
    """
    Unified instrumentation layer combining guardrails and profiler.

    Features:
    - Automatic span creation around guardrail checks
    - Event recording for blocked/modified content
    - Easy integration via context managers
    - Configurable via settings
    """

    def __init__(
        self,
        guardrails_settings: Optional[GuardrailsSettings] = None,
        profiler_settings: Optional[ObservabilitySettings] = None,
    ):
        settings = get_settings()
        self._guardrails_settings = guardrails_settings or settings.guardrails
        self._profiler_settings = profiler_settings or settings.observability

        # Initialize guardrails
        if self._guardrails_settings.enabled:
            guardrails_config = GuardrailsConfig(
                injection_detection=self._guardrails_settings.injection_detection,
                content_moderation=self._guardrails_settings.content_moderation,
                code_safety=self._guardrails_settings.code_safety,
                pii_filtering=self._guardrails_settings.pii_filtering,
                blocked_commands=self._guardrails_settings.blocked_commands,
                injection_threshold=self._guardrails_settings.injection_threshold,
                endpoint=self._guardrails_settings.nemoguard_endpoint,
            )
            self._guardrails = NemoGuard(guardrails_config)
        else:
            self._guardrails = None

        # Initialize profiler
        if self._profiler_settings.enabled:
            profiler_config = ProfilerConfig(
                enabled=True,
                profile_tokens=self._profiler_settings.profile_tokens,
                profile_latency=self._profiler_settings.profile_latency,
                output_dir=self._profiler_settings.output_dir,
                max_traces_in_memory=self._profiler_settings.max_traces_in_memory,
                trace_sampling_rate=self._profiler_settings.trace_sampling_rate,
            )
            self._profiler = AgentProfiler(profiler_config)
        else:
            self._profiler = None

        # Event storage
        self._events: List[GuardrailEvent] = []
        self._max_events = 10000

        # Callbacks for real-time notifications (supports both sync and async)
        self._event_callbacks: List[Callable[[GuardrailEvent], None]] = []
        self._async_event_callbacks: List[Callable] = []

    @property
    def guardrails_enabled(self) -> bool:
        return self._guardrails is not None

    @property
    def profiler_enabled(self) -> bool:
        return self._profiler is not None

    def on_event(self, callback: Callable[[GuardrailEvent], None]):
        """Register a callback for guardrail events (for WebSocket notifications).

        Supports both sync and async callbacks. Async callbacks are scheduled
        in the event loop.
        """
        if inspect.iscoroutinefunction(callback):
            self._async_event_callbacks.append(callback)
        else:
            self._event_callbacks.append(callback)

    def _record_event(
        self,
        check_type: GuardrailType,
        location: str,
        action: GuardrailAction,
        original_content: str,
        modified_content: Optional[str] = None,
        confidence: float = 1.0,
        model_source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> GuardrailEvent:
        """Record a guardrail event."""
        event = GuardrailEvent(
            timestamp=datetime.now(timezone.utc),
            check_type=check_type,
            location=location,
            action_taken=action,
            original_content=original_content,
            modified_content=modified_content,
            confidence=confidence,
            model_source=model_source,
            trace_id=self._profiler.current_trace_id if self._profiler else None,
            details=details or {},
        )

        # Store event
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Notify sync callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors break instrumentation

        # Schedule async callbacks
        for callback in self._async_event_callbacks:
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(callback(event), loop=loop)
            except RuntimeError:
                # No running event loop - try to run directly
                try:
                    asyncio.run(callback(event))
                except Exception:
                    pass

        return event

    # =========================================================================
    # Trace Management
    # =========================================================================

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new trace (e.g., for an episode or task)."""
        if self._profiler:
            return self._profiler.start_trace(name, metadata)
        return ""

    def end_trace(self, trace_id: Optional[str] = None):
        """End a trace."""
        if self._profiler:
            return self._profiler.end_trace(trace_id)
        return None

    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of a trace."""
        if self._profiler:
            return self._profiler.get_trace_summary(trace_id)
        return {}

    # =========================================================================
    # Command Instrumentation (for Gym, Runner)
    # =========================================================================

    @asynccontextmanager
    async def instrument_command(
        self,
        command: str,
        location: str = "unknown",
        model_source: Optional[str] = None,
    ):
        """
        Instrument a command execution with guardrails + profiling.

        Usage:
            async with inst.instrument_command("ls -la", "gym.execute_bash") as ctx:
                if ctx.allowed:
                    result = sandbox.execute(ctx.content)
                    ctx.set_result(success=True, output=result)
        """
        span = None
        check_result = None
        allowed = True
        action = GuardrailAction.ALLOW
        content = command

        # Start profiler span
        if self._profiler:
            span = self._profiler.start_span(
                name=f"command:{location}",
                kind=SpanKind.TOOL_CALL,
                attributes={"command_preview": command[:100], "location": location}
            )

        # Run guardrails check
        if self._guardrails and self._guardrails_settings.code_safety:
            check_result = await self._guardrails.check_command(command)
            action = check_result.action
            allowed = check_result.passed

            # Record span for guardrail check
            if self._profiler:
                gr_span = self._profiler.start_span(
                    name="guardrails.check_command",
                    kind=SpanKind.CUSTOM,
                    attributes={
                        "action": action.value,
                        "passed": allowed,
                    }
                )
                self._profiler.end_span(gr_span, status="success")

            # Record event if blocked or warned
            if action in (GuardrailAction.BLOCK, GuardrailAction.WARN):
                self._record_event(
                    check_type=GuardrailType.CODE_SAFETY,
                    location=location,
                    action=action,
                    original_content=command,
                    confidence=check_result.results[0].confidence if check_result.results else 1.0,
                    model_source=model_source,
                    details={"reason": check_result.blocked_reason},
                )

        ctx = InstrumentationContext(
            allowed=allowed,
            content=content,
            action=action,
            span=span,
            check_result=check_result,
            _instrumentation=self,
            _location=location,
        )

        try:
            yield ctx
        finally:
            # End span
            if span:
                status = "success" if ctx.allowed else "blocked"
                self._profiler.end_span(span, status=status)

    # =========================================================================
    # Input/Output Instrumentation (for Router, Runner)
    # =========================================================================

    @asynccontextmanager
    async def instrument_input(
        self,
        content: str,
        location: str = "unknown",
    ):
        """
        Instrument input content (prompts, requests) with injection detection.

        Usage:
            async with inst.instrument_input(user_prompt, "runner.task_prompt") as ctx:
                if ctx.allowed:
                    response = await call_llm(ctx.content)
        """
        span = None
        check_result = None
        allowed = True
        action = GuardrailAction.ALLOW
        final_content = content

        # Start profiler span
        if self._profiler:
            span = self._profiler.start_span(
                name=f"input:{location}",
                kind=SpanKind.CUSTOM,
                attributes={"content_preview": content[:100], "location": location}
            )

        # Run guardrails check
        if self._guardrails:
            check_result = await self._guardrails.check_input(content)
            action = check_result.action
            allowed = check_result.passed
            final_content = check_result.final_content

            # Record event if blocked
            if action == GuardrailAction.BLOCK:
                self._record_event(
                    check_type=GuardrailType.INJECTION_DETECTION,
                    location=location,
                    action=action,
                    original_content=content,
                    confidence=check_result.results[0].confidence if check_result.results else 1.0,
                    details={"reason": check_result.blocked_reason},
                )

        ctx = InstrumentationContext(
            allowed=allowed,
            content=final_content,
            action=action,
            span=span,
            check_result=check_result,
            _instrumentation=self,
            _location=location,
        )

        try:
            yield ctx
        finally:
            if span:
                status = "success" if ctx.allowed else "blocked"
                self._profiler.end_span(span, status=status)

    @asynccontextmanager
    async def instrument_output(
        self,
        content: str,
        location: str = "unknown",
        model_source: Optional[str] = None,
        filter_pii: bool = True,
    ):
        """
        Instrument output content (responses) with safety + PII filtering.

        Usage:
            async with inst.instrument_output(response, "router.student", "student") as ctx:
                if ctx.allowed:
                    return ctx.content  # PII-filtered if enabled
        """
        span = None
        check_result = None
        allowed = True
        action = GuardrailAction.ALLOW
        final_content = content
        modified_content = None

        # Start profiler span
        if self._profiler:
            span = self._profiler.start_span(
                name=f"output:{location}",
                kind=SpanKind.CUSTOM,
                attributes={"content_preview": content[:100], "location": location}
            )

        # Run guardrails check
        if self._guardrails:
            check_result = await self._guardrails.check_output(content)
            action = check_result.action
            allowed = check_result.passed
            final_content = check_result.final_content

            # Check if content was modified (PII filtered)
            if final_content != content:
                modified_content = final_content
                self._record_event(
                    check_type=GuardrailType.PII_FILTER,
                    location=location,
                    action=GuardrailAction.MODIFY,
                    original_content=content,
                    modified_content=modified_content,
                    model_source=model_source,
                )

            # Record event if blocked
            if action == GuardrailAction.BLOCK:
                self._record_event(
                    check_type=GuardrailType.CODE_SAFETY,
                    location=location,
                    action=action,
                    original_content=content,
                    confidence=check_result.results[0].confidence if check_result.results else 1.0,
                    model_source=model_source,
                    details={"reason": check_result.blocked_reason},
                )

        ctx = InstrumentationContext(
            allowed=allowed,
            content=final_content,
            action=action,
            span=span,
            check_result=check_result,
            _instrumentation=self,
            _location=location,
        )

        try:
            yield ctx
        finally:
            if span:
                status = "success" if ctx.allowed else "blocked"
                self._profiler.end_span(span, status=status)

    # =========================================================================
    # PII Filtering (for Trace Import)
    # =========================================================================

    async def filter_pii(self, content: str, location: str = "unknown") -> str:
        """
        Filter PII from content and record the event.

        Returns the filtered content.
        """
        if not self._guardrails or not self._guardrails_settings.pii_filtering:
            return content

        # Use the guardrails PII filter
        result, filtered = await self._guardrails._filter_pii(content)

        if result.triggered:
            self._record_event(
                check_type=GuardrailType.PII_FILTER,
                location=location,
                action=GuardrailAction.MODIFY,
                original_content=content,
                modified_content=filtered,
                details=result.details,
            )

        return filtered

    async def check_injection(self, content: str, location: str = "unknown") -> bool:
        """
        Check content for injection attempts.

        Returns True if content is safe, False if injection detected.
        """
        if not self._guardrails or not self._guardrails_settings.injection_detection:
            return True

        result = await self._guardrails._check_injection(content)

        if result.triggered:
            self._record_event(
                check_type=GuardrailType.INJECTION_DETECTION,
                location=location,
                action=result.action,
                original_content=content,
                confidence=result.confidence,
                details={"reason": result.reason},
            )
            return False

        return True

    # =========================================================================
    # LLM Call Instrumentation (for Router)
    # =========================================================================

    def record_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_source: Optional[str] = None,
        **kwargs
    ):
        """Record an LLM API call for profiling."""
        if self._profiler:
            span = self._profiler.record_llm_call(
                model=model,
                prompt=prompt,
                response=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                model_source=model_source or "unknown",
                **kwargs
            )
            return span
        return None

    # =========================================================================
    # Event Access (for API, Training)
    # =========================================================================

    def get_guardrail_events(
        self,
        action: Optional[str] = None,
        check_type: Optional[str] = None,
        model_source: Optional[str] = None,
        limit: int = 100,
    ) -> List[GuardrailEvent]:
        """
        Get recent guardrail events, optionally filtered.

        Args:
            action: Filter by action ("block", "warn", "modify")
            check_type: Filter by type ("injection", "code_safety", "pii")
            model_source: Filter by source ("student", "teacher")
            limit: Max events to return

        Returns:
            List of GuardrailEvent objects
        """
        events = self._events.copy()

        if action:
            action_enum = GuardrailAction(action)
            events = [e for e in events if e.action_taken == action_enum]

        if check_type:
            type_enum = GuardrailType(check_type)
            events = [e for e in events if e.check_type == type_enum]

        if model_source:
            events = [e for e in events if e.model_source == model_source]

        return events[-limit:]

    def get_blocked_events_for_dpo(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get blocked events formatted for DPO negative examples.

        Returns events where student model was blocked, suitable for
        creating DPO training pairs (blocked response = rejected).
        """
        events = [
            e for e in self._events
            if e.action_taken == GuardrailAction.BLOCK and e.model_source == "student"
        ]

        return [
            {
                "rejected_response": e.original_content,
                "reason": e.details.get("reason", "Safety violation"),
                "check_type": e.check_type.value,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated guardrail statistics."""
        if not self._events:
            return {"total_events": 0}

        stats = {
            "total_events": len(self._events),
            "by_action": {},
            "by_type": {},
            "by_source": {},
            "block_rate": 0.0,
        }

        for event in self._events:
            # By action
            action = event.action_taken.value
            stats["by_action"][action] = stats["by_action"].get(action, 0) + 1

            # By type
            check_type = event.check_type.value
            stats["by_type"][check_type] = stats["by_type"].get(check_type, 0) + 1

            # By source
            if event.model_source:
                stats["by_source"][event.model_source] = stats["by_source"].get(event.model_source, 0) + 1

        # Calculate block rate
        blocks = stats["by_action"].get("block", 0)
        stats["block_rate"] = blocks / len(self._events) if self._events else 0.0

        return stats

    # =========================================================================
    # Configuration
    # =========================================================================

    def update_settings(
        self,
        guardrails: Optional[Dict[str, Any]] = None,
        profiler: Optional[Dict[str, Any]] = None,
    ):
        """Update settings at runtime."""
        if guardrails:
            for key, value in guardrails.items():
                if hasattr(self._guardrails_settings, key):
                    setattr(self._guardrails_settings, key, value)

            # Rebuild guardrails with new settings
            if self._guardrails_settings.enabled:
                guardrails_config = GuardrailsConfig(
                    injection_detection=self._guardrails_settings.injection_detection,
                    content_moderation=self._guardrails_settings.content_moderation,
                    code_safety=self._guardrails_settings.code_safety,
                    pii_filtering=self._guardrails_settings.pii_filtering,
                    blocked_commands=self._guardrails_settings.blocked_commands,
                    injection_threshold=self._guardrails_settings.injection_threshold,
                )
                self._guardrails = NemoGuard(guardrails_config)
            else:
                self._guardrails = None

        if profiler:
            for key, value in profiler.items():
                if hasattr(self._profiler_settings, key):
                    setattr(self._profiler_settings, key, value)

    async def close(self):
        """Clean up resources."""
        if self._guardrails:
            await self._guardrails.close()


# Global instance
_instrumentation: Optional[Instrumentation] = None


def get_instrumentation() -> Instrumentation:
    """Get or create the global instrumentation instance."""
    global _instrumentation
    if _instrumentation is None:
        _instrumentation = Instrumentation()
    return _instrumentation


def reset_instrumentation():
    """Reset the global instrumentation instance (for testing)."""
    global _instrumentation
    _instrumentation = None
