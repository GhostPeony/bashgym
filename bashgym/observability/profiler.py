"""
Agent Profiler for Observability

Provides profiling, tracing, and performance analysis for agentic workflows
with integration support for Phoenix, Langfuse, and OpenTelemetry.

Module: Observability - Agent Performance Tracking
"""

import os
import json
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
import threading


class ObservabilityBackend(Enum):
    """Supported observability backends."""
    LOCAL = "local"  # Local file storage only


class SpanKind(Enum):
    """Types of trace spans."""
    AGENT = "agent"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"
    CUSTOM = "custom"


@dataclass
class ProfilerConfig:
    """Configuration for the Agent Profiler."""

    # Core settings
    enabled: bool = True
    trace_sampling_rate: float = 1.0  # 1.0 = trace all

    # What to profile
    profile_tokens: bool = True
    profile_latency: bool = True
    profile_guardrails: bool = True

    # Output settings (local storage)
    output_dir: str = "data/profiler_traces"
    max_traces_in_memory: int = 1000


@dataclass
class TraceSpan:
    """A single span in a trace."""

    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: str = "in_progress"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def set_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        """Set token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens

    def finish(self, status: str = "success"):
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
        self.latency_ms = self.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "metrics": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "latency_ms": self.latency_ms
            }
        }


@dataclass
class Trace:
    """A complete trace containing multiple spans."""

    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total trace duration."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        """Total tokens across all spans."""
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_llm_calls(self) -> int:
        """Count of LLM calls."""
        return sum(1 for s in self.spans if s.kind == SpanKind.LLM_CALL)

    @property
    def total_tool_calls(self) -> int:
        """Count of tool calls."""
        return sum(1 for s in self.spans if s.kind == SpanKind.TOOL_CALL)

    def add_span(self, span: TraceSpan):
        """Add a span to the trace."""
        self.spans.append(span)

    def finish(self):
        """Finish the trace."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "summary": {
                "total_spans": len(self.spans),
                "total_tokens": self.total_tokens,
                "llm_calls": self.total_llm_calls,
                "tool_calls": self.total_tool_calls
            }
        }


class AgentProfiler:
    """
    Profiler for agentic workflows.

    Features:
    - Trace creation and span management
    - Token counting and latency tracking
    - Multiple backend support (Phoenix, Langfuse, OTEL)
    - Performance analysis and bottleneck identification
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """Initialize the profiler."""
        self.config = config or ProfilerConfig()

        # Trace storage
        self.traces: Dict[str, Trace] = {}
        self.current_trace_id: Optional[str] = None
        self.current_span_id: Optional[str] = None

        # Thread safety
        self._lock = threading.Lock()

        # Counter for IDs
        self._span_counter = 0
        self._trace_counter = 0

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize backend connection
        self._backend_client = None
        if self.config.enabled:
            self._init_backend()

    def _init_backend(self):
        """Initialize local storage backend."""
        # Local storage only - ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        with self._lock:
            self._trace_counter += 1
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            return f"trace_{timestamp}_{self._trace_counter:06d}"

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        with self._lock:
            self._span_counter += 1
            return f"span_{self._span_counter:08d}"

    def start_trace(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new trace.

        Args:
            name: Name of the trace (e.g., task description)
            metadata: Optional metadata

        Returns:
            Trace ID
        """
        if not self.config.enabled:
            return ""

        trace_id = self._generate_trace_id()

        trace = Trace(
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            metadata=metadata or {}
        )

        with self._lock:
            self.traces[trace_id] = trace
            self.current_trace_id = trace_id

            # Cleanup old traces
            if len(self.traces) > self.config.max_traces_in_memory:
                oldest = min(self.traces.keys())
                del self.traces[oldest]

        return trace_id

    def end_trace(self, trace_id: Optional[str] = None) -> Optional[Trace]:
        """
        End a trace.

        Args:
            trace_id: ID of trace to end (uses current if not specified)

        Returns:
            Completed Trace
        """
        if not self.config.enabled:
            return None

        trace_id = trace_id or self.current_trace_id
        if not trace_id or trace_id not in self.traces:
            return None

        with self._lock:
            trace = self.traces[trace_id]
            trace.finish()

            # Clear current if matching
            if self.current_trace_id == trace_id:
                self.current_trace_id = None

        # Export to backend
        self._export_trace(trace)

        return trace

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """
        Start a new span.

        Args:
            name: Span name
            kind: Type of span
            trace_id: Trace to add span to
            parent_id: Parent span ID
            attributes: Initial attributes

        Returns:
            New TraceSpan
        """
        if not self.config.enabled:
            return TraceSpan(
                span_id="", trace_id="", parent_id=None,
                name=name, kind=kind, start_time=time.time()
            )

        trace_id = trace_id or self.current_trace_id
        if not trace_id:
            trace_id = self.start_trace(name)

        span_id = self._generate_span_id()
        parent_id = parent_id or self.current_span_id

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_id,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )

        with self._lock:
            if trace_id in self.traces:
                self.traces[trace_id].add_span(span)
            self.current_span_id = span_id

        return span

    def end_span(
        self,
        span: TraceSpan,
        status: str = "success",
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """
        End a span.

        Args:
            span: Span to end
            status: Final status
            input_tokens: Input token count
            output_tokens: Output token count
        """
        if not self.config.enabled or not span.span_id:
            return

        span.set_tokens(input_tokens, output_tokens)
        span.finish(status)

        with self._lock:
            if span.parent_id:
                self.current_span_id = span.parent_id
            else:
                self.current_span_id = None

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        **attributes
    ):
        """
        Context manager for spans.

        Usage:
            with profiler.span("my_operation", kind=SpanKind.TOOL_CALL):
                # do work
        """
        span = self.start_span(name, kind, attributes=attributes)
        try:
            yield span
            span.finish("success")
        except Exception as e:
            span.add_event("error", {"message": str(e)})
            span.finish("error")
            raise

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        **attributes
    ):
        """Async context manager for spans."""
        span = self.start_span(name, kind, attributes=attributes)
        try:
            yield span
            span.finish("success")
        except Exception as e:
            span.add_event("error", {"message": str(e)})
            span.finish("error")
            raise

    def record_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        **kwargs
    ) -> TraceSpan:
        """
        Record an LLM API call.

        Args:
            model: Model name
            prompt: Input prompt
            response: Model response
            input_tokens: Input token count
            output_tokens: Output token count
            latency_ms: Call latency
            **kwargs: Additional attributes

        Returns:
            Created span
        """
        span = self.start_span(
            name=f"llm_call:{model}",
            kind=SpanKind.LLM_CALL,
            attributes={
                "model": model,
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response_preview": response[:100] + "..." if len(response) > 100 else response,
                **kwargs
            }
        )

        span.set_tokens(input_tokens, output_tokens)
        span.latency_ms = latency_ms
        span.finish("success")

        return span

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        tool_output: str,
        success: bool,
        latency_ms: float,
        **kwargs
    ) -> TraceSpan:
        """
        Record a tool/function call.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input
            tool_output: Tool output
            success: Whether call succeeded
            latency_ms: Call latency
            **kwargs: Additional attributes

        Returns:
            Created span
        """
        span = self.start_span(
            name=f"tool:{tool_name}",
            kind=SpanKind.TOOL_CALL,
            attributes={
                "tool_name": tool_name,
                "input_preview": str(tool_input)[:100],
                "output_preview": str(tool_output)[:100],
                "success": success,
                **kwargs
            }
        )

        span.latency_ms = latency_ms
        span.finish("success" if success else "error")

        return span

    def get_trace_summary(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of a trace.

        Args:
            trace_id: Trace ID (uses current if not specified)

        Returns:
            Summary dictionary
        """
        trace_id = trace_id or self.current_trace_id
        if not trace_id or trace_id not in self.traces:
            return {}

        trace = self.traces[trace_id]

        # Analyze performance
        llm_spans = [s for s in trace.spans if s.kind == SpanKind.LLM_CALL]
        tool_spans = [s for s in trace.spans if s.kind == SpanKind.TOOL_CALL]

        return {
            "trace_id": trace_id,
            "name": trace.name,
            "duration_ms": trace.duration_ms,
            "total_spans": len(trace.spans),
            "llm_calls": {
                "count": len(llm_spans),
                "total_tokens": sum(s.total_tokens for s in llm_spans),
                "total_latency_ms": sum(s.latency_ms for s in llm_spans),
                "avg_latency_ms": sum(s.latency_ms for s in llm_spans) / max(len(llm_spans), 1)
            },
            "tool_calls": {
                "count": len(tool_spans),
                "success_rate": sum(1 for s in tool_spans if s.status == "success") / max(len(tool_spans), 1),
                "total_latency_ms": sum(s.latency_ms for s in tool_spans)
            },
            "bottlenecks": self._identify_bottlenecks(trace)
        }

    def _identify_bottlenecks(self, trace: Trace) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in a trace."""
        bottlenecks = []

        for span in trace.spans:
            # Flag slow LLM calls (>5s)
            if span.kind == SpanKind.LLM_CALL and span.latency_ms > 5000:
                bottlenecks.append({
                    "span_id": span.span_id,
                    "type": "slow_llm_call",
                    "latency_ms": span.latency_ms,
                    "suggestion": "Consider using a faster model or caching"
                })

            # Flag high token usage (>4K)
            if span.total_tokens > 4000:
                bottlenecks.append({
                    "span_id": span.span_id,
                    "type": "high_token_usage",
                    "tokens": span.total_tokens,
                    "suggestion": "Consider prompt optimization or context trimming"
                })

        return bottlenecks

    def _export_trace(self, trace: Trace):
        """Export trace to local file."""
        output_path = Path(self.config.output_dir) / f"{trace.trace_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(trace.to_dict(), indent=2))

    def get_all_traces(self) -> List[Trace]:
        """Get all traces in memory."""
        return list(self.traces.values())


# Global profiler instance
_profiler: Optional[AgentProfiler] = None


def get_profiler() -> AgentProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = AgentProfiler()
    return _profiler


def main():
    """Example usage of the Agent Profiler."""
    config = ProfilerConfig(
        backend=ObservabilityBackend.LOCAL,
        profile_tokens=True
    )

    profiler = AgentProfiler(config)

    # Start a trace for an agent task
    trace_id = profiler.start_trace(
        "Debug Python script",
        metadata={"user": "test", "priority": "high"}
    )

    # Simulate some agent work
    with profiler.span("analyze_error", SpanKind.LLM_CALL) as span:
        time.sleep(0.1)  # Simulate LLM call
        span.set_tokens(input_tokens=500, output_tokens=200)

    profiler.record_tool_call(
        tool_name="read_file",
        tool_input="script.py",
        tool_output="def main(): ...",
        success=True,
        latency_ms=50
    )

    profiler.record_llm_call(
        model="claude-sonnet-4-20250514",
        prompt="Fix the bug in this code...",
        response="Here's the fix...",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=2500
    )

    # End trace and get summary
    trace = profiler.end_trace()
    summary = profiler.get_trace_summary(trace_id)

    print("Trace Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
