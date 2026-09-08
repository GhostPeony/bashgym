"""Bash Gym API - FastAPI endpoints for frontend integration"""

from importlib import import_module


def __getattr__(name):
    """Importing a lightweight API utility must not construct the whole server."""
    if name not in __all__:
        raise AttributeError(name)
    if name in {"app", "create_app"}:
        module = "routes"
    elif name.startswith("broadcast_") or name in {
        "ConnectionManager",
        "MessageType",
        "TrainingProgressCallback",
        "handle_websocket",
        "manager",
    }:
        module = "websocket"
    else:
        module = "schemas"
    value = getattr(import_module(f"bashgym.api.{module}"), name)
    globals()[name] = value
    return value


__all__ = [
    # App
    "app",
    "create_app",
    # Task schemas
    "TaskRequest",
    "TaskResponse",
    "TaskStatus",
    # Training schemas
    "TrainingRequest",
    "TrainingResponse",
    "TrainingStatus",
    "TrainingStrategy",
    "TrainingProgress",
    # Model schemas
    "ModelInfo",
    "ExportRequest",
    "ExportResponse",
    "ExportFormat",
    # Trace schemas
    "TraceInfo",
    "TraceDetail",
    "TraceStep",
    "TraceQuality",
    "TraceStatus",
    # Router schemas
    "RouterStats",
    "RoutingDecisionInfo",
    "RoutingStrategyEnum",
    # System schemas
    "SystemStats",
    "HealthCheck",
    # WebSocket schemas
    "WSMessage",
    # WebSocket utilities
    "manager",
    "ConnectionManager",
    "MessageType",
    "handle_websocket",
    "TrainingProgressCallback",
    "broadcast_training_complete",
    "broadcast_training_failed",
    "broadcast_training_queued",
    "broadcast_workspace_canvas_intent",
    "broadcast_workspace_context_updated",
    "broadcast_task_status",
    "broadcast_trace_event",
    "broadcast_router_stats",
    "broadcast_verification_result",
]
