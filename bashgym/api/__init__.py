"""Bash Gym API - FastAPI endpoints for frontend integration"""

from bashgym.api.routes import app, create_app
from bashgym.api.schemas import (
    TaskRequest, TaskResponse, TaskStatus,
    TrainingRequest, TrainingResponse, TrainingStatus, TrainingStrategy,
    ModelInfo, ExportRequest, ExportResponse, ExportFormat,
    SystemStats, HealthCheck,
    TraceInfo, TraceDetail, TraceStep, TraceQuality, TraceStatus,
    RouterStats, RoutingDecisionInfo, RoutingStrategyEnum,
    WSMessage, TrainingProgress
)
from bashgym.api.websocket import (
    manager, ConnectionManager, MessageType,
    handle_websocket, TrainingProgressCallback,
    broadcast_training_complete, broadcast_training_failed,
    broadcast_task_status, broadcast_trace_event,
    broadcast_router_stats, broadcast_verification_result
)

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
    "broadcast_task_status",
    "broadcast_trace_event",
    "broadcast_router_stats",
    "broadcast_verification_result",
]
