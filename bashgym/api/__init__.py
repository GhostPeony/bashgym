"""Bash Gym API - FastAPI endpoints for frontend integration"""

from bashgym.api.routes import app, create_app
from bashgym.api.schemas import (
    ExportFormat,
    ExportRequest,
    ExportResponse,
    HealthCheck,
    ModelInfo,
    RouterStats,
    RoutingDecisionInfo,
    RoutingStrategyEnum,
    SystemStats,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    TraceDetail,
    TraceInfo,
    TraceQuality,
    TraceStatus,
    TraceStep,
    TrainingProgress,
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    TrainingStrategy,
    WSMessage,
)
from bashgym.api.websocket import (
    ConnectionManager,
    MessageType,
    TrainingProgressCallback,
    broadcast_router_stats,
    broadcast_task_status,
    broadcast_trace_event,
    broadcast_training_complete,
    broadcast_training_failed,
    broadcast_verification_result,
    handle_websocket,
    manager,
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
