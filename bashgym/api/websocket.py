"""
Bash Gym WebSocket Handler

Provides real-time updates for:
- Training progress (loss, metrics)
- Task status changes
- Trace events
- Router statistics
"""

import asyncio
import json
from typing import Dict, Set, Any, Optional, Callable, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(str, Enum):
    """WebSocket message types."""
    # Training events
    TRAINING_PROGRESS = "training:progress"
    TRAINING_COMPLETE = "training:complete"
    TRAINING_FAILED = "training:failed"
    TRAINING_LOG = "training:log"  # Raw log output

    # Task events
    TASK_STATUS = "task:status"
    TASK_COMPLETE = "task:complete"

    # Trace events
    TRACE_ADDED = "trace:added"
    TRACE_PROMOTED = "trace:promoted"
    TRACE_DEMOTED = "trace:demoted"

    # Router events
    ROUTER_STATS = "router:stats"
    ROUTER_DECISION = "router:decision"

    # Verification events
    VERIFICATION_RESULT = "verification:result"

    # Guardrail events
    GUARDRAIL_BLOCKED = "guardrail:blocked"
    GUARDRAIL_WARN = "guardrail:warn"
    GUARDRAIL_PII_REDACTED = "guardrail:pii_redacted"

    # System events
    SYSTEM_STATUS = "system:status"
    ERROR = "error"

    # HuggingFace events
    HF_JOB_STARTED = "hf:job:started"
    HF_JOB_LOG = "hf:job:log"
    HF_JOB_COMPLETED = "hf:job:completed"
    HF_JOB_FAILED = "hf:job:failed"
    HF_SPACE_READY = "hf:space:ready"
    HF_SPACE_ERROR = "hf:space:error"

    # Orchestration events
    ORCHESTRATION_DECOMPOSING = "orchestration:decomposing"
    ORCHESTRATION_READY = "orchestration:ready"
    ORCHESTRATION_TASK_STARTED = "orchestration:task:started"
    ORCHESTRATION_TASK_COMPLETED = "orchestration:task:completed"
    ORCHESTRATION_TASK_FAILED = "orchestration:task:failed"
    ORCHESTRATION_TASK_RETRYING = "orchestration:task:retrying"
    ORCHESTRATION_MERGE_RESULT = "orchestration:merge:result"
    ORCHESTRATION_BUDGET_UPDATE = "orchestration:budget:update"
    ORCHESTRATION_COMPLETE = "orchestration:complete"
    ORCHESTRATION_CANCELLED = "orchestration:cancelled"

    # Bashbros Integration events
    INTEGRATION_TRACE_RECEIVED = "integration:trace:received"
    INTEGRATION_TRACE_PROCESSED = "integration:trace:processed"
    INTEGRATION_MODEL_EXPORTED = "integration:model:exported"
    INTEGRATION_MODEL_ROLLBACK = "integration:model:rollback"
    INTEGRATION_LINKED = "integration:linked"
    INTEGRATION_UNLINKED = "integration:unlinked"
    INTEGRATION_TRAINING_TRIGGERED = "integration:training:triggered"


@dataclass
class WSMessage:
    """WebSocket message structure."""
    type: str
    payload: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.

    Supports:
    - Multiple concurrent connections
    - Topic-based subscriptions
    - Broadcast to all or specific clients
    """

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a new connection."""
        import logging
        logger = logging.getLogger(__name__)

        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"[WebSocket] Client connected. Total connections: {len(self.active_connections)}")

        # Send welcome message
        welcome = WSMessage(
            type="connected",
            payload={"message": "Connected to Bash Gym WebSocket"}
        )
        await websocket.send_text(welcome.to_json())

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a connection."""
        import logging
        logger = logging.getLogger(__name__)

        self.active_connections.discard(websocket)
        logger.info(f"[WebSocket] Client disconnected. Total connections: {len(self.active_connections)}")

        # Remove from all subscriptions
        for topic_subs in self.subscriptions.values():
            topic_subs.discard(websocket)

    def subscribe(self, websocket: WebSocket, topic: str) -> None:
        """Subscribe a connection to a topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)

    def unsubscribe(self, websocket: WebSocket, topic: str) -> None:
        """Unsubscribe a connection from a topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)

    async def send_personal(self, websocket: WebSocket, message: WSMessage) -> None:
        """Send a message to a specific connection."""
        try:
            await websocket.send_text(message.to_json())
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, message: WSMessage) -> None:
        """Broadcast a message to all connections."""
        import logging
        logger = logging.getLogger(__name__)

        num_connections = len(self.active_connections)
        if num_connections == 0:
            logger.debug(f"[WebSocket] No active connections, skipping broadcast: {message.type}")
            return

        logger.debug(f"[WebSocket] Broadcasting {message.type} to {num_connections} connections")
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message.to_json())
            except Exception as e:
                logger.warning(f"[WebSocket] Failed to send to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_to_topic(self, topic: str, message: WSMessage) -> None:
        """Broadcast a message to all subscribers of a topic."""
        if topic not in self.subscriptions:
            return

        disconnected = set()

        for connection in self.subscriptions[topic]:
            try:
                await connection.send_text(message.to_json())
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
manager = ConnectionManager()


class TrainingProgressCallback:
    """
    Callback for training progress that broadcasts via WebSocket.

    Usage:
        callback = TrainingProgressCallback(run_id)
        trainer.train_sft(dataset, callback=callback.on_progress)
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._loop = None

    def _get_loop(self):
        """Get the event loop, creating one if needed for sync context."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    async def on_progress(self, metrics: Dict[str, Any]) -> None:
        """Called on each training step/epoch."""
        message = WSMessage(
            type=MessageType.TRAINING_PROGRESS,
            payload={
                "run_id": self.run_id,
                "epoch": metrics.get("epoch"),
                "total_epochs": metrics.get("total_epochs"),
                "step": metrics.get("step", 0),
                "total_steps": metrics.get("total_steps", 0),
                "loss": metrics.get("loss"),
                "learning_rate": metrics.get("learning_rate"),
                "grad_norm": metrics.get("grad_norm", 0),
                "eta": metrics.get("eta")
            }
        )
        await manager.broadcast(message)

    def on_progress_sync(self, metrics: Dict[str, Any]) -> None:
        """Synchronous version for non-async training loops."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run in the existing loop
                asyncio.ensure_future(self.on_progress(metrics), loop=loop)
            else:
                # Run in a new loop if none is running
                asyncio.run(self.on_progress(metrics))
        except RuntimeError:
            # No event loop in current thread - create one
            try:
                asyncio.run(self.on_progress(metrics))
            except Exception as e:
                logger.warning(f"Failed to send progress via WebSocket: {e}")

        # Also log progress for visibility
        step = metrics.get("step", 0)
        total = metrics.get("total_steps", 0)
        loss = metrics.get("loss")
        loss_str = f", loss={loss:.4f}" if loss else ""
        logger.info(f"[Progress] Step {step}/{total}{loss_str}")

    async def on_log(self, log_line: str) -> None:
        """Called for each raw log line."""
        await broadcast_training_log(self.run_id, log_line)

    def on_log_sync(self, log_line: str) -> None:
        """Synchronous version for log lines."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.on_log(log_line), loop=loop)
            else:
                asyncio.run(self.on_log(log_line))
        except RuntimeError:
            try:
                asyncio.run(self.on_log(log_line))
            except Exception:
                pass  # Silently fail for log lines


async def broadcast_training_complete(run_id: str, metrics: Dict[str, Any]) -> None:
    """Broadcast training completion."""
    message = WSMessage(
        type=MessageType.TRAINING_COMPLETE,
        payload={
            "run_id": run_id,
            "metrics": metrics
        }
    )
    await manager.broadcast(message)


async def broadcast_training_failed(run_id: str, error: str) -> None:
    """Broadcast training failure."""
    message = WSMessage(
        type=MessageType.TRAINING_FAILED,
        payload={
            "run_id": run_id,
            "error": error
        }
    )
    await manager.broadcast(message)


async def broadcast_training_log(run_id: str, log_line: str, level: str = "info") -> None:
    """Broadcast a training log line."""
    message = WSMessage(
        type=MessageType.TRAINING_LOG,
        payload={
            "run_id": run_id,
            "message": log_line,
            "level": level
        }
    )
    await manager.broadcast(message)


async def broadcast_task_status(task_id: str, status: str, result: Any = None) -> None:
    """Broadcast task status update."""
    message = WSMessage(
        type=MessageType.TASK_STATUS,
        payload={
            "task_id": task_id,
            "status": status,
            "result": result
        }
    )
    await manager.broadcast(message)


async def broadcast_trace_event(
    event_type: MessageType,
    trace_id: str,
    trace_data: Dict[str, Any] = None
) -> None:
    """Broadcast trace-related events."""
    message = WSMessage(
        type=event_type,
        payload={
            "trace_id": trace_id,
            "data": trace_data or {}
        }
    )
    await manager.broadcast(message)


async def broadcast_router_stats(stats: Dict[str, Any]) -> None:
    """Broadcast router statistics."""
    message = WSMessage(
        type=MessageType.ROUTER_STATS,
        payload=stats
    )
    await manager.broadcast(message)


async def broadcast_verification_result(
    task_id: str,
    passed: bool,
    details: Dict[str, Any] = None
) -> None:
    """Broadcast verification results."""
    message = WSMessage(
        type=MessageType.VERIFICATION_RESULT,
        payload={
            "task_id": task_id,
            "passed": passed,
            "details": details or {}
        }
    )
    await manager.broadcast(message)


async def broadcast_guardrail_event(
    event_type: MessageType,
    check_type: str,
    location: str,
    action: str,
    confidence: float,
    content_preview: str = None,
    details: Dict[str, Any] = None
) -> None:
    """Broadcast guardrail events (blocked, warned, PII redacted)."""
    message = WSMessage(
        type=event_type,
        payload={
            "check_type": check_type,
            "location": location,
            "action": action,
            "confidence": confidence,
            "content_preview": content_preview[:100] if content_preview else None,
            "details": details or {}
        }
    )
    await manager.broadcast(message)


async def broadcast_guardrail_blocked(
    check_type: str,
    location: str,
    confidence: float,
    content_preview: str = None,
    details: Dict[str, Any] = None
) -> None:
    """Broadcast that content was blocked by guardrails."""
    await broadcast_guardrail_event(
        MessageType.GUARDRAIL_BLOCKED,
        check_type=check_type,
        location=location,
        action="block",
        confidence=confidence,
        content_preview=content_preview,
        details=details
    )


async def broadcast_guardrail_warn(
    check_type: str,
    location: str,
    confidence: float,
    content_preview: str = None,
    details: Dict[str, Any] = None
) -> None:
    """Broadcast that content triggered a guardrail warning."""
    await broadcast_guardrail_event(
        MessageType.GUARDRAIL_WARN,
        check_type=check_type,
        location=location,
        action="warn",
        confidence=confidence,
        content_preview=content_preview,
        details=details
    )


async def broadcast_pii_redacted(
    location: str,
    redaction_count: int,
    pii_types: List[str] = None,
    details: Dict[str, Any] = None
) -> None:
    """Broadcast that PII was redacted from content."""
    message = WSMessage(
        type=MessageType.GUARDRAIL_PII_REDACTED,
        payload={
            "location": location,
            "redaction_count": redaction_count,
            "pii_types": pii_types or [],
            "details": details or {}
        }
    )
    await manager.broadcast(message)


# =============================================================================
# HuggingFace Event Broadcasts
# =============================================================================

async def broadcast_hf_job_started(job_id: str, hardware: str, repo_id: str = None) -> None:
    """Broadcast when a HuggingFace job starts."""
    message = WSMessage(
        type=MessageType.HF_JOB_STARTED,
        payload={
            "job_id": job_id,
            "hardware": hardware,
            "repo_id": repo_id,
        }
    )
    await manager.broadcast(message)


async def broadcast_hf_job_log(job_id: str, log_line: str, level: str = "info") -> None:
    """Broadcast a HuggingFace job log line."""
    message = WSMessage(
        type=MessageType.HF_JOB_LOG,
        payload={
            "job_id": job_id,
            "log": log_line,
            "level": level,
        }
    )
    await manager.broadcast(message)


async def broadcast_hf_job_completed(job_id: str, model_repo: str = None, metrics: Dict[str, Any] = None) -> None:
    """Broadcast when a HuggingFace job completes successfully."""
    message = WSMessage(
        type=MessageType.HF_JOB_COMPLETED,
        payload={
            "job_id": job_id,
            "model_repo": model_repo,
            "metrics": metrics or {},
        }
    )
    await manager.broadcast(message)


async def broadcast_hf_job_failed(job_id: str, error: str, logs: str = None) -> None:
    """Broadcast when a HuggingFace job fails."""
    message = WSMessage(
        type=MessageType.HF_JOB_FAILED,
        payload={
            "job_id": job_id,
            "error": error,
            "logs": logs,
        }
    )
    await manager.broadcast(message)


async def broadcast_hf_space_ready(space_name: str, url: str) -> None:
    """Broadcast when a HuggingFace Space is ready."""
    message = WSMessage(
        type=MessageType.HF_SPACE_READY,
        payload={
            "space_name": space_name,
            "url": url,
        }
    )
    await manager.broadcast(message)


async def broadcast_hf_space_error(space_name: str, error: str) -> None:
    """Broadcast when a HuggingFace Space encounters an error."""
    message = WSMessage(
        type=MessageType.HF_SPACE_ERROR,
        payload={
            "space_name": space_name,
            "error": error,
        }
    )
    await manager.broadcast(message)


# =============================================================================
# Bashbros Integration Event Broadcasts
# =============================================================================

async def broadcast_integration_trace_received(
    filename: str,
    task: str,
    source: str = "bashbros",
    steps: int = 0,
    verified: bool = False
) -> None:
    """Broadcast when a new trace is received from bashbros."""
    message = WSMessage(
        type=MessageType.INTEGRATION_TRACE_RECEIVED,
        payload={
            "filename": filename,
            "task": task[:100] if task else "",
            "source": source,
            "steps": steps,
            "verified": verified,
        }
    )
    await manager.broadcast(message)


async def broadcast_integration_trace_processed(
    filename: str,
    success: bool,
    trace_id: str = None,
    error: str = None
) -> None:
    """Broadcast when a trace has been processed."""
    message = WSMessage(
        type=MessageType.INTEGRATION_TRACE_PROCESSED,
        payload={
            "filename": filename,
            "success": success,
            "trace_id": trace_id,
            "error": error,
        }
    )
    await manager.broadcast(message)


async def broadcast_integration_model_exported(
    version: str,
    gguf_path: str,
    ollama_registered: bool = False,
    traces_used: int = 0,
    quality_avg: float = 0.0
) -> None:
    """Broadcast when a model has been exported to GGUF for bashbros sidekick."""
    message = WSMessage(
        type=MessageType.INTEGRATION_MODEL_EXPORTED,
        payload={
            "version": version,
            "gguf_path": gguf_path,
            "ollama_registered": ollama_registered,
            "traces_used": traces_used,
            "quality_avg": quality_avg,
        }
    )
    await manager.broadcast(message)


async def broadcast_integration_model_rollback(
    version: str,
    previous_version: str = None
) -> None:
    """Broadcast when model has been rolled back to a previous version."""
    message = WSMessage(
        type=MessageType.INTEGRATION_MODEL_ROLLBACK,
        payload={
            "version": version,
            "previous_version": previous_version,
        }
    )
    await manager.broadcast(message)


async def broadcast_integration_linked() -> None:
    """Broadcast when bashbros integration is linked."""
    message = WSMessage(
        type=MessageType.INTEGRATION_LINKED,
        payload={"linked": True}
    )
    await manager.broadcast(message)


async def broadcast_integration_unlinked() -> None:
    """Broadcast when bashbros integration is unlinked."""
    message = WSMessage(
        type=MessageType.INTEGRATION_UNLINKED,
        payload={"linked": False}
    )
    await manager.broadcast(message)


async def broadcast_integration_training_triggered(
    gold_traces: int,
    threshold: int,
    run_id: str = None
) -> None:
    """Broadcast when auto-training is triggered by bashbros integration."""
    message = WSMessage(
        type=MessageType.INTEGRATION_TRAINING_TRIGGERED,
        payload={
            "gold_traces": gold_traces,
            "threshold": threshold,
            "run_id": run_id,
        }
    )
    await manager.broadcast(message)


# =============================================================================
# Orchestration Broadcasts
# =============================================================================

async def broadcast_orchestration_task_started(
    job_id: str, task_id: str, task_title: str, worker_count: int = 0
) -> None:
    """Broadcast when an orchestration task worker is spawned."""
    message = WSMessage(
        type=MessageType.ORCHESTRATION_TASK_STARTED,
        payload={
            "job_id": job_id,
            "task_id": task_id,
            "task_title": task_title,
            "active_workers": worker_count,
        }
    )
    await manager.broadcast(message)


async def broadcast_orchestration_task_completed(
    job_id: str, task_id: str, cost_usd: float,
    duration_seconds: float, newly_unblocked: int = 0
) -> None:
    """Broadcast when an orchestration task completes successfully."""
    message = WSMessage(
        type=MessageType.ORCHESTRATION_TASK_COMPLETED,
        payload={
            "job_id": job_id,
            "task_id": task_id,
            "cost_usd": round(cost_usd, 4),
            "duration_seconds": round(duration_seconds, 1),
            "newly_unblocked": newly_unblocked,
        }
    )
    await manager.broadcast(message)


async def broadcast_orchestration_task_failed(
    job_id: str, task_id: str, error: str, will_retry: bool = False
) -> None:
    """Broadcast when an orchestration task fails."""
    message = WSMessage(
        type=MessageType.ORCHESTRATION_TASK_FAILED,
        payload={
            "job_id": job_id,
            "task_id": task_id,
            "error": error[:500],
            "will_retry": will_retry,
        }
    )
    await manager.broadcast(message)


async def broadcast_orchestration_budget_update(
    job_id: str, spent_usd: float, budget_usd: float, task_count: int = 0
) -> None:
    """Broadcast budget status update during orchestration."""
    message = WSMessage(
        type=MessageType.ORCHESTRATION_BUDGET_UPDATE,
        payload={
            "job_id": job_id,
            "spent_usd": round(spent_usd, 4),
            "budget_usd": round(budget_usd, 2),
            "remaining_usd": round(budget_usd - spent_usd, 4),
            "tasks_completed": task_count,
        }
    )
    await manager.broadcast(message)


async def broadcast_orchestration_complete(
    job_id: str, completed: int, failed: int,
    total_cost: float, total_time: float,
    merge_successes: int = 0, merge_failures: int = 0,
) -> None:
    """Broadcast when entire orchestration job finishes."""
    message = WSMessage(
        type=MessageType.ORCHESTRATION_COMPLETE,
        payload={
            "job_id": job_id,
            "completed": completed,
            "failed": failed,
            "total_cost_usd": round(total_cost, 4),
            "total_time_seconds": round(total_time, 1),
            "merge_successes": merge_successes,
            "merge_failures": merge_failures,
        }
    )
    await manager.broadcast(message)


async def handle_websocket(websocket: WebSocket) -> None:
    """
    Main WebSocket handler.

    Handles:
    - Connection lifecycle
    - Subscription management
    - Incoming messages
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive and parse messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                payload = message.get("payload", {})

                # Handle subscription requests
                if msg_type == "subscribe":
                    topic = payload.get("topic")
                    if topic:
                        manager.subscribe(websocket, topic)
                        await manager.send_personal(websocket, WSMessage(
                            type="subscribed",
                            payload={"topic": topic}
                        ))

                elif msg_type == "unsubscribe":
                    topic = payload.get("topic")
                    if topic:
                        manager.unsubscribe(websocket, topic)
                        await manager.send_personal(websocket, WSMessage(
                            type="unsubscribed",
                            payload={"topic": topic}
                        ))

                # Handle ping/pong for connection health
                elif msg_type == "ping":
                    await manager.send_personal(websocket, WSMessage(
                        type="pong",
                        payload={}
                    ))

            except json.JSONDecodeError:
                await manager.send_personal(websocket, WSMessage(
                    type=MessageType.ERROR,
                    payload={"message": "Invalid JSON"}
                ))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
