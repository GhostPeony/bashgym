"""
Bash Gym WebSocket Handler

Provides real-time updates for:
- Training progress (loss, metrics)
- Task status changes
- Trace events
- Router statistics
"""

import asyncio
import hashlib
import json
import os
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field


class CampaignHintV1(BaseModel):
    """Low-entropy notification; durable REST state remains authoritative."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="campaign_hint.v1", pattern=r"^campaign_hint\.v1$")
    workspace_id: str = Field(
        min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$"
    )
    campaign_id: str = Field(
        min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$"
    )
    event_cursor: int = Field(ge=1)
    aggregate_version: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")
    correlation_id: str = Field(
        min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$"
    )
    emitted_at: datetime


def build_campaign_hint(source: Any, *, emitted_at: datetime | None = None) -> CampaignHintV1:
    """Project one payload-free persistence row at hint-send time."""

    return CampaignHintV1(
        workspace_id=source.workspace_id,
        campaign_id=source.campaign_id,
        event_cursor=source.cursor,
        aggregate_version=source.aggregate_version,
        event_type=source.event_type,
        correlation_id=source.correlation_id,
        emitted_at=emitted_at or datetime.now(UTC),
    )


@dataclass(frozen=True)
class CampaignLiveTicketBinding:
    workspace_id: str
    after_cursor: int
    expires_at: datetime
    credential_id: str
    authorization_revision: int
    repository: Any


@dataclass
class CampaignLiveSubscription:
    binding: CampaignLiveTicketBinding
    cursor: int


class MessageType(str, Enum):
    """WebSocket message types."""

    # Training events
    TRAINING_QUEUED = "training:queued"
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

    # Workspace canvas events
    WORKSPACE_CANVAS_INTENT = "workspace:canvas:intent"
    WORKSPACE_CONTEXT_UPDATED = "workspace:context:updated"

    # Durable campaign notification; payload is CampaignHintV1 only.
    CAMPAIGN_HINT = "campaign:hint"

    # HuggingFace events
    HF_JOB_STARTED = "hf:job:started"
    HF_JOB_LOG = "hf:job:log"
    HF_JOB_COMPLETED = "hf:job:completed"
    HF_JOB_FAILED = "hf:job:failed"
    HF_JOB_METRICS = "hf:job:metrics"
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

    # Pipeline events (auto-import pipeline)
    PIPELINE_IMPORT = "pipeline:import"
    PIPELINE_CLASSIFIED = "pipeline:classified"
    PIPELINE_THRESHOLD_REACHED = "pipeline:threshold_reached"
    PIPELINE_STAGE_STARTED = "pipeline:stage_started"

    # AutoResearch events (hyperparameter search)
    AUTORESEARCH_EXPERIMENT = "autoresearch:experiment"
    AUTORESEARCH_COMPLETE = "autoresearch:complete"
    AUTORESEARCH_FAILED = "autoresearch:failed"

    # Trace Research events (data-centric autoresearch)
    TRACE_RESEARCH_EXPERIMENT = "autoresearch:trace-experiment"
    TRACE_RESEARCH_COMPLETE = "autoresearch:trace-research-complete"
    TRACE_RESEARCH_FAILED = "autoresearch:trace-research-failed"

    # Schema Research events (Data Designer schema evolution)
    SCHEMA_RESEARCH_EXPERIMENT = "schema-research:experiment"
    SCHEMA_RESEARCH_STATUS = "schema-research:status"
    SCHEMA_RESEARCH_COMPLETE = "schema-research:complete"
    SCHEMA_RESEARCH_FAILED = "schema-research:failed"

    # Cascade RL events (domain-by-domain sequential training)
    CASCADE_STAGE_STARTED = "cascade:stage-started"
    CASCADE_STAGE_COMPLETED = "cascade:stage-completed"
    CASCADE_STAGE_FAILED = "cascade:stage-failed"
    CASCADE_STAGE_SKIPPED = "cascade:stage-skipped"
    CASCADE_COMPLETED = "cascade:completed"
    CASCADE_PROGRESS = "cascade:progress"
    MOPD_DATASET_READY = "cascade:mopd-dataset-ready"
    MOPD_TRAINING_STARTED = "cascade:mopd-training-started"
    MOPD_COMPLETED = "cascade:mopd-completed"
    MOPD_FAILED = "cascade:mopd-failed"


@dataclass
class WSMessage:
    """WebSocket message structure."""

    type: str
    payload: dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()

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
        self.active_connections: set[WebSocket] = set()
        self.subscriptions: dict[str, set[WebSocket]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.campaign_tickets: dict[str, CampaignLiveTicketBinding] = {}
        self.campaign_ticket_mint_windows: dict[
            tuple[str, str], tuple[datetime, int]
        ] = {}
        self.campaign_subscriptions: dict[
            WebSocket, dict[str, CampaignLiveSubscription]
        ] = {}
        self.campaign_poll_task: asyncio.Task[None] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a new connection."""
        import logging

        logger = logging.getLogger(__name__)

        await websocket.accept()
        self.loop = asyncio.get_running_loop()
        self.active_connections.add(websocket)
        logger.info(
            f"[WebSocket] Client connected. Total connections: {len(self.active_connections)}"
        )

        # Send welcome message
        welcome = WSMessage(
            type="connected", payload={"message": "Connected to Bash Gym WebSocket"}
        )
        await websocket.send_text(welcome.to_json())

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a connection."""
        import logging

        logger = logging.getLogger(__name__)

        self.active_connections.discard(websocket)
        logger.info(
            f"[WebSocket] Client disconnected. Total connections: {len(self.active_connections)}"
        )

        # Remove from all subscriptions
        for topic_subs in self.subscriptions.values():
            topic_subs.discard(websocket)
        self.campaign_subscriptions.pop(websocket, None)
        if not self.campaign_subscriptions and self.campaign_poll_task is not None:
            self.campaign_poll_task.cancel()
            self.campaign_poll_task = None

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
        """Broadcast a message to all connections + optional Discord webhook."""
        import logging

        logger = logging.getLogger(__name__)

        num_connections = len(self.active_connections)
        if num_connections > 0:
            logger.debug(
                f"[WebSocket] Broadcasting {message.type} to {num_connections} connections"
            )
            disconnected = set()

            for connection in self.active_connections:
                try:
                    await connection.send_text(message.to_json())
                except Exception as e:
                    logger.warning(f"[WebSocket] Failed to send to connection: {e}")
                    disconnected.add(connection)

            for conn in disconnected:
                self.disconnect(conn)

        # Discord webhook for training events
        await self._discord_notify(message)

    # Discord webhook notification types (only send these, not every WS message)
    DISCORD_EVENT_TYPES = {
        "training:complete",
        "training:failed",
        "cascade:stage-completed",
        "cascade:stage-failed",
        "cascade:completed",
        "schema-research:experiment",
        "schema-research:status",
        "classify:completed",
        "cascade:mopd-completed",
        "cascade:mopd-failed",
    }

    async def _discord_notify(self, message: WSMessage) -> None:
        """Send training events to Discord webhook if configured."""
        import os

        webhook_url = os.environ.get("DISCORD_TRAINING_WEBHOOK")
        if not webhook_url:
            return

        # Only send significant events, not every progress tick
        if message.type not in self.DISCORD_EVENT_TYPES:
            return

        try:
            import httpx

            # Format a clean Discord embed
            data = message.data if hasattr(message, "data") else {}
            if not isinstance(data, dict):
                data = getattr(message, "payload", {}) or {}

            # Build embed based on event type
            color = 0x2ECC71  # green
            if "failed" in message.type:
                color = 0xE74C3C  # red
            elif "experiment" in message.type:
                color = 0x3498DB  # blue

            embed = {
                "title": f"BashGym: {message.type}",
                "color": color,
                "fields": [],
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Add relevant fields from the data
            for key, value in list(data.items())[:6]:
                embed["fields"].append(
                    {
                        "name": str(key).replace("_", " ").title(),
                        "value": str(value)[:200],
                        "inline": True,
                    }
                )

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(webhook_url, json={"embeds": [embed]})
        except Exception:
            pass  # Discord notifications are best-effort

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

    @staticmethod
    def _ticket_hash(ticket: str) -> str:
        return hashlib.sha256(ticket.encode("utf-8")).hexdigest()

    def issue_campaign_live_ticket(
        self,
        repository: Any,
        principal: Any,
        workspace_id: str,
        *,
        after_cursor: int,
        ttl: timedelta = timedelta(seconds=30),
    ) -> tuple[str, CampaignLiveTicketBinding]:
        now = datetime.now(UTC)
        expires_at = min(now + ttl, principal.expires_at)
        for digest, existing in tuple(self.campaign_tickets.items()):
            if (
                existing.expires_at <= now
                or (
                    existing.credential_id == principal.credential_id
                    and existing.workspace_id == workspace_id
                )
            ):
                self.campaign_tickets.pop(digest, None)
        while len(self.campaign_tickets) >= 1_024:
            oldest_digest = next(iter(self.campaign_tickets))
            self.campaign_tickets.pop(oldest_digest, None)
        ticket = f"bgclt.{secrets.token_urlsafe(32)}"
        binding = CampaignLiveTicketBinding(
            workspace_id=workspace_id,
            after_cursor=after_cursor,
            expires_at=expires_at,
            credential_id=principal.credential_id,
            authorization_revision=principal.authorization_revision,
            repository=repository,
        )
        self.campaign_tickets[self._ticket_hash(ticket)] = binding
        return ticket, binding

    def allow_campaign_ticket_mint(
        self,
        credential_id: str,
        workspace_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        checked_at = now or datetime.now(UTC)
        window = timedelta(seconds=30)
        for scope, (started_at, _count) in tuple(
            self.campaign_ticket_mint_windows.items()
        ):
            if started_at + window <= checked_at:
                self.campaign_ticket_mint_windows.pop(scope, None)
        scope = (credential_id, workspace_id)
        started_at, count = self.campaign_ticket_mint_windows.get(
            scope, (checked_at, 0)
        )
        if started_at + window <= checked_at:
            started_at, count = checked_at, 0
        if count >= 30:
            return False
        while len(self.campaign_ticket_mint_windows) >= 1_024 and scope not in self.campaign_ticket_mint_windows:
            oldest_scope = next(iter(self.campaign_ticket_mint_windows))
            self.campaign_ticket_mint_windows.pop(oldest_scope, None)
        self.campaign_ticket_mint_windows[scope] = (started_at, count + 1)
        return True

    def _binding_authorized(self, binding: CampaignLiveTicketBinding) -> bool:
        now = datetime.now(UTC)
        if binding.expires_at <= now:
            return False
        parent = binding.repository.get_actor_credential(binding.credential_id)
        if parent is None or parent.revoked_at is not None or parent.expires_at <= now:
            return False
        if parent.authorization_revision != binding.authorization_revision:
            return False
        return (
            "desktop-local" in parent.workspace_ids
            or binding.workspace_id in parent.workspace_ids
        )

    def consume_campaign_live_ticket(
        self, ticket: str | None
    ) -> CampaignLiveTicketBinding | None:
        if not isinstance(ticket, str) or len(ticket) > 256:
            return None
        binding = self.campaign_tickets.pop(self._ticket_hash(ticket), None)
        return binding if binding is not None and self._binding_authorized(binding) else None

    async def subscribe_campaign(self, websocket: WebSocket, ticket: str | None) -> bool:
        binding = self.consume_campaign_live_ticket(ticket)
        if binding is None:
            return False
        self.campaign_subscriptions.setdefault(websocket, {})[
            binding.workspace_id
        ] = CampaignLiveSubscription(
            binding=binding,
            cursor=binding.after_cursor,
        )
        await self.send_personal(
            websocket,
            WSMessage(
                type="campaign:subscribed",
                payload={
                    "workspace_id": binding.workspace_id,
                    "accepted_cursor": binding.after_cursor,
                },
            ),
        )
        self._ensure_campaign_poller()
        return True

    def unsubscribe_campaign(self, websocket: WebSocket, workspace_id: str) -> None:
        subscriptions = self.campaign_subscriptions.get(websocket)
        if subscriptions is not None:
            subscriptions.pop(workspace_id, None)
            if not subscriptions:
                self.campaign_subscriptions.pop(websocket, None)
        if not self.campaign_subscriptions and self.campaign_poll_task is not None:
            self.campaign_poll_task.cancel()
            self.campaign_poll_task = None

    def _ensure_campaign_poller(self) -> None:
        if self.campaign_poll_task is None or self.campaign_poll_task.done():
            self.campaign_poll_task = asyncio.create_task(self._campaign_poll_loop())

    async def _campaign_poll_loop(self) -> None:
        try:
            while self.campaign_subscriptions:
                try:
                    await self.poll_campaign_subscriptions_once()
                except Exception:
                    # A transient repository/projection failure must not advance
                    # subscriber cursors or permanently stop every workspace.
                    import logging

                    logging.getLogger(__name__).exception(
                        "Campaign hint poll failed; retrying without cursor advance"
                    )
                await asyncio.sleep(0.35)
        except asyncio.CancelledError:
            return
        finally:
            self.campaign_poll_task = None

    async def poll_campaign_subscriptions_once(self) -> None:
        grouped: dict[str, list[tuple[WebSocket, CampaignLiveSubscription]]] = {}
        for websocket, subscriptions in tuple(self.campaign_subscriptions.items()):
            for workspace_id, subscription in tuple(subscriptions.items()):
                if not self._binding_authorized(subscription.binding):
                    subscriptions.pop(workspace_id, None)
                    continue
                grouped.setdefault(workspace_id, []).append((websocket, subscription))
            if not subscriptions:
                self.campaign_subscriptions.pop(websocket, None)
        for workspace_id, subscribers in grouped.items():
            after_cursor = min(subscription.cursor for _, subscription in subscribers)
            repository = subscribers[0][1].binding.repository
            sources = await asyncio.to_thread(
                repository.list_workspace_campaign_hint_sources,
                workspace_id,
                after_cursor=after_cursor,
                limit=200,
            )
            for websocket, subscription in subscribers:
                for source in sources:
                    if source.cursor <= subscription.cursor:
                        continue
                    hint = build_campaign_hint(source)
                    try:
                        await websocket.send_text(
                            WSMessage(
                                type=MessageType.CAMPAIGN_HINT,
                                payload=hint.model_dump(mode="json"),
                            ).to_json()
                        )
                    except Exception:
                        self.disconnect(websocket)
                        break
                    subscription.cursor = source.cursor


# Global connection manager
manager = ConnectionManager()


# Per-step metrics the trainer emits beyond the always-present core fields.
# Forwarded to the WS payload and persisted to metrics.jsonl when present, so the
# dashboard and run-analysis health gates see eval-loss, throughput, resource
# use, and session-distillation series instead of them being dropped in transit.
_OPTIONAL_PROGRESS_KEYS = (
    "eval_loss",
    "samples_processed",
    "tokens_per_second",
    "gpu_memory_gb",
    "gpu_utilization",
    "compute_target",
    "session_distillation_loss",
    "session_distillation_kl",
    "session_distillation_ce",
    "session_distillation_masked_tokens",
)


class TrainingProgressCallback:
    """
    Callback for training progress that broadcasts via WebSocket.

    Usage:
        callback = TrainingProgressCallback(run_id)
        trainer.train_sft(dataset, callback=callback.on_progress)
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str | None = None,
        static_payload: dict[str, Any] | None = None,
        metric_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.run_id = run_id
        # When set, each progress point is persisted to <output_dir>/metrics.jsonl
        # so loss curves survive the session (backs the run-comparison API).
        self.output_dir = output_dir
        self._loop = None
        # WebSocket connections live on the loop this callback was created on
        # (the server's main loop). Sync callbacks fire from trainer worker
        # threads — broadcasts must be scheduled back onto this loop, not
        # whatever loop is running in the worker.
        self._main_loop = self._get_loop()
        self._last_recorded_step = -1
        self.static_payload = static_payload or {}
        self.metric_sink = metric_sink

    def _get_loop(self):
        """Get the event loop, creating one if needed for sync context."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    async def on_progress(self, metrics: dict[str, Any]) -> None:
        """Called on each training step/epoch."""
        payload = {
            "run_id": self.run_id,
            "epoch": metrics.get("epoch"),
            "total_epochs": metrics.get("total_epochs"),
            "step": metrics.get("step", 0),
            "total_steps": metrics.get("total_steps", 0),
            "loss": metrics.get("loss"),
            "learning_rate": metrics.get("learning_rate"),
            "grad_norm": metrics.get("grad_norm", 0),
            "eta": metrics.get("eta"),
        }
        # Forward the richer per-step metrics the trainer emits (eval-loss,
        # throughput, resource use, session-distillation) instead of dropping
        # them here — only the ones actually present, to avoid null noise.
        for key in _OPTIONAL_PROGRESS_KEYS:
            if metrics.get(key) is not None:
                payload[key] = metrics[key]
            elif self.static_payload.get(key) is not None:
                payload[key] = self.static_payload[key]

        # Persist metrics before broadcasting so a broadcast problem can
        # never cost the on-disk loss curve.
        self._persist_point(metrics)

        message = WSMessage(type=MessageType.TRAINING_PROGRESS, payload=payload)
        await manager.broadcast(message)

    def _persist_point(self, metrics: dict[str, Any]) -> None:
        if self.metric_sink is not None:
            try:
                self.metric_sink(metrics)
            except Exception:
                # Ledger availability must never interrupt the training process.
                pass
        step = metrics.get("step")
        loss = metrics.get("loss")
        if (
            self.output_dir
            and isinstance(step, int)
            and loss is not None
            and step != self._last_recorded_step
        ):
            self._last_recorded_step = step
            from bashgym.gym.run_metrics import record_run_metric

            point = {
                "step": step,
                "loss": loss,
                "epoch": metrics.get("epoch"),
                "learning_rate": metrics.get("learning_rate"),
                "grad_norm": metrics.get("grad_norm"),
                "eval_loss": metrics.get("eval_loss"),
            }
            for key in _OPTIONAL_PROGRESS_KEYS:
                if metrics.get(key) is not None:
                    point[key] = metrics[key]
                elif self.static_payload.get(key) is not None:
                    point[key] = self.static_payload[key]
            record_run_metric(self.output_dir, point)

    def _schedule(self, coro) -> bool:
        """Run a callback coroutine on the server's main loop from any thread.

        Returns False if there is no usable loop (caller falls back to
        asyncio.run in the current thread).
        """
        loop = self._main_loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
            return True
        loop = self._get_loop()
        if loop is not None and loop.is_running():
            asyncio.ensure_future(coro, loop=loop)
            return True
        return False

    def on_progress_sync(self, metrics: dict[str, Any]) -> None:
        """Synchronous version for non-async training loops."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            if not self._schedule(self.on_progress(metrics)):
                asyncio.run(self.on_progress(metrics))
        except Exception as e:
            # Never let a broadcast problem interrupt training; keep the
            # on-disk metric point at minimum.
            logger.warning(f"Failed to send progress via WebSocket: {e}")
            try:
                self._persist_point(metrics)
            except Exception:
                pass

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
            if not self._schedule(self.on_log(log_line)):
                asyncio.run(self.on_log(log_line))
        except Exception:
            pass  # Silently fail for log lines


async def broadcast_training_complete(run_id: str, metrics: dict[str, Any]) -> None:
    """Broadcast training completion."""
    message = WSMessage(
        type=MessageType.TRAINING_COMPLETE, payload={"run_id": run_id, "metrics": metrics}
    )
    await manager.broadcast(message)


async def broadcast_training_queued(run_id: str, payload: dict[str, Any]) -> None:
    """Broadcast that a training run has been accepted and queued."""
    message = WSMessage(
        type=MessageType.TRAINING_QUEUED,
        payload={"run_id": run_id, **payload},
    )
    await manager.broadcast(message)


async def broadcast_training_failed(run_id: str, error: str) -> None:
    """Broadcast training failure."""
    message = WSMessage(
        type=MessageType.TRAINING_FAILED, payload={"run_id": run_id, "error": error}
    )
    await manager.broadcast(message)


async def broadcast_training_log(run_id: str, log_line: str, level: str = "info") -> None:
    """Broadcast a training log line."""
    message = WSMessage(
        type=MessageType.TRAINING_LOG,
        payload={"run_id": run_id, "message": log_line, "level": level},
    )
    await manager.broadcast(message)


async def broadcast_workspace_canvas_intent(payload: dict[str, Any]) -> None:
    """Broadcast a semantic canvas intent emitted by an agent or workspace tool."""
    await manager.broadcast(WSMessage(type=MessageType.WORKSPACE_CANVAS_INTENT, payload=payload))


async def broadcast_workspace_context_updated(payload: dict[str, Any]) -> None:
    """Broadcast a lightweight notification that workspace context changed."""
    await manager.broadcast(WSMessage(type=MessageType.WORKSPACE_CONTEXT_UPDATED, payload=payload))


async def broadcast_task_status(task_id: str, status: str, result: Any = None) -> None:
    """Broadcast task status update."""
    message = WSMessage(
        type=MessageType.TASK_STATUS,
        payload={"task_id": task_id, "status": status, "result": result},
    )
    await manager.broadcast(message)


async def broadcast_trace_event(
    event_type: MessageType, trace_id: str, trace_data: dict[str, Any] = None
) -> None:
    """Broadcast trace-related events."""
    message = WSMessage(type=event_type, payload={"trace_id": trace_id, "data": trace_data or {}})
    await manager.broadcast(message)


async def broadcast_router_stats(stats: dict[str, Any]) -> None:
    """Broadcast router statistics."""
    message = WSMessage(type=MessageType.ROUTER_STATS, payload=stats)
    await manager.broadcast(message)


async def broadcast_verification_result(
    task_id: str, passed: bool, details: dict[str, Any] = None
) -> None:
    """Broadcast verification results."""
    message = WSMessage(
        type=MessageType.VERIFICATION_RESULT,
        payload={"task_id": task_id, "passed": passed, "details": details or {}},
    )
    await manager.broadcast(message)


async def broadcast_guardrail_event(
    event_type: MessageType,
    check_type: str,
    location: str,
    action: str,
    confidence: float,
    content_preview: str = None,
    details: dict[str, Any] = None,
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
            "details": details or {},
        },
    )
    await manager.broadcast(message)


async def broadcast_guardrail_blocked(
    check_type: str,
    location: str,
    confidence: float,
    content_preview: str = None,
    details: dict[str, Any] = None,
) -> None:
    """Broadcast that content was blocked by guardrails."""
    await broadcast_guardrail_event(
        MessageType.GUARDRAIL_BLOCKED,
        check_type=check_type,
        location=location,
        action="block",
        confidence=confidence,
        content_preview=content_preview,
        details=details,
    )


async def broadcast_guardrail_warn(
    check_type: str,
    location: str,
    confidence: float,
    content_preview: str = None,
    details: dict[str, Any] = None,
) -> None:
    """Broadcast that content triggered a guardrail warning."""
    await broadcast_guardrail_event(
        MessageType.GUARDRAIL_WARN,
        check_type=check_type,
        location=location,
        action="warn",
        confidence=confidence,
        content_preview=content_preview,
        details=details,
    )


async def broadcast_pii_redacted(
    location: str, redaction_count: int, pii_types: list[str] = None, details: dict[str, Any] = None
) -> None:
    """Broadcast that PII was redacted from content."""
    message = WSMessage(
        type=MessageType.GUARDRAIL_PII_REDACTED,
        payload={
            "location": location,
            "redaction_count": redaction_count,
            "pii_types": pii_types or [],
            "details": details or {},
        },
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
        },
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
        },
    )
    await manager.broadcast(message)


async def broadcast_hf_job_completed(
    job_id: str, model_repo: str = None, metrics: dict[str, Any] = None
) -> None:
    """Broadcast when a HuggingFace job completes successfully."""
    message = WSMessage(
        type=MessageType.HF_JOB_COMPLETED,
        payload={
            "job_id": job_id,
            "model_repo": model_repo,
            "metrics": metrics or {},
        },
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
        },
    )
    await manager.broadcast(message)


async def broadcast_hf_job_metrics(job_id: str, metrics: dict[str, Any]) -> None:
    """Broadcast real-time training metrics from a HuggingFace cloud job."""
    message = WSMessage(
        type=MessageType.HF_JOB_METRICS,
        payload={
            "job_id": job_id,
            "metrics": metrics,
        },
    )
    await manager.broadcast(message)


async def broadcast_hf_space_ready(space_name: str, url: str) -> None:
    """Broadcast when a HuggingFace Space is ready."""
    message = WSMessage(
        type=MessageType.HF_SPACE_READY,
        payload={
            "space_name": space_name,
            "url": url,
        },
    )
    await manager.broadcast(message)


async def broadcast_hf_space_error(space_name: str, error: str) -> None:
    """Broadcast when a HuggingFace Space encounters an error."""
    message = WSMessage(
        type=MessageType.HF_SPACE_ERROR,
        payload={
            "space_name": space_name,
            "error": error,
        },
    )
    await manager.broadcast(message)


# =============================================================================
# Bashbros Integration Event Broadcasts
# =============================================================================


async def broadcast_integration_trace_received(
    filename: str, task: str, source: str = "bashbros", steps: int = 0, verified: bool = False
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
        },
    )
    await manager.broadcast(message)


async def broadcast_integration_trace_processed(
    filename: str, success: bool, trace_id: str = None, error: str = None
) -> None:
    """Broadcast when a trace has been processed."""
    message = WSMessage(
        type=MessageType.INTEGRATION_TRACE_PROCESSED,
        payload={
            "filename": filename,
            "success": success,
            "trace_id": trace_id,
            "error": error,
        },
    )
    await manager.broadcast(message)


async def broadcast_integration_model_exported(
    version: str,
    gguf_path: str,
    ollama_registered: bool = False,
    traces_used: int = 0,
    quality_avg: float = 0.0,
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
        },
    )
    await manager.broadcast(message)


async def broadcast_integration_model_rollback(version: str, previous_version: str = None) -> None:
    """Broadcast when model has been rolled back to a previous version."""
    message = WSMessage(
        type=MessageType.INTEGRATION_MODEL_ROLLBACK,
        payload={
            "version": version,
            "previous_version": previous_version,
        },
    )
    await manager.broadcast(message)


async def broadcast_integration_linked() -> None:
    """Broadcast when bashbros integration is linked."""
    message = WSMessage(type=MessageType.INTEGRATION_LINKED, payload={"linked": True})
    await manager.broadcast(message)


async def broadcast_integration_unlinked() -> None:
    """Broadcast when bashbros integration is unlinked."""
    message = WSMessage(type=MessageType.INTEGRATION_UNLINKED, payload={"linked": False})
    await manager.broadcast(message)


async def broadcast_integration_training_triggered(
    gold_traces: int, threshold: int, run_id: str = None
) -> None:
    """Broadcast when auto-training is triggered by bashbros integration."""
    message = WSMessage(
        type=MessageType.INTEGRATION_TRAINING_TRIGGERED,
        payload={
            "gold_traces": gold_traces,
            "threshold": threshold,
            "run_id": run_id,
        },
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
        },
    )
    await manager.broadcast(message)


async def broadcast_orchestration_task_completed(
    job_id: str, task_id: str, cost_usd: float, duration_seconds: float, newly_unblocked: int = 0
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
        },
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
        },
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
        },
    )
    await manager.broadcast(message)


async def broadcast_orchestration_complete(
    job_id: str,
    completed: int,
    failed: int,
    total_cost: float,
    total_time: float,
    merge_successes: int = 0,
    merge_failures: int = 0,
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
        },
    )
    await manager.broadcast(message)


async def broadcast_pipeline_event(event_type: MessageType, payload: dict[str, Any]) -> None:
    """Broadcast a pipeline event to all connected clients."""
    message = WSMessage(
        type=event_type,
        payload=payload,
    )
    await manager.broadcast(message)


async def broadcast_import_progress(processed: int, total: int, current_item: str = "") -> None:
    """Broadcast per-item trace import progress (pipeline:import)."""
    message = WSMessage(
        type=MessageType.PIPELINE_IMPORT,
        payload={
            "processed": processed,
            "total": total,
            "current_item": current_item,
            "phase": "importing",
        },
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
    # In web mode, verify session cookie before accepting connection
    if os.environ.get("BASHGYM_MODE", "").lower() == "web":
        from bashgym.api.auth_routes import COOKIE_NAME
        from bashgym.api.database import get_session_user

        token = websocket.cookies.get(COOKIE_NAME)
        if not token or not get_session_user(token):
            await websocket.close(code=4401, reason="Authentication required")
            return

    await manager.connect(websocket)

    async def invalid_frame() -> None:
        await manager.send_personal(
            websocket,
            WSMessage(type=MessageType.ERROR, payload={"code": "invalid_frame"}),
        )

    try:
        while True:
            # Receive and parse messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if not isinstance(message, dict):
                    await invalid_frame()
                    continue
                msg_type = message.get("type", "")
                payload = message.get("payload", {})
                if not isinstance(msg_type, str) or not isinstance(payload, dict):
                    await invalid_frame()
                    continue

                if msg_type == "campaign:subscribe":
                    if set(payload) != {"ticket"}:
                        await invalid_frame()
                        continue
                    if not await manager.subscribe_campaign(websocket, payload.get("ticket")):
                        await manager.send_personal(
                            websocket,
                            WSMessage(
                                type="campaign:subscription-error",
                                payload={"code": "campaign_subscription_denied"},
                            ),
                        )
                    continue

                if msg_type == "campaign:unsubscribe":
                    if set(payload) != {"workspace_id"}:
                        await invalid_frame()
                        continue
                    workspace_id = payload.get("workspace_id")
                    if isinstance(workspace_id, str):
                        manager.unsubscribe_campaign(websocket, workspace_id)
                    continue

                # Handle subscription requests
                if msg_type == "subscribe":
                    topic = payload.get("topic")
                    if topic:
                        manager.subscribe(websocket, topic)
                        await manager.send_personal(
                            websocket, WSMessage(type="subscribed", payload={"topic": topic})
                        )

                elif msg_type == "unsubscribe":
                    topic = payload.get("topic")
                    if topic:
                        manager.unsubscribe(websocket, topic)
                        await manager.send_personal(
                            websocket, WSMessage(type="unsubscribed", payload={"topic": topic})
                        )

                # Handle ping/pong for connection health
                elif msg_type == "ping":
                    await manager.send_personal(websocket, WSMessage(type="pong", payload={}))

            except json.JSONDecodeError:
                await invalid_frame()

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
