"""Foreground resident campaign worker with reconcile-before-claim semantics."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import (
    ActionAttempt,
    AttemptStatus,
    CampaignStatus,
    utc_now,
)
from bashgym.campaigns.executors import (
    DevelopmentEvaluationConfig,
    DevelopmentEvaluationExecutor,
    FakeExecutionRequest,
    FakeExecutor,
    RemoteOutputSealer,
)
from bashgym.campaigns.persistence import CampaignPersistenceError, LeaseBusyError, LeaseRecord
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
    RemoteCapacityPolicy,
    RemoteLaunchRequest,
    RemoteRunState,
    RemoteTrainingAdapter,
    remote_executor_config,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository


class SimulatedWorkerCrashError(RuntimeError):
    """Test-only fault boundary after external side effect and before DB commit."""


def scheduler_lease_key(data_directory: Path) -> str:
    """Return the stable leader key for one canonical BashGym data directory."""

    canonical = str(data_directory.resolve()).casefold()
    directory_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"scheduler:{directory_digest}"


class CampaignWorker:
    """Own the global scheduler lease and execute durable actions one at a time."""

    def __init__(
        self,
        repository: CampaignRuntimeRepository,
        artifact_root: Path,
        sealer: ArtifactSealer,
        *,
        data_directory: Path,
        worker_id: str | None = None,
        leader_ttl: timedelta = timedelta(seconds=15),
        action_ttl: timedelta = timedelta(seconds=15),
        remote_adapters: dict[str, RemoteTrainingAdapter] | None = None,
        remote_executor_profiles: Mapping[
            tuple[str, str], ApprovedRemoteExecutorProfile
        ] | None = None,
    ):
        self.repository = repository
        self.artifact_root = artifact_root.resolve()
        self.sealer = sealer
        self.worker_id = worker_id or f"worker-{uuid4().hex}"
        self.leader_key = scheduler_lease_key(data_directory)
        self.leader_ttl = leader_ttl
        self.action_ttl = action_ttl
        self._leader: LeaseRecord | None = None
        self._stop_requested = False
        self.executor = FakeExecutor(self.artifact_root, sealer)
        self.remote_output_sealer = RemoteOutputSealer(self.artifact_root, sealer)
        self.development_evaluation_executor = DevelopmentEvaluationExecutor(
            self.artifact_root, sealer
        )
        self.remote_adapters = dict(remote_adapters or {})
        self.remote_executor_profiles = dict(remote_executor_profiles or {})

    @property
    def leader(self) -> LeaseRecord | None:
        return self._leader

    def request_stop(self) -> None:
        """Stop taking new claims after reconciliation of already-owned work."""

        self._stop_requested = True

    def _ensure_leader(self, now: datetime) -> LeaseRecord:
        if self._leader is None:
            self._leader = self.repository.acquire_lease(
                self.leader_key,
                self.worker_id,
                ttl=self.leader_ttl,
                now=now,
            )
        else:
            self._leader = self.repository.heartbeat_lease(
                self._leader.lease_key,
                self._leader.owner_id,
                self._leader.generation,
                ttl=self.leader_ttl,
                now=now,
            )
        return self._leader

    def sealed_path(self, attempt: ActionAttempt) -> Path:
        return (
            self.artifact_root
            / attempt.workspace_id
            / attempt.campaign_id
            / attempt.study_id
            / attempt.action_id
            / attempt.attempt_id
        )

    def _verify(self, attempt: ActionAttempt, sealed_path: Path):
        return self.sealer.verify(
            sealed_path,
            expected_workspace_id=attempt.workspace_id,
            expected_campaign_id=attempt.campaign_id,
            expected_study_id=attempt.study_id,
            expected_action_id=attempt.action_id,
            expected_attempt_id=attempt.attempt_id,
            expected_manifest_revision=attempt.manifest_revision,
            expected_candidate_digest=attempt.candidate_digest,
            expected_input_digest=attempt.input_digest,
            expected_claim_generation=attempt.claim_generation,
        )

    def _ingest_sealed_metrics(self, attempt: ActionAttempt, sealed_path: Path, *, now: datetime) -> None:
        metrics_path = sealed_path / "training_metrics.jsonl"
        if not metrics_path.is_file() or metrics_path.is_symlink():
            return
        payload = metrics_path.read_text(encoding="utf-8")
        self.repository.append_remote_metrics(
            attempt,
            tuple(payload.splitlines()),
            source="training_metrics.jsonl",
            cursor_end=len(payload.encode("utf-8")),
            now=now,
        )

    def reconcile_once(self, *, now: datetime) -> str | None:
        """Register sealed results before marking expired uncertain work."""

        for attempt in self.repository.list_unfinished_attempts():
            if attempt.executor.get("kind") == "ssh_remote":
                if (
                    attempt.lease_owner != self.worker_id
                    or attempt.lease_expires_at is None
                    or attempt.lease_expires_at <= now
                ):
                    attempt = self.repository.adopt_remote_attempt(
                        attempt,
                        self._leader,
                        ttl=self.action_ttl,
                        now=now,
                    )
                return asyncio.run(self._remote_tick(attempt, now=now))
            sealed_path = self.sealed_path(attempt)
            if sealed_path.is_dir():
                manifest = self._verify(attempt, sealed_path)
                self._ingest_sealed_metrics(attempt, sealed_path, now=now)
                self.repository.complete_from_seal(
                    manifest,
                    sealed_path,
                    worker_id=self.worker_id,
                    reconcile=True,
                    now=now,
                )
                return "reconciled"
            if (
                attempt.status == AttemptStatus.RUNNING
                and attempt.lease_expires_at is not None
                and attempt.lease_expires_at <= now
            ):
                self.repository.mark_expired_unknown(attempt, now=now)
                return "unknown"
        return None

    def controller_once(self, leader: LeaseRecord, *, now: datetime) -> str | None:
        """Select one proposal and schedule its next safe stage under the leader fence."""

        campaign = self.repository.next_controller_campaign()
        if campaign is None:
            return None
        if campaign.active_study_id is None:
            selected = self.repository.select_next_proposal_as_controller(
                campaign.workspace_id,
                campaign.campaign_id,
                expected_version=campaign.version,
                controller_id="campaign-controller",
                correlation_id=f"worker-{self.worker_id}",
                idempotency_key=f"controller-select-v{campaign.version}",
            )
            if selected is None:
                return None
            campaign = selected.campaign
        if campaign.active_study_id is None:
            return "proposal_selected"
        if self.repository.skip_not_applicable_stages_under_leader(
            campaign.workspace_id,
            campaign.campaign_id,
            campaign.active_study_id,
            leader,
            expected_campaign_version=campaign.version,
            now=now,
        ):
            return "stage_skipped"
        try:
            spec = self.repository.next_action_spec(
                campaign.workspace_id,
                campaign.campaign_id,
                campaign.active_study_id,
                executor_profiles=self.remote_executor_profiles,
            )
        except CampaignPersistenceError as exc:
            self.repository.record_controller_blocker_under_leader(
                campaign.workspace_id,
                campaign.campaign_id,
                campaign.active_study_id,
                leader,
                code=str(exc),
                now=now,
            )
            return "action_blocked"
        self.repository.schedule_action_under_leader(
            spec,
            leader,
            expected_campaign_version=campaign.version,
            now=now,
        )
        return "action_scheduled"

    def _remote_request(self, attempt: ActionAttempt) -> RemoteLaunchRequest:
        executor = attempt.executor
        required = {
            "compute_profile_id",
            "script_path",
            "input_files",
            "script_args",
            "recipe_digest",
            "profile_id",
            "profile_revision",
            "profile_digest",
            "target_contract_key",
            "target_model_digest",
        }
        if not required.issubset(executor):
            raise RuntimeError("campaign_remote_executor_contract_incomplete")
        profile_key = (executor["compute_profile_id"], executor["target_contract_key"])
        try:
            profile = self.remote_executor_profiles[profile_key]
        except KeyError as exc:
            raise RuntimeError("campaign_remote_executor_profile_unavailable") from exc
        try:
            expected = remote_executor_config(
                profile,
                attempt.stage,
                recipe_digest=executor["recipe_digest"],
            )
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError("campaign_remote_executor_material_invalid") from exc
        if executor != {"kind": "ssh_remote", **expected}:
            raise RuntimeError("campaign_remote_executor_profile_mismatch")
        return RemoteLaunchRequest(
            compute_profile_id=executor["compute_profile_id"],
            run_id=attempt.attempt_id,
            script_path=Path(executor["script_path"]),
            input_files=tuple(Path(value) for value in executor["input_files"]),
            script_args=tuple(executor["script_args"]),
            python_executable=executor["python_executable"],
            recipe_digest=executor["recipe_digest"],
            output_paths=tuple(
                executor.get(
                    "output_paths",
                    ("final", "training_manifest.json", "training_metrics.jsonl"),
                )
            ),
        )

    async def _remote_tick(self, attempt: ActionAttempt, *, now: datetime) -> str:
        request = self._remote_request(attempt)
        try:
            adapter = self.remote_adapters[request.compute_profile_id]
        except KeyError as exc:
            raise RuntimeError("campaign_remote_compute_profile_unavailable") from exc
        record = self.repository.get_remote_run(attempt.workspace_id, attempt.attempt_id)
        campaign = self.repository.get_campaign(attempt.workspace_id, attempt.campaign_id)
        if record is None:
            identity = await adapter.discover(request)
            if identity is None:
                if campaign.status == CampaignStatus.CANCELLING:
                    sealed_path, manifest = (
                        self.remote_output_sealer.seal_unlaunched_cancelled(
                            attempt, compute_profile_id=request.compute_profile_id
                        )
                    )
                    verified = self._verify(attempt, sealed_path)
                    self.repository.settle_terminal_from_seal(
                        verified,
                        sealed_path,
                        worker_id=self.worker_id,
                        now=now,
                    )
                    return "remote_cancelled"
                capacity_config = attempt.executor.get("capacity_policy", {})
                capacity = await adapter.capacity_preflight(
                    RemoteCapacityPolicy.model_validate(capacity_config)
                )
                if not capacity.admitted:
                    self.repository.defer_unlaunched_remote_attempt(
                        attempt,
                        worker_id=self.worker_id,
                        reasons=capacity.blocking_reasons,
                        now=now,
                    )
                    return "remote_capacity_blocked"
                identity = await adapter.launch(request)
            record = self.repository.register_remote_identity(attempt, identity, now=now)

        force_stop_request = self.repository.pending_force_stop_request(
            attempt.workspace_id, attempt.action_id, record.identity
        )
        if force_stop_request is not None:
            executed = await adapter.force_stop(record.identity)
            self.repository.settle_force_stop_request(
                attempt.workspace_id, force_stop_request, executed=executed
            )
            if executed:
                return "remote_force_stopping"

        observation = await adapter.observe(record.identity)
        metric_cursor = record.metric_cursor
        log_cursor = record.log_cursor
        metric_lines: list[str] = []
        for source, cursor_name in (
            ("training_metrics.jsonl", "metric"),
            ("training.log", "log"),
        ):
            cursor = metric_cursor if cursor_name == "metric" else log_cursor
            try:
                chunk = await adapter.read_stream(record.identity, source, cursor)
            except RuntimeError:
                continue
            if cursor_name == "metric":
                metric_cursor = chunk.next_cursor
                metric_lines.extend(chunk.complete_lines)
            else:
                log_cursor = chunk.next_cursor
        self.repository.append_remote_metrics(
            attempt,
            tuple(metric_lines),
            source="training_metrics.jsonl",
            cursor_end=metric_cursor.byte_offset,
            now=now,
        )
        collection_ttl = timedelta(hours=1) if observation.state == RemoteRunState.COMPLETED else self.action_ttl
        record = self.repository.update_remote_run(
            record,
            observation,
            metric_cursor=metric_cursor,
            log_cursor=log_cursor,
            worker_id=self.worker_id,
            lease_ttl=collection_ttl,
            now=now,
        )
        if campaign.status == CampaignStatus.CANCELLING and observation.state in {
            RemoteRunState.RUNNING,
            RemoteRunState.PAUSED,
        }:
            await adapter.terminate(record.identity)
            return "remote_cancelling"
        if observation.state == RemoteRunState.RUNNING:
            return "remote_running"
        if observation.state == RemoteRunState.PAUSED:
            return "remote_paused"
        if observation.state == RemoteRunState.UNKNOWN:
            return "remote_unknown"
        if observation.state == RemoteRunState.FAILED:
            temporary = (
                self.artifact_root
                / ".tmp"
                / f"{attempt.action_id}.{attempt.attempt_id}.{uuid4().hex}"
            )
            temporary.mkdir(parents=True, exist_ok=False)
            await adapter.collect_terminal_evidence(
                record.identity,
                temporary,
                observation=observation,
            )
            outcome = (
                "cancelled"
                if campaign.status == CampaignStatus.CANCELLING
                else "failed"
            )
            sealed_path, _manifest = self.remote_output_sealer.seal_terminal(
                attempt,
                record.identity,
                observation,
                temporary,
                outcome=outcome,
            )
            verified = self._verify(attempt, sealed_path)
            self.repository.settle_terminal_from_seal(
                verified,
                sealed_path,
                worker_id=self.worker_id,
                now=now,
            )
            return "remote_cancelled" if outcome == "cancelled" else "remote_failed"

        temporary = (
            self.artifact_root
            / ".tmp"
            / f"{attempt.action_id}.{attempt.attempt_id}.{uuid4().hex}"
        )
        temporary.mkdir(parents=True, exist_ok=False)
        await adapter.collect_outputs(
            record.identity,
            request,
            temporary,
            observation=observation,
        )
        sealed_path, _manifest = self.remote_output_sealer.seal_completed(
            attempt, record.identity, observation, temporary
        )
        verified = self._verify(attempt, sealed_path)
        self.repository.complete_from_seal(
            verified,
            sealed_path,
            worker_id=self.worker_id,
            now=now,
        )
        return "completed"

    def _development_evaluation_tick(self, attempt: ActionAttempt, *, now: datetime) -> str:
        config = DevelopmentEvaluationConfig.model_validate(
            {key: value for key, value in attempt.executor.items() if key != "kind"}
        )
        champion = (
            self.repository.get_retrieval_evaluation(
                attempt.workspace_id, config.champion_evaluation_id
            )
            if config.champion_evaluation_id
            else None
        )
        execution = self.development_evaluation_executor.execute(
            attempt, config, champion=champion
        )
        self.repository.store_retrieval_evaluation(
            attempt.workspace_id,
            attempt.campaign_id,
            execution.evaluation,
            now=now,
        )
        if execution.comparison is not None:
            self.repository.store_development_comparison(
                attempt.workspace_id,
                attempt.campaign_id,
                execution.comparison,
                now=now,
            )
        verified = self._verify(attempt, execution.sealed_path)
        self.repository.complete_from_seal(
            verified,
            execution.sealed_path,
            worker_id=self.worker_id,
            now=now,
        )
        return "completed"

    def run_once(
        self,
        *,
        now: datetime | None = None,
        crash_after_seal: bool = False,
    ) -> str:
        """Reconcile, claim, execute outside SQLite, then commit the sealed result."""

        tick_at = now or utc_now()
        try:
            leader = self._ensure_leader(tick_at)
        except LeaseBusyError:
            return "not_leader"
        reconciled = self.reconcile_once(now=tick_at)
        if reconciled is not None:
            return reconciled
        if self._stop_requested:
            return "stopped"
        controller_result = self.controller_once(leader, now=tick_at)
        attempt = self.repository.claim_next_action(
            leader,
            ttl=self.action_ttl,
            now=tick_at,
        )
        if attempt is None:
            return controller_result or "idle"
        if attempt.executor.get("kind") == "ssh_remote":
            return asyncio.run(self._remote_tick(attempt, now=tick_at))
        if attempt.executor.get("kind") == "development_evaluation":
            return self._development_evaluation_tick(attempt, now=tick_at)
        request = FakeExecutionRequest(
            workspace_id=attempt.workspace_id,
            campaign_id=attempt.campaign_id,
            study_id=attempt.study_id,
            action_id=attempt.action_id,
            attempt_id=attempt.attempt_id,
            manifest_revision=attempt.manifest_revision,
            candidate_digest=attempt.candidate_digest,
            input_digest=attempt.input_digest,
            claim_generation=attempt.claim_generation,
            steps=int(attempt.executor.get("steps", 8)),
        )
        sealed_path, _manifest = self.executor.execute(request)
        if crash_after_seal:
            raise SimulatedWorkerCrashError(
                "simulated crash after seal and before completion commit"
            )
        verified = self._verify(attempt, sealed_path)
        self._ingest_sealed_metrics(attempt, sealed_path, now=tick_at)
        self.repository.complete_from_seal(
            verified,
            sealed_path,
            worker_id=self.worker_id,
            now=tick_at,
        )
        return "completed"

    def run_forever(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utc_now,
        heartbeat_seconds: float = 5.0,
        ready_poll_seconds: float = 2.0,
        idle_poll_seconds: float = 30.0,
    ) -> None:
        """Maintain the leader heartbeat while backing off full work checks when idle."""

        if heartbeat_seconds <= 0 or ready_poll_seconds <= 0 or idle_poll_seconds <= 0:
            raise ValueError("worker polling intervals must be positive")
        next_work_check: datetime | None = None
        try:
            while not self._stop_requested:
                now = clock()
                try:
                    self._ensure_leader(now)
                except LeaseBusyError:
                    sleep(heartbeat_seconds)
                    continue
                if next_work_check is None or now >= next_work_check:
                    result = self.run_once(now=now)
                    interval = (
                        ready_poll_seconds
                        if result
                        in {
                            "completed",
                            "reconciled",
                            "unknown",
                            "remote_running",
                            "remote_paused",
                            "remote_unknown",
                            "remote_failed",
                            "remote_cancelling",
                            "remote_cancelled",
                            "remote_capacity_blocked",
                            "remote_force_stopping",
                            "stage_skipped",
                        }
                        else idle_poll_seconds
                    )
                    next_work_check = now + timedelta(seconds=interval)
                remaining = max(0.0, (next_work_check - now).total_seconds())
                sleep(min(heartbeat_seconds, remaining or heartbeat_seconds))
        finally:
            if self._leader is not None:
                try:
                    self.repository.release_lease(
                        self._leader.lease_key,
                        self._leader.owner_id,
                        self._leader.generation,
                        now=clock(),
                    )
                finally:
                    self._leader = None


__all__ = ["CampaignWorker", "SimulatedWorkerCrashError", "scheduler_lease_key"]
