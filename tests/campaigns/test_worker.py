"""Resident worker fencing, completion, pause, and restart reconciliation tests."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.contracts import (
    AttemptStatus,
    AutonomyProfile,
    CampaignTrigger,
    CredentialKind,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyStatus,
    canonical_hash,
)
from bashgym.campaigns.evaluation import load_retrieval_evaluation_artifact
from bashgym.campaigns.executors import FakeExecutionRequest, fake_digest
from bashgym.campaigns.human_oversight import HumanOversightRepository
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    build_nemo_gym_campaign_evidence,
    write_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
    RemoteCapacitySnapshot,
    RemoteObservation,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamChunk,
    remote_executor_config,
)
from bashgym.campaigns.runtime import (
    ActionIdentityMismatchError,
    ActionSpec,
    CampaignRuntimeRepository,
)
from bashgym.campaigns.worker import (
    CampaignWorker,
    SimulatedWorkerCrashError,
    _controller_selection_idempotency_key,
)
from bashgym.environments.nemo_gym import export_star_count_nemo_gym_bundle
from bashgym.environments.star_count import (
    generate_star_count_dataset,
    star_count_environment_spec,
)
from tests.campaigns.test_persistence import campaign, create, transition

START = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def test_controller_selection_idempotency_is_campaign_scoped():
    first = _controller_selection_idempotency_key("workspace-a", "campaign-1", 5)
    replay = _controller_selection_idempotency_key("workspace-a", "campaign-1", 5)
    second = _controller_selection_idempotency_key("workspace-a", "campaign-2", 5)

    assert first == replay
    assert first != second
    assert len(first) <= 160


def active_repository(path) -> CampaignRuntimeRepository:
    repository = CampaignRuntimeRepository(path)
    repository.initialize()
    create(repository)
    transition(repository, CampaignTrigger.VALIDATE, 1, key="validate-worker")
    transition(repository, CampaignTrigger.VALIDATION_PASSED, 2, key="ready-worker")
    transition(repository, CampaignTrigger.START, 3, key="start-worker")
    return repository


def seed_validated_study(
    repository: CampaignRuntimeRepository,
    study_id: str = "study-1",
    *,
    sequence: int = 1,
    stage: StageKind = StageKind.FULL_TRAINING,
) -> StagePlan:
    """Insert already-validated controller fixtures; proposal planning is a later slice."""

    proposal_id = f"proposal-{study_id}"
    plan = StagePlan(
        items=(
            StagePlanItem(
                stage=stage,
                disposition=StageDisposition.REQUIRED,
                reason="Prove one durable fake execution lifecycle.",
                input_contract={"fixture": study_id},
                output_contract={"schema": "training_metrics_jsonl.v1"},
            ),
        )
    )
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_proposals(
                workspace_id, campaign_id, proposal_id, status, priority,
                estimated_cost, creation_sequence, proposal_json, created_at
            ) VALUES (?, ?, ?, 'accepted', 50, 0.1, ?, '{}', ?)
            """,
            ("workspace-a", "campaign-1", proposal_id, sequence, START.isoformat()),
        )
        connection.execute(
            """
            INSERT INTO campaign_studies(
                workspace_id, campaign_id, study_id, proposal_id, status,
                current_stage_index, stage_plan_json, candidate_digest,
                version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, 1, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                study_id,
                proposal_id,
                StudyStatus.VALIDATED.value,
                plan.model_dump_json(),
                fake_digest(f"candidate:{study_id}"),
                START.isoformat(),
                START.isoformat(),
            ),
        )
    return plan


def make_worker(repository, tmp_path, worker_id):
    return CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id=worker_id,
    )


def schedule(repository, worker, plan, *, study_id="study-1", version=4):
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"
    return repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id=study_id,
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest(f"candidate:{study_id}"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            fake_steps=6,
        ),
        worker.leader,
        expected_campaign_version=version,
        now=START,
    )


class FakeRemoteAdapter:
    def __init__(self, *, admitted=True, states=(RemoteRunState.RUNNING,)):
        self.admitted = admitted
        self.states = list(states)
        self.identity = None
        self.launch_count = 0
        self.discover_count = 0
        self.collect_count = 0
        self.terminate_count = 0
        self.force_stop_count = 0
        self.last_request = None

    async def capacity_preflight(self, policy):
        reasons = () if self.admitted else ("external_gpu_process_limit_exceeded",)
        return RemoteCapacitySnapshot(
            compute_profile_id="ssh-gpu-lab",
            available_memory_gib=80 if self.admitted else 42,
            available_disk_gib=160,
            external_gpu_processes=() if self.admitted else ("111, llama-server",),
            admitted=self.admitted,
            blocking_reasons=reasons,
            observed_at=START,
        )

    async def discover(self, request):
        self.discover_count += 1
        return self.identity

    async def launch(self, request):
        self.launch_count += 1
        self.last_request = request
        self.identity = RemoteRunIdentity(
            compute_profile_id=request.compute_profile_id,
            run_id=request.run_id,
            remote_run_directory=f"/home/trainer/bashgym-training/{request.run_id}",
            remote_pid=4242,
            process_group_id=4242,
            process_start_ticks=9001,
            boot_id="boot-1",
            command_hash="a" * 64,
            launch_manifest_sha256="b" * 64,
            launched_at=START,
        )
        return self.identity

    async def observe(self, identity):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        exit_code = 0 if state == RemoteRunState.COMPLETED else None
        if state == RemoteRunState.FAILED:
            exit_code = 7
        return RemoteObservation(
            identity=identity,
            state=state,
            observed_at=START + timedelta(seconds=2),
            exit_code=exit_code,
            safe_reason=(
                "remote_exit_code_recorded"
                if state in {RemoteRunState.COMPLETED, RemoteRunState.FAILED}
                else "remote_process_alive"
            ),
        )

    async def read_stream(self, identity, source, cursor):
        return RemoteStreamChunk(
            source=source,
            start_offset=cursor.byte_offset,
            end_offset=cursor.byte_offset,
            complete_lines=(),
            next_cursor=cursor,
        )

    async def force_stop(self, identity):
        self.force_stop_count += 1
        return identity == self.identity

    async def collect_outputs(self, identity, request, local_directory, *, observation):
        self.collect_count += 1
        final = local_directory / "final"
        final.mkdir(parents=True)
        (final / "config.json").write_text("{}", encoding="utf-8")
        (local_directory / "training_manifest.json").write_text(
            json.dumps({"run_id": identity.run_id}), encoding="utf-8"
        )
        (local_directory / "training_metrics.jsonl").write_text(
            '{"step":1,"loss":0.5}\n', encoding="utf-8"
        )
        (local_directory / "training.log").write_text("complete\n", encoding="utf-8")
        (local_directory / "exit_code").write_text("0\n", encoding="utf-8")
        (local_directory / "launch_manifest.json").write_text("{}", encoding="utf-8")
        return tuple(path for path in local_directory.rglob("*") if path.is_file())

    async def collect_terminal_evidence(self, identity, local_directory, *, observation):
        self.collect_count += 1
        (local_directory / "training.log").write_text("failed\n", encoding="utf-8")
        (local_directory / "exit_code").write_text("7\n", encoding="utf-8")
        (local_directory / "launch_manifest.json").write_text("{}", encoding="utf-8")
        (local_directory / "training_metrics.jsonl").write_text(
            '{"step":1,"loss":3.0}\n', encoding="utf-8"
        )
        return tuple(path for path in local_directory.rglob("*") if path.is_file())

    async def terminate(self, identity):
        self.terminate_count += 1
        return True


def approved_remote_profile(
    tmp_path,
    *,
    stage=StageKind.FULL_TRAINING,
    code_lineage_binding: ApprovedCodeLineageExecutionBinding | None = None,
):
    script = tmp_path / "train.py"
    data = tmp_path / "train.jsonl"
    key = tmp_path / "campaign-key"
    script.write_text("print('training')\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    pinned = PinnedRemoteStageProfile(
        stage=stage,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(data,),
        input_sha256={data.name: hashlib.sha256(data.read_bytes()).hexdigest()},
        script_args=("--grouped-jsonl", data.name),
        output_paths=("final", "training_manifest.json", "training_metrics.jsonl"),
        budget_unit="gpu_hours",
        budget_reservation=0.25,
        python_executable="/approved/venv/bin/python",
        code_lineage_binding=code_lineage_binding,
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="memexai-approved-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        remote_work_dir="~/bashgym-training",
        stages=(pinned,),
    )


def schedule_remote(repository, worker, plan, tmp_path):
    profile = approved_remote_profile(tmp_path)
    worker.remote_executor_profiles[(profile.compute_profile_id, profile.target_contract_key)] = (
        profile
    )
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"
    return repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-1"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="ssh_remote",
            executor_config=remote_executor_config(
                profile, StageKind.FULL_TRAINING, recipe_digest="e" * 64
            ),
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )


def activate_controller_live_study(repository, plan) -> None:
    proposal_payload = {
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": {
            "schema_version": "recipe.v1",
            "runtime": {"executor_kind": "registered_training"},
        },
        "evaluation_recipe": {"schema_version": "recipe.v1"},
    }
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET stage_plan_json = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (plan.model_dump_json(), "workspace-a", "study-1"),
        )
        connection.execute(
            """
            UPDATE campaign_proposals SET proposal_json = ?
            WHERE workspace_id = ? AND proposal_id = ?
            """,
            (json.dumps(proposal_payload), "workspace-a", "proposal-study-1"),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("study-1", "workspace-a", "campaign-1"),
        )


def test_only_leader_claims_and_completion_is_atomic(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    first = make_worker(repository, tmp_path, "worker-a")
    scheduled = schedule(repository, first, plan)
    competing = make_worker(repository, tmp_path, "worker-b")

    assert competing.run_once(now=START + timedelta(seconds=1)) == "not_leader"
    assert first.run_once(now=START + timedelta(seconds=1)) == "completed"

    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert first.executor.execution_count == 1
    assert competing.executor.execution_count == 0
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": 0.25,
        "limit_delta": 0.0,
    }
    events = repository.list_events("workspace-a", "campaign-1")
    assert sum(event.event_type == "campaign:action-completed" for _, event in events) == 1


def test_controller_skips_not_applicable_stage_then_executes_next_required_stage(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    seed_validated_study(repository)
    plan = StagePlan(
        items=(
            StagePlanItem(
                stage=StageKind.CONTRACT_EVALUATION,
                disposition=StageDisposition.NOT_APPLICABLE,
                reason="The approved inputs already satisfy this contract.",
            ),
            StagePlanItem(
                stage=StageKind.FULL_TRAINING,
                disposition=StageDisposition.REQUIRED,
                reason="Run the bounded fake training proof.",
                input_contract={"fixture": "study-1"},
            ),
        )
    )
    proposal_payload = {
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": {
            "schema_version": "recipe.v1",
            "runtime": {
                "executor_kind": "fake",
                "budget_unit": "gpu_hours",
                "budget_reservation": 0.01,
                "fake_steps": 4,
            },
        },
        "evaluation_recipe": {"schema_version": "recipe.v1"},
    }
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET stage_plan_json = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (plan.model_dump_json(), "workspace-a", "study-1"),
        )
        connection.execute(
            """
            UPDATE campaign_proposals SET proposal_json = ?
            WHERE workspace_id = ? AND proposal_id = ?
            """,
            (json.dumps(proposal_payload), "workspace-a", "proposal-study-1"),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_study_id = ?
            WHERE workspace_id = ? AND campaign_id = ?
            """,
            ("study-1", "workspace-a", "campaign-1"),
        )
    worker = make_worker(repository, tmp_path, "worker-a")

    assert worker.run_once(now=START) == "stage_skipped"
    study = repository.get_study("workspace-a", "campaign-1", "study-1")
    assert study.current_stage_index == 1
    assert repository.get_campaign("workspace-a", "campaign-1").version == 5
    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"
    events = repository.list_events("workspace-a", "campaign-1")
    skipped = [event for _, event in events if event.event_type == "campaign:stages-skipped"]
    assert len(skipped) == 1
    assert skipped[0].payload["skipped"] == [{"stage_index": 0, "stage": "contract_evaluation"}]


def test_restart_registers_sealed_result_without_reexecution(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    before = active_repository(path)
    plan = seed_validated_study(before)
    crashed = make_worker(before, tmp_path, "worker-before")
    scheduled = schedule(before, crashed, plan)

    with pytest.raises(SimulatedWorkerCrashError):
        crashed.run_once(now=START + timedelta(seconds=1), crash_after_seal=True)
    assert crashed.executor.execution_count == 1
    assert before.get_attempt("workspace-a", scheduled.attempt_id).status == AttemptStatus.RUNNING

    after = CampaignRuntimeRepository(path)
    after.initialize()
    successor = make_worker(after, tmp_path, "worker-after")
    assert successor.run_once(now=START + timedelta(seconds=17)) == "reconciled"

    completed = after.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert successor.executor.execution_count == 0
    with after._connection() as connection:
        artifact_count = connection.execute(
            "SELECT COUNT(*) FROM campaign_artifacts WHERE producer_action_id = ?",
            (scheduled.action_id,),
        ).fetchone()[0]
        settlement_count = connection.execute(
            "SELECT COUNT(*) FROM campaign_budget_ledger WHERE entry_id = ?",
            (f"budget-settle-{scheduled.action_id}",),
        ).fetchone()[0]
    assert artifact_count == 2
    assert settlement_count == 1


def test_expired_attempt_without_seal_becomes_unknown_and_is_not_retried(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    first = make_worker(repository, tmp_path, "worker-a")
    scheduled = schedule(repository, first, plan)
    claimed = repository.claim_next_action(
        first.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=1),
    )
    assert claimed.attempt_id == scheduled.attempt_id

    successor = make_worker(repository, tmp_path, "worker-b")
    assert successor.run_once(now=START + timedelta(seconds=17)) == "unknown"
    unknown = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert unknown.status == AttemptStatus.UNKNOWN
    assert successor.executor.execution_count == 0
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours")["reserved"] == 0.25


def test_pause_blocks_new_claim_but_reconciliation_can_drain_sealed_work(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    scheduled = schedule(repository, worker, plan)
    paused = repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.PAUSE,
        expected_version=5,
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="pause-test",
        idempotency_key="pause-test",
    )
    assert paused.campaign.status.value == "paused"
    assert worker.run_once(now=START + timedelta(seconds=1)) == "idle"
    assert (
        repository.get_attempt("workspace-a", scheduled.attempt_id).status
        == AttemptStatus.SCHEDULED
    )


def test_remote_worker_launches_once_then_completes_on_a_later_tick(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING, RemoteRunState.COMPLETED))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    running = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert running.status == AttemptStatus.RUNNING
    assert repository.get_remote_run("workspace-a", scheduled.attempt_id) is not None
    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"

    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert adapter.launch_count == 1
    assert adapter.collect_count == 1
    assert (Path(completed.sealed_result_uri) / "final" / "config.json").is_file()
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": 0.25,
        "limit_delta": 0.0,
    }


def test_nemo_gym_receipt_is_registered_with_bounded_metadata(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    schedule(repository, worker, plan)
    claimed = repository.claim_next_action(
        worker.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=1),
    )
    assert claimed is not None

    dataset = tmp_path / "star-count-dataset"
    generate_star_count_dataset(
        dataset,
        train_size=1,
        validation_size=1,
        heldout_size=1,
        seed=7,
    )
    bundle = export_star_count_nemo_gym_bundle(
        dataset,
        tmp_path / "nemo-gym-bundle",
        nemo_gym_revision="a" * 40,
        bashgym_revision="b" * 40,
        dataset_license="MIT",
    )
    environment = star_count_environment_spec()
    rollout = {
        "session_id": "session-a",
        "example_index": 0,
        "environment_id": environment.id,
        "environment_digest": canonical_hash(environment.to_dict()),
        "response": {
            "output": [
                {
                    "id": "message-a",
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ]
        },
        "reward_components": {"count_accuracy": 1.0, "format_accuracy": 1.0},
        "total_reward": 1.0,
        "refit": {
            "refit_id": "refit-4",
            "training_step": 4,
            "source_checkpoint_sha256": "c" * 64,
            "policy_revision": 4,
            "generation_revision": 4,
            "synchronized": True,
        },
    }
    evidence = build_nemo_gym_campaign_evidence(
        claimed,
        bundle_manifest=bundle,
        environment=environment,
        rollout_payloads=[rollout],
    )
    temporary = tmp_path / "artifacts" / ".tmp" / "nemo-evidence"
    temporary.mkdir(parents=True)
    write_nemo_gym_campaign_evidence(
        temporary / NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
        evidence,
    )
    identity = RemoteRunIdentity(
        compute_profile_id="private-compute-a",
        run_id=claimed.attempt_id,
        remote_run_directory=f"/private/{claimed.attempt_id}",
        remote_pid=42,
        process_group_id=42,
        process_start_ticks=7,
        boot_id="boot-a",
        command_hash="d" * 64,
        launch_manifest_sha256="e" * 64,
        launched_at=START,
    )
    observation = RemoteObservation(
        identity=identity,
        state=RemoteRunState.COMPLETED,
        observed_at=START + timedelta(seconds=2),
        exit_code=0,
        safe_reason="completed",
    )
    sealed, _manifest = worker.remote_output_sealer.seal_completed(
        claimed,
        identity,
        observation,
        temporary,
    )
    verified = worker._verify(claimed, sealed)
    repository.complete_from_seal(
        verified,
        sealed,
        worker_id=worker.worker_id,
        now=START + timedelta(seconds=2),
    )

    artifact = repository.list_artifacts("workspace-a", "campaign-1")[0]
    reference = artifact.metadata["nemo_gym"]
    assert reference["artifact_id"] == artifact.artifact_id
    assert reference["bundle_digest"] == bundle["bundle_digest"]
    assert reference["token_evidence_digest"] == evidence.token_evidence_digest
    assert reference["refit_receipt_digest"] == evidence.refit_receipt_digest
    assert reference["rollout_count"] == 1


def test_controller_resolves_server_profile_and_launches_without_actor_material(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    activate_controller_live_study(repository, plan)
    profile = approved_remote_profile(tmp_path)
    registry = {(profile.compute_profile_id, profile.target_contract_key): profile}
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles=registry,
    )

    assert worker.run_once(now=START) == "remote_running"
    attempts = repository.list_attempts("workspace-a", "campaign-1")
    assert len(attempts) == 1
    executor = attempts[0].executor
    assert executor["profile_id"] == profile.profile_id
    assert executor["profile_digest"] == profile.profile_digest
    assert executor["python_executable"] == "/approved/venv/bin/python"
    assert adapter.last_request is not None
    assert adapter.last_request.python_executable == "/approved/venv/bin/python"
    with repository._connection() as connection:
        actor_recipe = json.loads(
            connection.execute(
                "SELECT proposal_json FROM campaign_proposals WHERE proposal_id = ?",
                ("proposal-study-1",),
            ).fetchone()[0]
        )["training_recipe"]
    assert actor_recipe == {
        "schema_version": "recipe.v1",
        "runtime": {"executor_kind": "registered_training"},
    }


def test_controller_missing_profile_blocks_once_without_budget_or_launch(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    activate_controller_live_study(repository, plan)
    adapter = FakeRemoteAdapter()
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )

    assert worker.run_once(now=START) == "action_blocked"
    assert worker.run_once(now=START + timedelta(seconds=1)) == "action_blocked"
    assert repository.list_attempts("workspace-a", "campaign-1") == ()
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours")["reserved"] == 0.0
    assert adapter.launch_count == 0
    blocked = [
        event
        for _, event in repository.list_events("workspace-a", "campaign-1")
        if event.event_type == "campaign:action-blocked"
    ]
    assert len(blocked) == 1
    assert blocked[0].payload["code"] == "campaign_controller_action_blocked"


def test_registered_compute_resolves_remote_development_evaluation(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    activate_controller_live_study(repository, plan)
    proposal_payload = {
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": {"schema_version": "recipe.v1"},
        "evaluation_recipe": {
            "schema_version": "recipe.v1",
            "runtime": {"executor_kind": "registered_compute"},
        },
    }
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_proposals SET proposal_json = ? WHERE proposal_id = ?",
            (json.dumps(proposal_payload), "proposal-study-1"),
        )
    profile = approved_remote_profile(tmp_path, stage=StageKind.DEVELOPMENT_EVALUATION)

    action = repository.next_action_spec(
        "workspace-a",
        "campaign-1",
        "study-1",
        executor_profiles={
            (profile.compute_profile_id, profile.target_contract_key): profile
        },
    )

    assert action.stage == StageKind.DEVELOPMENT_EVALUATION
    assert action.executor_kind == "ssh_remote"
    assert action.executor_config["stage"] == "development_evaluation"


def test_controller_profile_hash_change_blocks_before_remote_side_effect(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    activate_controller_live_study(repository, plan)
    profile = approved_remote_profile(tmp_path)
    profile.stages[0].script_path.write_text("print('tampered')\n", encoding="utf-8")
    adapter = FakeRemoteAdapter()
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={
            (profile.compute_profile_id, profile.target_contract_key): profile
        },
    )

    assert worker.run_once(now=START) == "action_blocked"
    assert repository.list_attempts("workspace-a", "campaign-1") == ()
    assert adapter.launch_count == 0


def test_remote_worker_executes_only_persisted_exact_identity_force_stop(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)
    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    remote = repository.get_remote_run("workspace-a", scheduled.attempt_id)
    assert remote is not None
    version = repository.get_campaign("workspace-a", "campaign-1").version
    requested = repository.request_force_stop(
        "workspace-a",
        "campaign-1",
        scheduled.action_id,
        remote.identity,
        reason="Exact persisted identity confirmed.",
        expected_version=version,
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="force-stop",
        idempotency_key="force-stop",
    )

    assert worker.run_once(now=START + timedelta(seconds=2)) == "remote_force_stopping"
    assert adapter.force_stop_count == 1
    assert (
        repository.pending_force_stop_request("workspace-a", scheduled.action_id, remote.identity)
        is None
    )
    with repository._connection() as connection:
        state = connection.execute(
            "SELECT state FROM campaign_action_control_requests WHERE request_id = ?",
            (requested.details["request_id"],),
        ).fetchone()["state"]
    assert state == "executed"


def test_capacity_block_never_launches_and_returns_claim_to_queue(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(admitted=False)
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_capacity_blocked"
    deferred = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert deferred.status == AttemptStatus.SCHEDULED
    assert deferred.claim_generation == 0
    assert adapter.launch_count == 0
    assert repository.get_remote_run("workspace-a", scheduled.attempt_id) is None


def test_successor_discovers_and_adopts_crash_after_remote_launch(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    repository = active_repository(path)
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    first = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-before",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, first, plan, tmp_path)
    claimed = repository.claim_next_action(
        first.leader, ttl=timedelta(seconds=15), now=START + timedelta(seconds=1)
    )
    assert claimed is not None
    asyncio.run(adapter.launch(first._remote_request(claimed)))
    assert repository.get_remote_run("workspace-a", scheduled.attempt_id) is None

    successor_repository = CampaignRuntimeRepository(path)
    successor_repository.initialize()
    successor = CampaignWorker(
        successor_repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-after",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles=first.remote_executor_profiles,
    )
    assert successor.run_once(now=START + timedelta(seconds=17)) == "remote_running"
    adopted = successor_repository.get_attempt("workspace-a", scheduled.attempt_id)
    remote = successor_repository.get_remote_run("workspace-a", scheduled.attempt_id)
    assert adopted.claim_generation == 2
    assert adopted.lease_owner == "worker-after"
    assert remote is not None and remote.claim_generation == 2
    assert adapter.launch_count == 1
    assert adapter.discover_count >= 1


def test_failed_remote_attempt_seals_evidence_and_settles_budget(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.FAILED,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_failed"
    failed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert failed.status == AttemptStatus.FAILED
    assert (Path(failed.sealed_result_uri) / "training.log").is_file()
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": 0.25,
        "limit_delta": 0.0,
    }
    with repository._connection() as connection:
        study_status = connection.execute(
            "SELECT status FROM campaign_studies WHERE study_id = 'study-1'"
        ).fetchone()[0]
    assert study_status == StudyStatus.EXECUTION_FAILED.value


def test_campaign_cancel_terminates_remote_group_and_settles_cancelled(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(
        states=(
            RemoteRunState.RUNNING,
            RemoteRunState.RUNNING,
            RemoteRunState.FAILED,
        )
    )
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)
    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    transition(repository, CampaignTrigger.CANCEL, 5, key="cancel-running-remote")

    assert worker.run_once(now=START + timedelta(seconds=2)) == "remote_cancelling"
    assert adapter.terminate_count == 1
    assert worker.run_once(now=START + timedelta(seconds=3)) == "remote_cancelled"
    cancelled = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert cancelled.status == AttemptStatus.CANCELLED
    assert repository.get_campaign("workspace-a", "campaign-1").status.value == "cancelled"
    assert (Path(cancelled.sealed_result_uri) / "exit_code").is_file()


def test_development_evaluation_stage_validates_seals_and_persists_rows(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    worker = make_worker(repository, tmp_path, "worker-a")
    assert worker.run_once(now=START) == "idle"
    development = tmp_path / "heldout-dev.jsonl"
    scored = tmp_path / "scored-dev.jsonl"
    dev_rows = [
        {
            "eval_id": f"dev-{index}",
            "split": "dev",
            "positive_video_id": f"video-{index % 3}",
        }
        for index in range(18)
    ]
    scored_rows = [
        {
            **row,
            "positive_rank_exact": 1,
            "positive_rank_local_window": 1,
            "top_retrieved_video_id": row["positive_video_id"],
            "query_type": "natural_question",
            "channel": "Channel A",
            "source_set": "fixture",
        }
        for row in dev_rows
    ]
    dev_payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows)
    development.write_bytes(dev_payload.encode())
    scored.write_bytes(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored_rows).encode()
    )
    dev_sha256 = hashlib.sha256(dev_payload.encode()).hexdigest()
    champion = load_retrieval_evaluation_artifact(
        scored,
        candidate_digest="a" * 64,
        corpus_sha256="c" * 64,
        development_sha256=dev_sha256,
        representation_contract={"query_prefix_mode": "memexai_youtube"},
        median_latency_ms=10.0,
        model_footprint_bytes=1000,
    )
    champion_id = repository.store_retrieval_evaluation(
        "workspace-a", "campaign-1", champion, now=START
    )
    with repository._connection(immediate=True) as connection:
        manifest_row = connection.execute(
            """
            SELECT manifest_json FROM campaign_manifest_revisions
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1' AND revision = 1
            """
        ).fetchone()
        manifest_payload = json.loads(manifest_row["manifest_json"])
        manifest_payload["promotion_gates"]["quality_claim_eligible"] = True
        connection.execute(
            """
            UPDATE campaign_manifest_revisions SET manifest_json = ?, manifest_hash = ?
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1' AND revision = 1
            """,
            (
                json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")),
                canonical_hash(manifest_payload),
            ),
        )
    scheduled = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-1"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="development_evaluation",
            executor_config={
                "development_path": str(development),
                "expected_development_sha256": dev_sha256,
                "protected_hashes": ["f" * 64],
                "protected_path_fragments": ["heldout-test", "heldout-dev-test"],
                "scored_rows_path": str(scored),
                "corpus_sha256": "c" * 64,
                "representation_contract": {"query_prefix_mode": "memexai_youtube"},
                "median_latency_ms": 10.0,
                "model_footprint_bytes": 1000,
                "champion_evaluation_id": champion_id,
                "gate_contract": {"bootstrap_samples": 100},
            },
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )

    original_persist = worker._persist_development_evaluation_evidence
    worker._persist_development_evaluation_evidence = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SimulatedWorkerCrashError("crash after development seal")
    )
    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=START + timedelta(seconds=1))
    worker._persist_development_evaluation_evidence = original_persist
    successor = make_worker(repository, tmp_path, "worker-b")
    assert successor.run_once(now=START + timedelta(seconds=17)) == "reconciled"
    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert (Path(completed.sealed_result_uri) / "evaluation.json").is_file()
    with repository._connection() as connection:
        row = connection.execute(
            "SELECT evaluation_id FROM campaign_evaluations WHERE evaluation_id != ?",
            (champion_id,),
        ).fetchone()
        decision_count = connection.execute(
            "SELECT COUNT(*) FROM campaign_gate_decisions"
        ).fetchone()[0]
    evaluation = repository.get_retrieval_evaluation("workspace-a", row["evaluation_id"])
    assert len(evaluation.rows) == 18
    assert {item.eval_id for item in evaluation.rows} == {f"dev-{index}" for index in range(18)}
    assert decision_count == 0

    oversight = HumanOversightRepository(
        repository,
        sealer=ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
    )
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="desktop-reviewer",
        autonomy_profile=AutonomyProfile.DESKTOP_USER,
        workspace_ids=("workspace-a",),
    )
    reviewer = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)
    queue = oversight.read_queue(
        "workspace-a", "campaign-1", reviewer, now=START + timedelta(seconds=18)
    )
    assert len(queue["items"]) == 1
    assert champion.candidate_digest not in json.dumps(queue)
    work = queue["items"][0]
    claimed = oversight.claim(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id=work["work_id"], expected_campaign_revision=1,
        expected_version=1, expected_state="pending", principal=reviewer,
        correlation_id="worker-review-claim",
        idempotency_key=work["claim_idempotency_key"],
        now=START + timedelta(seconds=18),
    )
    oversight.submit(
        workspace_id="workspace-a", campaign_id="campaign-1",
        work_id=work["work_id"], expected_campaign_revision=1,
        expected_version=2, expected_rubric_version=1,
        decision="no_material_difference", rationale="Equivalent under the blinded rubric.",
        principal=reviewer, correlation_id="worker-review-submit",
        idempotency_key=claimed.queue["items"][0]["submit_idempotency_key"],
        now=START + timedelta(seconds=19),
    )
    with repository._connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM campaign_gate_decisions"
        ).fetchone()[0] == 1


def test_development_evaluation_invokes_hash_pinned_physical_dev_scorer(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    worker = make_worker(repository, tmp_path, "worker-a")
    assert worker.run_once(now=START) == "idle"

    development = tmp_path / "heldout-dev.jsonl"
    dev_rows = [
        {
            "eval_id": f"dev-{index}",
            "split": "dev",
            "positive_video_id": f"video-{index % 3}",
        }
        for index in range(18)
    ]
    development.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    scorer = tmp_path / "fixture_scorer.py"
    scorer.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--queries-jsonl', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args, _unknown = parser.parse_known_args()
args.output_dir.mkdir(parents=True)
rows = [json.loads(line) for line in args.queries_jsonl.read_text(encoding='utf-8').splitlines()]
for row in rows:
    row.update({
        'positive_rank_exact': 1,
        'positive_rank_local_window': 1,
        'top_retrieved_video_id': row['positive_video_id'],
        'query_type': 'natural_question',
        'channel': 'Channel A',
        'source_set': 'fixture',
    })
rows_path = args.output_dir / 'memexai_youtube-retrieval_eval_queries.jsonl'
rows_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\\n' for row in rows), encoding='utf-8')
manifest = {
    'model_footprint_bytes': 4321,
    'runs': {'memexai_youtube': {'median_query_latency_ms': 12.5}},
}
(args.output_dir / 'query_format_ablation_manifest.json').write_text(
    json.dumps(manifest, sort_keys=True), encoding='utf-8'
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.jsonl"
    matrix = tmp_path / "corpus.npy"
    chunk_ids = tmp_path / "chunk_ids.json"
    model = tmp_path / "model"
    corpus.write_text("{}\n", encoding="utf-8")
    matrix.write_bytes(b"fixture matrix")
    chunk_ids.write_text("[]\n", encoding="utf-8")
    model.mkdir()

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    scheduled = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-1"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="development_evaluation",
            executor_config={
                "development_path": str(development),
                "expected_development_sha256": digest(development),
                "protected_hashes": ["f" * 64],
                "protected_path_fragments": ["heldout-test"],
                "corpus_sha256": digest(corpus),
                "representation_contract": {"query_prefix_mode": "memexai_youtube"},
                "gate_contract": {"bootstrap_samples": 100},
                "scorer": {
                    "scorer_script_path": str(scorer),
                    "expected_scorer_sha256": digest(scorer),
                    "embedding_model_path": str(model),
                    "corpus_path": str(corpus),
                    "expected_corpus_sha256": digest(corpus),
                    "corpus_embedding_matrix": str(matrix),
                    "expected_matrix_sha256": digest(matrix),
                    "corpus_embedding_chunk_ids": str(chunk_ids),
                    "expected_chunk_ids_sha256": digest(chunk_ids),
                    "query_prefix_mode": "memexai_youtube",
                    "embedding_device": "cpu",
                },
            },
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )

    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"
    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    sealed = Path(completed.sealed_result_uri)
    assert (sealed / "scoring" / "query_format_ablation_manifest.json").is_file()
    evaluation = json.loads((sealed / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["median_latency_ms"] == 12.5
    assert evaluation["model_footprint_bytes"] == 4321


def test_completion_rejects_full_identity_mismatch_without_writes(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    scheduled = schedule(repository, worker, plan)
    claimed = repository.claim_next_action(
        worker.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=1),
    )
    sealed, manifest = worker.executor.execute(
        FakeExecutionRequest(
            workspace_id=claimed.workspace_id,
            campaign_id=claimed.campaign_id,
            study_id=claimed.study_id,
            action_id=claimed.action_id,
            attempt_id=claimed.attempt_id,
            manifest_revision=claimed.manifest_revision,
            candidate_digest=claimed.candidate_digest,
            input_digest=claimed.input_digest,
            claim_generation=claimed.claim_generation,
        )
    )
    wrong = manifest.model_copy(update={"candidate_digest": fake_digest("wrong-candidate")})

    with pytest.raises(ActionIdentityMismatchError):
        repository.complete_from_seal(wrong, sealed, worker_id=worker.worker_id)
    assert (
        repository.get_attempt("workspace-a", scheduled.attempt_id).status == AttemptStatus.RUNNING
    )
    with repository._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_artifacts").fetchone()[0] == 0


def test_one_worker_completes_three_studies_without_duplicate_actions(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plans = {
        study_id: seed_validated_study(repository, study_id, sequence=index)
        for index, study_id in enumerate(("study-1", "study-2", "study-3"), start=1)
    }
    worker = make_worker(repository, tmp_path, "worker-a")
    version = 4
    attempt_ids = []
    for index, (study_id, plan) in enumerate(plans.items(), start=1):
        scheduled = schedule(
            repository,
            worker,
            plan,
            study_id=study_id,
            version=version,
        )
        attempt_ids.append(scheduled.attempt_id)
        assert worker.run_once(now=START + timedelta(seconds=index)) == "completed"
        version += 2

    assert len(set(attempt_ids)) == 3
    assert worker.executor.execution_count == 3
    with repository._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_actions").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM campaign_artifacts").fetchone()[0] == 6
    events = repository.list_events("workspace-a", "campaign-1")
    assert sum(event.event_type == "campaign:action-completed" for _, event in events) == 3
    assert sum(event.event_type == "campaign:training-metrics-appended" for _, event in events) == 3
    for attempt_id in attempt_ids:
        loss = repository.get_metric_series(
            "workspace-a",
            attempt_id,
            "loss",
            source="training_metrics.jsonl",
        )
        assert len(loss) == 6
        assert loss[-1].value < loss[0].value


def test_resident_loop_heartbeats_during_idle_backoff_and_releases_leader(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    worker = make_worker(repository, tmp_path, "worker-a")
    current = START
    sleeps = []

    def clock():
        return current

    def fake_sleep(seconds):
        nonlocal current
        sleeps.append(seconds)
        current += timedelta(seconds=seconds)
        if len(sleeps) == 3:
            worker.request_stop()

    worker.run_forever(sleep=fake_sleep, clock=clock)

    assert sleeps == [5.0, 5.0, 5.0]
    assert worker.leader is None
    replacement = repository.acquire_lease(
        worker.leader_key,
        "worker-b",
        ttl=timedelta(seconds=15),
        now=current,
    )
    assert replacement.generation == 2


def test_worker_reacquires_an_expired_cached_scheduler_lease(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    worker = make_worker(repository, tmp_path, "worker-a")

    assert worker.run_once(now=START) == "idle"
    assert worker.leader is not None
    assert worker.leader.generation == 1

    assert worker.run_once(now=START + timedelta(seconds=16)) == "idle"
    assert worker.leader is not None
    assert worker.leader.generation == 2
    assert worker.leader.expires_at == START + timedelta(seconds=31)


def test_worker_drops_an_expired_cached_lease_after_a_successor_takes_over(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    worker = make_worker(repository, tmp_path, "worker-a")

    assert worker.run_once(now=START) == "idle"
    successor = repository.acquire_lease(
        worker.leader_key,
        "worker-b",
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=16),
    )

    assert successor.generation == 2
    assert worker.run_once(now=START + timedelta(seconds=17)) == "not_leader"
    assert worker.leader is None
    assert repository.get_lease(worker.leader_key) == successor
