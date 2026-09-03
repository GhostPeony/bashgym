"""Resident worker fencing, completion, pause, and restart reconciliation tests."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from bashgym._compat import UTC
from bashgym.campaigns import remote as remote_contracts
from bashgym.campaigns.artifacts import SEAL_FILENAME, ArtifactSealer, ArtifactSealError
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchCampaignSpec,
    AutoResearchProposalControl,
    AutoResearchRepository,
    AutoResearchStopRules,
    ExperimentRole,
    MetricDirection,
)
from bashgym.campaigns.autoresearch_dataset import (
    AUTORESEARCH_DATASET_RECEIPT_FILENAME,
    AutoResearchDatasetFile,
    AutoResearchDatasetQuality,
    AutoResearchDatasetReceipt,
)
from bashgym.campaigns.autoresearch_evidence import (
    AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
    AUTORESEARCH_EVALUATION_FILENAME,
    AutoResearchEvaluationContext,
)
from bashgym.campaigns.contracts import (
    ActionAttempt,
    ActionStatus,
    AttemptStatus,
    AutonomyProfile,
    CampaignTrigger,
    CredentialKind,
    FailureClass,
    ManifestRevision,
    SealedActionResult,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyStatus,
    canonical_hash,
)
from bashgym.campaigns.diagnostic_actions import (
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA,
    AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME,
    AutoResearchDiagnosticRecipe,
    AutoResearchDiagnosticRequest,
    diagnostic_recipe_digest,
)
from bashgym.campaigns.evaluation import load_retrieval_evaluation_artifact
from bashgym.campaigns.executor_adapters import (
    DevelopmentEvaluationExecutorAdapter,
    FakeExecutorAdapter,
    SshRemoteExecutorAdapter,
    build_default_registry,
)
from bashgym.campaigns.executor_registry import ExecutorRegistry
from bashgym.campaigns.executors import FakeExecutionRequest, RemoteOutputSealer, fake_digest
from bashgym.campaigns.human_oversight import HumanOversightRepository
from bashgym.campaigns.lineage import canonical_model_manifest_digest
from bashgym.campaigns.nemo_gym_evidence import (
    NEMO_GYM_CAMPAIGN_EVIDENCE_FILENAME,
    build_nemo_gym_campaign_evidence,
    write_nemo_gym_campaign_evidence,
)
from bashgym.campaigns.persistence import CampaignPersistenceError, RecordNotFoundError
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    DiagnosticStageContract,
    PinnedRemoteStageProfile,
    RegisteredRemoteEvaluationDatasetSource,
    RegisteredRemoteModelSource,
    RemoteCapacitySnapshot,
    RemoteObservation,
    RemoteOutputFile,
    RemoteOutputInventory,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamChunk,
    SealedStageArtifactInput,
    remote_executor_config,
)
from bashgym.campaigns.runtime import (
    ActionIdentityMismatchError,
    ActionSpec,
    CampaignArtifactRecord,
    CampaignRuntimeRepository,
    _artifact_reference,
)
from bashgym.campaigns.service import CampaignService
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
from bashgym.ledger.contracts import (
    DatasetSpec,
    DatasetVersionSpec,
    EvaluationSuiteSpec,
    ProjectSpec,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository
from tests.campaigns.reuse_helpers import (
    derived_data_build_result_key,
    rewrite_result_manifest,
    set_action_status,
    set_reuse_link,
)
from tests.campaigns.test_persistence import campaign, create, manifest, transition
from tests.campaigns.test_proposals import principal

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
    campaign_id: str = "campaign-1",
    input_contract: dict | None = None,
) -> StagePlan:
    """Insert already-validated controller fixtures; proposal planning is a later slice."""

    proposal_id = f"proposal-{study_id}"
    plan = StagePlan(
        items=(
            StagePlanItem(
                stage=stage,
                disposition=StageDisposition.REQUIRED,
                reason="Prove one durable fake execution lifecycle.",
                input_contract=input_contract or {"fixture": study_id},
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
            ("workspace-a", campaign_id, proposal_id, sequence, START.isoformat()),
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
                campaign_id,
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
    def __init__(self, *, admitted=True, states=(RemoteRunState.RUNNING,), failed_exit_code=7):
        self.admitted = admitted
        self.states = list(states)
        self.failed_exit_code = failed_exit_code
        self.identity = None
        self.launch_count = 0
        self.discover_count = 0
        self.collect_count = 0
        self.terminate_count = 0
        self.force_stop_count = 0
        self.last_request = None
        self.seal_payload = None
        self.persist_count = 0
        self.stream_sources = []

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
        self.last_request = request
        if self.identity is None or self.identity.run_id != request.run_id:
            return None
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
            exit_code = self.failed_exit_code
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
        self.stream_sources.append(source)
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

    async def inventory_outputs(self, identity, request, *, observation):
        self.collect_count += 1
        payloads = {
            "exit_code": b"0\n",
            "final/config.json": b"{}",
            "launch_manifest.json": b"{}",
            "training.log": b"complete\n",
            "training_manifest.json": json.dumps({"run_id": identity.run_id}).encode(),
            "training_metrics.jsonl": b'{"step":1,"loss":0.5}\n',
        }
        return RemoteOutputInventory(
            compute_profile_id=identity.compute_profile_id,
            run_id=identity.run_id,
            files=tuple(
                RemoteOutputFile(
                    path=path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
                for path, payload in sorted(payloads.items())
            ),
        )

    async def inventory_terminal_evidence(self, identity, *, observation):
        self.collect_count += 1
        payloads = {
            "exit_code": f"{self.failed_exit_code}\n".encode(),
            "launch_manifest.json": b"{}",
            "training.log": b"failed\n",
        }
        return RemoteOutputInventory(
            compute_profile_id=identity.compute_profile_id,
            run_id=identity.run_id,
            files=tuple(
                RemoteOutputFile(
                    path=path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
                for path, payload in sorted(payloads.items())
            ),
        )

    async def persist_action_seal(self, identity, envelope):
        self.persist_count += 1
        self.seal_payload = bytes(envelope)
        return f"{identity.remote_run_directory}/{SEAL_FILENAME}"

    async def read_action_seal(self, identity):
        del identity
        return self.seal_payload

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
    target_model_digest = canonical_hash(campaign().target_model.model_dump(mode="json"))
    registered_base_model = (
        RegisteredRemoteModelSource(
            source_id="registered-base-v1",
            compute_profile_id="ssh-gpu-lab",
            target_contract_key="memexai-embedding-v1",
            model_digest=target_model_digest,
            remote_model_path="/models/registered-base-v1",
        )
        if stage == StageKind.FULL_TRAINING
        else None
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="memexai-approved-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=target_model_digest,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        remote_work_dir="~/bashgym-training",
        stages=(pinned,),
        registered_base_model=registered_base_model,
    )


def test_diagnostic_remote_request_is_worker_owned_and_restart_deterministic(tmp_path):
    repository = active_repository(tmp_path / "diagnostic.sqlite3")
    seed_validated_study(repository, stage=StageKind.CONTRACT_EVALUATION)
    script = tmp_path / "diagnose.py"
    key = tmp_path / "diagnostic-key"
    script.write_text("print('diagnose')\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.CONTRACT_EVALUATION,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(),
        input_sha256={},
        output_paths=(AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,),
        budget_unit="gpu_hours",
        budget_reservation=0.1,
        diagnostic_contract=DiagnosticStageContract(
            runner_id="generic-diagnostic-runner",
            runner_version="1",
            max_sample_limit=128,
            max_measurements=4,
        ),
    )
    target_model_digest = canonical_hash(campaign().target_model.model_dump(mode="json"))
    profile = ApprovedRemoteExecutorProfile(
        profile_id="diagnostic-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=target_model_digest,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
    )
    recipe = AutoResearchDiagnosticRecipe(
        probe_family="agent_authored_conflict_probe",
        question="Does one aggregate explain the fixed-suite failure cluster?",
        hypothesis="The aggregate separates retained and failed behavior.",
        informs_methods=("sft",),
        measurements=({"name": "conflict_rate", "interpretation": "minimize", "unit": "fraction"},),
        sample_limit=64,
        seed=9,
        data_scope_ids=("memexai-approved-training",),
    )
    executor = repository._attempt_bound_executor_contract(
        {
            "kind": "ssh_remote",
            **remote_executor_config(
                profile,
                StageKind.CONTRACT_EVALUATION,
                recipe_digest=diagnostic_recipe_digest(recipe),
                diagnostic_recipe=recipe,
                approved_data_scopes=frozenset({"memexai-approved-training"}),
            ),
            "diagnostic_proposal_id": "proposal-study-1",
        },
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-diagnostic-1",
        attempt_id="attempt-diagnostic-1",
        candidate_digest=fake_digest("candidate:study-1"),
    )
    attempt = ActionAttempt(
        attempt_id="attempt-diagnostic-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-diagnostic-1",
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.RUNNING,
        input_digest="1" * 64,
        candidate_digest=fake_digest("candidate:study-1"),
        manifest_revision=1,
        stage=StageKind.CONTRACT_EVALUATION,
        stage_index=0,
        executor=executor,
        created_at=START,
        updated_at=START,
    )
    workers = [
        CampaignWorker(
            repository,
            tmp_path / f"artifacts-{index}",
            ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
            data_directory=tmp_path / "data-root",
            worker_id=f"worker-{index}",
            remote_executor_profiles={
                (profile.compute_profile_id, profile.target_contract_key): profile
            },
        )
        for index in (1, 2)
    ]

    first = workers[0]._remote_request(attempt)
    replay = workers[1]._remote_request(attempt)

    assert [path.name for path in first.input_files] == [AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME]
    assert first.script_args == (
        "--request",
        AUTORESEARCH_DIAGNOSTIC_REQUEST_FILENAME,
        "--output",
        AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
    )
    request = AutoResearchDiagnosticRequest.model_validate_json(first.input_files[0].read_bytes())
    assert request.recipe == recipe
    assert request.proposal_id == "proposal-study-1"
    assert request.runner_id == "generic-diagnostic-runner"
    assert replay.request_digest == first.request_digest
    assert replay.input_files[0].read_bytes() == first.input_files[0].read_bytes()


def test_remote_output_sealer_classifies_diagnostic_evidence() -> None:
    assert (
        RemoteOutputSealer._schema_for_relative(AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME)
        == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_SCHEMA
    )


def test_remote_diagnostic_completion_stores_signed_aggregate_projection(tmp_path):
    repository = active_repository(tmp_path / "diagnostic-completion.sqlite3")
    plan = seed_validated_study(repository, stage=StageKind.CONTRACT_EVALUATION)
    script = tmp_path / "diagnose-completion.py"
    key = tmp_path / "diagnostic-completion-key"
    script.write_text("print('diagnose')\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.CONTRACT_EVALUATION,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(),
        input_sha256={},
        output_paths=(AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,),
        budget_unit="gpu_hours",
        budget_reservation=0.1,
        diagnostic_contract=DiagnosticStageContract(
            runner_id="generic-diagnostic-runner",
            runner_version="1",
            max_sample_limit=128,
            max_measurements=4,
        ),
    )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="diagnostic-completion-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
    )
    recipe = AutoResearchDiagnosticRecipe(
        probe_family="agent_authored_loss_probe",
        question="Does loss separate the observed failure clusters?",
        hypothesis="Loss is higher for the failed cluster.",
        informs_methods=("sft",),
        measurements=({"name": "loss_gap", "interpretation": "maximize", "unit": "loss"},),
        sample_limit=64,
        seed=9,
        data_scope_ids=("memexai-approved-training",),
    )
    sealer = ArtifactSealer(b"w" * 32, key_version="worker-test-v1")

    class DiagnosticAdapter(FakeRemoteAdapter):
        payload: bytes

        async def inventory_outputs(self, identity, request, *, observation):
            del request, observation
            return RemoteOutputInventory(
                compute_profile_id=identity.compute_profile_id,
                run_id=identity.run_id,
                files=(
                    RemoteOutputFile(
                        path=AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,
                        sha256=hashlib.sha256(self.payload).hexdigest(),
                        size_bytes=len(self.payload),
                    ),
                ),
            )

        async def read_output_bytes(
            self, identity, path, *, expected_sha256, expected_size_bytes, max_bytes
        ):
            del identity
            assert path == AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME
            assert expected_sha256 == hashlib.sha256(self.payload).hexdigest()
            assert expected_size_bytes == len(self.payload)
            assert len(self.payload) <= max_bytes
            return self.payload

    adapter = DiagnosticAdapter(states=(RemoteRunState.COMPLETED,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="worker-diagnostic",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={("ssh-gpu-lab", "memexai-embedding-v1"): profile},
    )
    assert worker.run_once(now=START) == "idle"
    scheduled = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.CONTRACT_EVALUATION,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-1"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.1,
            executor_kind="ssh_remote",
            executor_config={
                **remote_executor_config(
                    profile,
                    StageKind.CONTRACT_EVALUATION,
                    recipe_digest=diagnostic_recipe_digest(recipe),
                    diagnostic_recipe=recipe,
                    approved_data_scopes=frozenset({"memexai-approved-training"}),
                ),
                "diagnostic_proposal_id": "proposal-study-1",
            },
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )
    adapter.payload = json.dumps(
        {
            "schema_version": "bashgym.autoresearch_diagnostic_evidence.v1",
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-1",
            "proposal_id": "proposal-study-1",
            "study_id": "study-1",
            "action_id": scheduled.action_id,
            "attempt_id": scheduled.attempt_id,
            "recipe_digest": diagnostic_recipe_digest(recipe),
            "runner_id": "generic-diagnostic-runner",
            "runner_version": "1",
            "status": "completed",
            "measurements": [
                {
                    "name": "loss_gap",
                    "value": 0.2,
                    "sample_count": 64,
                    "unit": "loss",
                }
            ],
            "resource_usage": [
                {
                    "unit": "gpu_hours",
                    "amount": 0.02,
                    "source": "runner",
                    "confidence": "measured",
                }
            ],
        },
        sort_keys=True,
    ).encode()

    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"
    artifact = next(
        item
        for item in repository.list_artifacts("workspace-a", "campaign-1")
        if item.producer_action_id == scheduled.action_id
    )
    normalized = artifact.metadata["normalized_diagnostic"]
    assert normalized["projection"]["probe_family"] == "agent_authored_loss_probe"
    assert normalized["projection"]["measurements"][0]["value"] == 0.2
    assert artifact.metadata["projection_signature"] == sealer.sign_canonical_payload(
        normalized,
        domain="bashgym.autoresearch.normalized-diagnostic.v1",
    )


def test_runtime_schedules_diagnostic_with_profile_budget_and_open_recipe(tmp_path) -> None:
    repository = active_repository(tmp_path / "diagnostic-runtime.sqlite3")
    seed_validated_study(repository, stage=StageKind.CONTRACT_EVALUATION)
    recipe = {
        "schema_version": "bashgym.autoresearch_diagnostic_recipe.v1",
        "probe_family": "unseen_agent_probe",
        "question": "Which aggregate would distinguish the failure clusters?",
        "hypothesis": "The requested aggregate separates the clusters.",
        "informs_methods": ["dpo"],
        "measurements": [
            {"name": "cluster_separation", "interpretation": "maximize", "unit": "score"}
        ],
        "sample_limit": 32,
        "seed": 4,
        "data_scope_ids": ["memexai-approved-training"],
        "parameters": {"minimum_cluster_size": 3},
        "runtime": {"executor_kind": "registered_compute"},
    }
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_proposals SET proposal_json = ?
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND proposal_id = 'proposal-study-1'
            """,
            (
                json.dumps(
                    {
                        "primary_variable": "diagnostic.cluster_separation",
                        "dataset_recipe": {"schema_version": "dataset.v1"},
                        "training_recipe": {"schema_version": "training.v1"},
                        "evaluation_recipe": recipe,
                    }
                ),
            ),
        )
    script = tmp_path / "runtime-diagnose.py"
    key = tmp_path / "runtime-diagnostic-key"
    script.write_text("print('diagnose')\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.CONTRACT_EVALUATION,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(),
        input_sha256={},
        output_paths=(AUTORESEARCH_DIAGNOSTIC_EVIDENCE_FILENAME,),
        budget_unit="gpu_hours",
        budget_reservation=0.125,
        diagnostic_contract=DiagnosticStageContract(
            runner_id="generic-diagnostic-runner",
            runner_version="1",
            max_sample_limit=128,
            max_measurements=4,
        ),
    )
    target_model_digest = canonical_hash(campaign().target_model.model_dump(mode="json"))
    profile = ApprovedRemoteExecutorProfile(
        profile_id="diagnostic-runtime-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=target_model_digest,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
    )

    spec = repository.next_action_spec(
        "workspace-a",
        "campaign-1",
        "study-1",
        executor_profiles={("ssh-gpu-lab", "memexai-embedding-v1"): profile},
    )

    assert spec.stage == StageKind.CONTRACT_EVALUATION
    assert spec.executor_kind == "ssh_remote"
    assert spec.budget_unit == "gpu_hours"
    assert spec.budget_reservation == 0.125
    assert spec.executor_config["diagnostic_recipe"]["probe_family"] == "unseen_agent_probe"
    assert spec.executor_config["diagnostic_proposal_id"] == "proposal-study-1"


def test_evaluation_remote_request_regenerates_context_and_revalidates_sealed_model(tmp_path):
    sealer = ArtifactSealer(b"w" * 32, key_version="worker-test-v1")
    training_attempt = ActionAttempt(
        attempt_id="attempt-training-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-training-1",
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.COMPLETED,
        input_digest="1" * 64,
        candidate_digest=fake_digest("candidate:study-1"),
        manifest_revision=1,
        stage=StageKind.FULL_TRAINING,
        stage_index=1,
        sealed_result_uri=str(tmp_path / "sealed-training"),
        created_at=START,
        updated_at=START,
    )
    temporary = tmp_path / "temporary-training"
    (temporary / "final").mkdir(parents=True)
    (temporary / "final" / "config.json").write_text("{}", encoding="utf-8")
    (temporary / "final" / "weights.safetensors").write_bytes(b"weights")
    outputs = sealer.describe_outputs(
        temporary,
        {
            "final/config.json": "huggingface_model_file.v1",
            "final/weights.safetensors": "huggingface_model_file.v1",
        },
    )
    training_manifest = SealedActionResult(
        workspace_id=training_attempt.workspace_id,
        campaign_id=training_attempt.campaign_id,
        study_id=training_attempt.study_id,
        action_id=training_attempt.action_id,
        attempt_id=training_attempt.attempt_id,
        manifest_revision=training_attempt.manifest_revision,
        candidate_digest=training_attempt.candidate_digest,
        input_digest=training_attempt.input_digest,
        claim_generation=training_attempt.claim_generation,
        executor_id="campaign-ssh-remote-executor",
        executor_version="1",
        compute_profile_id="ssh-gpu-lab",
        started_at=START,
        ended_at=START + timedelta(seconds=1),
        outcome="completed",
        exit_code=0,
        exit_reason="remote_exit_code_recorded",
        outputs=outputs,
    )
    sealed_training = Path(training_attempt.sealed_result_uri)
    sealer.seal(temporary, sealed_training, training_manifest)
    artifacts = tuple(
        CampaignArtifactRecord(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            artifact_id=f"artifact-model-{index}",
            producer_action_id=training_attempt.action_id,
            uri=str(sealed_training / output.path),
            sha256=output.sha256,
            size_bytes=output.size_bytes,
            schema_name=output.schema_name,
            sealed=True,
            valid=True,
            metadata={
                "attempt_id": training_attempt.attempt_id,
                "relative_path": output.path.removeprefix("final/"),
            },
            created_at=START,
        )
        for index, output in enumerate(outputs, start=1)
    )
    evaluator = tmp_path / "evaluate.py"
    dataset_path = tmp_path / "development.jsonl"
    key = tmp_path / "campaign-key"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    dataset_path.write_text("{}\n", encoding="utf-8")
    key.write_text("fixture-key\n", encoding="utf-8")
    dataset = DatasetVersionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        dataset_id="dataset-a",
        dataset_version_id="dataset-version-a",
        source_uri="bashgym-remote-dataset://development-v1",
        content_digest=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="suite-a",
        name="Development evaluator",
        task_type="retrieval",
        dataset_version_id=dataset.dataset_version_id,
        metric_contract={"primary_metric": "score"},
        code_digest=hashlib.sha256(evaluator.read_bytes()).hexdigest(),
    )
    heldout = RegisteredRemoteEvaluationDatasetSource(
        source_id="development-v1",
        compute_profile_id="ssh-gpu-lab",
        dataset_version_id=dataset.dataset_version_id,
        content_digest=dataset.content_digest,
        remote_dataset_path="/datasets/development.jsonl",
    )
    stage = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=evaluator,
        script_sha256=suite.code_digest,
        input_files=(),
        input_sha256={},
        script_args=("--batch-size", "8"),
        output_paths=(AUTORESEARCH_EVALUATION_FILENAME,),
        budget_reservation=0.25,
    )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="evaluation-profile-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(stage,),
        registered_evaluation_dataset=heldout,
    )
    sealed_inputs = tuple(
        SealedStageArtifactInput(
            campaign_artifact_id=artifact.artifact_id,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            schema_name=artifact.schema_name,
            local_sealed_path=Path(artifact.uri),
            remote_relative_path=f"model/{artifact.metadata['relative_path']}",
        )
        for artifact in artifacts
    )
    source_training = {
        "campaign_id": training_attempt.campaign_id,
        "study_id": training_attempt.study_id,
        "action_id": training_attempt.action_id,
        "attempt_id": training_attempt.attempt_id,
        "stage_index": 1,
    }
    executor_config = remote_executor_config(
        profile,
        StageKind.DEVELOPMENT_EVALUATION,
        recipe_digest="e" * 64,
        sealed_stage_artifact_inputs=sealed_inputs,
        evaluation_suite=suite,
        dataset_version=dataset,
        source_training=source_training,
    )
    persisted_context = AutoResearchEvaluationContext(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-evaluation-1",
        attempt_id="attempt-evaluation-1",
        candidate_digest=training_attempt.candidate_digest,
        evaluation_suite_id=suite.evaluation_suite_id,
        evaluation_code_digest=suite.code_digest,
        dataset_version_id=dataset.dataset_version_id,
        dataset_content_digest=dataset.content_digest,
        evaluated_model_manifest_digest=canonical_model_manifest_digest(sealed_inputs),
    )
    executor_config["evaluation_context_sha256"] = hashlib.sha256(
        json.dumps(
            persisted_context.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation_attempt = ActionAttempt(
        attempt_id="attempt-evaluation-1",
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        study_id="study-1",
        action_id="action-evaluation-1",
        attempt_number=1,
        claim_generation=1,
        status=AttemptStatus.RUNNING,
        input_digest="3" * 64,
        candidate_digest=training_attempt.candidate_digest,
        manifest_revision=1,
        stage=StageKind.DEVELOPMENT_EVALUATION,
        stage_index=2,
        executor={"kind": "ssh_remote", **executor_config},
        created_at=START,
        updated_at=START,
    )

    class Repository:
        def get_study(self, *_args):
            return SimpleNamespace(proposal_id="proposal-1")

        def get_code_lineage(self, *_args):
            raise RecordNotFoundError("missing")

        def get_artifact(self, _workspace_id, _campaign_id, artifact_id):
            return {artifact.artifact_id: artifact for artifact in artifacts}[artifact_id]

        def get_attempt(self, _workspace_id, attempt_id):
            assert attempt_id == training_attempt.attempt_id
            return training_attempt

        def get_immediately_preceding_training_attempt(self, *_args):
            return training_attempt, 1

        def get_evaluation_suite_spec(self, *_args):
            return suite

        def get_dataset_version_spec(self, *_args):
            return dataset

    worker = CampaignWorker.__new__(CampaignWorker)
    worker.repository = Repository()
    worker.sealer = sealer
    worker.remote_executor_profiles = {
        (profile.compute_profile_id, profile.target_contract_key): profile
    }
    worker.source_repository_profiles = {}
    worker._lineage_snapshots = {}
    worker.evaluation_context_root = tmp_path / "evaluation-contexts"

    request = worker._remote_request(evaluation_attempt)

    context_path = next(
        path
        for path in request.input_files
        if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
    )
    context = AutoResearchEvaluationContext.model_validate_json(context_path.read_bytes())
    assert context.attempt_id == evaluation_attempt.attempt_id
    assert context.evaluation_code_digest == suite.code_digest
    assert context.dataset_content_digest == dataset.content_digest
    assert request.sealed_stage_artifact_inputs == sealed_inputs
    assert (
        request.source_training.model_dump(mode="json", exclude={"schema_version"})
        == source_training
    )
    assert request.evaluation_context_sha256 == executor_config["evaluation_context_sha256"]
    assert evaluation_attempt.executor["source_training"] == source_training
    assert request.script_args == (
        "--batch-size",
        "8",
        "--context",
        AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
        "--model-dir",
        "model",
        "--dataset",
        heldout.remote_dataset_path,
        "--output",
        AUTORESEARCH_EVALUATION_FILENAME,
    )

    tampered = evaluation_attempt.model_copy(deep=True)
    tampered.executor["sealed_stage_artifact_inputs"][0][
        "remote_relative_path"
    ] = "model/renamed.json"
    with pytest.raises(RuntimeError, match="profile mismatch|artifact"):
        worker._remote_request(tampered)

    cross_study = evaluation_attempt.model_copy(deep=True)
    cross_study.executor["source_training"]["study_id"] = "study-other"
    with pytest.raises(RuntimeError, match="training_checkpoint_invalid"):
        worker._remote_request(cross_study)

    non_immediate = evaluation_attempt.model_copy(deep=True)
    non_immediate.executor["source_training"]["stage_index"] = 0
    with pytest.raises(RuntimeError, match="training_checkpoint_invalid"):
        worker._remote_request(non_immediate)

    preceding_training = worker.repository.get_immediately_preceding_training_attempt

    def missing_predecessor(*_args):
        raise RecordNotFoundError("immediately preceding training attempt not found")

    worker.repository.get_immediately_preceding_training_attempt = missing_predecessor
    with pytest.raises(RuntimeError, match="campaign_remote_training_checkpoint_invalid"):
        worker._remote_request(evaluation_attempt)
    worker.repository.get_immediately_preceding_training_attempt = preceding_training

    retry_repository = active_repository(tmp_path / "retry.sqlite3")
    retry_plan = seed_validated_study(retry_repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    retry_scheduler = make_worker(retry_repository, tmp_path / "retry-worker", "retry-scheduler")
    assert retry_scheduler.run_once(now=START) == "idle"
    scheduled = retry_repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract=retry_plan.items[0].input_contract,
            candidate_digest=training_attempt.candidate_digest,
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="ssh_remote",
            executor_config=executor_config,
        ),
        retry_scheduler.leader,
        expected_campaign_version=4,
        now=START,
    )
    with retry_repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_attempts SET status = 'failed' WHERE attempt_id = ?",
            (scheduled.attempt_id,),
        )
        connection.execute(
            "UPDATE campaign_actions SET status = 'failed' WHERE action_id = ?",
            (scheduled.action_id,),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_action_id = NULL, active_study_id = NULL,
                version = version + 1 WHERE workspace_id = ? AND campaign_id = ?
            """,
            (scheduled.workspace_id, scheduled.campaign_id),
        )
    retry_mutation = retry_repository.retry_action(
        scheduled.workspace_id,
        scheduled.campaign_id,
        scheduled.action_id,
        expected_version=retry_repository.get_campaign(
            scheduled.workspace_id, scheduled.campaign_id
        ).version,
        actor_id="operator-a",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="retry-evaluation",
        idempotency_key="retry-evaluation",
    )
    retry_attempt = retry_repository.get_attempt(
        scheduled.workspace_id, retry_mutation.details["attempt_id"]
    )
    assert (
        retry_attempt.executor["evaluation_context_sha256"]
        != scheduled.executor["evaluation_context_sha256"]
    )
    retry_request = worker._remote_request(retry_attempt)
    retry_context_path = next(
        path
        for path in retry_request.input_files
        if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
    )
    retry_context = AutoResearchEvaluationContext.model_validate_json(
        retry_context_path.read_bytes()
    )
    assert retry_context.attempt_id == retry_attempt.attempt_id


def test_evaluation_context_write_recovers_after_pre_replace_crash(tmp_path, monkeypatch):
    worker = CampaignWorker.__new__(CampaignWorker)
    worker_module = __import__("importlib").import_module("bashgym.campaigns.worker")
    destination = tmp_path / "attempt-1" / AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
    payload = b'{"schema_version":"autoresearch_evaluation_context.v1"}'
    original_replace = __import__("os").replace

    def crash_before_replace(_source, _destination):
        raise OSError("injected pre-replace crash")

    monkeypatch.setattr(worker_module.os, "replace", crash_before_replace)
    with pytest.raises(OSError, match="pre-replace crash"):
        worker._write_evaluation_context(destination, payload)
    assert not destination.exists()

    destination.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="evaluation_context_mismatch"):
        worker._write_evaluation_context(destination, payload)
    destination.unlink()

    monkeypatch.setattr(worker_module.os, "replace", original_replace)
    worker._write_evaluation_context(destination, payload)
    assert destination.read_bytes() == payload


def test_persisted_evaluation_retry_restart_adopts_identical_remote_request(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = active_repository(database)
    training_plan = seed_validated_study(repository)
    evaluation_item = StagePlanItem(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        disposition=StageDisposition.REQUIRED,
        reason="Evaluate the immediately preceding sealed checkpoint.",
        input_contract={"fixture": "development-evaluation"},
        output_contract={"schema": "autoresearch_evaluation_evidence.v1"},
    )
    plan = StagePlan(items=(*training_plan.items, evaluation_item))
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET stage_plan_json = ?
            WHERE workspace_id = ? AND study_id = ?
            """,
            (plan.model_dump_json(), "workspace-a", "study-1"),
        )

    sealer = ArtifactSealer(b"w" * 32, key_version="worker-test-v1")
    scheduler = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="training-scheduler",
    )
    training = schedule(repository, scheduler, plan)
    claimed_training = repository.claim_next_action(
        scheduler.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=1),
    )
    assert claimed_training is not None
    assert claimed_training.attempt_id == training.attempt_id
    temporary = tmp_path / "training-output"
    (temporary / "final").mkdir(parents=True)
    (temporary / "final" / "config.json").write_text("{}", encoding="utf-8")
    (temporary / "final" / "weights.safetensors").write_bytes(b"weights")
    (temporary / "checkpoints" / "step-20").mkdir(parents=True)
    (temporary / "checkpoints" / "step-20" / "adapter_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (temporary / "checkpoints" / "step-20" / "adapter_model.safetensors").write_bytes(b"checkpoint")
    outputs = sealer.describe_outputs(
        temporary,
        {
            "checkpoints/step-20/adapter_config.json": "huggingface_checkpoint_file.v1",
            "checkpoints/step-20/adapter_model.safetensors": ("huggingface_checkpoint_file.v1"),
            "final/config.json": "huggingface_model_file.v1",
            "final/weights.safetensors": "huggingface_model_file.v1",
        },
    )
    training_manifest = SealedActionResult(
        workspace_id=claimed_training.workspace_id,
        campaign_id=claimed_training.campaign_id,
        study_id=claimed_training.study_id,
        action_id=claimed_training.action_id,
        attempt_id=claimed_training.attempt_id,
        manifest_revision=claimed_training.manifest_revision,
        candidate_digest=claimed_training.candidate_digest,
        input_digest=claimed_training.input_digest,
        claim_generation=claimed_training.claim_generation,
        executor_id="campaign-test-training-executor",
        executor_version="1",
        compute_profile_id="fake-local",
        started_at=START + timedelta(seconds=1),
        ended_at=START + timedelta(seconds=2),
        outcome="completed",
        exit_code=0,
        exit_reason="test_training_completed",
        outputs=outputs,
    )
    sealed_training = scheduler.sealed_path(claimed_training)
    sealer.seal(temporary, sealed_training, training_manifest)
    verified_training = scheduler._verify(claimed_training, sealed_training)
    repository.complete_from_seal(
        verified_training,
        sealed_training,
        worker_id=scheduler.worker_id,
        now=START + timedelta(seconds=2),
    )

    artifacts, _cursor, _has_more = repository.list_artifact_page("workspace-a", "campaign-1")
    model_artifacts = tuple(
        artifact
        for artifact in artifacts
        if artifact.producer_action_id == claimed_training.action_id
        and artifact.schema_name == "huggingface_model_file.v1"
    )
    assert len(model_artifacts) == 2
    assert {artifact.metadata["attempt_id"] for artifact in model_artifacts} == {
        claimed_training.attempt_id
    }
    checkpoint_artifacts = tuple(
        artifact
        for artifact in artifacts
        if artifact.producer_action_id == claimed_training.action_id
        and artifact.schema_name == "huggingface_checkpoint_file.v1"
    )
    assert len(checkpoint_artifacts) == 2
    assert {artifact.metadata["checkpoint_step"] for artifact in checkpoint_artifacts} == {20}
    assert {artifact.metadata["relative_path"] for artifact in checkpoint_artifacts} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }

    evaluator = tmp_path / "evaluate.py"
    dataset_path = tmp_path / "development.jsonl"
    key = tmp_path / "evaluation-key"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    dataset_path.write_text("{}\n", encoding="utf-8")
    key.write_text("fixture-key\n", encoding="utf-8")
    dataset = DatasetVersionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        dataset_id="dataset-a",
        dataset_version_id="dataset-version-a",
        source_uri="bashgym-remote-dataset://development-v1",
        content_digest=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="suite-a",
        name="Development evaluator",
        task_type="retrieval",
        dataset_version_id=dataset.dataset_version_id,
        metric_contract={"primary_metric": "score"},
        code_digest=hashlib.sha256(evaluator.read_bytes()).hexdigest(),
    )
    ledger = ExperimentLedgerRepository(database)
    ledger.initialize()
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="Evaluation retry fixture",
            owner_actor_id="operator-a",
        )
    )
    ledger.register_dataset(
        DatasetSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            dataset_id=dataset.dataset_id,
            display_name="Development dataset",
            task_type="retrieval",
        )
    )
    ledger.register_dataset_version(dataset)
    ledger.register_evaluation_suite(suite)
    heldout = RegisteredRemoteEvaluationDatasetSource(
        source_id="development-v1",
        compute_profile_id="ssh-gpu-lab",
        dataset_version_id=dataset.dataset_version_id,
        content_digest=dataset.content_digest,
        remote_dataset_path="/datasets/development.jsonl",
    )
    evaluation_stage = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=evaluator,
        script_sha256=suite.code_digest,
        input_files=(),
        input_sha256={},
        script_args=("--batch-size", "8"),
        output_paths=(AUTORESEARCH_EVALUATION_FILENAME,),
        budget_reservation=0.25,
    )
    evaluation_profile = ApprovedRemoteExecutorProfile(
        profile_id="evaluation-retry-profile-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(evaluation_stage,),
        registered_evaluation_dataset=heldout,
    )
    source_training, sealed_inputs = repository.sealed_full_training_launch_inputs(
        "workspace-a", "campaign-1", "study-1", 1
    )
    evaluation_executor = remote_executor_config(
        evaluation_profile,
        StageKind.DEVELOPMENT_EVALUATION,
        recipe_digest="e" * 64,
        sealed_stage_artifact_inputs=sealed_inputs,
        evaluation_suite=suite,
        dataset_version=dataset,
        source_training=source_training,
    )
    evaluation = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=1,
            stage=StageKind.DEVELOPMENT_EVALUATION,
            input_contract=evaluation_item.input_contract,
            candidate_digest=claimed_training.candidate_digest,
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="ssh_remote",
            executor_config=evaluation_executor,
        ),
        scheduler.leader,
        expected_campaign_version=repository.get_campaign("workspace-a", "campaign-1").version,
        now=START + timedelta(seconds=3),
    )
    first_evaluation_attempt = repository.claim_next_action(
        scheduler.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(seconds=4),
    )
    assert first_evaluation_attempt is not None
    assert first_evaluation_attempt.attempt_id == evaluation.attempt_id
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_attempts SET status = 'failed' WHERE attempt_id = ?",
            (evaluation.attempt_id,),
        )
        connection.execute(
            "UPDATE campaign_actions SET status = 'failed' WHERE action_id = ?",
            (evaluation.action_id,),
        )
        connection.execute(
            """
            UPDATE campaigns SET active_action_id = NULL, active_study_id = NULL,
                version = version + 1 WHERE workspace_id = ? AND campaign_id = ?
            """,
            (evaluation.workspace_id, evaluation.campaign_id),
        )
    retry = repository.retry_action(
        evaluation.workspace_id,
        evaluation.campaign_id,
        evaluation.action_id,
        expected_version=repository.get_campaign(
            evaluation.workspace_id, evaluation.campaign_id
        ).version,
        actor_id="operator-a",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="persisted-evaluation-retry",
        idempotency_key="persisted-evaluation-retry",
    )

    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    precrash_repository = CampaignRuntimeRepository(database)
    precrash_repository.initialize()
    precrash = CampaignWorker(
        precrash_repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="evaluation-before-restart",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={
            (evaluation_profile.compute_profile_id, evaluation_profile.target_contract_key): (
                evaluation_profile
            )
        },
    )
    restart_base = START + timedelta(days=90)
    precrash_leader = precrash._ensure_leader(restart_base)
    claimed_retry = precrash_repository.claim_next_action(
        precrash_leader,
        ttl=timedelta(seconds=15),
        now=restart_base,
    )
    assert claimed_retry is not None
    assert claimed_retry.attempt_id == retry.details["attempt_id"]
    assert claimed_retry.executor["evaluation_context_sha256"] != (
        first_evaluation_attempt.executor["evaluation_context_sha256"]
    )
    initial_request = precrash._remote_request(claimed_retry)
    initial_context = next(
        path
        for path in initial_request.input_files
        if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
    ).read_bytes()
    asyncio.run(adapter.launch(initial_request))
    assert (
        precrash_repository.get_remote_run(claimed_retry.workspace_id, claimed_retry.attempt_id)
        is None
    )

    successor_repository = CampaignRuntimeRepository(database)
    successor_repository.initialize()
    successor = CampaignWorker(
        successor_repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="evaluation-after-restart",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={
            (evaluation_profile.compute_profile_id, evaluation_profile.target_contract_key): (
                evaluation_profile
            )
        },
    )
    assert successor.run_once(now=restart_base + timedelta(seconds=17)) == "remote_running"
    adopted_request = adapter.last_request
    assert adopted_request.request_digest == initial_request.request_digest
    adopted_context = next(
        path
        for path in adopted_request.input_files
        if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
    ).read_bytes()
    assert adopted_context == initial_context
    assert adapter.launch_count == 1
    assert adapter.discover_count >= 1
    remote = successor_repository.get_remote_run(
        claimed_retry.workspace_id, claimed_retry.attempt_id
    )
    assert remote is not None
    assert remote.identity == adapter.identity


def test_runtime_model_manifest_input_digest_is_order_independent(tmp_path):
    first = tmp_path / "config.json"
    second = tmp_path / "weights.safetensors"
    first.write_text("{}", encoding="utf-8")
    second.write_bytes(b"weights")
    inputs = tuple(
        SealedStageArtifactInput(
            campaign_artifact_id=f"artifact-{path.stem}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            schema_name="huggingface_model_file.v1",
            local_sealed_path=path,
            remote_relative_path=f"model/{path.name}",
        )
        for path in (first, second)
    )

    assert CampaignRuntimeRepository._evaluation_model_manifest_digest(inputs) == (
        CampaignRuntimeRepository._evaluation_model_manifest_digest(tuple(reversed(inputs)))
    )
    assert CampaignRuntimeRepository._evaluation_model_manifest_digest(inputs) == (
        canonical_model_manifest_digest(inputs)
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


def activate_controller_live_study(repository, plan, *, training_recipe=None) -> None:
    proposal_payload = {
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": training_recipe
        or {
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
    artifact_root = tmp_path / "artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    assert adapter.stream_sources == ["training_metrics.jsonl"]
    running = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert running.status == AttemptStatus.RUNNING
    assert repository.get_remote_run("workspace-a", scheduled.attempt_id) is not None
    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"
    assert adapter.stream_sources == ["training_metrics.jsonl", "training_metrics.jsonl"]

    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert adapter.launch_count == 1
    assert adapter.collect_count == 1
    assert completed.sealed_result_uri.startswith(
        f"bashgym-remote-seal://ssh-gpu-lab/{completed.attempt_id}/sha256/"
    )
    assert not artifact_root.exists()
    artifacts = repository.list_artifacts("workspace-a", "campaign-1")
    assert artifacts
    assert all(
        artifact.uri.startswith(f"bashgym-remote-artifact://ssh-gpu-lab/{completed.attempt_id}/")
        for artifact in artifacts
    )
    source = repository.remote_resident_full_training_source(
        "workspace-a", "campaign-1", "study-1", 1
    )
    assert source.compute_profile_id == "ssh-gpu-lab"
    assert source.remote_model_path.endswith(f"/{completed.attempt_id}/final")
    assert [item.remote_relative_path for item in source.files] == ["model/config.json"]
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": pytest.approx(2 / 3600),
        "limit_delta": 0.0,
    }


def test_remote_completion_reuses_seal_after_worker_restart_boundary(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.COMPLETED,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)
    original_complete = repository.complete_from_seal
    repository.complete_from_seal = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SimulatedWorkerCrashError("worker crashed after remote seal")
    )

    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=START + timedelta(seconds=1))

    repository.complete_from_seal = original_complete
    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"
    completed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert adapter.persist_count == 1
    assert not (tmp_path / "artifacts").exists()


def test_successor_rebinds_remote_seal_after_terminal_lease_expiry(tmp_path):
    path = tmp_path / "campaigns.sqlite3"
    repository = active_repository(path)
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.COMPLETED,))
    sealer = ArtifactSealer(b"w" * 32, key_version="worker-test-v1")
    predecessor = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="worker-before",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, predecessor, plan, tmp_path)
    original_complete = repository.complete_from_seal
    repository.complete_from_seal = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SimulatedWorkerCrashError("worker crashed after remote seal")
    )

    with pytest.raises(SimulatedWorkerCrashError):
        predecessor.run_once(now=START + timedelta(seconds=1))

    repository.complete_from_seal = original_complete
    prior_envelope = adapter.seal_payload
    assert prior_envelope is not None
    prior_manifest = sealer.verify_envelope_bytes(
        prior_envelope,
        expected_attempt_id=scheduled.attempt_id,
        expected_claim_generation=1,
    )

    successor_repository = CampaignRuntimeRepository(path)
    successor_repository.initialize()
    successor = CampaignWorker(
        successor_repository,
        tmp_path / "artifacts",
        sealer,
        data_directory=tmp_path / "data-root",
        worker_id="worker-after",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles=predecessor.remote_executor_profiles,
    )

    assert successor.run_once(now=START + timedelta(hours=1, seconds=2)) == "completed"
    completed = successor_repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert completed.status == AttemptStatus.COMPLETED
    assert completed.claim_generation == 2
    assert adapter.launch_count == 1
    assert adapter.persist_count == 2
    assert adapter.seal_payload is not None
    rebound_manifest = sealer.verify_envelope_bytes(
        adapter.seal_payload,
        expected_attempt_id=scheduled.attempt_id,
        expected_claim_generation=2,
    )
    assert rebound_manifest.outputs == prior_manifest.outputs
    assert rebound_manifest.remote_process_identity == prior_manifest.remote_process_identity
    assert completed.sealed_result_uri.endswith(
        f"/sha256/{hashlib.sha256(adapter.seal_payload).hexdigest()}"
    )
    assert not (tmp_path / "artifacts").exists()


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


def test_first_generation_full_training_binds_registered_base_for_plain_recipe(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    activate_controller_live_study(repository, plan)
    profile = approved_remote_profile(tmp_path)
    base_model = RegisteredRemoteModelSource(
        source_id="registered-base-v1",
        compute_profile_id=profile.compute_profile_id,
        target_contract_key=profile.target_contract_key,
        model_digest=profile.target_model_digest,
        remote_model_path="/models/registered-base-v1",
    )
    profile = ApprovedRemoteExecutorProfile(
        **profile.model_dump(exclude={"profile_digest", "registered_base_model"}),
        registered_base_model=base_model,
    )
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
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

    assert worker.run_once(now=START) == "remote_running"
    request = adapter.last_request
    assert request is not None
    assert request.registered_base_model == base_model
    assert request.remote_resident_model is None
    assert request.script_args == (
        *profile.stages[0].script_args,
        "--model-dir",
        base_model.remote_model_path,
    )
    attempt = repository.list_attempts("workspace-a", "campaign-1")[0]
    assert attempt.executor["training_base_model"] == base_model.model_dump(mode="json")


def test_remote_executor_config_binds_first_generation_training_base(tmp_path):
    profile = approved_remote_profile(tmp_path)

    executor = remote_executor_config(
        profile,
        StageKind.FULL_TRAINING,
        recipe_digest="e" * 64,
    )

    assert executor["training_base_model"] == profile.registered_base_model.model_dump(mode="json")
    assert "remote_resident_model" not in executor


def test_worker_passes_typed_training_recipe_arguments_to_compute(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    recipe_args = (
        "--algorithm",
        "grpo",
        "--sft-enabled",
        "false",
        "--learning-rate",
        "2e-05",
        "--max-steps",
        "250",
        "--group-size",
        "16",
        "--temperature",
        "0.7",
        "--seed",
        "7",
    )
    activate_controller_live_study(
        repository,
        plan,
        training_recipe={
            "schema_version": "bashgym.tmax_composite_training_recipe.v1",
            "runtime": {"executor_kind": "registered_training"},
            "algorithm": "grpo",
            "sft_enabled": False,
            "learning_rate": 0.00002,
            "max_steps": 250,
            "group_size": 16,
            "temperature": 0.7,
            "seed": 7,
        },
    )
    profile = approved_remote_profile(tmp_path)
    profile = ApprovedRemoteExecutorProfile(
        **profile.model_dump(exclude={"profile_digest", "registered_base_model"}),
        registered_base_model=RegisteredRemoteModelSource(
            source_id="registered-base-v1",
            compute_profile_id=profile.compute_profile_id,
            target_contract_key=profile.target_contract_key,
            model_digest=profile.target_model_digest,
            remote_model_path="/models/registered-base-v1",
        ),
    )
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
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

    assert worker.run_once(now=START) == "remote_running"
    assert adapter.last_request is not None
    assert adapter.last_request.script_args == (
        *profile.stages[0].script_args,
        *recipe_args,
        "--model-dir",
        "/models/registered-base-v1",
    )
    attempt = repository.list_attempts("workspace-a", "campaign-1")[0]
    assert attempt.executor["recipe_script_args"] == list(recipe_args)


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


def test_unbound_registered_compute_rejects_remote_development_evaluation(tmp_path):
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

    with pytest.raises(CampaignPersistenceError, match="profile_material_invalid"):
        repository.next_action_spec(
            "workspace-a",
            "campaign-1",
            "study-1",
            executor_profiles={(profile.compute_profile_id, profile.target_contract_key): profile},
        )


def test_controller_launches_evaluation_only_baseline_from_registered_remote_model(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()
    create(repository)
    AutoResearchCampaignCore(repository).register(
        AutoResearchCampaignSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="score",
            metric_direction=MetricDirection.MAXIMIZE,
            ledger_project_id="project-a",
            evaluation_suite_id="suite-a",
            stop_rules=AutoResearchStopRules(
                max_attempts=3,
                budget_unit="gpu_hours",
                max_total_cost=3.0,
                minimum_improvement=0.0,
            ),
            created_at=START,
        )
    )
    transition(repository, CampaignTrigger.VALIDATE, 1, key="validate-baseline")
    transition(repository, CampaignTrigger.VALIDATION_PASSED, 2, key="ready-baseline")
    transition(repository, CampaignTrigger.START, 3, key="start-baseline")
    plan = seed_validated_study(repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    repository.register_autoresearch_proposal(
        AutoResearchProposalControl(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="proposal-study-1",
            role=ExperimentRole.BASELINE,
            created_at=START,
        )
    )

    evaluator = tmp_path / "evaluate.py"
    dataset_path = tmp_path / "development.jsonl"
    key = tmp_path / "evaluation-key"
    evaluator.write_text("print('evaluate')\n", encoding="utf-8")
    dataset_path.write_text("{}\n", encoding="utf-8")
    key.write_text("fixture-key\n", encoding="utf-8")
    dataset = DatasetVersionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        dataset_id="dataset-a",
        dataset_version_id="dataset-version-a",
        source_uri=str(dataset_path),
        content_digest=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="suite-a",
        name="Fixed development evaluator",
        task_type="retrieval",
        dataset_version_id=dataset.dataset_version_id,
        metric_contract={"primary_metric": "score"},
        code_digest=hashlib.sha256(evaluator.read_bytes()).hexdigest(),
    )
    ledger = ExperimentLedgerRepository(database)
    ledger.initialize()
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="Baseline evaluation fixture",
            owner_actor_id="operator-a",
        )
    )
    ledger.register_dataset(
        DatasetSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            dataset_id=dataset.dataset_id,
            display_name="Fixed development dataset",
            task_type="retrieval",
        )
    )
    ledger.register_dataset_version(dataset)
    ledger.register_evaluation_suite(suite)

    target_model_digest = canonical_hash(campaign().target_model.model_dump(mode="json"))
    base_model = remote_contracts.RegisteredRemoteModelSource(
        source_id="memexai-base-model",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        model_digest=target_model_digest,
        remote_model_path="/models/memexai-base-model",
    )
    heldout = RegisteredRemoteEvaluationDatasetSource(
        source_id="development-v1",
        compute_profile_id="ssh-gpu-lab",
        dataset_version_id=dataset.dataset_version_id,
        content_digest=dataset.content_digest,
        remote_dataset_path="/datasets/development.jsonl",
    )
    evaluation_stage = PinnedRemoteStageProfile(
        stage=StageKind.DEVELOPMENT_EVALUATION,
        script_path=evaluator,
        script_sha256=suite.code_digest,
        input_files=(),
        input_sha256={},
        script_args=("--batch-size", "8"),
        output_paths=(AUTORESEARCH_EVALUATION_FILENAME,),
        budget_reservation=0.25,
    )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="evaluation-baseline-profile-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=target_model_digest,
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=(evaluation_stage,),
        registered_base_model=base_model,
        registered_evaluation_dataset=heldout,
    )
    proposal_payload = {
        "dataset_recipe": {"schema_version": "recipe.v1"},
        "training_recipe": {"schema_version": "recipe.v1"},
        "evaluation_recipe": {
            "schema_version": "recipe.v1",
            "runtime": {"executor_kind": "registered_compute"},
        },
    }
    with repository._connection(immediate=True) as connection:
        manifest_row = connection.execute(
            """
            SELECT manifest_json FROM campaign_manifest_revisions
            WHERE workspace_id = ? AND campaign_id = ? AND revision = 1
            """,
            ("workspace-a", "campaign-1"),
        ).fetchone()
        manifest_payload = json.loads(manifest_row["manifest_json"])
        manifest_payload["evaluation_plan"].update(
            {
                "ledger_project_id": "project-a",
                "evaluation_suite_id": suite.evaluation_suite_id,
                "dataset_binding_id": dataset.dataset_version_id,
            }
        )
        connection.execute(
            """
            UPDATE campaign_manifest_revisions SET manifest_json = ?, manifest_hash = ?
            WHERE workspace_id = ? AND campaign_id = ? AND revision = 1
            """,
            (
                json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")),
                canonical_hash(manifest_payload),
                "workspace-a",
                "campaign-1",
            ),
        )
        connection.execute(
            "UPDATE campaign_studies SET stage_plan_json = ? WHERE study_id = ?",
            (plan.model_dump_json(), "study-1"),
        )
        connection.execute(
            "UPDATE campaign_proposals SET proposal_json = ? WHERE proposal_id = ?",
            (json.dumps(proposal_payload), "proposal-study-1"),
        )
        connection.execute(
            "UPDATE campaigns SET active_study_id = ? WHERE campaign_id = ?",
            ("study-1", "campaign-1"),
        )

    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="baseline-worker",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={("ssh-gpu-lab", "memexai-embedding-v1"): profile},
    )

    assert worker.run_once(now=START) == "remote_running"
    request = adapter.last_request
    assert request.registered_base_model == base_model
    assert request.registered_evaluation_dataset == heldout
    assert request.input_files == (
        tmp_path
        / "data-root"
        / "campaigns"
        / "evaluation-contexts"
        / request.run_id
        / AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
    )
    assert request.source_training is None
    assert request.sealed_stage_artifact_inputs == ()
    assert request.script_args == (
        "--batch-size",
        "8",
        "--context",
        AUTORESEARCH_EVALUATION_CONTEXT_FILENAME,
        "--model-dir",
        base_model.remote_model_path,
        "--dataset",
        heldout.remote_dataset_path,
        "--output",
        AUTORESEARCH_EVALUATION_FILENAME,
    )
    attempt = repository.list_attempts("workspace-a", "campaign-1")[0]
    assert attempt.executor["evaluation_binding"] == {
        "ledger_project_id": "project-a",
        "evaluation_suite_id": suite.evaluation_suite_id,
        "evaluation_code_digest": suite.code_digest,
        "dataset_version_id": dataset.dataset_version_id,
        "dataset_content_digest": dataset.content_digest,
        "dataset_remote_path": heldout.remote_dataset_path,
        "dataset_remote_name": heldout.remote_dataset_path,
    }


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
    artifact_root = tmp_path / "artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_failed"
    failed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert failed.status == AttemptStatus.FAILED
    assert failed.sealed_result_uri.startswith(
        f"bashgym-remote-seal://ssh-gpu-lab/{failed.attempt_id}/sha256/"
    )
    assert not artifact_root.exists()
    assert repository.budget_totals("workspace-a", "campaign-1", "gpu_hours") == {
        "reserved": 0.0,
        "actual": pytest.approx(2 / 3600),
        "limit_delta": 0.0,
    }
    with repository._connection() as connection:
        study_status = connection.execute(
            "SELECT status FROM campaign_studies WHERE study_id = 'study-1'"
        ).fetchone()[0]
    assert study_status == StudyStatus.EXECUTION_FAILED.value


def test_killed_remote_attempt_is_classified_as_infrastructure(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.FAILED,), failed_exit_code=137)
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
    manifest = repository.get_attempt_result_manifest("workspace-a", scheduled.attempt_id)
    assert manifest.failure_class == FailureClass.INFRASTRUCTURE
    events = repository.list_events("workspace-a", "campaign-1")
    failed_events = [event for _, event in events if event.event_type == "campaign:action-failed"]
    assert failed_events[-1].payload["failure_class"] == "infrastructure"


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
    artifact_root = tmp_path / "artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
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
    assert cancelled.sealed_result_uri.startswith(
        f"bashgym-remote-seal://ssh-gpu-lab/{cancelled.attempt_id}/sha256/"
    )
    assert not artifact_root.exists()


def test_unlaunched_remote_cancellation_writes_no_controller_evidence(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    adapter = FakeRemoteAdapter(states=(RemoteRunState.RUNNING,))
    artifact_root = tmp_path / "artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
    )
    scheduled = schedule_remote(repository, worker, plan, tmp_path)
    claimed = repository.claim_next_action(
        worker.leader,
        ttl=timedelta(seconds=15),
        now=START + timedelta(milliseconds=1),
    )
    assert claimed is not None
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaigns SET status = ? WHERE campaign_id = ?",
            ("cancelling", "campaign-1"),
        )

    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_cancelled"

    cancelled = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert cancelled.status == AttemptStatus.CANCELLED
    assert cancelled.sealed_result_uri.startswith("bashgym-controller-state://")
    assert not artifact_root.exists()
    assert adapter.launch_count == 0


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
        manifest_row = connection.execute("""
            SELECT manifest_json FROM campaign_manifest_revisions
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1' AND revision = 1
            """).fetchone()
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
    worker._persist_development_evaluation_evidence = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(SimulatedWorkerCrashError("crash after development seal"))
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
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id=work["work_id"],
        expected_campaign_revision=1,
        expected_version=1,
        expected_state="pending",
        principal=reviewer,
        correlation_id="worker-review-claim",
        idempotency_key=work["claim_idempotency_key"],
        now=START + timedelta(seconds=18),
    )
    oversight.submit(
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        work_id=work["work_id"],
        expected_campaign_revision=1,
        expected_version=2,
        expected_rubric_version=1,
        decision="no_material_difference",
        rationale="Equivalent under the blinded rubric.",
        principal=reviewer,
        correlation_id="worker-review-submit",
        idempotency_key=claimed.queue["items"][0]["submit_idempotency_key"],
        now=START + timedelta(seconds=19),
    )
    with repository._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_gate_decisions").fetchone()[0] == 1


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
rows_path = args.output_dir / 'domain_retrieval-retrieval_eval_queries.jsonl'
rows_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\\n' for row in rows), encoding='utf-8')
manifest = {
    'model_footprint_bytes': 4321,
    'runs': {'domain_retrieval': {'median_query_latency_ms': 12.5}},
}
(args.output_dir / 'query_format_ablation_manifest.json').write_text(
    json.dumps(manifest, sort_keys=True), encoding='utf-8'
)
""".strip() + "\n",
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
                "representation_contract": {"query_prefix_mode": "domain_retrieval"},
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
                    "query_prefix_mode": "domain_retrieval",
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
    envelope = json.loads((sealed / SEAL_FILENAME).read_text(encoding="utf-8"))
    output_schemas = {
        output["path"]: output["schema_name"] for output in envelope["manifest"]["outputs"]
    }
    assert (
        output_schemas["scoring/query_format_ablation_manifest.json"]
        == "query_format_ablation_manifest.v2"
    )
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


def test_reconcile_skips_remote_attempt_leased_by_a_live_foreign_worker(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    predecessor = make_worker(repository, tmp_path, "worker-a")
    schedule_remote(repository, predecessor, plan, tmp_path)
    claimed = repository.claim_next_action(
        predecessor.leader, ttl=timedelta(hours=1), now=START + timedelta(seconds=1)
    )
    assert claimed is not None
    assert claimed.executor.get("kind") == "ssh_remote"

    successor = make_worker(repository, tmp_path, "worker-b")
    later = START + timedelta(minutes=2)
    successor._ensure_leader(later)

    assert successor.reconcile_once(now=later) is None

    refreshed = {item.attempt_id: item for item in repository.list_unfinished_attempts()}[
        claimed.attempt_id
    ]
    assert refreshed.lease_owner == claimed.lease_owner
    assert refreshed.claim_generation == claimed.claim_generation


def test_find_reusable_completion_returns_newest_completed_match(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    key = "d" * 64
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"
    scheduled = repository.schedule_action_under_leader(
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
            fake_steps=6,
            result_key=key,
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )
    assert (
        repository.find_reusable_completion(
            "workspace-a", key, stage=StageKind.FULL_TRAINING, exclude_action_id="none"
        )
        is None
    )

    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"

    found = repository.find_reusable_completion(
        "workspace-a", key, stage=StageKind.FULL_TRAINING, exclude_action_id="none"
    )
    assert found is not None
    assert found.attempt.attempt_id == scheduled.attempt_id
    assert found.manifest.attempt_id == scheduled.attempt_id
    assert set(found.artifact_metadata_by_path) == {
        output.path for output in found.manifest.outputs
    }
    for value in found.artifact_metadata_by_path.values():
        assert "attempt_id" not in value

    # Give one output real metadata beyond the attempt_id every artifact carries, so
    # the mapping above cannot be satisfied by the {} fallback alone.
    target_path = found.manifest.outputs[0].path
    target_uri = _artifact_reference(str(found.attempt.sealed_result_uri), target_path)
    with repository._connection(immediate=True) as connection:
        connection.execute(
            "UPDATE campaign_artifacts SET metadata_json = ? WHERE workspace_id = ? AND uri = ?",
            (
                json.dumps({"attempt_id": found.attempt.attempt_id, "checkpoint_step": 3}),
                "workspace-a",
                target_uri,
            ),
        )
    enriched = repository.find_reusable_completion(
        "workspace-a", key, stage=StageKind.FULL_TRAINING, exclude_action_id="none"
    )
    assert enriched is not None
    assert enriched.artifact_metadata_by_path[target_path] == {"checkpoint_step": 3}

    assert (
        repository.find_reusable_completion(
            "workspace-a",
            key,
            stage=StageKind.FULL_TRAINING,
            exclude_action_id=scheduled.action_id,
        )
        is None
    )
    outputs = repository.completed_stage_outputs(
        "workspace-a", "campaign-1", "study-1", StageKind.FULL_TRAINING
    )
    assert outputs == found.manifest.outputs


def test_find_reusable_completion_prefers_the_newest_match_in_the_same_workspace(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plans = {
        study_id: seed_validated_study(repository, study_id, sequence=index)
        for index, study_id in enumerate(("study-1", "study-2"), start=1)
    }
    worker = make_worker(repository, tmp_path, "worker-a")
    key = "e" * 64
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"

    def schedule_with_key(study_id, version, now):
        return repository.schedule_action_under_leader(
            ActionSpec(
                workspace_id="workspace-a",
                campaign_id="campaign-1",
                study_id=study_id,
                stage_index=0,
                stage=StageKind.FULL_TRAINING,
                input_contract=plans[study_id].items[0].input_contract,
                candidate_digest=fake_digest(f"candidate:{study_id}"),
                manifest_revision=1,
                budget_unit="gpu_hours",
                budget_reservation=0.25,
                fake_steps=6,
                result_key=key,
            ),
            worker.leader,
            expected_campaign_version=version,
            now=now,
        )

    older = schedule_with_key("study-1", 4, START)
    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"

    newer = schedule_with_key("study-2", 6, START + timedelta(seconds=1))
    # Training never reuses, so this executes even though it shares a content key.
    assert worker.run_once(now=START + timedelta(seconds=30)) == "completed"

    found = repository.find_reusable_completion(
        "workspace-a", key, stage=StageKind.FULL_TRAINING, exclude_action_id="none"
    )
    assert found is not None
    assert found.attempt.attempt_id == newer.attempt_id

    excluding_newer = repository.find_reusable_completion(
        "workspace-a", key, stage=StageKind.FULL_TRAINING, exclude_action_id=newer.action_id
    )
    assert excluding_newer is not None
    assert excluding_newer.attempt.attempt_id == older.attempt_id


def test_completed_stage_outputs_filters_by_study_and_stage(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plans = {
        study_id: seed_validated_study(repository, study_id, sequence=index)
        for index, study_id in enumerate(("study-1", "study-2"), start=1)
    }
    worker = make_worker(repository, tmp_path, "worker-a")
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"

    first = repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract=plans["study-1"].items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-1"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            fake_steps=6,
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )
    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"

    repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-2",
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract=plans["study-2"].items[0].input_contract,
            candidate_digest=fake_digest("candidate:study-2"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            fake_steps=6,
        ),
        worker.leader,
        expected_campaign_version=6,
        now=START + timedelta(seconds=1),
    )

    assert (
        repository.completed_stage_outputs(
            "workspace-a", "campaign-1", "study-1", StageKind.DATA_BUILD
        )
        is None
    )
    assert (
        repository.completed_stage_outputs(
            "workspace-a", "campaign-1", "study-2", StageKind.FULL_TRAINING
        )
        is None
    )

    outputs = repository.completed_stage_outputs(
        "workspace-a", "campaign-1", "study-1", StageKind.FULL_TRAINING
    )
    manifest = repository.get_attempt_result_manifest("workspace-a", first.attempt_id)
    assert outputs == manifest.outputs


def test_next_action_spec_carries_a_result_key_only_for_reusable_stages(tmp_path):
    def memoize_recipes(repository: CampaignRuntimeRepository) -> None:
        runtime = {"executor_kind": "fake", "memoize": True}
        with repository._connection(immediate=True) as connection:
            connection.execute(
                """
                UPDATE campaign_proposals SET proposal_json = ?
                WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
                  AND proposal_id = 'proposal-study-1'
                """,
                (
                    json.dumps(
                        {
                            "primary_variable": "data.mixture",
                            "dataset_recipe": {"schema_version": "dataset.v1", "runtime": runtime},
                            "training_recipe": {
                                "schema_version": "training.v1",
                                "runtime": runtime,
                            },
                            "evaluation_recipe": {
                                "schema_version": "evaluation.v1",
                                "runtime": runtime,
                            },
                        }
                    ),
                ),
            )

    repository = active_repository(tmp_path / "campaigns.sqlite3")
    seed_validated_study(repository, stage=StageKind.DATA_BUILD)
    memoize_recipes(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    assert worker.run_once(now=START) == "idle"

    spec = repository.next_action_spec(
        "workspace-a",
        "campaign-1",
        "study-1",
        executor_profiles=worker.remote_executor_profiles,
    )

    assert spec.stage == StageKind.DATA_BUILD
    assert spec.result_key is not None
    assert len(spec.result_key) == 64

    training = active_repository(tmp_path / "training.sqlite3")
    seed_validated_study(training, stage=StageKind.FULL_TRAINING)
    memoize_recipes(training)
    training_worker = make_worker(training, tmp_path / "training", "worker-b")
    assert training_worker.run_once(now=START) == "idle"

    training_spec = training.next_action_spec(
        "workspace-a",
        "campaign-1",
        "study-1",
        executor_profiles=training_worker.remote_executor_profiles,
    )

    assert training_spec.stage == StageKind.FULL_TRAINING
    assert training_spec.result_key is None

    # A reusable stage with an unresolved upstream edge carries no key either. Only the data
    # build is reusable, so the unresolved edge is declared rather than positional.
    unresolved_plan = StagePlan(
        items=(
            StagePlanItem(
                stage=StageKind.CONTRACT_EVALUATION,
                disposition=StageDisposition.REQUIRED,
                reason="Diagnose the failure cluster the data build targets.",
                input_contract={"fixture": "study-1"},
                output_contract={"schema": "autoresearch_diagnostic_evidence.v1"},
            ),
            StagePlanItem(
                stage=StageKind.DATA_BUILD,
                disposition=StageDisposition.REQUIRED,
                reason="Build data from the diagnostic the previous stage produced.",
                input_contract={"fixture": "study-1"},
                output_contract={"schema": "autoresearch_dataset_receipt.v1"},
                consumes=(StageKind.CONTRACT_EVALUATION,),
            ),
        )
    )
    assert unresolved_plan.consumed_stages(1) == (StageKind.CONTRACT_EVALUATION,)
    with training._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET stage_plan_json = ?, current_stage_index = 1
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND study_id = 'study-1'
            """,
            (unresolved_plan.model_dump_json(),),
        )

    unresolved_spec = training.next_action_spec(
        "workspace-a",
        "campaign-1",
        "study-1",
        executor_profiles=training_worker.remote_executor_profiles,
    )

    assert unresolved_spec.stage == StageKind.DATA_BUILD
    assert unresolved_spec.result_key is None


def seed_memoized_fake_data_builds(tmp_path):
    """One repository and worker whose two data-build studies share a content key."""

    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plans = {
        study_id: seed_validated_study(
            repository, study_id, sequence=index, stage=StageKind.DATA_BUILD
        )
        for index, study_id in enumerate(("study-1", "study-2"), start=1)
    }
    worker = make_worker(repository, tmp_path, "worker-a")
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"
    return repository, worker, plans


def schedule_fake_data_build(repository, worker, study_id, plan, now, *, result_key):
    """Schedule one locally sealed fake data build under an explicit content key."""

    return repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id=study_id,
            stage_index=0,
            stage=StageKind.DATA_BUILD,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest(f"candidate:{study_id}"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            fake_steps=6,
            result_key=result_key,
        ),
        worker.leader,
        expected_campaign_version=repository.get_campaign("workspace-a", "campaign-1").version,
        now=now,
    )


def test_local_reuse_rejects_a_tampered_producer_manifest(tmp_path):
    """A locally sealed producer row is bound to its sealed bytes before re-signing."""

    repository, worker, plans = seed_memoized_fake_data_builds(tmp_path)
    key = "e" * 64
    producer = schedule_fake_data_build(
        repository, worker, "study-1", plans["study-1"], START, result_key=key
    )
    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"

    def tamper(payload):
        payload["executor_id"] = "tampered-executor"
        payload["outputs"][0]["schema_name"] = "tampered_schema.v1"

    rewrite_result_manifest(repository, producer.attempt_id, tamper)
    consumer = schedule_fake_data_build(
        repository,
        worker,
        "study-2",
        plans["study-2"],
        START + timedelta(seconds=2),
        result_key=key,
    )

    with pytest.raises(ArtifactSealError, match="campaign_artifact_seal_invalid"):
        worker.run_once(now=START + timedelta(seconds=3))

    assert worker.executor.execution_count == 1
    reused = repository.get_attempt("workspace-a", consumer.attempt_id)
    assert reused.status != AttemptStatus.COMPLETED
    assert reused.sealed_result_uri is None
    with pytest.raises(RecordNotFoundError):
        repository.get_attempt_result_manifest("workspace-a", consumer.attempt_id)


def test_identical_data_build_is_reused_across_studies_without_execution(tmp_path):
    repository, worker, plans = seed_memoized_fake_data_builds(tmp_path)
    key = "e" * 64

    schedule_fake_data_build(repository, worker, "study-1", plans["study-1"], START, result_key=key)
    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"
    schedule_fake_data_build(
        repository,
        worker,
        "study-2",
        plans["study-2"],
        START + timedelta(seconds=2),
        result_key=key,
    )

    assert worker.run_once(now=START + timedelta(seconds=3)) == "reused"

    assert worker.executor.execution_count == 1
    attempts = repository.list_attempts("workspace-a", "campaign-1")
    completed = [item for item in attempts if item.status == AttemptStatus.COMPLETED]
    assert len(completed) == 2
    reused = next(item for item in completed if item.study_id == "study-2")
    source = next(item for item in completed if item.study_id == "study-1")
    manifest = repository.get_attempt_result_manifest("workspace-a", reused.attempt_id)
    source_manifest = repository.get_attempt_result_manifest("workspace-a", source.attempt_id)
    assert manifest.attempt_id == reused.attempt_id
    assert manifest.action_id == reused.action_id
    assert manifest.study_id == "study-2"
    assert manifest.remote_process_identity == {
        "kind": "reused",
        "reused_from_attempt_id": source.attempt_id,
        "reused_from_action_id": source.action_id,
        "compute_profile_id": source_manifest.compute_profile_id,
    }
    assert manifest.log_reference is None
    assert manifest.outputs == source_manifest.outputs
    assert reused.sealed_result_uri != source.sealed_result_uri
    reused_seal = Path(str(reused.sealed_result_uri))
    for output in manifest.outputs:
        assert (reused_seal / output.path).read_bytes() == (
            Path(str(source.sealed_result_uri)) / output.path
        ).read_bytes()
    worker.sealer.verify(
        reused_seal,
        expected_workspace_id="workspace-a",
        expected_attempt_id=reused.attempt_id,
        expected_action_id=reused.action_id,
    )
    with repository._connection() as connection:
        actual = connection.execute(
            """
            SELECT actual_delta FROM campaign_budget_ledger
            WHERE entry_id = ?
            """,
            (f"budget-settle-{reused.action_id}",),
        ).fetchone()[0]
    assert actual == 0.0
    events = repository.list_events("workspace-a", "campaign-1")
    reuse_events = [
        event
        for _, event in events
        if event.event_type == "campaign:action-completed"
        and event.payload.get("reused_from_attempt_id") == source.attempt_id
    ]
    assert len(reuse_events) == 1
    assert reuse_events[0].payload["attempt_id"] == reused.attempt_id


class FakeDataBuildRemoteAdapter(FakeRemoteAdapter):
    """Complete a remote DATA_BUILD with a dataset receipt and its generated rows."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        train = b'{"task":"terminal"}\n'
        validation = b'{"task":"terminal-validation"}\n'
        self.receipt = AutoResearchDatasetReceipt(
            files=(
                AutoResearchDatasetFile(
                    path="dataset/train.jsonl",
                    sha256=hashlib.sha256(train).hexdigest(),
                    size_bytes=len(train),
                    split="train",
                    row_count=1,
                ),
                AutoResearchDatasetFile(
                    path="dataset/validation.jsonl",
                    sha256=hashlib.sha256(validation).hexdigest(),
                    size_bytes=len(validation),
                    split="validation",
                    row_count=1,
                ),
            ),
            generator={"kind": "nvidia_data_designer", "pipeline": "terminal_env_generation"},
            quality=AutoResearchDatasetQuality(
                generated_rows=3,
                accepted_rows=2,
                deterministic_verified_rows=2,
                verification_failed_rows=1,
                duplicate_rows_removed=0,
                contamination_rows_removed=0,
                verifier_digest="e" * 64,
            ),
        )
        self.payloads = {
            "exit_code": b"0\n",
            "launch_manifest.json": b"{}",
            "training.log": b"complete\n",
            "dataset/train.jsonl": train,
            "dataset/validation.jsonl": validation,
            AUTORESEARCH_DATASET_RECEIPT_FILENAME: self.receipt.model_dump_json().encode(),
        }

    async def inventory_outputs(self, identity, request, *, observation):
        self.collect_count += 1
        return RemoteOutputInventory(
            compute_profile_id=identity.compute_profile_id,
            run_id=identity.run_id,
            files=tuple(
                RemoteOutputFile(
                    path=path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
                for path, payload in sorted(self.payloads.items())
            ),
        )

    async def read_output_bytes(
        self, identity, path, *, expected_sha256, expected_size_bytes, max_bytes
    ):
        del identity
        payload = self.payloads[path]
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        assert len(payload) == expected_size_bytes
        assert expected_size_bytes <= max_bytes
        return payload


def approved_data_build_profile(tmp_path):
    script = tmp_path / "build_data.py"
    config = tmp_path / "data-designer-config.json"
    key = tmp_path / "data-build-key"
    script.write_text("print('build data')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    key.write_text("test-only-key\n", encoding="utf-8")
    stage = PinnedRemoteStageProfile(
        stage=StageKind.DATA_BUILD,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(config,),
        input_sha256={config.name: hashlib.sha256(config.read_bytes()).hexdigest()},
        output_paths=(AUTORESEARCH_DATASET_RECEIPT_FILENAME, "dataset"),
        budget_unit="gpu_hours",
        budget_reservation=0.25,
    )
    return ApprovedRemoteExecutorProfile(
        profile_id="data-designer-v1",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key="memexai-embedding-v1",
        target_model_digest=canonical_hash(campaign().target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        remote_work_dir="~/bashgym-training",
        stages=(stage,),
    )


def bind_generated_dataset_project(
    repository, database, project_id="project-a", campaign_id="campaign-1"
):
    """Give the campaign manifest a ledger project so a data build can register its dataset."""

    ledger = ExperimentLedgerRepository(database)
    ledger.initialize()
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id=project_id,
            display_name="Generated dataset fixture",
            owner_actor_id="operator-a",
        )
    )
    with repository._connection(immediate=True) as connection:
        row = connection.execute(
            """
            SELECT manifest_json FROM campaign_manifest_revisions
            WHERE workspace_id = ? AND campaign_id = ? AND revision = 1
            """,
            ("workspace-a", campaign_id),
        ).fetchone()
        payload = json.loads(row["manifest_json"])
        payload["evaluation_plan"]["ledger_project_id"] = project_id
        connection.execute(
            """
            UPDATE campaign_manifest_revisions SET manifest_json = ?, manifest_hash = ?
            WHERE workspace_id = ? AND campaign_id = ? AND revision = 1
            """,
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                canonical_hash(payload),
                "workspace-a",
                campaign_id,
            ),
        )


def test_remote_data_build_seal_is_reused_without_launching_a_second_run(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = active_repository(database)
    plans = {
        study_id: seed_validated_study(
            repository, study_id, sequence=index, stage=StageKind.DATA_BUILD
        )
        for index, study_id in enumerate(("study-1", "study-2"), start=1)
    }
    bind_generated_dataset_project(repository, database)
    profile = approved_data_build_profile(tmp_path)
    adapter = FakeDataBuildRemoteAdapter(states=(RemoteRunState.RUNNING, RemoteRunState.COMPLETED))
    artifact_root = tmp_path / "artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={("ssh-gpu-lab", "memexai-embedding-v1"): profile},
    )
    key = "f" * 64
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"

    def schedule_with_key(study_id, now):
        return repository.schedule_action_under_leader(
            ActionSpec(
                workspace_id="workspace-a",
                campaign_id="campaign-1",
                study_id=study_id,
                stage_index=0,
                stage=StageKind.DATA_BUILD,
                input_contract=plans[study_id].items[0].input_contract,
                candidate_digest=fake_digest(f"candidate:{study_id}"),
                manifest_revision=1,
                budget_unit="gpu_hours",
                budget_reservation=0.25,
                executor_kind="ssh_remote",
                executor_config=remote_executor_config(
                    profile, StageKind.DATA_BUILD, recipe_digest="e" * 64
                ),
                result_key=key,
            ),
            worker.leader,
            expected_campaign_version=repository.get_campaign("workspace-a", "campaign-1").version,
            now=now,
        )

    source_attempt = schedule_with_key("study-1", START)
    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"
    reused_attempt = schedule_with_key("study-2", START + timedelta(seconds=3))

    assert worker.run_once(now=START + timedelta(seconds=4)) == "reused"

    assert adapter.launch_count == 1
    assert repository.get_remote_run("workspace-a", reused_attempt.attempt_id) is None
    assert repository.get_remote_run("workspace-a", source_attempt.attempt_id) is not None
    assert not artifact_root.exists()
    source = repository.get_attempt("workspace-a", source_attempt.attempt_id)
    reused = repository.get_attempt("workspace-a", reused_attempt.attempt_id)
    assert reused.status == AttemptStatus.COMPLETED
    assert reused.stage == StageKind.DATA_BUILD
    manifest = repository.get_attempt_result_manifest("workspace-a", reused.attempt_id)
    source_manifest = repository.get_attempt_result_manifest("workspace-a", source.attempt_id)
    assert manifest.remote_process_identity == {
        "kind": "reused",
        "reused_from_attempt_id": source.attempt_id,
        "reused_from_action_id": source.action_id,
        "compute_profile_id": source_manifest.compute_profile_id,
    }
    assert source_manifest.remote_process_identity["run_id"] == source.attempt_id
    assert manifest.log_reference is None
    assert manifest.resource_usage == ()
    assert manifest.outputs == source_manifest.outputs
    assert any(output.path == AUTORESEARCH_DATASET_RECEIPT_FILENAME for output in manifest.outputs)
    expected_digest = hashlib.sha256(worker.sealer.envelope_bytes(manifest)).hexdigest()
    assert reused.sealed_result_uri == (
        f"bashgym-remote-seal://ssh-gpu-lab/{reused.attempt_id}/sha256/{expected_digest}"
    )
    assert reused.sealed_result_uri != source.sealed_result_uri
    with repository._connection() as connection:
        settled = connection.execute(
            "SELECT actual_delta FROM campaign_budget_ledger WHERE entry_id = ?",
            (f"budget-settle-{reused.action_id}",),
        ).fetchone()[0]
    assert settled == 0.0


def approved_data_build_and_training_profile(tmp_path):
    """One approved profile whose data build and full training share a compute target."""

    data_build = approved_data_build_profile(tmp_path)
    script = tmp_path / "train.py"
    config = tmp_path / "trainer-config.json"
    script.write_text("print('train')\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    training = PinnedRemoteStageProfile(
        stage=StageKind.FULL_TRAINING,
        script_path=script,
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input_files=(config,),
        input_sha256={config.name: hashlib.sha256(config.read_bytes()).hexdigest()},
        output_paths=("final",),
        budget_unit="gpu_hours",
        budget_reservation=0.25,
    )
    return ApprovedRemoteExecutorProfile(
        **data_build.model_dump(
            exclude={"profile_digest", "stages", "registered_base_model"},
        ),
        stages=(*data_build.stages, training),
        registered_base_model=RegisteredRemoteModelSource(
            source_id="registered-base-v1",
            compute_profile_id="ssh-gpu-lab",
            target_contract_key="memexai-embedding-v1",
            model_digest=data_build.target_model_digest,
            remote_model_path="/models/registered-base-v1",
        ),
    )


SHARED_DATA_BUILD_INPUT = {"dataset_scope": "memexai-approved-training", "rows": 1000}


def seed_data_build_then_training_study(
    repository, study_id, *, sequence, campaign_id="campaign-1", input_contract=None
):
    """Seed a validated study whose full training consumes its own data build."""

    plan = seed_validated_study(
        repository,
        study_id,
        sequence=sequence,
        stage=StageKind.DATA_BUILD,
        campaign_id=campaign_id,
        input_contract=input_contract,
    )
    extended = StagePlan(
        items=(
            *plan.items,
            StagePlanItem(
                stage=StageKind.FULL_TRAINING,
                disposition=StageDisposition.REQUIRED,
                reason="Consume the study's data build without copying its rows.",
                input_contract={"fixture": study_id},
                output_contract={"schema": "training_metrics_jsonl.v1"},
                consumes=(StageKind.DATA_BUILD,),
            ),
        )
    )
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_studies SET stage_plan_json = ?
            WHERE workspace_id = ? AND campaign_id = ? AND study_id = ?
            """,
            (extended.model_dump_json(), "workspace-a", campaign_id, study_id),
        )
    return extended


def start_campaign(repository, campaign_id, *, campaign_manifest=None):
    """Create and start one more campaign in the same workspace."""

    value = campaign("workspace-a", campaign_id)
    if campaign_manifest is None:
        create(repository, value)
    else:
        repository.create_campaign(
            value,
            ManifestRevision(
                workspace_id="workspace-a",
                campaign_id=campaign_id,
                revision=1,
                manifest=campaign_manifest,
                actor_id="codex-agent",
                correlation_id=f"correlation-create-{campaign_id}",
            ),
            actor_id="codex-agent",
            credential_kind=CredentialKind.ACCESS,
            correlation_id=f"correlation-create-{campaign_id}",
            idempotency_key=f"create-{campaign_id}",
        )
    for trigger, version, key in (
        (CampaignTrigger.VALIDATE, 1, "validate"),
        (CampaignTrigger.VALIDATION_PASSED, 2, "ready"),
        (CampaignTrigger.START, 3, "start"),
    ):
        repository.transition_campaign(
            "workspace-a",
            campaign_id,
            trigger,
            expected_version=version,
            actor_id="codex-agent",
            credential_kind=CredentialKind.ACCESS,
            correlation_id=f"correlation-{key}-{campaign_id}",
            idempotency_key=f"{key}-{campaign_id}",
        )


def schedule_remote_data_build(
    repository,
    worker,
    profile,
    campaign_id,
    study_id,
    plan,
    now,
    *,
    recipe_digest="e" * 64,
):
    """Schedule one data build under the content key its own campaign manifest derives."""

    executor_config = remote_executor_config(
        profile, StageKind.DATA_BUILD, recipe_digest=recipe_digest
    )
    return repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id=campaign_id,
            study_id=study_id,
            stage_index=0,
            stage=StageKind.DATA_BUILD,
            input_contract=plan.items[0].input_contract,
            candidate_digest=fake_digest(f"candidate:{study_id}"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="ssh_remote",
            executor_config=executor_config,
            result_key=derived_data_build_result_key(
                repository,
                campaign_id,
                stage_input=plan.items[0].input_contract,
                executor_config=executor_config,
                recipe_digest=recipe_digest,
            ),
        ),
        worker.leader,
        expected_campaign_version=repository.get_campaign("workspace-a", campaign_id).version,
        now=now,
    )


def reuse_remote_data_build(tmp_path, *, consumer_campaign_id="campaign-1", consumer_manifest=None):
    """Complete one remote data build, then let a second study reuse its sealed result."""

    database = tmp_path / "campaigns.sqlite3"
    repository = active_repository(database)
    producer_plan = seed_validated_study(
        repository,
        "study-1",
        sequence=1,
        stage=StageKind.DATA_BUILD,
        input_contract=SHARED_DATA_BUILD_INPUT,
    )
    bind_generated_dataset_project(repository, database)
    if consumer_campaign_id != "campaign-1":
        start_campaign(repository, consumer_campaign_id, campaign_manifest=consumer_manifest)
    consumer_plan = seed_data_build_then_training_study(
        repository,
        "study-2",
        sequence=2,
        campaign_id=consumer_campaign_id,
        input_contract=SHARED_DATA_BUILD_INPUT,
    )
    profile = approved_data_build_and_training_profile(tmp_path)
    adapter = FakeDataBuildRemoteAdapter(states=(RemoteRunState.RUNNING, RemoteRunState.COMPLETED))
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={("ssh-gpu-lab", "memexai-embedding-v1"): profile},
    )
    if worker.leader is None:
        assert worker.run_once(now=START) == "idle"
    producer = schedule_remote_data_build(
        repository, worker, profile, "campaign-1", "study-1", producer_plan, START
    )
    assert worker.run_once(now=START + timedelta(seconds=1)) == "remote_running"
    assert worker.run_once(now=START + timedelta(seconds=2)) == "completed"
    consumer = schedule_remote_data_build(
        repository,
        worker,
        profile,
        consumer_campaign_id,
        "study-2",
        consumer_plan,
        START + timedelta(seconds=3),
    )
    assert worker.run_once(now=START + timedelta(seconds=4)) == "reused"
    assert adapter.launch_count == 1
    return SimpleNamespace(
        repository=repository,
        worker=worker,
        adapter=adapter,
        profile=profile,
        producer=producer,
        consumer=consumer,
        consumer_campaign_id=consumer_campaign_id,
        consumer_plan=consumer_plan,
    )


def schedule_resident_dataset_training(fixture, resident, now):
    """Schedule the consuming study's training against one resolved dataset source."""

    return fixture.repository.schedule_action_under_leader(
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id=fixture.consumer_campaign_id,
            study_id="study-2",
            stage_index=1,
            stage=StageKind.FULL_TRAINING,
            input_contract=fixture.consumer_plan.items[1].input_contract,
            candidate_digest=fake_digest("candidate:study-2"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="ssh_remote",
            executor_config=remote_executor_config(
                fixture.profile,
                StageKind.FULL_TRAINING,
                recipe_digest="e" * 64,
                remote_resident_dataset=resident,
            ),
        ),
        fixture.worker.leader,
        expected_campaign_version=fixture.repository.get_campaign(
            "workspace-a", fixture.consumer_campaign_id
        ).version,
        now=now,
    )


def test_training_after_a_reused_data_build_launches_against_the_producing_run(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    worker = fixture.worker
    adapter = fixture.adapter

    resident = repository.remote_resident_data_build_source(
        "workspace-a", "campaign-1", "study-2", 1
    )
    assert resident.attempt_id == fixture.producer.attempt_id
    assert resident.action_id == fixture.producer.action_id
    assert resident.study_id == "study-1"
    assert resident.stage_index == 0
    assert resident.remote_dataset_path == (
        f"/home/trainer/bashgym-training/{fixture.producer.attempt_id}/dataset"
    )
    assert resident.content_digest == adapter.receipt.content_digest

    training_attempt = schedule_resident_dataset_training(
        fixture, resident, START + timedelta(seconds=5)
    )
    # Holding the training run at RUNNING keeps this test on the launch inputs rather
    # than output ingestion.
    adapter.states = [RemoteRunState.RUNNING]

    assert worker.run_once(now=START + timedelta(seconds=6)) == "remote_running"

    request = adapter.last_request
    assert request.run_id == training_attempt.attempt_id
    assert request.remote_resident_dataset == resident
    assert request.remote_resident_dataset.remote_dataset_path == (
        f"/home/trainer/bashgym-training/{fixture.producer.attempt_id}/dataset"
    )
    assert adapter.launch_count == 2
    assert repository.get_remote_run("workspace-a", fixture.consumer.attempt_id) is None
    assert repository.get_remote_run("workspace-a", training_attempt.attempt_id) is not None

    names_the_reusing_attempt = training_attempt.model_copy(deep=True)
    names_the_reusing_attempt.executor["remote_resident_dataset"][
        "attempt_id"
    ] = fixture.consumer.attempt_id
    with pytest.raises(RuntimeError, match="campaign_remote_training_dataset_invalid"):
        worker._remote_request(names_the_reusing_attempt)

    names_another_directory = training_attempt.model_copy(deep=True)
    names_another_directory.executor["remote_resident_dataset"][
        "remote_dataset_path"
    ] = "/home/trainer/bashgym-training/elsewhere/dataset"
    with pytest.raises(RuntimeError, match="campaign_remote_training_dataset_invalid"):
        worker._remote_request(names_another_directory)

    consumes_a_foreign_stage_edge = training_attempt.model_copy(update={"stage_index": 2})
    with pytest.raises(RuntimeError, match="campaign_remote_training_dataset_invalid"):
        worker._remote_request(consumes_a_foreign_stage_edge)


def test_training_preflight_rejects_a_tampered_hop_in_the_reuse_chain(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    worker = fixture.worker
    resident = fixture.repository.remote_resident_data_build_source(
        "workspace-a", "campaign-1", "study-2", 1
    )
    training_attempt = schedule_resident_dataset_training(
        fixture, resident, START + timedelta(seconds=5)
    )
    assert worker._remote_request(training_attempt).remote_resident_dataset == resident

    # The producing manifest still describes the same outputs, dataset, and remote run, so
    # resolution still succeeds; only its sealed envelope stops matching its seal reference.
    rewrite_result_manifest(
        fixture.repository,
        fixture.producer.attempt_id,
        lambda payload: payload.update({"exit_reason": "rewritten after sealing"}),
    )
    assert (
        fixture.repository.remote_resident_data_build_source(
            "workspace-a", "campaign-1", "study-2", 1
        )
        == resident
    )

    with pytest.raises(RuntimeError, match="campaign_remote_training_dataset_invalid"):
        worker._remote_request(training_attempt)


def test_reuse_link_resolves_only_to_a_completed_same_stage_source(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    consumer = repository.get_attempt("workspace-a", fixture.consumer.attempt_id)
    producer = repository.get_attempt("workspace-a", fixture.producer.attempt_id)

    chain = repository.reuse_source_chain(consumer)
    assert [item.attempt_id for item, _manifest in chain] == [producer.attempt_id]
    assert [manifest.attempt_id for _item, manifest in chain] == [producer.attempt_id]
    assert repository.reuse_source_chain(producer) == ()
    assert repository.reuse_source_attempt(consumer).attempt_id == producer.attempt_id
    assert repository.reuse_source_attempt(producer).attempt_id == producer.attempt_id
    assert (
        repository.completed_data_build_attempt(
            "workspace-a", "campaign-1", "study-2", 0
        ).attempt_id
        == consumer.attempt_id
    )
    with pytest.raises(CampaignPersistenceError, match="campaign_training_dataset_missing"):
        repository.completed_data_build_attempt("workspace-a", "campaign-1", "study-2", 1)

    set_reuse_link(repository, consumer.attempt_id, "attempt-does-not-exist")
    with pytest.raises(CampaignPersistenceError, match="campaign_reuse_source_invalid"):
        repository.reuse_source_attempt(consumer)

    set_reuse_link(repository, consumer.attempt_id, consumer.attempt_id)
    with pytest.raises(CampaignPersistenceError, match="campaign_reuse_source_invalid"):
        repository.reuse_source_attempt(consumer)

    pending_plan = seed_validated_study(
        repository,
        "study-3",
        sequence=3,
        stage=StageKind.DATA_BUILD,
        input_contract=SHARED_DATA_BUILD_INPUT,
    )
    pending = schedule_remote_data_build(
        repository,
        fixture.worker,
        fixture.profile,
        "campaign-1",
        "study-3",
        pending_plan,
        START + timedelta(seconds=5),
    )
    set_reuse_link(repository, consumer.attempt_id, pending.attempt_id)
    with pytest.raises(CampaignPersistenceError, match="campaign_reuse_source_invalid"):
        repository.reuse_source_attempt(consumer)

    set_reuse_link(repository, consumer.attempt_id, producer.attempt_id)
    assert repository.reuse_source_attempt(consumer).attempt_id == producer.attempt_id
    set_action_status(repository, producer.action_id, ActionStatus.FAILED.value)
    with pytest.raises(CampaignPersistenceError, match="campaign_reuse_source_invalid"):
        repository.reuse_source_attempt(consumer)
    set_action_status(repository, producer.action_id, ActionStatus.COMPLETED.value)

    other_stage = consumer.model_copy(update={"stage": StageKind.FULL_TRAINING})
    with pytest.raises(CampaignPersistenceError, match="campaign_reuse_source_invalid"):
        repository.reuse_source_attempt(other_stage)


def test_repeated_reuse_of_one_build_records_one_hop_to_the_executing_attempt(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    reusing = [fixture.consumer]
    for index, study_id in enumerate(("study-3", "study-4"), start=3):
        plan = seed_data_build_then_training_study(
            repository, study_id, sequence=index, input_contract=SHARED_DATA_BUILD_INPUT
        )
        reusing.append(
            schedule_remote_data_build(
                repository,
                fixture.worker,
                fixture.profile,
                "campaign-1",
                study_id,
                plan,
                START + timedelta(seconds=index + 2),
            )
        )
        assert fixture.worker.run_once(now=START + timedelta(seconds=index + 3)) == "reused"

    assert fixture.adapter.launch_count == 1
    for attempt in reusing:
        manifest = repository.get_attempt_result_manifest("workspace-a", attempt.attempt_id)
        assert manifest.remote_process_identity["reused_from_attempt_id"] == (
            fixture.producer.attempt_id
        )
        assert manifest.remote_process_identity["reused_from_action_id"] == (
            fixture.producer.action_id
        )
        assert len(repository.reuse_source_chain(attempt)) == 1
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {
        attempt.attempt_id: fixture.producer.attempt_id for attempt in reusing
    }
    assert repository.reuse_source_links("workspace-a", "campaign-1") == {
        attempt.attempt_id: fixture.producer.attempt_id for attempt in reusing
    }
    assert repository.resolved_reuse_links("workspace-a", "campaign-2") == {}


def test_collapsed_map_follows_a_legacy_multi_hop_reuse_chain(tmp_path):
    """Links written before write-time collapse name the matched attempt, not the producer."""

    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    third_plan = seed_data_build_then_training_study(
        repository, "study-3", sequence=3, input_contract=SHARED_DATA_BUILD_INPUT
    )
    third = schedule_remote_data_build(
        repository,
        fixture.worker,
        fixture.profile,
        "campaign-1",
        "study-3",
        third_plan,
        START + timedelta(seconds=5),
    )
    assert fixture.worker.run_once(now=START + timedelta(seconds=6)) == "reused"
    set_reuse_link(repository, third.attempt_id, fixture.consumer.attempt_id)

    chain = repository.reuse_source_chain(repository.get_attempt("workspace-a", third.attempt_id))

    assert [item.attempt_id for item, _manifest in chain] == [
        fixture.consumer.attempt_id,
        fixture.producer.attempt_id,
    ]
    assert repository.reuse_source_links("workspace-a", "campaign-1") == {
        fixture.consumer.attempt_id: fixture.producer.attempt_id,
        third.attempt_id: fixture.consumer.attempt_id,
    }
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {
        fixture.consumer.attempt_id: fixture.producer.attempt_id,
        third.attempt_id: fixture.producer.attempt_id,
    }


def test_reuse_lookup_prefers_the_attempt_that_executed(tmp_path):
    """The newest match is often itself a reusing attempt; the executed row wins."""

    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    consumer = repository.get_attempt("workspace-a", fixture.consumer.attempt_id)
    producer = repository.get_attempt("workspace-a", fixture.producer.attempt_id)
    assert consumer.updated_at > producer.updated_at

    match = repository.find_reusable_completion(
        "workspace-a",
        str(producer.result_key),
        stage=StageKind.DATA_BUILD,
        exclude_action_id="action-not-scheduled",
    )

    assert match is not None
    assert match.attempt.attempt_id == producer.attempt_id


def test_a_damaged_link_on_the_only_match_falls_through_to_execution(tmp_path, caplog):
    """A cache miss costs one real launch, never a worker crash."""

    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    set_reuse_link(repository, fixture.consumer.attempt_id, "attempt-does-not-exist")
    set_action_status(repository, fixture.producer.action_id, ActionStatus.FAILED.value)
    third_plan = seed_data_build_then_training_study(
        repository, "study-3", sequence=3, input_contract=SHARED_DATA_BUILD_INPUT
    )
    third = schedule_remote_data_build(
        repository,
        fixture.worker,
        fixture.profile,
        "campaign-1",
        "study-3",
        third_plan,
        START + timedelta(seconds=5),
    )
    fixture.adapter.states = [RemoteRunState.RUNNING]

    with caplog.at_level(logging.WARNING, logger="bashgym.campaigns.worker"):
        assert fixture.worker.run_once(now=START + timedelta(seconds=6)) == "remote_running"

    assert fixture.adapter.launch_count == 2
    assert repository.get_remote_run("workspace-a", third.attempt_id) is not None
    assert fixture.consumer.attempt_id in caplog.text
    assert "campaign_reuse_source_invalid" in caplog.text


def test_reuse_links_drop_from_the_collapsed_map_without_failing_the_projection(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    consumer = fixture.consumer
    stored = {consumer.attempt_id: fixture.producer.attempt_id}

    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == stored

    set_reuse_link(repository, consumer.attempt_id, consumer.attempt_id)
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {}
    assert repository.reuse_source_links("workspace-a", "campaign-1") == {
        consumer.attempt_id: consumer.attempt_id
    }

    set_reuse_link(repository, consumer.attempt_id, "attempt-does-not-exist")
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {}
    assert repository.reuse_source_links("workspace-a", "campaign-1") == {
        consumer.attempt_id: "attempt-does-not-exist"
    }

    set_reuse_link(repository, consumer.attempt_id, fixture.producer.attempt_id)
    set_action_status(repository, fixture.producer.action_id, ActionStatus.FAILED.value)
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {}
    set_action_status(repository, fixture.producer.action_id, ActionStatus.COMPLETED.value)
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == stored


def test_reused_attempt_projects_its_producer_through_the_campaign_service(tmp_path):
    fixture = reuse_remote_data_build(tmp_path)
    service = CampaignService(fixture.repository, export_root=tmp_path / "reports")
    actor = principal(fixture.repository)

    projected = {
        item.attempt_id: item.reused_from_attempt_id
        for item in service.attempts("workspace-a", "campaign-1", actor)
    }

    assert projected[fixture.consumer.attempt_id] == fixture.producer.attempt_id
    assert projected[fixture.producer.attempt_id] is None


def test_remote_reuse_rejects_a_tampered_producer_manifest(tmp_path):
    """A stored producer row is re-signed under the consumer only after it verifies."""

    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    third_plan = seed_data_build_then_training_study(
        repository, "study-3", sequence=3, input_contract=SHARED_DATA_BUILD_INPUT
    )
    third = schedule_remote_data_build(
        repository,
        fixture.worker,
        fixture.profile,
        "campaign-1",
        "study-3",
        third_plan,
        START + timedelta(seconds=5),
    )

    def tamper(payload):
        payload["exit_reason"] = "rewritten in storage"

    rewrite_result_manifest(repository, fixture.producer.attempt_id, tamper)

    with pytest.raises(ValueError, match="remote dataset seal mismatch"):
        fixture.worker.run_once(now=START + timedelta(seconds=6))

    assert fixture.adapter.launch_count == 1
    reused = repository.get_attempt("workspace-a", third.attempt_id)
    assert reused.status != AttemptStatus.COMPLETED
    assert reused.sealed_result_uri is None
    with pytest.raises(RecordNotFoundError):
        repository.get_attempt_result_manifest("workspace-a", third.attempt_id)


def test_reuse_across_campaigns_ignores_budgets_and_the_evaluation_plan(tmp_path):
    """Manifest sections a data build never reads do not partition the content key."""

    relaxed = manifest().model_copy(
        update={
            "budget_limits": {"gpu_hours": 40.0},
            "evaluation_plan": {"development_query_set": "dev-99-v9"},
            "promotion_gates": {"mrr_at_10_delta_min": 0.5},
            "max_proposal_rounds": 9,
            "retention_days_failed": 7,
        }
    )
    fixture = reuse_remote_data_build(
        tmp_path, consumer_campaign_id="campaign-2", consumer_manifest=relaxed
    )
    repository = fixture.repository

    producer = repository.get_attempt("workspace-a", fixture.producer.attempt_id)
    consumer = repository.get_attempt("workspace-a", fixture.consumer.attempt_id)
    assert repository.get_manifest_revision("workspace-a", "campaign-1", 1).manifest != (
        repository.get_manifest_revision("workspace-a", "campaign-2", 1).manifest
    )
    assert producer.result_key == consumer.result_key
    assert fixture.adapter.launch_count == 1
    assert repository.resolved_reuse_links("workspace-a", "campaign-2") == {
        fixture.consumer.attempt_id: fixture.producer.attempt_id
    }


def test_a_different_dataset_recipe_derives_a_different_key_and_executes(tmp_path):
    """Recipe content is inside the key, so a changed build runs instead of reusing."""

    fixture = reuse_remote_data_build(tmp_path)
    repository = fixture.repository
    third_plan = seed_data_build_then_training_study(
        repository, "study-3", sequence=3, input_contract=SHARED_DATA_BUILD_INPUT
    )
    third = schedule_remote_data_build(
        repository,
        fixture.worker,
        fixture.profile,
        "campaign-1",
        "study-3",
        third_plan,
        START + timedelta(seconds=5),
        recipe_digest="d" * 64,
    )
    fixture.adapter.states = [RemoteRunState.RUNNING]

    assert third.result_key != fixture.producer.result_key
    assert fixture.worker.run_once(now=START + timedelta(seconds=6)) == "remote_running"
    assert fixture.adapter.launch_count == 2
    assert repository.get_remote_run("workspace-a", third.attempt_id) is not None


def test_reuse_across_campaigns_binds_the_producing_campaign_dataset(tmp_path):
    fixture = reuse_remote_data_build(tmp_path, consumer_campaign_id="campaign-2")
    repository = fixture.repository
    adapter = fixture.adapter

    assert repository.resolved_reuse_links("workspace-a", "campaign-2") == {
        fixture.consumer.attempt_id: fixture.producer.attempt_id
    }
    assert repository.resolved_reuse_links("workspace-a", "campaign-1") == {}
    resident = repository.remote_resident_data_build_source(
        "workspace-a", "campaign-2", "study-2", 1
    )
    assert resident.campaign_id == "campaign-1"
    assert resident.study_id == "study-1"
    assert resident.attempt_id == fixture.producer.attempt_id
    assert resident.action_id == fixture.producer.action_id
    assert resident.remote_dataset_path == (
        f"/home/trainer/bashgym-training/{fixture.producer.attempt_id}/dataset"
    )
    version = repository.get_dataset_version_spec(
        "workspace-a", "project-a", resident.dataset_version_id
    )
    assert version.metadata["producer_action_id"] == fixture.producer.action_id
    assert version.metadata["producer_attempt_id"] == fixture.producer.attempt_id
    assert version.content_digest == resident.content_digest

    training_attempt = schedule_resident_dataset_training(
        fixture, resident, START + timedelta(seconds=5)
    )
    adapter.states = [RemoteRunState.RUNNING]

    assert fixture.worker.run_once(now=START + timedelta(seconds=6)) == "remote_running"

    request = adapter.last_request
    assert request.run_id == training_attempt.attempt_id
    assert request.remote_resident_dataset == resident
    assert request.remote_resident_dataset.campaign_id == "campaign-1"
    assert adapter.launch_count == 2
    assert repository.get_remote_run("workspace-a", fixture.consumer.attempt_id) is None


def rewrite_executor_kind(repository, attempt_id: str, kind: str) -> None:
    """Store an executor kind that the local worker registry does not contain."""

    with repository._connection(immediate=True) as connection:
        row = connection.execute(
            "SELECT executor_json FROM campaign_attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        executor = json.loads(row[0])
        executor["kind"] = kind
        connection.execute(
            "UPDATE campaign_attempts SET executor_json=? WHERE attempt_id=?",
            (json.dumps(executor), attempt_id),
        )


def write_recipe_runtimes(repository, **runtimes):
    """Replace the seeded proposal with one whose named recipes carry the given runtimes."""

    recipes = {
        f"{name}_recipe": {"schema_version": f"{name}.v1"}
        for name in ("dataset", "training", "evaluation")
    }
    for name, runtime in runtimes.items():
        recipes[f"{name}_recipe"]["runtime"] = runtime
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_proposals SET proposal_json = ?
            WHERE workspace_id = 'workspace-a' AND campaign_id = 'campaign-1'
              AND proposal_id = 'proposal-study-1'
            """,
            (json.dumps({"primary_variable": "data.mixture", **recipes}),),
        )


def materialization_error(repository, tmp_path, worker_id="worker-a"):
    """Return the persistence error code raised while materializing the seeded study."""

    worker = make_worker(repository, tmp_path, worker_id)
    assert worker.run_once(now=START) == "idle"
    with pytest.raises(CampaignPersistenceError) as raised:
        repository.next_action_spec(
            "workspace-a",
            "campaign-1",
            "study-1",
            executor_profiles=worker.remote_executor_profiles,
        )
    return str(raised.value)


def test_next_action_spec_rejects_an_unregistered_recipe_executor(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    seed_validated_study(repository)
    write_recipe_runtimes(repository, training={"executor_kind": "mystery"})

    assert materialization_error(repository, tmp_path) == ("campaign_executor_kind_not_registered")


def test_next_action_spec_rejects_a_registered_kind_it_cannot_materialize(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    seed_validated_study(repository, stage=StageKind.DEVELOPMENT_EVALUATION)
    write_recipe_runtimes(repository, evaluation={"executor_kind": "development_evaluation"})

    assert materialization_error(repository, tmp_path) == (
        "campaign_executor_kind_not_materializable"
    )


def test_next_action_spec_rejects_a_registered_kind_on_an_unregistered_stage(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    seed_validated_study(repository, stage=StageKind.PROMOTION)
    write_recipe_runtimes(repository, evaluation={"executor_kind": "registered_compute"})

    assert materialization_error(repository, tmp_path) == "campaign_remote_stage_not_allowed"


def registry_with(*adapters):
    """Build a frozen registry from the built-in adapters plus the given extras."""

    defaults = build_default_registry()
    registry = ExecutorRegistry()
    for kind in defaults.kinds():
        registry.register(defaults.get(kind))
    for adapter in adapters:
        registry.register(adapter)
    registry.freeze()
    return registry


def test_worker_dispatches_through_the_executor_registry(tmp_path):
    ticked = []

    class RecordingAdapter:
        kind = "recording"
        allowed_stages = frozenset({StageKind.FULL_TRAINING})
        reuses_completed_results = False

        def tick(self, worker, attempt, *, now):
            ticked.append(attempt.attempt_id)
            return worker._fake_tick(attempt, now=now)

        def reconcile(self, worker, attempt, *, now):
            return None

        def repair_allowed(self):
            return True

    registry = registry_with(RecordingAdapter())
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        executor_registry=registry,
    )
    assert worker.run_once(now=START) == "idle"
    scheduled = repository.schedule_action_under_leader(
        ActionSpec.model_validate(
            {
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-1",
                "study_id": "study-1",
                "stage_index": 0,
                "stage": StageKind.FULL_TRAINING,
                "input_contract": plan.items[0].input_contract,
                "candidate_digest": fake_digest("candidate:study-1"),
                "manifest_revision": 1,
                "budget_unit": "gpu_hours",
                "budget_reservation": 0.25,
                "executor_kind": "recording",
            },
            context={"executor_registry": registry},
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )

    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"
    assert ticked == [scheduled.attempt_id]
    assert repository.get_attempt("workspace-a", scheduled.attempt_id).status == (
        AttemptStatus.COMPLETED
    )


def test_fake_attempts_dispatch_through_the_fake_executor_adapter(tmp_path):
    ticked = []

    class RecordingFakeAdapter(FakeExecutorAdapter):
        def tick(self, worker, attempt, *, now):
            ticked.append(attempt.attempt_id)
            return super().tick(worker, attempt, now=now)

    registry = ExecutorRegistry()
    registry.register(RecordingFakeAdapter())
    registry.register(SshRemoteExecutorAdapter())
    registry.register(DevelopmentEvaluationExecutorAdapter())
    registry.freeze()

    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        executor_registry=registry,
    )
    scheduled = schedule(repository, worker, plan)

    assert worker.run_once(now=START + timedelta(seconds=1)) == "completed"
    assert ticked == [scheduled.attempt_id]


def test_reconcile_leaves_an_expired_attempt_of_a_non_repairable_kind_untouched(tmp_path):
    class InertAdapter:
        kind = "inert"
        allowed_stages = frozenset({StageKind.FULL_TRAINING})
        reuses_completed_results = False

        def tick(self, worker, attempt, *, now):
            return "inert_running"

        def reconcile(self, worker, attempt, *, now):
            return None

        def repair_allowed(self):
            return False

    registry = registry_with(InertAdapter())
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"w" * 32, key_version="worker-test-v1"),
        data_directory=tmp_path / "data-root",
        worker_id="worker-a",
        executor_registry=registry,
    )
    assert worker.run_once(now=START) == "idle"
    scheduled = repository.schedule_action_under_leader(
        ActionSpec.model_validate(
            {
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-1",
                "study_id": "study-1",
                "stage_index": 0,
                "stage": StageKind.FULL_TRAINING,
                "input_contract": plan.items[0].input_contract,
                "candidate_digest": fake_digest("candidate:study-1"),
                "manifest_revision": 1,
                "budget_unit": "gpu_hours",
                "budget_reservation": 0.25,
                "executor_kind": "inert",
            },
            context={"executor_registry": registry},
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )
    assert worker.run_once(now=START + timedelta(seconds=1)) == "inert_running"
    claimed = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert claimed.status == AttemptStatus.RUNNING
    assert claimed.lease_expires_at is not None
    assert claimed.lease_expires_at <= START + timedelta(minutes=5)

    assert worker.reconcile_once(now=START + timedelta(minutes=5)) is None

    assert repository.get_attempt("workspace-a", scheduled.attempt_id) == claimed


def test_unknown_executor_kind_is_rejected_at_spec_validation():
    with pytest.raises(ValueError, match="campaign_executor_kind_not_registered"):
        ActionSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            study_id="study-1",
            stage_index=0,
            stage=StageKind.FULL_TRAINING,
            input_contract={},
            candidate_digest=fake_digest("candidate"),
            manifest_revision=1,
            budget_unit="gpu_hours",
            budget_reservation=0.25,
            executor_kind="not-a-kind",
        )


def test_run_once_settles_a_claimed_attempt_whose_executor_kind_is_unregistered(tmp_path):
    class RecordingAdapter:
        kind = "recording"
        allowed_stages = frozenset({StageKind.FULL_TRAINING})
        reuses_completed_results = False

        def tick(self, worker, attempt, *, now):
            return worker._fake_tick(attempt, now=now)

        def reconcile(self, worker, attempt, *, now):
            return None

        def repair_allowed(self):
            return True

    scheduling_registry = registry_with(RecordingAdapter())
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    assert worker.run_once(now=START) == "idle"
    scheduled = repository.schedule_action_under_leader(
        ActionSpec.model_validate(
            {
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-1",
                "study_id": "study-1",
                "stage_index": 0,
                "stage": StageKind.FULL_TRAINING,
                "input_contract": plan.items[0].input_contract,
                "candidate_digest": fake_digest("candidate:study-1"),
                "manifest_revision": 1,
                "budget_unit": "gpu_hours",
                "budget_reservation": 0.25,
                "executor_kind": "recording",
            },
            context={"executor_registry": scheduling_registry},
        ),
        worker.leader,
        expected_campaign_version=4,
        now=START,
    )
    assert repository.get_campaign("workspace-a", "campaign-1").active_action_id == (
        scheduled.action_id
    )

    assert worker.run_once(now=START + timedelta(seconds=1)) == "blocked"

    settled = repository.get_attempt("workspace-a", scheduled.attempt_id)
    assert settled.status == AttemptStatus.FAILED
    assert repository.get_campaign("workspace-a", "campaign-1").active_action_id is None
    failures = [
        event
        for _, event in repository.list_events("workspace-a", "campaign-1")
        if event.event_type == "campaign:action-failed"
    ]
    assert len(failures) == 1
    assert failures[0].payload["attempt_id"] == scheduled.attempt_id
    assert failures[0].payload["exit_reason"] == "campaign_executor_kind_not_registered"
    assert repository.list_unfinished_attempts() == []
    assert worker.run_once(now=START + timedelta(seconds=2)) == "idle"


def test_reconcile_marks_an_expired_unregistered_kind_attempt_unknown(tmp_path):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    schedule(repository, worker, plan)
    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=START, crash_after_seal=True)
    stranded = repository.list_unfinished_attempts()[0]
    rewrite_executor_kind(repository, stranded.attempt_id, "vendor_gpu")

    assert worker.reconcile_once(now=START + timedelta(minutes=5)) == "unknown"

    after = repository.get_attempt("workspace-a", stranded.attempt_id)
    assert after.status == AttemptStatus.UNKNOWN


def test_reconcile_skips_and_warns_for_an_unregistered_executor_kind(tmp_path, caplog):
    repository = active_repository(tmp_path / "campaigns.sqlite3")
    plan = seed_validated_study(repository)
    worker = make_worker(repository, tmp_path, "worker-a")
    schedule(repository, worker, plan)
    with pytest.raises(SimulatedWorkerCrashError):
        worker.run_once(now=START, crash_after_seal=True)
    unfinished = repository.list_unfinished_attempts()
    assert len(unfinished) == 1
    stranded = unfinished[0]
    rewrite_executor_kind(repository, stranded.attempt_id, "vendor_gpu")

    with caplog.at_level(logging.WARNING, logger="bashgym.campaigns.worker"):
        assert worker.reconcile_once(now=START + timedelta(seconds=1)) is None

    assert "campaign_executor_kind_not_registered" in caplog.text
    assert stranded.attempt_id in caplog.text
    assert "vendor_gpu" in caplog.text
    after = repository.get_attempt("workspace-a", stranded.attempt_id)
    assert after.status == AttemptStatus.RUNNING
