"""Credential-free proof of the simplified AutoResearch discovery loop."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchCampaignSpec,
    AutoResearchInvariantError,
    AutoResearchNextAction,
    AutoResearchRepository,
    AutoResearchStopRules,
    MetricDirection,
    ResultDecision,
    _validate_controlled_candidate_change,
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
    AutoResearchEvaluationEvidence,
    AutoResearchEvaluatorReadiness,
    CampaignEvaluationProjector,
    SealedEvaluationReader,
)
from bashgym.campaigns.autoresearch_loop import AutoResearchLoopCoordinator
from bashgym.campaigns.contracts import (
    CampaignStatus,
    CampaignTrigger,
    Capability,
    CodeMutationKind,
    CredentialKind,
    ManifestRevision,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    canonical_hash,
)
from bashgym.campaigns.failure_observations import AutoResearchFailureObservation
from bashgym.campaigns.remote import (
    ApprovedRemoteExecutorProfile,
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
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker
from bashgym.ledger.contracts import (
    DatasetSpec,
    DatasetVersionSpec,
    EvaluationSuiteSpec,
    ProjectSpec,
)
from tests.campaigns.test_persistence import campaign, manifest
from tests.campaigns.test_proposals import principal, proposal

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def test_controlled_change_can_add_one_declared_leaf_to_empty_recipe_mapping() -> None:
    parent = SimpleNamespace(
        study_family="terminal-agent",
        dataset_recipe={"schema_version": "recipe.v1"},
        training_recipe={"optimizer": {}},
        evaluation_recipe={"schema_version": "recipe.v1"},
    )
    candidate = SimpleNamespace(
        study_family="terminal-agent",
        dataset_recipe={"schema_version": "recipe.v1"},
        training_recipe={"optimizer": {"learning_rate": 0.001}},
        evaluation_recipe={"schema_version": "recipe.v1"},
    )

    _validate_controlled_candidate_change(
        parent,
        candidate,
        declared_variable="training_recipe.optimizer.learning_rate",
        code_mutation_kind=None,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_recipe() -> dict[str, object]:
    return {
        "schema_version": "bashgym.autoresearch_data_design_recipe.v1",
        "runtime": {"executor_kind": "registered_training"},
        "hypothesis": "Target the measured high-count failure slice.",
        "pipeline": "terminal_env_generation",
        "generation_brief": "Generate balanced examples targeting the measured failure slice.",
        "target_rows": 64,
        "train_fraction": 0.8,
        "seed": 17,
    }


def _training_recipe(learning_rate: float, *, seed: int = 42) -> dict[str, object]:
    return {
        "schema_version": "bashgym.tmax_composite_training_recipe.v1",
        "algorithm": "grpo",
        "sft_enabled": False,
        "learning_rate": learning_rate,
        "max_steps": 100,
        "group_size": 8,
        "temperature": 0.8,
        "seed": seed,
        "runtime": {"executor_kind": "registered_training"},
    }


def _baseline_submission():
    return proposal("baseline-registered", estimated_cost=0.1).model_copy(
        update={
            # Baseline executes evaluation only, but freezes the scientific recipe
            # against which the first candidate declares one controlled change.
            "dataset_recipe": _dataset_recipe(),
            "training_recipe": _training_recipe(0.001),
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset({Capability.EVAL_DEVELOPMENT}),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Measure the registered base model on the fixed suite.",
                    ),
                )
            ),
        }
    )


def _candidate_submission(
    proposal_id: str,
    *,
    learning_rate: float,
    prerequisite_study_id: str,
):
    return proposal(proposal_id, estimated_cost=0.2).model_copy(
        update={
            "prerequisite_study_ids": (prerequisite_study_id,),
            "dataset_recipe": _dataset_recipe(),
            "training_recipe": _training_recipe(learning_rate),
            "evaluation_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {"executor_kind": "registered_compute"},
            },
            "required_capabilities": frozenset(
                {
                    Capability.DATA_BUILD,
                    Capability.COMPUTE_TRAIN_WITHIN_BUDGET,
                    Capability.EVAL_DEVELOPMENT,
                }
            ),
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.DATA_BUILD,
                        disposition=StageDisposition.REQUIRED,
                        reason="Generate and validate one compute-resident training dataset.",
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Train the single controlled learning-rate change.",
                    ),
                    StagePlanItem(
                        stage=StageKind.DEVELOPMENT_EVALUATION,
                        disposition=StageDisposition.REQUIRED,
                        reason="Compare the candidate on the unchanged fixed suite.",
                    ),
                )
            ),
        }
    )


class _CredentialFreeRemoteAdapter:
    """Deterministic external-execution boundary; the worker owns all mechanics."""

    def __init__(self, evaluation_metrics: tuple[float, ...]) -> None:
        self.evaluation_metrics = list(evaluation_metrics)
        self.identities: dict[str, RemoteRunIdentity] = {}
        self.launch_requests = []
        self.collected_run_ids: list[str] = []
        self.remote_files: dict[str, dict[str, bytes]] = {}
        self.remote_seals: dict[str, bytes] = {}

    @staticmethod
    def _baseline_readiness(request, metric: float):
        if request.registered_base_model is None or request.remote_resident_model is not None:
            return None
        return AutoResearchEvaluatorReadiness(
            known_good_case_id="known-good",
            known_good_passed=True,
            known_bad_case_id="known-bad",
            known_bad_rejected=True,
            baseline_scores=(metric - 0.005, metric, metric + 0.005),
        )

    @staticmethod
    def _failure_observations(metric: float):
        return (
            AutoResearchFailureObservation(
                observation_id="task-failure",
                category="task_failure",
                summary="The fixed evaluator marked the task response incorrect.",
                slice_path="behavior.task_failure",
                count=round((1.0 - metric) * 100),
            ),
        )

    async def discover(self, request):
        return self.identities.get(request.run_id)

    async def capacity_preflight(self, _policy):
        return RemoteCapacitySnapshot(
            compute_profile_id="ssh-gpu-lab",
            available_memory_gib=128,
            available_disk_gib=512,
            external_gpu_processes=(),
            admitted=True,
            blocking_reasons=(),
            observed_at=NOW,
        )

    async def launch(self, request):
        launched_at = NOW + timedelta(seconds=1 + 2 * len(self.launch_requests))
        identity = RemoteRunIdentity(
            compute_profile_id=request.compute_profile_id,
            run_id=request.run_id,
            remote_run_directory=f"/fixture-runs/{request.run_id}",
            remote_pid=42 + len(self.launch_requests),
            process_group_id=42 + len(self.launch_requests),
            process_start_ticks=1 + len(self.launch_requests),
            boot_id="fixture-boot",
            command_hash="a" * 64,
            launch_manifest_sha256="b" * 64,
            launched_at=launched_at,
        )
        self.identities[request.run_id] = identity
        self.launch_requests.append(request)
        files = {
            "exit_code": b"0\n",
            "launch_manifest.json": b"{}",
            "training.log": b"complete\n",
        }
        if AUTORESEARCH_DATASET_RECEIPT_FILENAME in request.output_paths:
            train_rows = (json.dumps({"task": "terminal", "run": request.run_id}) + "\n").encode()
            validation_rows = (
                json.dumps({"task": "terminal-validation", "run": request.run_id}) + "\n"
            ).encode()
            files["dataset/train.jsonl"] = train_rows
            files["dataset/validation.jsonl"] = validation_rows
            receipt = AutoResearchDatasetReceipt(
                files=(
                    AutoResearchDatasetFile(
                        path="dataset/train.jsonl",
                        sha256=hashlib.sha256(train_rows).hexdigest(),
                        size_bytes=len(train_rows),
                        split="train",
                        row_count=1,
                    ),
                    AutoResearchDatasetFile(
                        path="dataset/validation.jsonl",
                        sha256=hashlib.sha256(validation_rows).hexdigest(),
                        size_bytes=len(validation_rows),
                        split="validation",
                        row_count=1,
                    ),
                ),
                generator={
                    "kind": "nvidia_data_designer",
                    "pipeline": "terminal_env_generation",
                },
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
            files[AUTORESEARCH_DATASET_RECEIPT_FILENAME] = receipt.model_dump_json().encode()
        elif request.evaluation_context_sha256 is not None:
            context_path = next(
                path
                for path in request.input_files
                if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
            )
            context = AutoResearchEvaluationContext.model_validate_json(
                context_path.read_text(encoding="utf-8")
            )
            metric = self.evaluation_metrics.pop(0)
            evidence = AutoResearchEvaluationEvidence(
                campaign_id=context.campaign_id,
                study_id=context.study_id,
                action_id=context.action_id,
                attempt_id=context.attempt_id,
                candidate_digest=context.candidate_digest,
                evaluation_suite_id=context.evaluation_suite_id,
                evaluation_code_digest=context.evaluation_code_digest,
                dataset_version_id=context.dataset_version_id,
                evaluated_model_manifest_digest=context.evaluated_model_manifest_digest,
                metrics={"mrr_at_10": metric},
                failure_observations=self._failure_observations(metric),
                evaluator_readiness=self._baseline_readiness(request, metric),
                started_at=identity.launched_at,
                completed_at=identity.launched_at + timedelta(seconds=1),
            )
            files[AUTORESEARCH_EVALUATION_FILENAME] = evidence.model_dump_json().encode()
        else:
            assert request.remote_resident_dataset is not None
            files["final/config.json"] = json.dumps(
                {"fixture_run": request.run_id}, sort_keys=True
            ).encode()
            files["final/model.safetensors"] = request.run_id.encode("utf-8")
        self.remote_files[request.run_id] = files
        return identity

    async def observe(self, identity):
        return RemoteObservation(
            identity=identity,
            state=RemoteRunState.COMPLETED,
            observed_at=identity.launched_at + timedelta(seconds=1),
            exit_code=0,
            safe_reason="credential_free_fixture_completed",
        )

    async def read_stream(self, _identity, source, cursor):
        return RemoteStreamChunk(
            source=source,
            start_offset=cursor.byte_offset,
            end_offset=cursor.byte_offset,
            complete_lines=(),
            next_cursor=cursor,
        )

    async def collect_outputs(self, identity, request, local_directory, *, observation):
        del observation
        if request.evaluation_context_sha256 is not None:
            context_path = next(
                path
                for path in request.input_files
                if path.name == AUTORESEARCH_EVALUATION_CONTEXT_FILENAME
            )
            context = AutoResearchEvaluationContext.model_validate_json(
                context_path.read_text(encoding="utf-8")
            )
            metric = self.evaluation_metrics.pop(0)
            evidence = AutoResearchEvaluationEvidence(
                campaign_id=context.campaign_id,
                study_id=context.study_id,
                action_id=context.action_id,
                attempt_id=context.attempt_id,
                candidate_digest=context.candidate_digest,
                evaluation_suite_id=context.evaluation_suite_id,
                evaluation_code_digest=context.evaluation_code_digest,
                dataset_version_id=context.dataset_version_id,
                evaluated_model_manifest_digest=context.evaluated_model_manifest_digest,
                metrics={"mrr_at_10": metric},
                failure_observations=self._failure_observations(metric),
                evaluator_readiness=self._baseline_readiness(request, metric),
                started_at=identity.launched_at,
                completed_at=identity.launched_at + timedelta(seconds=1),
            )
            (local_directory / AUTORESEARCH_EVALUATION_FILENAME).write_text(
                evidence.model_dump_json(), encoding="utf-8"
            )
        else:
            final = local_directory / "final"
            final.mkdir()
            (final / "config.json").write_text(
                json.dumps({"fixture_run": request.run_id}, sort_keys=True),
                encoding="utf-8",
            )
            (final / "model.safetensors").write_bytes(request.run_id.encode("utf-8"))
        self.collected_run_ids.append(identity.run_id)
        return tuple(path for path in local_directory.rglob("*") if path.is_file())

    async def inventory_outputs(self, identity, request, *, observation):
        del request, observation
        self.collected_run_ids.append(identity.run_id)
        return RemoteOutputInventory(
            compute_profile_id=identity.compute_profile_id,
            run_id=identity.run_id,
            files=tuple(
                RemoteOutputFile(
                    path=path,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                )
                for path, payload in sorted(self.remote_files[identity.run_id].items())
            ),
        )

    async def persist_action_seal(self, identity, envelope):
        self.remote_seals[identity.run_id] = bytes(envelope)
        return f"{identity.remote_run_directory}/sealed_action_result.v1.json"

    async def read_action_seal(self, identity):
        return self.remote_seals.get(identity.run_id)

    async def read_output_bytes(
        self,
        identity,
        relative_path,
        *,
        expected_sha256,
        expected_size_bytes,
        max_bytes,
    ):
        payload = self.remote_files[identity.run_id][relative_path]
        assert len(payload) <= max_bytes
        assert len(payload) == expected_size_bytes
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        return payload

    async def collect_terminal_evidence(self, *_args, **_kwargs):
        raise AssertionError("the successful discovery-loop fixture cannot collect a failure")

    async def force_stop(self, _identity):
        return False

    async def terminate(self, _identity):
        return True


def test_start_to_branched_candidate_decisions_stops_and_exports_without_compute(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()

    campaign_value = campaign()
    campaign_manifest = manifest().model_copy(
        update={
            "budget_limits": {"gpu_hours": 1.0, "study_count": 4.0},
            "evaluation_plan": {
                **manifest().evaluation_plan,
                "ledger_project_id": "project-a",
                "evaluation_suite_id": "suite-held-out",
                "dataset_binding_id": "dataset-held-out-v1",
            },
            "max_proposal_rounds": 4,
        }
    )
    repository.create_campaign(
        campaign_value,
        ManifestRevision(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            revision=1,
            manifest=campaign_manifest,
            actor_id="codex-agent",
            correlation_id="create-discovery-loop",
            created_at=NOW,
        ),
        actor_id="codex-agent",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="create-discovery-loop",
        idempotency_key="create-discovery-loop",
    )

    materials = tmp_path / "registered-materials"
    materials.mkdir()
    data_builder = materials / "build_data.py"
    train_script = materials / "train.py"
    evaluator = materials / "evaluate.py"
    data_designer_config = materials / "data-designer-config.json"
    train_config = materials / "tmax-runner-config.json"
    development_data = materials / "development.jsonl"
    key = materials / "ssh-key"
    data_builder.write_text("print('fixture data build')\n", encoding="utf-8")
    train_script.write_text("print('fixture train')\n", encoding="utf-8")
    evaluator.write_text("print('fixture evaluate')\n", encoding="utf-8")
    data_designer_config.write_text('{"pipeline":"terminal_env_generation"}\n', encoding="utf-8")
    train_config.write_text('{"runner":"tmax-composite"}\n', encoding="utf-8")
    development_data.write_text('{"text":"fixed held-out fixture"}\n', encoding="utf-8")
    key.write_text("unused fixture key\n", encoding="utf-8")

    core = AutoResearchCampaignCore(repository)
    core.ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="Credential-free discovery-loop proof",
            owner_actor_id="codex-agent",
            created_at=NOW,
        )
    )
    core.ledger.register_dataset(
        DatasetSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            dataset_id="dataset-held-out",
            display_name="Fixed held-out development tasks",
            task_type="text-retrieval",
            created_at=NOW,
        )
    )
    dataset_version = DatasetVersionSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        dataset_id="dataset-held-out",
        dataset_version_id="dataset-held-out-v1",
        source_uri="bashgym-remote-dataset://heldout-v1",
        content_digest=_sha256(development_data),
        created_at=NOW,
    )
    evaluation_suite = EvaluationSuiteSpec(
        workspace_id="workspace-a",
        project_id="project-a",
        evaluation_suite_id="suite-held-out",
        name="Fixed held-out evaluator",
        task_type="text-retrieval",
        dataset_version_id=dataset_version.dataset_version_id,
        metric_contract={
            "primary_metric": "mrr_at_10",
            "evaluator_readiness": {
                "known_good_case_id": "known-good",
                "known_bad_case_id": "known-bad",
                "baseline_repeat_count": 3,
                "maximum_baseline_spread": 0.02,
            },
        },
        code_digest=_sha256(evaluator),
        created_at=NOW,
    )
    core.ledger.register_dataset_version(dataset_version)
    core.ledger.register_evaluation_suite(evaluation_suite)
    core.register(
        AutoResearchCampaignSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="mrr_at_10",
            metric_direction=MetricDirection.MAXIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=4,
                budget_unit="gpu_hours",
                max_total_cost=1.0,
                minimum_improvement=0.01,
            ),
            ledger_project_id="project-a",
            evaluation_suite_id="suite-held-out",
            created_at=NOW,
        )
    )

    target_model_digest = canonical_hash(campaign_value.target_model.model_dump(mode="json"))
    base_model = RegisteredRemoteModelSource(
        schema_version="campaign_registered_remote_model_source.v2",
        source_id="registered-base-v1",
        compute_profile_id="ssh-gpu-lab",
        target_contract_key=campaign_value.target_model.target_contract_key,
        model_digest=target_model_digest,
        remote_model_path="/private/models/registered-base-v1",
        artifact_receipt={
            "model_id": "example/research-model",
            "revision": "a" * 40,
            "artifact_manifest_sha256": "f" * 64,
            "weight_file_count": 2,
            "total_size_bytes": 1024,
        },
    )
    heldout = RegisteredRemoteEvaluationDatasetSource(
        source_id="heldout-v1",
        compute_profile_id="ssh-gpu-lab",
        dataset_version_id=dataset_version.dataset_version_id,
        content_digest=dataset_version.content_digest,
        remote_dataset_path="/private/datasets/development.jsonl",
    )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="credential-free-fixture-profile",
        profile_revision=1,
        compute_profile_id="ssh-gpu-lab",
        target_contract_key=campaign_value.target_model.target_contract_key,
        target_model_digest=target_model_digest,
        host="fixture.invalid",
        username="fixture-user",
        key_path=str(key),
        stages=(
            PinnedRemoteStageProfile(
                stage=StageKind.DATA_BUILD,
                script_path=data_builder,
                script_sha256=_sha256(data_builder),
                input_files=(data_designer_config,),
                input_sha256={data_designer_config.name: _sha256(data_designer_config)},
                output_paths=(AUTORESEARCH_DATASET_RECEIPT_FILENAME, "dataset"),
                budget_reservation=0.05,
            ),
            PinnedRemoteStageProfile(
                stage=StageKind.DEVELOPMENT_EVALUATION,
                script_path=evaluator,
                script_sha256=evaluation_suite.code_digest,
                input_files=(),
                input_sha256={},
                output_paths=(AUTORESEARCH_EVALUATION_FILENAME,),
                budget_reservation=0.1,
            ),
            PinnedRemoteStageProfile(
                stage=StageKind.FULL_TRAINING,
                script_path=train_script,
                script_sha256=_sha256(train_script),
                input_files=(train_config,),
                input_sha256={train_config.name: _sha256(train_config)},
                output_paths=("final",),
                budget_reservation=0.1,
            ),
        ),
        registered_base_model=base_model,
        registered_evaluation_dataset=heldout,
    )

    ready = core.prepare(
        "workspace-a",
        "campaign-1",
        controller_id="autoresearch-controller",
        correlation_id="prepare-discovery-loop",
        idempotency_prefix="prepare-discovery-loop",
    )
    actor = principal(repository)
    active = CampaignService(repository).transition(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.START,
        expected_version=ready.version,
        principal=actor,
        correlation_id="start-discovery-loop",
        idempotency_key="start-discovery-loop",
    )
    assert active.campaign.status == CampaignStatus.ACTIVE

    sealer = ArtifactSealer(b"d" * 32, key_version="discovery-loop-proof-v1")
    reader = SealedEvaluationReader(sealer)
    core = AutoResearchCampaignCore(repository, evaluation_reader=reader)
    projector = CampaignEvaluationProjector(repository, core.ledger, reader)
    coordinator = AutoResearchLoopCoordinator(repository, projector, core)
    adapter = _CredentialFreeRemoteAdapter((0.50, 0.70, 0.65, 0.72))
    artifact_root = tmp_path / "sealed-artifacts"
    worker = CampaignWorker(
        repository,
        artifact_root,
        sealer,
        data_directory=tmp_path / "worker-data",
        worker_id="credential-free-discovery-loop",
        remote_adapters={"ssh-gpu-lab": adapter},
        remote_executor_profiles={
            (profile.compute_profile_id, profile.target_contract_key): profile
        },
        autoresearch_loop=coordinator,
    )

    submitted_baseline = core.submit_baseline(
        _baseline_submission(),
        expected_version=active.campaign.version,
        principal=actor,
        correlation_id="submit-baseline",
        idempotency_key="submit-baseline",
    )
    assert worker.run_once(now=NOW + timedelta(seconds=1)) == "completed"
    baseline_attempt = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert baseline_attempt.stage == StageKind.DEVELOPMENT_EVALUATION
    assert baseline_attempt.executor["registered_base_model"] == base_model.model_dump(mode="json")
    assert baseline_attempt.executor["evaluated_model_digest"] == "f" * 64
    assert baseline_attempt.executor["registered_evaluation_dataset"] == heldout.model_dump(
        mode="json"
    )
    assert adapter.launch_requests[-1].registered_evaluation_dataset == heldout
    assert heldout.remote_dataset_path in adapter.launch_requests[-1].script_args
    assert development_data not in adapter.launch_requests[-1].input_files
    # Rebuild every projection component from durable state before ingesting the
    # remote evaluation. Raw evaluator JSON remains only in the fake run store.
    reader = SealedEvaluationReader(sealer)
    core = AutoResearchCampaignCore(repository, evaluation_reader=reader)
    projector = CampaignEvaluationProjector(repository, core.ledger, reader)
    worker.autoresearch_loop = AutoResearchLoopCoordinator(repository, projector, core)
    assert worker.run_once(now=NOW + timedelta(seconds=3)) == ("autoresearch_evaluation_ingested")
    baseline_state = core.state("workspace-a", "campaign-1", now=NOW + timedelta(seconds=3))
    assert baseline_state.next_action == AutoResearchNextAction.PROPOSE_CANDIDATE

    baseline_versions = [
        item
        for item in core.ledger.list_model_versions("workspace-a", "project-a")
        if item["metadata"].get("source_kind") == "registered_base_model"
    ]
    assert len(baseline_versions) == 1
    assert (
        baseline_versions[0]["source_uri"] == "autoresearch-registered-model://registered-base-v1"
    )
    assert base_model.remote_model_path not in json.dumps(baseline_versions[0], sort_keys=True)
    assert baseline_versions[0]["metadata"]["evaluated_model_digest"] == "f" * 64
    assert baseline_versions[0]["metadata"]["target_model_digest"] == target_model_digest

    unchanged_learning_rate = _candidate_submission(
        "candidate-unchanged-learning-rate",
        learning_rate=0.001,
        prerequisite_study_id=baseline_state.best_study_id,
    )
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_candidate_declared_variable_unchanged",
    ):
        core.submit_controlled_candidate(
            unchanged_learning_rate,
            parent_proposal_id="baseline-registered",
            changed_variable="learning_rate",
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=actor,
            correlation_id="reject-unchanged-learning-rate",
            idempotency_key="reject-unchanged-learning-rate",
        )

    code_candidate = unchanged_learning_rate.model_copy(
        update={
            "proposal_id": "candidate-code-only",
            "primary_variable": "trainer.optimizer",
        }
    )
    _validate_controlled_candidate_change(
        repository.get_proposal("workspace-a", "campaign-1", "baseline-registered").proposal,
        code_candidate,
        declared_variable="trainer.optimizer",
        code_mutation_kind=CodeMutationKind.TRAINER,
    )
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_candidate_changed_undeclared_variable",
    ):
        _validate_controlled_candidate_change(
            repository.get_proposal("workspace-a", "campaign-1", "baseline-registered").proposal,
            code_candidate.model_copy(update={"training_recipe": _training_recipe(0.002)}),
            declared_variable="trainer.optimizer",
            code_mutation_kind=CodeMutationKind.TRAINER,
        )

    undeclared_seed_change = _candidate_submission(
        "candidate-undeclared-seed",
        learning_rate=0.002,
        prerequisite_study_id=baseline_state.best_study_id,
    ).model_copy(update={"training_recipe": _training_recipe(0.002, seed=43)})
    with pytest.raises(
        AutoResearchInvariantError,
        match="autoresearch_candidate_changed_undeclared_variable",
    ):
        core.submit_controlled_candidate(
            undeclared_seed_change,
            parent_proposal_id="baseline-registered",
            changed_variable="learning_rate",
            expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
            principal=actor,
            correlation_id="reject-undeclared-seed",
            idempotency_key="reject-undeclared-seed",
        )

    candidate_one = _candidate_submission(
        "candidate-learning-rate-1",
        learning_rate=0.002,
        prerequisite_study_id=baseline_state.best_study_id,
    )
    candidate_one_submission = core.submit_controlled_candidate(
        candidate_one,
        parent_proposal_id="baseline-registered",
        changed_variable="learning_rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-candidate-one",
        idempotency_key="submit-candidate-one",
    )
    assert worker.run_once(now=NOW + timedelta(seconds=4)) == "completed"
    first_data_build = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert first_data_build.stage == StageKind.DATA_BUILD
    assert first_data_build.executor["script_args"][-2] == "--recipe-json"
    rendered_data_recipe = json.loads(first_data_build.executor["script_args"][-1])
    assert rendered_data_recipe["generation_brief"].startswith("Generate balanced examples")
    assert worker.run_once(now=NOW + timedelta(seconds=5)) == "completed"
    first_training = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert first_training.stage == StageKind.FULL_TRAINING
    assert (
        first_training.executor["remote_resident_dataset"]["attempt_id"]
        == first_data_build.attempt_id
    )
    assert adapter.launch_requests[-1].remote_resident_dataset is not None
    assert adapter.launch_requests[-1].remote_resident_dataset.attempt_id == (
        first_data_build.attempt_id
    )
    assert adapter.launch_requests[-1].registered_base_model == base_model
    assert adapter.launch_requests[-1].script_args[:4] == (
        "--algorithm",
        "grpo",
        "--sft-enabled",
        "false",
    )
    assert adapter.launch_requests[-1].script_args[-4:] == (
        "--model-dir",
        base_model.remote_model_path,
        "--dataset-dir",
        adapter.launch_requests[-1].remote_resident_dataset.remote_dataset_path,
    )
    assert worker.run_once(now=NOW + timedelta(seconds=6)) == "completed"
    first_evaluation = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert first_evaluation.stage == StageKind.DEVELOPMENT_EVALUATION
    assert (
        first_evaluation.executor["remote_resident_model"]["attempt_id"]
        == first_training.attempt_id
    )
    assert "source_training" not in first_evaluation.executor
    assert "sealed_stage_artifact_inputs" not in first_evaluation.executor
    assert worker.run_once(now=NOW + timedelta(seconds=8)) == ("autoresearch_evaluation_ingested")
    first_state = core.state("workspace-a", "campaign-1", now=NOW + timedelta(seconds=8))
    assert first_state.latest_decision == ResultDecision.KEEP
    assert first_state.best_proposal_id == candidate_one.proposal_id

    candidate_two = _candidate_submission(
        "candidate-learning-rate-2",
        learning_rate=0.003,
        prerequisite_study_id=first_state.best_study_id,
    ).model_copy(update={"primary_variable": "training_recipe.learning_rate"})
    core.submit_controlled_candidate(
        candidate_two,
        parent_proposal_id=candidate_one.proposal_id,
        changed_variable="training_recipe.learning_rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-candidate-two",
        idempotency_key="submit-candidate-two",
    )
    assert worker.run_once(now=NOW + timedelta(seconds=9)) == "completed"
    second_data_build = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert second_data_build.stage == StageKind.DATA_BUILD
    assert worker.run_once(now=NOW + timedelta(seconds=10)) == "completed"
    second_training = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert second_training.stage == StageKind.FULL_TRAINING
    assert (
        second_training.executor["remote_resident_dataset"]["attempt_id"]
        == second_data_build.attempt_id
    )
    second_training_request = adapter.launch_requests[-1]
    assert second_training_request.registered_base_model is None
    assert second_training_request.remote_resident_model is not None
    assert second_training_request.remote_resident_model.attempt_id == first_training.attempt_id
    assert second_training_request.remote_resident_model.remote_model_path.endswith(
        f"/{first_training.attempt_id}/final"
    )
    assert second_training_request.script_args[-4:] == (
        "--model-dir",
        second_training_request.remote_resident_model.remote_model_path,
        "--dataset-dir",
        second_training_request.remote_resident_dataset.remote_dataset_path,
    )
    assert worker.run_once(now=NOW + timedelta(seconds=11)) == "completed"
    second_evaluation = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert second_evaluation.stage == StageKind.DEVELOPMENT_EVALUATION
    assert (
        second_evaluation.executor["remote_resident_model"]["attempt_id"]
        == second_training.attempt_id
    )
    assert "source_training" not in second_evaluation.executor
    assert "sealed_stage_artifact_inputs" not in second_evaluation.executor
    assert worker.run_once(now=NOW + timedelta(seconds=13)) == ("autoresearch_evaluation_ingested")
    second_state = core.state("workspace-a", "campaign-1", now=NOW + timedelta(seconds=13))
    assert second_state.latest_decision == ResultDecision.DISCARD
    assert second_state.best_proposal_id == candidate_one.proposal_id
    assert second_state.next_action == AutoResearchNextAction.PROPOSE_CANDIDATE
    failure_packet = core.failures("workspace-a", "campaign-1")
    assert failure_packet["comparison"] == [
        {
            "category": "task_failure",
            "reference_count": 30,
            "candidate_count": 35,
            "delta": 5,
            "status": "regressed",
        }
    ]
    assert "prediction" not in json.dumps(failure_packet, sort_keys=True).lower()
    candidate_two_outcome = repository.list_autoresearch_outcomes("workspace-a", "campaign-1")[-1]

    candidate_three = _candidate_submission(
        "candidate-learning-rate-branch",
        learning_rate=0.0025,
        prerequisite_study_id=candidate_two_outcome.result.study_id,
    ).model_copy(update={"primary_variable": "training_recipe.learning_rate"})
    core.submit_controlled_candidate(
        candidate_three,
        parent_proposal_id=candidate_two.proposal_id,
        changed_variable="training_recipe.learning_rate",
        expected_version=repository.get_campaign("workspace-a", "campaign-1").version,
        principal=actor,
        correlation_id="submit-candidate-three",
        idempotency_key="submit-candidate-three",
    )
    assert worker.run_once(now=NOW + timedelta(seconds=14)) == "completed"
    third_data_build = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert third_data_build.stage == StageKind.DATA_BUILD
    assert worker.run_once(now=NOW + timedelta(seconds=15)) == "completed"
    third_training = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert third_training.stage == StageKind.FULL_TRAINING
    third_training_request = adapter.launch_requests[-1]
    assert third_training_request.registered_base_model is None
    assert third_training_request.remote_resident_model is not None
    assert third_training_request.remote_resident_model.attempt_id == second_training.attempt_id
    assert worker.run_once(now=NOW + timedelta(seconds=16)) == "completed"
    third_evaluation = repository.list_attempts("workspace-a", "campaign-1")[-1]
    assert third_evaluation.stage == StageKind.DEVELOPMENT_EVALUATION
    assert worker.run_once(now=NOW + timedelta(seconds=18)) == ("autoresearch_evaluation_ingested")
    third_state = core.state("workspace-a", "campaign-1", now=NOW + timedelta(seconds=18))
    assert third_state.latest_decision == ResultDecision.KEEP
    assert third_state.best_proposal_id == candidate_three.proposal_id
    assert third_state.next_action == AutoResearchNextAction.STOP

    outcomes = repository.list_autoresearch_outcomes("workspace-a", "campaign-1")
    assert [item.result.proposal_id for item in outcomes] == [
        "baseline-registered",
        "candidate-learning-rate-1",
        "candidate-learning-rate-2",
        "candidate-learning-rate-branch",
    ]
    assert [item.decision.decision for item in outcomes] == [
        ResultDecision.BASELINE,
        ResultDecision.KEEP,
        ResultDecision.DISCARD,
        ResultDecision.KEEP,
    ]
    assert [item.result.metric_value for item in outcomes] == [0.50, 0.70, 0.65, 0.72]
    assert outcomes[-1].decision.previous_best_proposal_id == candidate_one.proposal_id
    assert outcomes[-1].decision.previous_best_metric == 0.70
    assert outcomes[-1].decision.improvement == pytest.approx(0.02)
    assert submitted_baseline.record.proposal.planner_actor_id == actor.actor_id
    assert candidate_one_submission.record.proposal.planner_actor_id == actor.actor_id
    assert len(adapter.launch_requests) == len(adapter.collected_run_ids) == 10
    assert adapter.evaluation_metrics == []
    assert not artifact_root.exists()

    generated_dataset_versions = [
        item
        for item in core.ledger.list_dataset_versions("workspace-a", "project-a")
        if item["metadata"].get("source_kind") == "remote_data_build"
    ]
    assert len(generated_dataset_versions) == 3
    assert all(
        item["metadata"]["data_quality"]["acceptance_rate"] == 2 / 3
        for item in generated_dataset_versions
    )
    assert all(
        item["source_uri"].startswith("autoresearch-remote-dataset://sha256/")
        for item in generated_dataset_versions
    )
    serialized_dataset_versions = json.dumps(generated_dataset_versions, sort_keys=True)
    assert "/private/" not in serialized_dataset_versions
    assert "remote_dataset_path" not in serialized_dataset_versions
    assert "remote_run_directory" not in serialized_dataset_versions
    assert '"run":' not in serialized_dataset_versions

    evaluation_attempts = [
        item
        for item in repository.list_attempts("workspace-a", "campaign-1")
        if item.stage == StageKind.DEVELOPMENT_EVALUATION
    ]
    assert len(evaluation_attempts) == 4
    assert {
        (
            item.executor["evaluation_binding"]["evaluation_suite_id"],
            item.executor["evaluation_binding"]["dataset_version_id"],
        )
        for item in evaluation_attempts
    } == {("suite-held-out", "dataset-held-out-v1")}
    assert all(
        item.executor["evaluation_binding"]["dataset_remote_path"] == heldout.remote_dataset_path
        for item in evaluation_attempts
    )

    assert worker.run_once(now=NOW + timedelta(seconds=19)) == "autoresearch_stop_enforced"
    stopped = core.state("workspace-a", "campaign-1", now=NOW + timedelta(seconds=19))
    assert stopped.campaign_status == CampaignStatus.EXHAUSTED
    assert stopped.reason_code == "attempt_limit_reached"

    export_root = tmp_path / "reports"
    terminal = repository.get_campaign("workspace-a", "campaign-1")
    api_repository = CampaignRuntimeRepository(repository.db_path)
    api_repository.initialize()
    exported = CampaignService(api_repository, export_root=export_root).export(
        "workspace-a",
        "campaign-1",
        ("markdown", "json"),
        expected_version=terminal.version,
        principal=actor,
        correlation_id="export-discovery-loop",
        idempotency_key="export-discovery-loop",
    )
    report_directory = export_root / "workspace-a" / "campaign-1" / exported.details["export_id"]
    report = (report_directory / "campaign_report.md").read_text(encoding="utf-8")
    evidence = json.loads((report_directory / "campaign_evidence.json").read_text(encoding="utf-8"))
    assert "# Campaign Evidence Report" in report
    assert "- Status: `exhausted`" in report
    assert evidence["campaign"]["stop_reason"] == "attempt_limit_reached"
    assert len(evidence["attempts"]) == 10
    history = evidence["autoresearch_history"]
    assert history["schema_version"] == "bashgym.autoresearch_history.v1"
    assert history["evaluation_suite_id"] == "suite-held-out"
    assert history["total_experiments"] == 4
    assert [item["proposal_id"] for item in history["experiments"]] == [
        "baseline-registered",
        "candidate-learning-rate-1",
        "candidate-learning-rate-2",
        "candidate-learning-rate-branch",
    ]
    assert [item["decision"]["decision"] for item in history["experiments"]] == [
        "baseline",
        "keep",
        "discard",
        "keep",
    ]
    assert [
        item["performance"]["primary"]["candidate_value"] for item in history["experiments"]
    ] == [0.50, 0.70, 0.65, 0.72]
    assert [
        item["performance"]["primary"]["reference_proposal_id"] for item in history["experiments"]
    ] == [
        None,
        "baseline-registered",
        "candidate-learning-rate-1",
        "candidate-learning-rate-1",
    ]
    branch_history = history["experiments"][-1]
    assert branch_history["performance"]["parent"]["proposal_id"] == ("candidate-learning-rate-2")
    assert branch_history["performance"]["parent"]["value"] == 0.65
    assert branch_history["performance"]["parent"]["improvement"] == pytest.approx(0.07)
    assert branch_history["performance"]["primary"]["reference_value"] == 0.70
    assert branch_history["performance"]["primary"]["improvement"] == pytest.approx(0.02)
    assert history["experiments"][1]["data"]["quality"]["acceptance_rate"] == 2 / 3
    assert history["experiments"][2]["data"]["quality"]["acceptance_rate"] == 2 / 3
    assert "## AutoResearch experiment history" in report
    assert "### 2. candidate-learning-rate-1" in report
    assert "### 3. candidate-learning-rate-2" in report
    assert {item["name"] for item in exported.details["files"]} >= {
        "campaign_evidence.json",
        "campaign_report.md",
    }
