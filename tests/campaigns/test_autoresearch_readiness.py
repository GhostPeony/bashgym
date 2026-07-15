"""Fail-closed installation binding checks for AutoResearch templates."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from bashgym.campaigns.autoresearch import (
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
    AutoResearchTemplatePolicy,
    MetricDirection,
    builtin_autoresearch_template_definitions,
)
from bashgym.campaigns.contracts import (
    CampaignManifest,
    StageKind,
    TargetModelContract,
    canonical_hash,
)
from bashgym.campaigns.readiness import doctor_autoresearch_template
from bashgym.campaigns.remote import (
    ApprovedCodeLineageExecutionBinding,
    ApprovedRemoteExecutorProfile,
    PinnedRemoteStageProfile,
)
from bashgym.campaigns.worker_service import ControllerStatusProjection
from bashgym.ledger.contracts import (
    DatasetSpec,
    DatasetVersionSpec,
    EvaluationSuiteSpec,
    ProjectSpec,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository
from tests.campaigns.test_lineage import initialized_repository, source_profile

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
WORKSPACE = "workspace-a"
PROJECT = "autoresearch-test"
DATASET_VERSION = "terminal-agent-approved-v1"
SUITE = "terminal-heldout-v1"


def definition() -> AutoResearchTemplateDefinition:
    target = TargetModelContract(
        target_contract_key="tiny-local-terminal-agent-v1",
        base_model_ref=f"hf://test/tiny-local-model@{'a' * 40}",
        task="terminal-agent-sft",
        representation_contract={
            "artifact_role": "trainable_base",
            "adapter": "lora",
            "quality_claim_eligible": True,
        },
    )
    return AutoResearchTemplateDefinition(
        template_id="autoresearch-installed-test-v1",
        objective="Improve a held-out terminal-agent metric with one controlled variable.",
        target_model=target,
        manifest=CampaignManifest(
            approved_data_scopes=(DATASET_VERSION,),
            compute_profile_id="registered-gpu-lab",
            budget_limits={"gpu_hours": 2.0, "study_count": 3.0},
            evaluation_plan={
                "schema_version": "autoresearch_evaluation_plan.v1",
                "primary_metric": "exact_task_accuracy",
                "metric_direction": "maximize",
                "ledger_project_id": PROJECT,
                "evaluation_suite_id": SUITE,
                "dataset_binding_id": DATASET_VERSION,
                "source_repository_binding_id": "bashgym-source-v1",
                "required_training_stages": ["smoke_training", "full_training"],
            },
            promotion_gates={"quality_claim_eligible": True},
            max_proposal_rounds=3,
        ),
        policy=AutoResearchTemplatePolicy(
            template_revision="1",
            primary_metric="exact_task_accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=3,
                budget_unit="gpu_hours",
                max_total_cost=2.0,
                minimum_improvement=0.01,
            ),
            ledger_project_id=PROJECT,
            evaluation_suite_id=SUITE,
            quality_claim_eligible=True,
        ),
    )


def online_controller() -> ControllerStatusProjection:
    return ControllerStatusProjection(
        online=True,
        state="online",
        code="controller_online",
        observed_at=NOW,
    )


def offline_controller() -> ControllerStatusProjection:
    return ControllerStatusProjection(
        online=False,
        state="offline",
        code="controller_offline",
        observed_at=NOW,
        guidance="Start the resident worker.",
    )


def initialized_ledger(tmp_path: Path) -> ExperimentLedgerRepository:
    ledger = ExperimentLedgerRepository(tmp_path / "campaigns.sqlite3")
    ledger.initialize()
    return ledger


def register_scientific_bindings(ledger: ExperimentLedgerRepository) -> None:
    ledger.register_project(
        ProjectSpec(
            workspace_id=WORKSPACE,
            project_id=PROJECT,
            display_name="AutoResearch test",
            owner_actor_id="test-operator",
        )
    )
    ledger.register_dataset(
        DatasetSpec(
            workspace_id=WORKSPACE,
            project_id=PROJECT,
            dataset_id="terminal-agent-approved",
            display_name="Approved terminal agent data",
            task_type="terminal-agent-sft",
        )
    )
    ledger.register_dataset_version(
        DatasetVersionSpec(
            workspace_id=WORKSPACE,
            project_id=PROJECT,
            dataset_id="terminal-agent-approved",
            dataset_version_id=DATASET_VERSION,
            source_uri="artifact://test/terminal-agent-approved-v1",
            content_digest="b" * 64,
        )
    )
    ledger.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id=WORKSPACE,
            project_id=PROJECT,
            evaluation_suite_id=SUITE,
            name="Terminal heldout",
            task_type="terminal-agent-sft",
            dataset_version_id=DATASET_VERSION,
            metric_contract={
                "primary_metric": "exact_task_accuracy",
                "metric_direction": "maximize",
            },
            code_digest="c" * 64,
        )
    )


def registered_profile(
    tmp_path: Path,
    template: AutoResearchTemplateDefinition,
    *,
    entrypoint_path: str = "bashgym/gym/trainer.py",
):
    key = tmp_path / "worker-key"
    data = tmp_path / "train.jsonl"
    key.write_text("test-only-key\n", encoding="utf-8")
    data.write_text("{}\n", encoding="utf-8")
    stages = []
    for stage_kind in (StageKind.FULL_TRAINING, StageKind.SMOKE_TRAINING):
        script = tmp_path / f"{stage_kind.value}.py"
        script.write_text("print('training')\n", encoding="utf-8")
        stages.append(
            PinnedRemoteStageProfile(
                stage=stage_kind,
                script_path=script,
                script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                input_files=(data,),
                input_sha256={data.name: hashlib.sha256(data.read_bytes()).hexdigest()},
                budget_unit="gpu_hours",
                budget_reservation=0.25,
                code_lineage_binding=ApprovedCodeLineageExecutionBinding(
                    binding_id=f"bashgym-{stage_kind.value}-entrypoint-v1",
                    binding_revision=1,
                    source_repository_profile_id="bashgym-source-v1",
                    entrypoint_path=entrypoint_path,
                ),
            )
        )
    profile = ApprovedRemoteExecutorProfile(
        profile_id="registered-terminal-agent-v1",
        profile_revision=1,
        compute_profile_id=template.manifest.compute_profile_id,
        target_contract_key=template.target_model.target_contract_key,
        target_model_digest=canonical_hash(template.target_model.model_dump(mode="json")),
        host="192.0.2.10",
        username="trainer",
        key_path=str(key),
        stages=tuple(stages),
    )
    return profile, stages[0].script_path


def test_control_template_is_materializable_but_not_launch_ready_offline(tmp_path: Path):
    report = doctor_autoresearch_template(
        builtin_autoresearch_template_definitions()[0],
        workspace_id=WORKSPACE,
        ledger=initialized_ledger(tmp_path),
        executor_profiles={},
        controller=offline_controller(),
    )

    assert report.materializable is True
    assert report.launch_ready is False
    assert report.blocking_codes == ("controller_offline",)
    assert report.quality_claim_eligible is False


def test_real_template_fails_closed_when_installation_bindings_are_missing(tmp_path: Path):
    report = doctor_autoresearch_template(
        definition(),
        workspace_id=WORKSPACE,
        ledger=initialized_ledger(tmp_path),
        executor_profiles={},
        controller=online_controller(),
    )

    assert report.materializable is False
    assert report.launch_ready is False
    assert report.blocking_codes == (
        "data_binding_unresolved",
        "evaluator_binding_unresolved",
        "compute_binding_unresolved",
        "source_repository_binding_unresolved",
        "code_lineage_execution_binding_unresolved",
    )


def test_real_template_requires_exact_ledger_profile_and_material_hashes(tmp_path: Path):
    template = definition()
    ledger = initialized_ledger(tmp_path)
    register_scientific_bindings(ledger)
    profile, script = registered_profile(tmp_path, template)
    registry = {(profile.compute_profile_id, profile.target_contract_key): profile}
    source_root = tmp_path / "source-fixture"
    source_root.mkdir()
    source_repository, _base_commit = initialized_repository(source_root)
    source = source_profile(source_repository)
    source_registry = {source.profile_id: source}

    ready = doctor_autoresearch_template(
        template,
        workspace_id=WORKSPACE,
        ledger=ledger,
        executor_profiles=registry,
        controller=online_controller(),
        source_profiles=source_registry,
    )
    assert ready.materializable is True
    assert ready.launch_ready is True
    assert ready.available is True
    assert ready.blocking_codes == ()

    missing_entrypoint_profile, _missing_script = registered_profile(
        tmp_path, template, entrypoint_path="scripts/missing.py"
    )
    missing_entrypoint = doctor_autoresearch_template(
        template,
        workspace_id=WORKSPACE,
        ledger=ledger,
        executor_profiles={
            (
                missing_entrypoint_profile.compute_profile_id,
                missing_entrypoint_profile.target_contract_key,
            ): missing_entrypoint_profile
        },
        controller=online_controller(),
        source_profiles=source_registry,
    )
    assert missing_entrypoint.blocking_codes == ("code_lineage_execution_binding_unresolved",)

    script.write_text("print('mutated')\n", encoding="utf-8")
    stale = doctor_autoresearch_template(
        template,
        workspace_id=WORKSPACE,
        ledger=ledger,
        executor_profiles=registry,
        controller=online_controller(),
        source_profiles=source_registry,
    )
    assert stale.materializable is False
    assert stale.blocking_codes == ("compute_binding_unresolved",)


def test_doctor_requires_declared_remote_evaluation_stage(tmp_path: Path):
    configured = definition().model_dump(mode="python", exclude={"definition_digest"})
    configured["manifest"]["evaluation_plan"]["required_compute_stages"] = [
        "development_evaluation",
        "full_training",
        "smoke_training",
    ]
    template = AutoResearchTemplateDefinition.model_validate(configured)
    ledger = initialized_ledger(tmp_path)
    register_scientific_bindings(ledger)
    profile, _script = registered_profile(tmp_path, template)

    report = doctor_autoresearch_template(
        template,
        workspace_id=WORKSPACE,
        ledger=ledger,
        executor_profiles={
            (profile.compute_profile_id, profile.target_contract_key): profile
        },
        controller=offline_controller(),
        source_profiles={},
    )

    compute = next(check for check in report.checks if check.check_id == "compute_binding")
    assert compute.ready is False
