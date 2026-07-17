"""Portable no-GPU smoke for the durable AutoResearch control plane."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.autoresearch import (
    AutoResearchCampaignCore,
    AutoResearchCampaignSpec,
    AutoResearchNextAction,
    AutoResearchRepository,
    AutoResearchResult,
    AutoResearchStopRules,
    ExperimentOutcome,
    ExperimentProvenance,
    ExperimentRole,
    MetricDirection,
    ResultDecision,
)
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    Campaign,
    CampaignKind,
    CampaignManifest,
    CampaignTrigger,
    CredentialKind,
    ManifestRevision,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
    StudyProposalSubmission,
    TargetModelContract,
)
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker

WORKSPACE_ID = "control-smoke-workspace"
CAMPAIGN_ID = "control-smoke-campaign"
ACTOR_ID = "control-smoke-operator"


def _campaign(now: datetime) -> tuple[Campaign, ManifestRevision]:
    campaign = Campaign(
        campaign_id=CAMPAIGN_ID,
        workspace_id=WORKSPACE_ID,
        title="Portable AutoResearch control smoke",
        kind=CampaignKind.GENERAL,
        objective="Prove durable scheduling and evidence without claiming model quality.",
        target_model=TargetModelContract(
            target_contract_key="control-smoke-model",
            base_model_ref=f"fixture://control-smoke@sha256:{'0' * 64}",
            task="control-plane-validation",
            representation_contract={"artifact_role": "non_trainable_fixture"},
        ),
        owner_actor_id=ACTOR_ID,
    )
    revision = ManifestRevision(
        workspace_id=WORKSPACE_ID,
        campaign_id=CAMPAIGN_ID,
        revision=1,
        manifest=CampaignManifest(
            approved_data_scopes=("control-smoke-data",),
            compute_profile_id="fake-control-smoke",
            budget_limits={"gpu_hours": 0.1},
            evaluation_plan={
                "primary_metric": "loss",
                "source_repository_binding_id": "control-smoke-source",
            },
            promotion_gates={"quality_claim_eligible": False},
            max_proposal_rounds=2,
        ),
        actor_id=ACTOR_ID,
        correlation_id=f"control-smoke-create-{int(now.timestamp())}",
    )
    return campaign, revision


def _proposal() -> StudyProposalSubmission:
    return StudyProposalSubmission(
        proposal_id="control-smoke-baseline",
        workspace_id=WORKSPACE_ID,
        campaign_id=CAMPAIGN_ID,
        hypothesis="The durable worker can execute and seal a bounded fake stage.",
        study_family="control-plane-validation",
        primary_variable="executor",
        controlled_variables=("campaign_contract",),
        expected_outcome="A sealed simulated attempt with restart-safe state.",
        falsification_criterion="Reject if scheduling, sealing, metrics, or reopen fails.",
        estimated_cost=0.01,
        priority=50,
        dataset_recipe={"schema_version": "recipe.v1", "data_scope_id": "control-smoke-data"},
        training_recipe={
            "schema_version": "recipe.v1",
            "runtime": {
                "executor_kind": "fake",
                "budget_unit": "gpu_hours",
                "budget_reservation": 0.01,
                "fake_steps": 4,
            },
        },
        evaluation_recipe={"schema_version": "recipe.v1", "primary_metric": "loss"},
        stage_plan=StagePlan(
            items=(
                StagePlanItem(
                    stage=StageKind.SMOKE_TRAINING,
                    disposition=StageDisposition.REQUIRED,
                    reason="Validate the durable worker without model execution.",
                    input_contract={"quality_claim": False},
                ),
            )
        ),
        rationale="A dependency-free control-plane check for fresh installations.",
    )


def run_autoresearch_control_smoke(directory: Path) -> dict[str, object]:
    """Execute the production campaign path with fake compute and real persistence."""

    root = directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    database = root / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()
    campaign, revision = _campaign(now)
    repository.create_campaign(
        campaign,
        revision,
        actor_id=ACTOR_ID,
        credential_kind=CredentialKind.ACCESS,
        correlation_id="control-smoke-create",
        idempotency_key="control-smoke-create",
    )

    core = AutoResearchCampaignCore(repository)
    core.register(
        AutoResearchCampaignSpec(
            workspace_id=WORKSPACE_ID,
            campaign_id=CAMPAIGN_ID,
            primary_metric="loss",
            metric_direction=MetricDirection.MINIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=2,
                budget_unit="gpu_hours",
                max_total_cost=0.1,
            ),
            created_at=now,
        )
    )
    ready = core.prepare(
        WORKSPACE_ID,
        CAMPAIGN_ID,
        controller_id="control-smoke-controller",
        correlation_id="control-smoke-prepare",
        idempotency_prefix="control-smoke-prepare",
    )
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id=ACTOR_ID,
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=(WORKSPACE_ID,),
    )
    principal = auth.authenticate_access(auth.exchange_refresh(refresh.raw_token).raw_token)
    active = (
        CampaignService(repository)
        .transition(
            WORKSPACE_ID,
            CAMPAIGN_ID,
            CampaignTrigger.START,
            expected_version=ready.version,
            principal=principal,
            correlation_id="control-smoke-start",
            idempotency_key="control-smoke-start",
        )
        .campaign
    )
    submitted = core.submit_baseline(
        _proposal(),
        expected_version=active.version,
        principal=principal,
        correlation_id="control-smoke-baseline",
        idempotency_key="control-smoke-baseline",
    )

    worker = CampaignWorker(
        repository,
        root / "artifacts",
        ArtifactSealer(b"control-smoke-seal-key-material-32", key_version="control-smoke-v1"),
        data_directory=root / "data",
        worker_id="control-smoke-worker",
    )
    worker_result = worker.run_once(now=now)
    attempts = repository.list_attempts(WORKSPACE_ID, CAMPAIGN_ID)
    if worker_result != "completed" or len(attempts) != 1:
        raise RuntimeError("control smoke worker did not complete exactly one attempt")
    loss = repository.get_metric_series(
        WORKSPACE_ID,
        attempts[0].attempt_id,
        "loss",
        source="training_metrics.jsonl",
    )
    artifacts = repository.list_artifacts(WORKSPACE_ID, CAMPAIGN_ID)
    if not loss or not artifacts:
        raise RuntimeError("control smoke did not persist metrics and sealed artifacts")

    outcome = core.record_result(
        AutoResearchResult(
            result_id="control-smoke-result",
            workspace_id=WORKSPACE_ID,
            campaign_id=CAMPAIGN_ID,
            proposal_id=submitted.record.proposal.proposal_id,
            study_id=attempts[0].study_id,
            role=ExperimentRole.BASELINE,
            provenance=ExperimentProvenance.SIMULATED,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name="loss",
            metric_value=loss[-1].value,
            actual_cost=0.01,
            attempt_ids=(attempts[0].attempt_id,),
            evidence_references=tuple(item.artifact_id for item in artifacts),
            recorded_at=now,
        )
    )
    reopened = AutoResearchRepository(database)
    reopened.initialize()
    restored = AutoResearchCampaignCore(reopened).state(WORKSPACE_ID, CAMPAIGN_ID, now=now)
    checks = {
        "campaign_ready_and_started": active.status.value == "active",
        "fake_attempt_completed": worker_result == "completed",
        "metrics_persisted": bool(loss),
        "artifact_sealed": bool(artifacts),
        "simulated_result_ineligible": outcome.decision.decision == ResultDecision.INELIGIBLE,
        "restart_recovered": restored.next_action == AutoResearchNextAction.SUBMIT_BASELINE,
        "quality_baseline_locked": not restored.baseline_verified
        and restored.best_proposal_id is None,
    }
    return {
        "schema_version": "autoresearch_control_smoke.v1",
        "ok": all(checks.values()),
        "checks": checks,
        "attempt_count": len(attempts),
        "artifact_count": len(artifacts),
        "metric_points": len(loss),
        "decision": outcome.decision.decision.value,
        "next_action": restored.next_action.value,
    }


__all__ = ["run_autoresearch_control_smoke"]
