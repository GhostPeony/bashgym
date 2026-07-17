"""Runnable no-GPU AutoResearch slice over the real campaign worker."""

from datetime import datetime

from bashgym._compat import UTC
from bashgym.campaigns.artifacts import ArtifactSealer
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
    CampaignTrigger,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
)
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker
from tests.campaigns.test_persistence import create
from tests.campaigns.test_proposals import principal, proposal

NOW = datetime(2026, 7, 14, 18, 0, tzinfo=UTC)


def test_worker_executes_restart_safe_smoke_without_unlocking_quality_search(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()
    create(repository)
    core = AutoResearchCampaignCore(repository)
    core.register(
        AutoResearchCampaignSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="loss",
            metric_direction=MetricDirection.MINIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=2,
                budget_unit="gpu_hours",
                max_total_cost=0.1,
            ),
            created_at=NOW,
        )
    )
    ready = core.prepare(
        "workspace-a",
        "campaign-1",
        controller_id="autoresearch-controller",
        correlation_id="worker-slice-prepare",
        idempotency_prefix="worker-slice-prepare",
    )
    actor = principal(repository)
    active = CampaignService(repository).transition(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.START,
        expected_version=ready.version,
        principal=actor,
        correlation_id="worker-slice-start",
        idempotency_key="worker-slice-start",
    ).campaign

    baseline = proposal("baseline-control-smoke", estimated_cost=0.01).model_copy(
        update={
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {
                    "executor_kind": "fake",
                    "budget_unit": "gpu_hours",
                    "budget_reservation": 0.01,
                    "fake_steps": 4,
                },
            },
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Prove the durable worker and metric-ingestion path.",
                        input_contract={"quality_claim": False},
                    ),
                )
            ),
        }
    )
    submitted = core.submit_baseline(
        baseline,
        expected_version=active.version,
        principal=actor,
        correlation_id="worker-slice-baseline",
        idempotency_key="worker-slice-baseline",
    )
    assert submitted.record.proposal.status.value == "submitted"

    worker = CampaignWorker(
        repository,
        tmp_path / "artifacts",
        ArtifactSealer(b"a" * 32, key_version="autoresearch-worker-slice-v1"),
        data_directory=tmp_path / "data",
        worker_id="autoresearch-worker-slice",
    )
    assert worker.run_once(now=NOW) == "completed"
    attempts = repository.list_attempts("workspace-a", "campaign-1")
    assert len(attempts) == 1
    loss = repository.get_metric_series(
        "workspace-a",
        attempts[0].attempt_id,
        "loss",
        source="training_metrics.jsonl",
    )
    assert loss
    artifacts = repository.list_artifacts("workspace-a", "campaign-1")
    assert artifacts

    outcome = core.record_result(
        AutoResearchResult(
            result_id="result-baseline-control-smoke",
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="baseline-control-smoke",
            study_id=attempts[0].study_id,
            role=ExperimentRole.BASELINE,
            provenance=ExperimentProvenance.SIMULATED,
            outcome=ExperimentOutcome.COMPLETED,
            metric_name="loss",
            metric_value=loss[-1].value,
            actual_cost=0.01,
            attempt_ids=(attempts[0].attempt_id,),
            evidence_references=tuple(item.artifact_id for item in artifacts),
            recorded_at=NOW,
        )
    )
    assert outcome.decision.decision == ResultDecision.INELIGIBLE

    reopened = AutoResearchRepository(database)
    reopened.initialize()
    restored = AutoResearchCampaignCore(reopened).state(
        "workspace-a", "campaign-1", now=NOW
    )
    assert restored.baseline_verified is False
    assert restored.best_proposal_id is None
    assert restored.next_action == AutoResearchNextAction.SUBMIT_BASELINE
