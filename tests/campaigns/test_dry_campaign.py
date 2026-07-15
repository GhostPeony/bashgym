"""End-to-end three-study dry campaign with resident-worker restart recovery."""

from datetime import timedelta

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.contracts import (
    CampaignTrigger,
    StageDisposition,
    StageKind,
    StagePlan,
    StagePlanItem,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker
from tests.campaigns.test_proposals import activate, principal, proposal
from tests.campaigns.test_worker import START


def dry_proposal(proposal_id: str, *, priority: int, cost: float):
    runtime = {
        "executor_kind": "fake",
        "budget_unit": "gpu_hours",
        "budget_reservation": 0.01,
        "fake_steps": 12,
    }
    return proposal(proposal_id, priority=priority, estimated_cost=cost).model_copy(
        update={
            "training_recipe": {"schema_version": "recipe.v1", "runtime": runtime},
            "stage_plan": StagePlan(
                items=(
                    StagePlanItem(
                        stage=StageKind.SMOKE_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Dry-run runtime and loss-stream proof.",
                        input_contract={"quality_claim": False},
                    ),
                    StagePlanItem(
                        stage=StageKind.FULL_TRAINING,
                        disposition=StageDisposition.REQUIRED,
                        reason="Dry-run full-stage orchestration proof.",
                        input_contract={"quality_claim": False},
                    ),
                )
            ),
        }
    )


def test_three_study_campaign_resumes_across_worker_restarts_and_concludes(tmp_path):
    database = tmp_path / "campaigns.sqlite3"
    repository = CampaignRuntimeRepository(database)
    repository.initialize()
    activate(repository)
    service = CampaignService(repository)
    actor = principal(repository)

    version = 4
    for value in (
        dry_proposal("study-high-cheap", priority=90, cost=1),
        dry_proposal("study-high-costly", priority=90, cost=2),
        dry_proposal("study-lower", priority=50, cost=1),
    ):
        submitted = service.submit_proposal(
            value,
            expected_version=version,
            principal=actor,
            correlation_id=f"submit-{value.proposal_id}",
            idempotency_key=f"submit-{value.proposal_id}",
        )
        version = submitted.campaign.version
    advanced = service.request_advance(
        "workspace-a",
        "campaign-1",
        expected_version=version,
        principal=actor,
        correlation_id="dry-advance",
        idempotency_key="dry-advance",
    )
    assert advanced.event.event_type == "campaign:advance-requested"

    results = []
    for index in range(6):
        reopened = CampaignRuntimeRepository(database)
        reopened.initialize()
        worker = CampaignWorker(
            reopened,
            tmp_path / "artifacts",
            ArtifactSealer(b"d" * 32, key_version="dry-campaign-v1"),
            data_directory=tmp_path / "data-root",
            worker_id=f"worker-restart-{index}",
        )
        results.append(worker.run_once(now=START + timedelta(seconds=20 * (index + 1))))

    final_repository = CampaignRuntimeRepository(database)
    final_repository.initialize()
    assert results == ["completed"] * 6
    attempts = final_repository.list_attempts("workspace-a", "campaign-1")
    assert len(attempts) == 6
    assert all(item.status.value == "completed" for item in attempts)
    assert [item.stage for item in attempts] == [
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
        StageKind.SMOKE_TRAINING,
        StageKind.FULL_TRAINING,
    ]
    with final_repository._connection() as connection:
        studies = connection.execute(
            """
            SELECT p.proposal_id, s.status
            FROM campaign_studies s
            JOIN campaign_proposals p
              ON p.workspace_id = s.workspace_id AND p.proposal_id = s.proposal_id
            ORDER BY s.created_at, s.study_id
            """
        ).fetchall()
    assert [row["proposal_id"] for row in studies] == [
        "study-high-cheap",
        "study-high-costly",
        "study-lower",
    ]
    assert {row["status"] for row in studies} == {"completed"}
    totals = final_repository.budget_totals("workspace-a", "campaign-1", "gpu_hours")
    assert totals["reserved"] == totals["limit_delta"] == 0.0
    assert round(totals["actual"], 6) == 0.06
    assert all(
        final_repository.get_metric_series(
            "workspace-a",
            attempt.attempt_id,
            "loss",
            source="training_metrics.jsonl",
        )
        for attempt in attempts
    )

    final_service = CampaignService(final_repository)
    final_actor = principal(final_repository)
    current = final_repository.get_campaign("workspace-a", "campaign-1")
    concluded = final_service.transition(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.CONCLUDE,
        expected_version=current.version,
        principal=final_actor,
        correlation_id="dry-conclude",
        idempotency_key="dry-conclude",
        stop_reason="Three-study restart-safe dry campaign completed.",
    )
    assert concluded.campaign.status.value == "completed"
    assert concluded.campaign.champion_ref is None
