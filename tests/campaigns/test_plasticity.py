from datetime import UTC, datetime, timedelta

from bashgym.campaigns.autoresearch import (
    AutoResearchDiagnosticResult,
    AutoResearchProposalControl,
    ExperimentRole,
)
from bashgym.campaigns.plasticity import build_plasticity_comparison

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
RECIPE_DIGEST = "a" * 64


def _control(
    proposal_id: str,
    role: ExperimentRole,
    parent_proposal_id: str | None,
) -> AutoResearchProposalControl:
    return AutoResearchProposalControl(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id=proposal_id,
        role=role,
        parent_proposal_id=parent_proposal_id,
        changed_variables=(
            ("training_recipe.max_steps",) if role == ExperimentRole.CANDIDATE else ()
        ),
        created_at=NOW,
    )


def _result(
    proposal_id: str,
    *,
    initial: float,
    final: float,
    retention_delta: float,
    cumulative_steps: int,
    recorded_at: datetime,
    recipe_digest: str = RECIPE_DIGEST,
    probe_sample_count: int = 64,
) -> AutoResearchDiagnosticResult:
    measurements = (
        ("initial_probe_metric", initial, probe_sample_count, "accuracy"),
        ("final_probe_metric", final, probe_sample_count, "accuracy"),
        ("retention_delta", retention_delta, probe_sample_count, "accuracy"),
        ("cumulative_training_steps", cumulative_steps, 1, "steps"),
        ("cumulative_training_tokens", cumulative_steps * 128, 1, "tokens"),
        ("dataset_revision_count", 1 if cumulative_steps == 160 else 2, 1, "revisions"),
    )
    return AutoResearchDiagnosticResult(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id=proposal_id,
        study_id=f"study-{proposal_id}",
        attempt_id=f"attempt-{proposal_id}",
        status="completed",
        projection={
            "schema_version": "bashgym.research_diagnostic_result.v1",
            "probe_family": "plasticity_probe",
            "status": "completed",
            "comparison_contract": {
                "metric_direction": "maximize",
                "fixed_step_budget": 20,
                "minimum_efficiency_ratio": 0.75,
                "maximum_retention_drop": 0.02,
                "sample_limit": 64,
                "seed": 7,
                "data_scope_ids": ["probe-split-v1"],
            },
            "measurements": [
                {
                    "name": name,
                    "value": value,
                    "sample_count": sample_count,
                    "unit": unit,
                }
                for name, value, sample_count, unit in measurements
            ],
            "evidence_reference": {
                "proposal_id": proposal_id,
                "study_id": f"study-{proposal_id}",
                "attempt_id": f"attempt-{proposal_id}",
                "recipe_digest": recipe_digest,
            },
        },
        actual_cost=0.05,
        recorded_at=recorded_at,
    )


def _controls() -> tuple[AutoResearchProposalControl, ...]:
    return (
        _control("baseline", ExperimentRole.BASELINE, None),
        _control("candidate-one", ExperimentRole.CANDIDATE, "baseline"),
        _control("candidate-two", ExperimentRole.CANDIDATE, "candidate-one"),
        _control("probe-one", ExperimentRole.DIAGNOSTIC, "candidate-one"),
        _control("probe-two", ExperimentRole.DIAGNOSTIC, "candidate-two"),
    )


def test_comparable_probes_distinguish_plasticity_loss_from_retention() -> None:
    comparison = build_plasticity_comparison(
        diagnostic_results=(
            _result(
                "probe-one",
                initial=0.2,
                final=0.6,
                retention_delta=-0.01,
                cumulative_steps=160,
                recorded_at=NOW,
            ),
            _result(
                "probe-two",
                initial=0.2,
                final=0.4,
                retention_delta=-0.01,
                cumulative_steps=240,
                recorded_at=NOW + timedelta(minutes=1),
            ),
        ),
        controls=_controls(),
    )

    assert comparison["status"] == "comparable"
    assert comparison["classification"] == "plasticity_loss_suspected"
    assert comparison["comparison"] == {
        "reference_parent_proposal_id": "candidate-one",
        "latest_parent_proposal_id": "candidate-two",
        "reference_lineage_depth": 1,
        "latest_lineage_depth": 2,
        "adaptation_efficiency_ratio": 0.5,
        "retention_delta_change": 0.0,
        "minimum_efficiency_ratio": 0.75,
        "maximum_retention_drop": 0.02,
    }
    assert comparison["observations"][1]["adaptation_gain_per_step"] == 0.01


def test_retention_drop_is_not_mislabeled_as_plasticity_loss() -> None:
    comparison = build_plasticity_comparison(
        diagnostic_results=(
            _result(
                "probe-one",
                initial=0.2,
                final=0.6,
                retention_delta=0.0,
                cumulative_steps=160,
                recorded_at=NOW,
            ),
            _result(
                "probe-two",
                initial=0.2,
                final=0.58,
                retention_delta=-0.08,
                cumulative_steps=240,
                recorded_at=NOW + timedelta(minutes=1),
            ),
        ),
        controls=_controls(),
    )

    assert comparison["classification"] == "retention_regression_observed"


def test_different_probe_contracts_remain_incomparable() -> None:
    comparison = build_plasticity_comparison(
        diagnostic_results=(
            _result(
                "probe-one",
                initial=0.2,
                final=0.6,
                retention_delta=0.0,
                cumulative_steps=160,
                recorded_at=NOW,
            ),
            _result(
                "probe-two",
                initial=0.2,
                final=0.4,
                retention_delta=0.0,
                cumulative_steps=240,
                recorded_at=NOW + timedelta(minutes=1),
                recipe_digest="b" * 64,
            ),
        ),
        controls=_controls(),
    )

    assert comparison == {
        "schema_version": "bashgym.autoresearch_plasticity_comparison.v1",
        "status": "insufficient_comparable_probes",
        "reason_code": "matching_fixed_probe_required",
        "classification": None,
        "observations": [],
        "comparison": None,
    }


def test_same_recipe_with_different_observed_sample_counts_remains_incomparable() -> None:
    comparison = build_plasticity_comparison(
        diagnostic_results=(
            _result(
                "probe-one",
                initial=0.2,
                final=0.6,
                retention_delta=0.0,
                cumulative_steps=160,
                recorded_at=NOW,
            ),
            _result(
                "probe-two",
                initial=0.2,
                final=0.4,
                retention_delta=0.0,
                cumulative_steps=240,
                recorded_at=NOW + timedelta(minutes=1),
                probe_sample_count=48,
            ),
        ),
        controls=_controls(),
    )

    assert comparison["status"] == "insufficient_comparable_probes"
    assert comparison["reason_code"] == "matching_fixed_probe_required"
