from __future__ import annotations

import pytest

from bashgym.campaigns.research_diagnostics import build_autoresearch_diagnostics


def outcome(
    role: str,
    decision: str,
    evaluation_id: str,
    *,
    attempt_ids: tuple[str, ...] = (),
) -> dict:
    return {
        "result": {
            "role": role,
            "evidence_references": [evaluation_id, f"run-{role}"],
            "attempt_ids": list(attempt_ids),
        },
        "decision": {"decision": decision},
    }


def evaluation(
    evaluation_id: str,
    run_id: str,
    value: float,
    *,
    slices: dict | None = None,
) -> dict:
    return {
        "evaluation_result_id": evaluation_id,
        "evaluation_suite_id": "suite-fixed",
        "run_id": run_id,
        "metrics": {"exact_accuracy": value},
        "slice_metrics": slices or {},
    }


def build(
    *,
    evaluations: list[dict],
    runs: list[dict] | None = None,
    dataset_versions: list[dict] | None = None,
    training_metrics: list[dict] | None = None,
):
    return build_autoresearch_diagnostics(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        primary_metric="exact_accuracy",
        metric_direction="maximize",
        evaluation_suite_id="suite-fixed",
        outcomes=[
            outcome("baseline", "baseline", "eval-baseline"),
            outcome(
                "candidate",
                "discard",
                "eval-candidate",
                attempt_ids=("attempt-data", "attempt-train", "attempt-eval"),
            ),
        ],
        evaluations=evaluations,
        runs=runs
        or [
            {"run_id": "run-baseline", "campaign_id": "campaign-a", "config": {}},
            {"run_id": "run-candidate", "campaign_id": "campaign-a", "config": {}},
        ],
        dataset_versions=dataset_versions or [],
        training_metrics=training_metrics or [],
    )


def test_flags_collapsed_candidate_and_ranks_bounded_next_work():
    diagnostics = build(
        evaluations=[
            evaluation(
                "eval-baseline",
                "run-baseline",
                0.25,
                slices={
                    "per_color": {"red": {"accuracy": 0.5, "mae": 0.6}},
                    "format_accuracy": 0.90,
                },
            ),
            evaluation(
                "eval-candidate",
                "run-candidate",
                0.20,
                slices={
                    "example_count": 64,
                    "unique_prediction_count": 1,
                    "modal_prediction_fraction": 1.0,
                    "degenerate_constant_output": True,
                    "format_accuracy": 1.0,
                    "per_color": {"red": {"accuracy": 0.4, "mae": 0.8}},
                },
            ),
        ]
    )

    assert diagnostics.low_signal is True
    assert [signal.code for signal in diagnostics.signals] == [
        "primary_metric_no_improvement",
        "degenerate_constant_output",
        "prediction_diversity_collapsed",
        "modal_prediction_dominates",
        "secondary_metrics_regressed",
        "format_only_gain",
        "checkpoint_evidence_missing",
    ]
    slices = {item.slice_path: item for item in diagnostics.error_slices}
    assert slices["per_color.red.accuracy"].status == "regressed"
    assert slices["per_color.red.accuracy"].improvement == pytest.approx(-0.1)
    assert slices["per_color.red.mae"].status == "regressed"
    assert slices["per_color.red.mae"].improvement == pytest.approx(-0.2)
    assert [item.action_kind for item in diagnostics.ranked_hypotheses[:2]] == [
        "diagnostic",
        "evaluation",
    ]
    assert all(not item.eligible_for_submission for item in diagnostics.ranked_hypotheses)


def test_compares_checkpoint_trajectory_in_training_order():
    diagnostics = build(
        evaluations=[
            evaluation("eval-baseline", "run-baseline", 0.25),
            evaluation("eval-candidate", "run-candidate", 0.30),
            evaluation(
                "eval-step-20",
                "run-candidate",
                0.28,
                slices={"autoresearch_role": "checkpoint", "checkpoint_step": 20},
            ),
            evaluation(
                "eval-step-10",
                "run-candidate",
                0.27,
                slices={"autoresearch_role": "checkpoint", "checkpoint_step": 10},
            ),
        ]
    )

    assert [item.evaluation_result_id for item in diagnostics.checkpoint_comparisons] == [
        "eval-step-10",
        "eval-step-20",
        "eval-candidate",
    ]
    assert [item.improvement_from_previous for item in diagnostics.checkpoint_comparisons] == [
        None,
        0.010000000000000009,
        0.019999999999999962,
    ]
    assert "checkpoint_evidence_missing" not in {item.code for item in diagnostics.signals}


def test_compares_checkpoint_trajectory_embedded_in_sealed_candidate_evaluation():
    diagnostics = build(
        evaluations=[
            evaluation("eval-baseline", "run-baseline", 0.25),
            evaluation(
                "eval-candidate",
                "run-candidate",
                0.30,
                slices={
                    "autoresearch_checkpoint_trajectory": [
                        {
                            "observation_id": "checkpoint-step-10",
                            "checkpoint_step": 10,
                            "evaluated_model_manifest_digest": "1" * 64,
                            "metrics": {"exact_accuracy": 0.27},
                            "slice_metrics": {},
                        },
                        {
                            "observation_id": "checkpoint-step-20",
                            "checkpoint_step": 20,
                            "evaluated_model_manifest_digest": "2" * 64,
                            "metrics": {"exact_accuracy": 0.28},
                            "slice_metrics": {},
                        },
                    ]
                },
            ),
        ]
    )

    assert [item.evaluation_result_id for item in diagnostics.checkpoint_comparisons] == [
        "checkpoint-step-10",
        "checkpoint-step-20",
        "eval-candidate",
    ]
    assert "checkpoint_evidence_missing" not in {item.code for item in diagnostics.signals}


def test_minimize_direction_reports_positive_improvement_for_lower_metric():
    diagnostics = build_autoresearch_diagnostics(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        primary_metric="mean_absolute_error",
        metric_direction="minimize",
        evaluation_suite_id="suite-fixed",
        outcomes=[
            outcome("baseline", "baseline", "eval-baseline"),
            outcome("candidate", "keep", "eval-candidate"),
        ],
        evaluations=[
            {
                **evaluation("eval-baseline", "run-baseline", 0.0),
                "metrics": {"mean_absolute_error": 0.8},
            },
            {
                **evaluation("eval-candidate", "run-candidate", 0.0),
                "metrics": {"mean_absolute_error": 0.6},
            },
        ],
        runs=[],
    )

    assert diagnostics.checkpoint_comparisons[0].improvement_from_baseline == pytest.approx(0.2)
    assert "primary_metric_no_improvement" not in {item.code for item in diagnostics.signals}


def test_projection_is_deterministic_for_identical_evidence():
    evaluations = [
        evaluation("eval-baseline", "run-baseline", 0.25),
        evaluation("eval-candidate", "run-candidate", 0.20),
    ]

    first = build(evaluations=evaluations)
    second = build(evaluations=list(reversed(evaluations)))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_data_quality_and_regressed_slice_rank_one_dataset_design_hypothesis():
    diagnostics = build(
        evaluations=[
            evaluation(
                "eval-baseline",
                "run-baseline",
                0.50,
                slices={"per_count": {"many": {"accuracy": 0.60}}},
            ),
            evaluation(
                "eval-candidate",
                "run-candidate",
                0.45,
                slices={"per_count": {"many": {"accuracy": 0.30}}},
            ),
        ],
        dataset_versions=[
            {
                "dataset_version_id": "dataset-generated-1",
                "metadata": {
                    "producer_attempt_id": "attempt-data",
                    "data_quality": {
                        "generated_rows": 100,
                        "accepted_rows": 40,
                        "acceptance_rate": 0.4,
                        "verification_failed_rows": 45,
                        "duplicate_rows_removed": 10,
                        "contamination_rows_removed": 5,
                    },
                },
            }
        ],
    )

    codes = {item.code for item in diagnostics.signals}
    assert "dataset_acceptance_low" in codes
    hypothesis = next(
        item
        for item in diagnostics.ranked_hypotheses
        if item.changed_variable == "dataset_recipe.generation_brief"
    )
    assert hypothesis.action_kind == "candidate"
    assert hypothesis.evidence_references == (
        "dataset-generated-1",
        "eval-candidate",
        "eval-baseline",
    )
    assert "overfit" not in hypothesis.hypothesis.casefold()


def test_train_validation_divergence_requires_paired_trajectory_evidence():
    evaluations = [
        evaluation("eval-baseline", "run-baseline", 0.50),
        evaluation("eval-candidate", "run-candidate", 0.45),
    ]
    without_metrics = build(evaluations=evaluations)
    with_metrics = build(
        evaluations=evaluations,
        training_metrics=[
            {"attempt_id": "attempt-train", "metric_name": "train_loss", "step": 1, "value": 1.0},
            {"attempt_id": "attempt-train", "metric_name": "train_loss", "step": 10, "value": 0.2},
            {
                "attempt_id": "attempt-train",
                "metric_name": "validation_loss",
                "step": 1,
                "value": 0.8,
            },
            {
                "attempt_id": "attempt-train",
                "metric_name": "validation_loss",
                "step": 10,
                "value": 1.1,
            },
        ],
    )

    assert "train_validation_divergence" not in {item.code for item in without_metrics.signals}
    assert "train_validation_divergence" in {item.code for item in with_metrics.signals}
