from __future__ import annotations

import pytest

from bashgym.campaigns.experiment_power import (
    AUTORESEARCH_EVALUATION_POWER_KEY,
    build_experiment_power_projection,
)


def _outcome(*references: str) -> dict[str, object]:
    return {
        "result": {
            "proposal_id": "candidate-1",
            "evidence_references": list(references),
        }
    }


def _evaluation(result_id: str, slices: dict[str, object]) -> dict[str, object]:
    return {
        "evaluation_result_id": result_id,
        "slice_metrics": slices,
    }


def test_observed_example_count_does_not_invent_sample_sufficiency() -> None:
    projection = build_experiment_power_projection(
        outcome=_outcome("evaluation-1"),
        evaluations=(_evaluation("evaluation-1", {"example_count": 64}),),
        hypothesis_family=None,
    )

    assert projection["evaluation"] == {
        "evaluation_result_id": "evaluation-1",
        "sample_count": 64,
        "sample_count_source": "slice_metrics.example_count",
        "comparison_design": None,
        "uncertainty": None,
        "sufficiency": {
            "status": "not_assessed",
            "criteria": [],
        },
    }
    assert projection["sequential_stopping"] == {
        "status": "not_predeclared",
        "evidence": None,
    }
    assert projection["seed_uncertainty"]["status"] == "not_grouped"
    assert (
        "An observed sample count is not evidence of adequate power." in projection["limitations"]
    )


def test_validated_precision_and_sequential_evidence_are_projected() -> None:
    evidence = {
        "schema_version": "bashgym.autoresearch_evaluation_power.v1",
        "sample_count": 192,
        "comparison_design": "paired",
        "uncertainty_method": "paired_bootstrap",
        "confidence_level": 0.95,
        "interval_lower": 0.02,
        "interval_upper": 0.08,
        "maximum_interval_width": 0.1,
        "minimum_detectable_effect": 0.03,
        "sequential_stopping": {
            "schema_version": "bashgym.autoresearch_sequential_stopping.v1",
            "plan_digest": "a" * 64,
            "method": "confidence_sequence",
            "looks_completed": 3,
            "maximum_sample_count": 256,
            "stopping_reason": "precision_reached",
        },
    }

    projection = build_experiment_power_projection(
        outcome=_outcome("evaluation-1"),
        evaluations=(
            _evaluation(
                "evaluation-1",
                {
                    "example_count": 192,
                    AUTORESEARCH_EVALUATION_POWER_KEY: evidence,
                },
            ),
        ),
        hypothesis_family=None,
    )

    assert projection["evaluation"]["sample_count_source"] == (
        f"slice_metrics.{AUTORESEARCH_EVALUATION_POWER_KEY}.sample_count"
    )
    assert projection["evaluation"]["uncertainty"] == {
        "method": "paired_bootstrap",
        "confidence_level": 0.95,
        "interval_lower": 0.02,
        "interval_upper": 0.08,
        "minimum_detectable_effect": 0.03,
    }
    assert projection["evaluation"]["sufficiency"] == {
        "status": "sufficient",
        "criteria": [
            {
                "criterion": "maximum_interval_width",
                "observed": pytest.approx(0.06),
                "target": 0.1,
                "passed": True,
            }
        ],
    }
    assert projection["sequential_stopping"] == {
        "status": "predeclared",
        "evidence": evidence["sequential_stopping"],
    }


@pytest.mark.parametrize(
    "updates",
    (
        {"confidence_level": 0.95, "interval_lower": 0.1},
        {"target_power": 0.8},
        {
            "sequential_stopping": {
                "schema_version": "bashgym.autoresearch_sequential_stopping.v1",
                "plan_digest": "a" * 64,
                "method": "confidence_sequence",
                "looks_completed": 2,
                "maximum_sample_count": 32,
                "stopping_reason": "precision_reached",
            }
        },
    ),
)
def test_incomplete_or_incoherent_power_evidence_fails_closed(
    updates: dict[str, object],
) -> None:
    evidence = {
        "schema_version": "bashgym.autoresearch_evaluation_power.v1",
        "sample_count": 64,
        "comparison_design": "paired",
        "uncertainty_method": "paired_bootstrap",
        **updates,
    }

    with pytest.raises(ValueError):
        build_experiment_power_projection(
            outcome=_outcome("evaluation-1"),
            evaluations=(
                _evaluation("evaluation-1", {AUTORESEARCH_EVALUATION_POWER_KEY: evidence}),
            ),
            hypothesis_family=None,
        )


def test_repeated_ordinary_evaluations_are_not_sequential_stopping_evidence() -> None:
    projection = build_experiment_power_projection(
        outcome=_outcome("evaluation-1", "evaluation-2"),
        evaluations=(
            _evaluation("evaluation-1", {"sample_count": 64}),
            _evaluation("evaluation-2", {"sample_count": 128}),
        ),
        hypothesis_family=None,
    )

    assert projection["evaluation"]["evaluation_result_id"] == "evaluation-1"
    assert projection["sequential_stopping"]["status"] == "not_predeclared"


def test_seed_uncertainty_is_between_runs_not_per_example_confidence() -> None:
    projection = build_experiment_power_projection(
        outcome=_outcome(),
        evaluations=(),
        hypothesis_family={
            "status": "replicated",
            "training_seeds": [11, 22, 33],
            "completed_real_results": 3,
            "primary_metric_summary": {
                "sample_standard_deviation": 0.03,
                "standard_error": 0.017320508,
                "uncertainty_method": "between_run_sample_standard_deviation",
            },
        },
    )

    assert projection["evaluation"]["sufficiency"]["status"] == "unavailable"
    assert projection["seed_uncertainty"] == {
        "status": "replicated",
        "completed_real_results": 3,
        "distinct_training_seeds": 3,
        "sample_standard_deviation": 0.03,
        "standard_error": 0.017320508,
        "uncertainty_method": "between_run_sample_standard_deviation",
        "limitation": "Between-run variation is not a per-example confidence interval.",
    }
