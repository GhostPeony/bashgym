from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from bashgym.campaigns.diagnostic_actions import (
    AutoResearchDiagnosticEvidence,
    AutoResearchDiagnosticRecipe,
    AutoResearchDiagnosticRequest,
    DiagnosticMeasurementRequest,
    diagnostic_recipe_digest,
    diagnostic_request_bytes,
    public_diagnostic_projection,
    validate_diagnostic_envelope,
    validated_diagnostic_evidence,
)


def _recipe(**updates: object) -> AutoResearchDiagnosticRecipe:
    payload: dict[str, object] = {
        "probe_family": "novel_gradient_conflict_probe",
        "question": "Do hard formatting examples conflict with count supervision?",
        "hypothesis": "Gradient conflict is concentrated in mixed count-and-format batches.",
        "informs_methods": ("sft", "rlvr"),
        "measurements": (
            {
                "name": "gradient_conflict_rate",
                "interpretation": "minimize",
                "unit": "fraction",
            },
            {
                "name": "hard_slice_coverage",
                "interpretation": "maximize",
                "unit": "fraction",
            },
        ),
        "sample_limit": 96,
        "seed": 17,
        "data_scope_ids": ("approved-train",),
        "parameters": {
            "batch_size": 8,
            "compare_layers": [8, 12, 16],
            "normalize": True,
            "note": "compare hard and ordinary examples",
        },
    }
    payload.update(updates)
    return AutoResearchDiagnosticRecipe.model_validate(payload)


def _evidence(recipe: AutoResearchDiagnosticRecipe, **updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-a",
        "proposal_id": "diagnostic-a",
        "study_id": "study-a",
        "action_id": "action-a",
        "attempt_id": "attempt-a",
        "recipe_digest": diagnostic_recipe_digest(recipe),
        "runner_id": "generic-diagnostic-runner",
        "runner_version": "1",
        "status": "completed",
        "measurements": (
            {
                "name": "gradient_conflict_rate",
                "value": 0.31,
                "sample_count": 96,
                "unit": "fraction",
            },
            {
                "name": "hard_slice_coverage",
                "value": 0.82,
                "sample_count": 96,
                "unit": "fraction",
            },
        ),
        "observations": (
            {
                "observation_id": "mixed-batch-conflict",
                "category": "optimization_conflict",
                "summary": "Mixed batches showed the highest aggregate conflict rate.",
                "count": 23,
            },
        ),
        "resource_usage": (
            {
                "unit": "wall_clock_seconds",
                "amount": 12.5,
                "source": "diagnostic_runner",
                "confidence": "measured",
            },
        ),
    }
    payload.update(updates)
    return payload


def test_recipe_accepts_agent_authored_probe_family_and_measurements() -> None:
    recipe = _recipe()

    assert recipe.probe_family == "novel_gradient_conflict_probe"
    assert [item.name for item in recipe.measurements] == [
        "gradient_conflict_rate",
        "hard_slice_coverage",
    ]
    assert recipe.parameters["compare_layers"] == (8, 12, 16)
    assert diagnostic_recipe_digest(recipe) == diagnostic_recipe_digest(
        AutoResearchDiagnosticRecipe.model_validate_json(recipe.model_dump_json())
    )


def test_model_only_diagnostic_does_not_require_an_artificial_data_scope() -> None:
    recipe = _recipe(data_scope_ids=())

    assert (
        validate_diagnostic_envelope(
            recipe,
            approved_data_scopes=frozenset(),
            max_sample_limit=128,
            max_measurements=4,
        )
        == recipe
    )


def test_runner_request_bytes_bind_recipe_attempt_and_runner_deterministically() -> None:
    recipe = _recipe()
    request = AutoResearchDiagnosticRequest(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id="diagnostic-a",
        study_id="study-a",
        action_id="action-a",
        attempt_id="attempt-a",
        recipe=recipe,
        recipe_digest=diagnostic_recipe_digest(recipe),
        runner_id="generic-diagnostic-runner",
        runner_version="1",
    )

    payload = diagnostic_request_bytes(request)

    assert payload == diagnostic_request_bytes(
        AutoResearchDiagnosticRequest.model_validate_json(request.model_dump_json())
    )
    assert payload.endswith(b"\n")
    assert b'"probe_family":"novel_gradient_conflict_probe"' in payload
    assert b"script_path" not in payload


@pytest.mark.parametrize(
    "parameters",
    [
        {"script_path": "runner.py"},
        {"command": "python runner.py"},
        {"token": "secret-value"},
        {"raw_rows": ["example"]},
        {"output": "results.json"},
        {"note": "file:///private/result.json"},
    ],
)
def test_recipe_rejects_execution_private_and_raw_material(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="parameters"):
        _recipe(parameters=parameters)


def test_recipe_rejects_duplicate_measurements_without_restricting_names() -> None:
    with pytest.raises(ValidationError, match="measurement names must be unique"):
        _recipe(
            measurements=(
                DiagnosticMeasurementRequest(name="new_metric", interpretation="observe"),
                DiagnosticMeasurementRequest(name="new_metric", interpretation="maximize"),
            )
        )


def test_recipe_rejects_unsafe_question_but_not_novel_science() -> None:
    with pytest.raises(ValidationError, match="private or secret-like"):
        _recipe(question="Inspect https://private.example/results before deciding.")


def test_envelope_checks_only_campaign_limits_and_scopes() -> None:
    recipe = _recipe()

    assert (
        validate_diagnostic_envelope(
            recipe,
            approved_data_scopes=frozenset({"approved-train", "approved-heldout"}),
            max_sample_limit=128,
            max_measurements=4,
        )
        is recipe
    )
    with pytest.raises(ValueError, match="diagnostic_data_scope_not_approved"):
        validate_diagnostic_envelope(
            _recipe(data_scope_ids=("unapproved",)),
            approved_data_scopes=frozenset({"approved-train"}),
            max_sample_limit=128,
            max_measurements=4,
        )
    with pytest.raises(ValueError, match="diagnostic_sample_limit_exceeded"):
        validate_diagnostic_envelope(
            recipe,
            approved_data_scopes=frozenset({"approved-train"}),
            max_sample_limit=64,
            max_measurements=4,
        )


def test_completed_evidence_requires_exact_requested_measurements_and_identity() -> None:
    recipe = _recipe()

    evidence = validated_diagnostic_evidence(
        _evidence(recipe),
        recipe=recipe,
        expected_identity={
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-a",
            "proposal_id": "diagnostic-a",
            "study_id": "study-a",
            "action_id": "action-a",
            "attempt_id": "attempt-a",
        },
        expected_runner_id="generic-diagnostic-runner",
        expected_runner_version="1",
    )

    assert isinstance(evidence, AutoResearchDiagnosticEvidence)
    assert [item.name for item in evidence.measurements] == [
        "gradient_conflict_rate",
        "hard_slice_coverage",
    ]

    wrong = _evidence(
        recipe,
        measurements=(
            {
                "name": "invented_unrequested_metric",
                "value": 1.0,
                "sample_count": 96,
            },
        ),
    )
    with pytest.raises(ValueError, match="diagnostic_measurements_mismatch"):
        validated_diagnostic_evidence(
            wrong,
            recipe=recipe,
            expected_identity={
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-a",
                "proposal_id": "diagnostic-a",
                "study_id": "study-a",
                "action_id": "action-a",
                "attempt_id": "attempt-a",
            },
            expected_runner_id="generic-diagnostic-runner",
            expected_runner_version="1",
        )


def test_evidence_rejects_nonfinite_values_and_private_observations() -> None:
    recipe = _recipe()
    payload = _evidence(recipe)
    payload["measurements"] = (
        {
            "name": "gradient_conflict_rate",
            "value": math.inf,
            "sample_count": 96,
            "unit": "fraction",
        },
        {
            "name": "hard_slice_coverage",
            "value": 0.82,
            "sample_count": 96,
            "unit": "fraction",
        },
    )
    with pytest.raises(ValidationError, match="finite"):
        AutoResearchDiagnosticEvidence.model_validate(payload)

    payload = _evidence(recipe)
    payload["observations"] = (
        {
            "observation_id": "leak",
            "category": "leak",
            "summary": "Read C:\\Users\\name\\private.json for the example.",
            "count": 1,
        },
    )
    with pytest.raises(ValidationError, match="private or secret-like"):
        AutoResearchDiagnosticEvidence.model_validate(payload)


def test_unsupported_evidence_is_honest_and_has_no_measurements() -> None:
    recipe = _recipe()
    payload = _evidence(
        recipe,
        status="unsupported",
        measurements=(),
        observations=(),
        unsupported_reason="runner_does_not_implement_probe_family",
    )

    evidence = validated_diagnostic_evidence(
        payload,
        recipe=recipe,
        expected_identity={
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-a",
            "proposal_id": "diagnostic-a",
            "study_id": "study-a",
            "action_id": "action-a",
            "attempt_id": "attempt-a",
        },
        expected_runner_id="generic-diagnostic-runner",
        expected_runner_version="1",
    )

    assert evidence.status == "unsupported"
    assert evidence.measurements == ()
    assert evidence.unsupported_reason == "runner_does_not_implement_probe_family"

    with pytest.raises(ValidationError, match="unsupported evidence cannot contain measurements"):
        AutoResearchDiagnosticEvidence.model_validate(
            _evidence(
                recipe,
                status="unsupported",
                unsupported_reason="runner_does_not_implement_probe_family",
            )
        )


def test_public_projection_is_aggregate_and_allowlisted() -> None:
    recipe = _recipe()
    evidence = AutoResearchDiagnosticEvidence.model_validate(_evidence(recipe))

    projected = public_diagnostic_projection(recipe, evidence)

    assert projected == {
        "schema_version": "bashgym.research_diagnostic_result.v1",
        "probe_family": "novel_gradient_conflict_probe",
        "question": "Do hard formatting examples conflict with count supervision?",
        "hypothesis": "Gradient conflict is concentrated in mixed count-and-format batches.",
        "informs_methods": ["sft", "rlvr"],
        "status": "completed",
        "measurements": [
            {
                "name": "gradient_conflict_rate",
                "value": 0.31,
                "sample_count": 96,
                "unit": "fraction",
            },
            {
                "name": "hard_slice_coverage",
                "value": 0.82,
                "sample_count": 96,
                "unit": "fraction",
            },
        ],
        "observations": [
            {
                "observation_id": "mixed-batch-conflict",
                "category": "optimization_conflict",
                "summary": "Mixed batches showed the highest aggregate conflict rate.",
                "count": 23,
            }
        ],
        "resource_usage": [
            {
                "unit": "wall_clock_seconds",
                "amount": 12.5,
                "source": "diagnostic_runner",
                "confidence": "measured",
            }
        ],
        "unsupported_reason": None,
        "evidence_reference": {
            "proposal_id": "diagnostic-a",
            "study_id": "study-a",
            "attempt_id": "attempt-a",
            "recipe_digest": diagnostic_recipe_digest(recipe),
        },
    }
    serialized = str(projected).casefold()
    assert "parameters" not in serialized
    assert "data_scope" not in serialized
    assert "path" not in serialized
    assert "uri" not in serialized


def test_plasticity_probe_projects_only_its_fixed_comparison_contract() -> None:
    recipe = _recipe(
        probe_family="plasticity_probe",
        measurements=(
            DiagnosticMeasurementRequest(
                name="initial_probe_metric", interpretation="observe", unit="accuracy"
            ),
            DiagnosticMeasurementRequest(
                name="final_probe_metric", interpretation="observe", unit="accuracy"
            ),
            DiagnosticMeasurementRequest(
                name="retention_delta", interpretation="observe", unit="accuracy"
            ),
            DiagnosticMeasurementRequest(
                name="cumulative_training_steps", interpretation="observe", unit="steps"
            ),
            DiagnosticMeasurementRequest(
                name="cumulative_training_tokens", interpretation="observe", unit="tokens"
            ),
            DiagnosticMeasurementRequest(
                name="dataset_revision_count", interpretation="observe", unit="revisions"
            ),
        ),
        parameters={
            "metric_direction": "maximize",
            "fixed_step_budget": 20,
            "minimum_efficiency_ratio": 0.75,
            "maximum_retention_drop": 0.02,
            "optimizer_label": "fixed_probe_optimizer",
        },
    )
    evidence = AutoResearchDiagnosticEvidence.model_validate(
        _evidence(
            recipe,
            measurements=(
                {
                    "name": "initial_probe_metric",
                    "value": 0.2,
                    "sample_count": 96,
                    "unit": "accuracy",
                },
                {
                    "name": "final_probe_metric",
                    "value": 0.5,
                    "sample_count": 96,
                    "unit": "accuracy",
                },
                {"name": "retention_delta", "value": -0.01, "sample_count": 96, "unit": "accuracy"},
                {
                    "name": "cumulative_training_steps",
                    "value": 160,
                    "sample_count": 1,
                    "unit": "steps",
                },
                {
                    "name": "cumulative_training_tokens",
                    "value": 4096,
                    "sample_count": 1,
                    "unit": "tokens",
                },
                {
                    "name": "dataset_revision_count",
                    "value": 2,
                    "sample_count": 1,
                    "unit": "revisions",
                },
            ),
        )
    )

    projected = public_diagnostic_projection(recipe, evidence)

    assert projected["comparison_contract"] == {
        "metric_direction": "maximize",
        "fixed_step_budget": 20,
        "minimum_efficiency_ratio": 0.75,
        "maximum_retention_drop": 0.02,
        "sample_limit": 96,
        "seed": 17,
        "data_scope_ids": ["approved-train"],
    }
    assert "optimizer_label" not in str(projected)


def test_plasticity_probe_rejects_missing_fixed_budget_fields() -> None:
    with pytest.raises(ValidationError, match="plasticity probe parameters"):
        _recipe(probe_family="plasticity_probe", parameters={"fixed_step_budget": 20})


def test_reward_integrity_probe_projects_spec_identity_and_bounded_measurements() -> None:
    recipe = _recipe(
        probe_family="reward_integrity_probe",
        informs_methods=("grpo", "rlvr"),
        measurements=(
            DiagnosticMeasurementRequest(
                name="reward_canary_cases", interpretation="maximize", unit="cases"
            ),
            DiagnosticMeasurementRequest(
                name="reward_canary_failure_rate", interpretation="minimize", unit="fraction"
            ),
            DiagnosticMeasurementRequest(
                name="hard_constraint_violation_rate",
                interpretation="minimize",
                unit="fraction",
            ),
        ),
        parameters={
            "reward_spec_digest": "a" * 64,
            "canary_suite_id": "reward-hacking-v1",
        },
    )
    evidence = AutoResearchDiagnosticEvidence.model_validate(
        _evidence(
            recipe,
            measurements=(
                {"name": "reward_canary_cases", "value": 4, "sample_count": 4, "unit": "cases"},
                {
                    "name": "reward_canary_failure_rate",
                    "value": 0.0,
                    "sample_count": 4,
                    "unit": "fraction",
                },
                {
                    "name": "hard_constraint_violation_rate",
                    "value": 0.0,
                    "sample_count": 64,
                    "unit": "fraction",
                },
            ),
        )
    )

    projected = public_diagnostic_projection(recipe, evidence)

    assert projected["comparison_contract"] == {
        "reward_spec_digest": "a" * 64,
        "canary_suite_id": "reward-hacking-v1",
        "sample_limit": 96,
        "seed": 17,
    }


def test_reward_integrity_probe_requires_spec_canary_and_constraint_measurements() -> None:
    with pytest.raises(ValidationError, match="reward integrity probe"):
        _recipe(
            probe_family="reward_integrity_probe",
            parameters={"reward_spec_digest": "not-a-digest"},
        )


@pytest.mark.parametrize(
    ("measurement_name", "invalid_value"),
    (
        ("reward_canary_cases", 0.0),
        ("reward_canary_cases", 1.5),
        ("reward_canary_failure_rate", -0.01),
        ("reward_canary_failure_rate", 1.01),
        ("hard_constraint_violation_rate", -0.01),
        ("hard_constraint_violation_rate", 1.01),
    ),
)
def test_reward_integrity_evidence_rejects_impossible_aggregates(
    measurement_name: str,
    invalid_value: float,
) -> None:
    recipe = _recipe(
        probe_family="reward_integrity_probe",
        informs_methods=("grpo", "rlvr"),
        measurements=tuple(
            DiagnosticMeasurementRequest(
                name=name,
                interpretation="maximize" if name == "reward_canary_cases" else "minimize",
            )
            for name in (
                "reward_canary_cases",
                "reward_canary_failure_rate",
                "hard_constraint_violation_rate",
            )
        ),
        parameters={
            "reward_spec_digest": "a" * 64,
            "canary_suite_id": "reward-hacking-v1",
        },
    )
    measurements = [
        {
            "name": name,
            "value": 4.0 if name == "reward_canary_cases" else 0.0,
            "sample_count": 4,
        }
        for name in (
            "reward_canary_cases",
            "reward_canary_failure_rate",
            "hard_constraint_violation_rate",
        )
    ]
    next(item for item in measurements if item["name"] == measurement_name)["value"] = invalid_value

    with pytest.raises(ValueError, match="reward_integrity_measurement_out_of_range"):
        validated_diagnostic_evidence(
            _evidence(recipe, measurements=tuple(measurements)),
            recipe=recipe,
            expected_identity={
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-a",
                "proposal_id": "diagnostic-a",
                "study_id": "study-a",
                "action_id": "action-a",
                "attempt_id": "attempt-a",
            },
            expected_runner_id="generic-diagnostic-runner",
            expected_runner_version="1",
        )


def test_preference_integrity_probe_projects_contract_identity_and_aggregates() -> None:
    recipe = _recipe(
        probe_family="preference_integrity_probe",
        informs_methods=("dpo",),
        measurements=(
            DiagnosticMeasurementRequest(
                name="preference_pairs", interpretation="maximize", unit="pairs"
            ),
            DiagnosticMeasurementRequest(
                name="preference_agreement_lower_bound",
                interpretation="maximize",
                unit="fraction",
            ),
            DiagnosticMeasurementRequest(
                name="ambiguous_pair_rate", interpretation="minimize", unit="fraction"
            ),
            DiagnosticMeasurementRequest(
                name="preference_position_bias_rate",
                interpretation="minimize",
                unit="fraction",
            ),
            DiagnosticMeasurementRequest(
                name="preference_label_conflict_rate",
                interpretation="minimize",
                unit="fraction",
            ),
            DiagnosticMeasurementRequest(
                name="preference_contamination_rate",
                interpretation="minimize",
                unit="fraction",
            ),
        ),
        parameters={
            "preference_dataset_digest": "b" * 64,
            "labeling_contract_digest": "c" * 64,
        },
        sample_limit=240,
    )
    evidence = validated_diagnostic_evidence(
        _evidence(
            recipe,
            measurements=(
                {"name": "preference_pairs", "value": 240, "sample_count": 240, "unit": "pairs"},
                {
                    "name": "preference_agreement_lower_bound",
                    "value": 0.72,
                    "sample_count": 240,
                    "unit": "fraction",
                },
                {
                    "name": "ambiguous_pair_rate",
                    "value": 0.04,
                    "sample_count": 240,
                    "unit": "fraction",
                },
                {
                    "name": "preference_position_bias_rate",
                    "value": 0.02,
                    "sample_count": 96,
                    "unit": "fraction",
                },
                {
                    "name": "preference_label_conflict_rate",
                    "value": 0.0,
                    "sample_count": 240,
                    "unit": "fraction",
                },
                {
                    "name": "preference_contamination_rate",
                    "value": 0.0,
                    "sample_count": 240,
                    "unit": "fraction",
                },
            ),
        ),
        recipe=recipe,
        expected_identity={
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-a",
            "proposal_id": "diagnostic-a",
            "study_id": "study-a",
            "action_id": "action-a",
            "attempt_id": "attempt-a",
        },
        expected_runner_id="generic-diagnostic-runner",
        expected_runner_version="1",
    )

    projected = public_diagnostic_projection(recipe, evidence)

    assert projected["comparison_contract"] == {
        "preference_dataset_digest": "b" * 64,
        "labeling_contract_digest": "c" * 64,
        "sample_limit": 240,
        "seed": 17,
    }
    assert "prompt" not in str(projected).casefold()
    assert "response" not in str(projected).casefold()


def test_preference_integrity_probe_requires_both_digests_and_all_measurements() -> None:
    with pytest.raises(ValidationError, match="preference integrity probe"):
        _recipe(
            probe_family="preference_integrity_probe",
            informs_methods=("dpo",),
            measurements=(
                DiagnosticMeasurementRequest(
                    name="preference_pairs", interpretation="maximize", unit="pairs"
                ),
            ),
            parameters={
                "preference_dataset_digest": "b" * 64,
                "labeling_contract_digest": "not-a-digest",
            },
        )


def test_preference_integrity_evidence_rejects_impossible_rates() -> None:
    recipe = _recipe(
        probe_family="preference_integrity_probe",
        informs_methods=("dpo",),
        measurements=tuple(
            DiagnosticMeasurementRequest(
                name=name,
                interpretation=(
                    "maximize"
                    if name
                    in {
                        "preference_pairs",
                        "preference_agreement_lower_bound",
                    }
                    else "minimize"
                ),
            )
            for name in (
                "preference_pairs",
                "preference_agreement_lower_bound",
                "ambiguous_pair_rate",
                "preference_position_bias_rate",
                "preference_label_conflict_rate",
                "preference_contamination_rate",
            )
        ),
        parameters={
            "preference_dataset_digest": "b" * 64,
            "labeling_contract_digest": "c" * 64,
        },
    )
    measurements = [
        {"name": name, "value": 240 if name == "preference_pairs" else 0.0, "sample_count": 96}
        for name in (
            "preference_pairs",
            "preference_agreement_lower_bound",
            "ambiguous_pair_rate",
            "preference_position_bias_rate",
            "preference_label_conflict_rate",
            "preference_contamination_rate",
        )
    ]
    measurements[3]["value"] = -0.1

    with pytest.raises(ValueError, match="preference_integrity_measurement_out_of_range"):
        validated_diagnostic_evidence(
            _evidence(recipe, measurements=tuple(measurements)),
            recipe=recipe,
            expected_identity={
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-a",
                "proposal_id": "diagnostic-a",
                "study_id": "study-a",
                "action_id": "action-a",
                "attempt_id": "attempt-a",
            },
            expected_runner_id="generic-diagnostic-runner",
            expected_runner_version="1",
        )
