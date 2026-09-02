from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.diagnostic_actions import (
    AutoResearchDiagnosticEvidence,
    AutoResearchDiagnosticRecipe,
    AutoResearchDiagnosticRequest,
    DiagnosticMeasurementRequest,
    diagnostic_recipe_digest,
)
from bashgym.campaigns.first_party_diagnostic_runner import (
    FIRST_PARTY_DIAGNOSTIC_RUNNER_ID,
    FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION,
    FirstPartyDiagnosticSourceBundle,
    PlasticityProbeSummary,
    PreferenceIntegritySummary,
    RewardIntegritySummary,
    run_first_party_diagnostic,
)
from bashgym.campaigns.reward_integrity import build_reward_integrity_evidence
from bashgym.environments.contracts import (
    EnvironmentSpec,
    RewardComponentSpec,
    RewardConstraintSpec,
    VerifierSpec,
)


def _recipe(
    probe_family: str,
    measurements: tuple[tuple[str, str], ...],
    *,
    parameters: dict | None = None,
    sample_limit: int = 100,
    seed: int = 17,
) -> AutoResearchDiagnosticRecipe:
    return AutoResearchDiagnosticRecipe(
        probe_family=probe_family,
        question="Which measured condition should determine the next experiment?",
        hypothesis="The pinned aggregate evidence distinguishes the available methods.",
        informs_methods=("sft",),
        measurements=tuple(
            DiagnosticMeasurementRequest(
                name=name,
                interpretation="observe",
                unit=unit,
            )
            for name, unit in measurements
        ),
        sample_limit=sample_limit,
        seed=seed,
        data_scope_ids=("scope-a",),
        parameters=parameters or {},
    )


def _run(
    tmp_path: Path,
    recipe: AutoResearchDiagnosticRecipe,
    bundle: FirstPartyDiagnosticSourceBundle,
) -> AutoResearchDiagnosticEvidence:
    request = AutoResearchDiagnosticRequest(
        workspace_id="workspace-a",
        campaign_id="campaign-a",
        proposal_id="diagnostic-a",
        study_id="study-a",
        action_id="action-a",
        attempt_id="attempt-a",
        recipe=recipe,
        recipe_digest=diagnostic_recipe_digest(recipe),
        runner_id=FIRST_PARTY_DIAGNOSTIC_RUNNER_ID,
        runner_version=FIRST_PARTY_DIAGNOSTIC_RUNNER_VERSION,
    )
    request_path = tmp_path / "autoresearch_diagnostic_request.json"
    source_path = tmp_path / "autoresearch_diagnostic_sources.json"
    output_path = tmp_path / "autoresearch_diagnostic.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    source_path.write_text(bundle.model_dump_json(), encoding="utf-8")

    run_first_party_diagnostic(request_path, output_path, source_path=source_path)

    return AutoResearchDiagnosticEvidence.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )


def test_runner_projects_fixed_budget_plasticity_receipt(tmp_path: Path) -> None:
    measurements = (
        ("initial_probe_metric", "accuracy"),
        ("final_probe_metric", "accuracy"),
        ("retention_delta", "accuracy"),
        ("cumulative_training_steps", "steps"),
        ("cumulative_training_tokens", "tokens"),
        ("dataset_revision_count", "revisions"),
    )
    recipe = _recipe(
        "plasticity_probe",
        measurements,
        parameters={
            "metric_direction": "maximize",
            "fixed_step_budget": 20,
            "minimum_efficiency_ratio": 0.75,
            "maximum_retention_drop": 0.02,
        },
        sample_limit=96,
    )
    bundle = FirstPartyDiagnosticSourceBundle(
        sources=(
            PlasticityProbeSummary(
                data_scope_id="scope-a",
                metric_direction="maximize",
                fixed_step_budget=20,
                seed=17,
                sample_count=96,
                initial_probe_metric=0.2,
                final_probe_metric=0.5,
                retention_delta=-0.01,
                cumulative_training_steps=160,
                cumulative_training_tokens=4096,
                dataset_revision_count=2,
                parent_model_digest="a" * 64,
                candidate_model_digest="b" * 64,
            ),
        )
    )

    evidence = _run(tmp_path, recipe, bundle)

    assert evidence.status == "completed"
    assert [(item.name, item.value, item.sample_count) for item in evidence.measurements] == [
        ("initial_probe_metric", 0.2, 96),
        ("final_probe_metric", 0.5, 96),
        ("retention_delta", -0.01, 96),
        ("cumulative_training_steps", 160.0, 1),
        ("cumulative_training_tokens", 4096.0, 1),
        ("dataset_revision_count", 2.0, 1),
    ]


def test_runner_projects_existing_reward_integrity_evidence(tmp_path: Path) -> None:
    environment = EnvironmentSpec(
        id="reward-env",
        instruction="Return a verified answer.",
        source="unit",
        verifier=VerifierSpec(
            command="python verify.py",
            reward_type="components",
            reward_components=[
                RewardComponentSpec(
                    name="correctness",
                    weight=1.0,
                    hard_constraint=RewardConstraintSpec(minimum=0.8),
                )
            ],
        ),
    )
    reward_evidence = build_reward_integrity_evidence(
        environment,
        reward_rows=({"correctness": 1.0}, {"correctness": 0.5}),
        canary_summary={"total": 4, "guarded": 3, "failed": 1, "guard_rate": 0.75},
    )
    recipe = _recipe(
        "reward_integrity_probe",
        (
            ("reward_canary_cases", "cases"),
            ("reward_canary_failure_rate", "fraction"),
            ("hard_constraint_violation_rate", "fraction"),
        ),
        parameters={
            "reward_spec_digest": reward_evidence.reward_spec.reward_spec_digest,
            "canary_suite_id": "reward-canaries-v1",
        },
    )
    bundle = FirstPartyDiagnosticSourceBundle(
        sources=(
            RewardIntegritySummary(
                data_scope_id="scope-a",
                canary_suite_id="reward-canaries-v1",
                evidence=reward_evidence,
            ),
        )
    )

    evidence = _run(tmp_path, recipe, bundle)

    assert [(item.name, item.value, item.sample_count) for item in evidence.measurements] == [
        ("reward_canary_cases", 4.0, 4),
        ("reward_canary_failure_rate", 0.25, 4),
        ("hard_constraint_violation_rate", 0.5, 2),
    ]


def test_runner_derives_preference_rates_and_wilson_lower_bound(tmp_path: Path) -> None:
    recipe = _recipe(
        "preference_integrity_probe",
        (
            ("preference_pairs", "pairs"),
            ("preference_agreement_lower_bound", "fraction"),
            ("ambiguous_pair_rate", "fraction"),
            ("preference_position_bias_rate", "fraction"),
            ("preference_label_conflict_rate", "fraction"),
            ("preference_contamination_rate", "fraction"),
        ),
        parameters={
            "preference_dataset_digest": "c" * 64,
            "labeling_contract_digest": "d" * 64,
        },
    )
    bundle = FirstPartyDiagnosticSourceBundle(
        sources=(
            PreferenceIntegritySummary(
                data_scope_id="scope-a",
                preference_dataset_digest="c" * 64,
                labeling_contract_digest="d" * 64,
                preference_pairs=100,
                agreement_cases=100,
                agreement_successes=90,
                ambiguous_pairs=5,
                position_swap_cases=50,
                position_swap_disagreements=2,
                label_conflicts=1,
                heldout_overlaps=0,
            ),
        )
    )

    evidence = _run(tmp_path, recipe, bundle)
    observed = {item.name: item for item in evidence.measurements}

    assert observed["preference_pairs"].value == 100
    assert observed["preference_agreement_lower_bound"].value == pytest.approx(0.8256343384950865)
    assert observed["ambiguous_pair_rate"].value == 0.05
    assert observed["preference_position_bias_rate"].value == 0.04
    assert observed["preference_label_conflict_rate"].value == 0.01
    assert observed["preference_contamination_rate"].value == 0.0


def test_runner_derives_teacher_gap_and_output_acceptance(tmp_path: Path) -> None:
    recipe = _recipe(
        "teacher_gap_probe",
        (
            ("teacher_metric_gap", "accuracy"),
            ("teacher_output_acceptance_rate", "fraction"),
        ),
        parameters={
            "evaluation_suite_id": "heldout-v1",
            "metric_direction": "maximize",
            "teacher_model_digest": "e" * 64,
            "student_model_digest": "f" * 64,
            "output_validation_contract_digest": "a" * 64,
        },
        sample_limit=100,
    )
    bundle = FirstPartyDiagnosticSourceBundle.model_validate(
        {
            "sources": [
                {
                    "source_kind": "teacher_gap_probe_summary",
                    "data_scope_id": "scope-a",
                    "evaluation_suite_id": "heldout-v1",
                    "metric_direction": "maximize",
                    "teacher_model_digest": "e" * 64,
                    "student_model_digest": "f" * 64,
                    "output_validation_contract_digest": "a" * 64,
                    "sample_count": 64,
                    "teacher_metric": 0.75,
                    "student_metric": 0.5,
                    "evaluated_outputs": 80,
                    "accepted_outputs": 72,
                }
            ]
        }
    )

    evidence = _run(tmp_path, recipe, bundle)

    assert [(item.name, item.value, item.sample_count) for item in evidence.measurements] == [
        ("teacher_metric_gap", 0.25, 64),
        ("teacher_output_acceptance_rate", 0.9, 80),
    ]


def test_runner_derives_paired_session_recovery_lower_bound(tmp_path: Path) -> None:
    recipe = _recipe(
        "recovery_trace_probe",
        (
            ("recovery_traces", "traces"),
            ("recovery_lift_lower_bound", "fraction"),
        ),
        parameters={
            "recovery_dataset_digest": "b" * 64,
            "reader_contract_digest": "c" * 64,
            "confidence_level": 0.95,
        },
        sample_limit=100,
    )
    bundle = FirstPartyDiagnosticSourceBundle.model_validate(
        {
            "sources": [
                {
                    "source_kind": "session_recovery_probe_summary",
                    "data_scope_id": "scope-a",
                    "recovery_dataset_digest": "b" * 64,
                    "reader_contract_digest": "c" * 64,
                    "confidence_level": 0.95,
                    "accepted_recovery_traces": 80,
                    "both_failed": 50,
                    "baseline_only_success": 5,
                    "hinted_only_success": 25,
                    "both_succeeded": 20,
                }
            ]
        }
    )

    evidence = _run(tmp_path, recipe, bundle)

    assert evidence.measurements[0].name == "recovery_traces"
    assert evidence.measurements[0].value == 80
    assert evidence.measurements[0].sample_count == 80
    assert evidence.measurements[1].name == "recovery_lift_lower_bound"
    assert evidence.measurements[1].value == pytest.approx(0.10006105396891199)
    assert evidence.measurements[1].sample_count == 100


def test_session_recovery_source_rejects_inconsistent_pair_counts() -> None:
    with pytest.raises(ValidationError, match="session recovery counts are inconsistent"):
        FirstPartyDiagnosticSourceBundle.model_validate(
            {
                "sources": [
                    {
                        "source_kind": "session_recovery_probe_summary",
                        "data_scope_id": "scope-a",
                        "recovery_dataset_digest": "b" * 64,
                        "reader_contract_digest": "c" * 64,
                        "confidence_level": 0.95,
                        "accepted_recovery_traces": 101,
                        "both_failed": 50,
                        "baseline_only_success": 5,
                        "hinted_only_success": 25,
                        "both_succeeded": 20,
                    }
                ]
            }
        )


def test_runner_returns_typed_unsupported_for_uninstalled_probe(tmp_path: Path) -> None:
    recipe = _recipe("new_agent_probe", (("novel_signal", "fraction"),))

    evidence = _run(tmp_path, recipe, FirstPartyDiagnosticSourceBundle(sources=()))

    assert evidence.status == "unsupported"
    assert evidence.unsupported_reason == "diagnostic_source_unavailable"
    assert evidence.measurements == ()


def test_preference_source_rejects_inconsistent_aggregate_counts() -> None:
    with pytest.raises(ValidationError, match="preference integrity counts are inconsistent"):
        PreferenceIntegritySummary(
            data_scope_id="scope-a",
            preference_dataset_digest="c" * 64,
            labeling_contract_digest="d" * 64,
            preference_pairs=10,
            agreement_cases=10,
            agreement_successes=11,
            ambiguous_pairs=0,
            position_swap_cases=4,
            position_swap_disagreements=0,
            label_conflicts=0,
            heldout_overlaps=0,
        )
