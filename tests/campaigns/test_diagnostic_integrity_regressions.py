"""Regression checks for receipt identity and finite-sample recovery evidence."""

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from bashgym.campaigns.decision_packet import (
    method_contracts_for_proposal,
    method_evidence_from_diagnostic_results,
)
from bashgym.campaigns.diagnostic_actions import DiagnosticInputBinding
from bashgym.campaigns.first_party_diagnostic_runner import (
    FirstPartyDiagnosticSourceBundle,
    PlasticityProbeSummary,
    SessionRecoveryProbeSummary,
    _plasticity_values,
    _session_recovery_values,
    diagnostic_input_binding_from_executor,
)
from bashgym.campaigns.persistence import CampaignPersistenceError
from bashgym.campaigns.runtime import CampaignRuntimeRepository


def recovery_request():
    return SimpleNamespace(
        recipe=SimpleNamespace(
            sample_limit=10000,
            parameters={
                "recovery_dataset_digest": "a" * 64,
                "reader_contract_digest": "b" * 64,
                "confidence_level": 0.95,
            },
        )
    )


def recovery_source(n=1, **kwargs):
    values = dict(
        data_scope_id="scope-a",
        recovery_dataset_digest="a" * 64,
        reader_contract_digest="b" * 64,
        accepted_recovery_traces=n,
        both_failed=0,
        baseline_only_success=0,
        hinted_only_success=n,
        both_succeeded=0,
    )
    values.update(kwargs)
    return SessionRecoveryProbeSummary(**values)


def test_single_favorable_pair_cannot_establish_positive_recovery_lift():
    source = recovery_source()
    # Legacy receipts stay parseable but cannot produce a new confidence claim.
    with pytest.raises(ValueError, match="independent paired"):
        _session_recovery_values(recovery_request(), source)


def test_plasticity_receipt_without_authoritative_binding_is_ineligible():
    source = PlasticityProbeSummary(
        data_scope_id="scope-a",
        metric_direction="maximize",
        fixed_step_budget=20,
        seed=17,
        sample_count=96,
        initial_probe_metric=0.2,
        final_probe_metric=0.5,
        retention_delta=0,
        cumulative_training_steps=20,
        cumulative_training_tokens=100,
        dataset_revision_count=1,
        parent_model_digest="a" * 64,
        candidate_model_digest="b" * 64,
    )
    request = SimpleNamespace(
        input_binding=None,
        recipe=SimpleNamespace(
            parameters={"metric_direction": "maximize", "fixed_step_budget": 20},
            seed=17,
            sample_limit=96,
        ),
    )
    with pytest.raises(ValueError, match="binding"):
        _plasticity_values(request, source)


@pytest.mark.parametrize("n", [1, 2, 5, 100, 10000])
def test_recovery_bound_remains_conservative_for_uniform_favorable_pairs(n):
    source = recovery_source(
        n,
        sampling_unit="independent_paired_case",
        independent_case_count=n,
        sampling_design_digest="c" * 64,
    )
    bound, count = _session_recovery_values(recovery_request(), source)["recovery_lift_lower_bound"]
    assert count == n
    assert bound == pytest.approx(max(-1, 1 - math.sqrt(2 * math.log(20) / n)))
    assert bound < 1
    if n == 1:
        assert bound <= 0


def test_repeated_pairs_do_not_count_as_independent_cases():
    source = recovery_source(
        100,
        sampling_unit="independent_paired_case",
        independent_case_count=1,
        sampling_design_digest="c" * 64,
    )
    with pytest.raises(ValueError, match="independent paired"):
        _session_recovery_values(recovery_request(), source)


def test_method_evidence_requires_current_matching_inputs_and_retains_contract():
    contract = dict(
        reward_spec_digest="a" * 64, canary_suite_id="canary-a", data_scope_ids=["scope-a"]
    )

    def result(contract, failures):
        return dict(
            probe_family="reward_integrity_probe",
            status="completed",
            comparison_contract=contract,
            measurements=[
                dict(name="reward_canary_cases", value=100),
                dict(name="reward_canary_failure_rate", value=failures),
                dict(name="hard_constraint_violation_rate", value=0),
            ],
        )

    current = result(contract, 0.2)
    stale = result({**contract, "reward_spec_digest": "b" * 64}, 0)
    assert method_evidence_from_diagnostic_results([current, stale]) == {}
    evidence = method_evidence_from_diagnostic_results(
        [current, stale], expected_contracts={"reward_integrity_probe": contract}
    )
    assert evidence["reward_canary_failure_rate"] == 0.2
    assert evidence["comparison_contracts"]["reward_integrity_probe"] == contract
    assert (
        method_evidence_from_diagnostic_results(
            [current],
            expected_contracts={
                "reward_integrity_probe": {**contract, "data_scope_ids": ["different-scope"]}
            },
        )
        == {}
    )


def test_pinned_plasticity_input_binding_rejects_changed_material_or_recipe(tmp_path):
    source = PlasticityProbeSummary(
        data_scope_id="scope-a",
        metric_direction="maximize",
        fixed_step_budget=20,
        seed=17,
        sample_count=96,
        initial_probe_metric=0.2,
        final_probe_metric=0.5,
        retention_delta=0,
        cumulative_training_steps=20,
        cumulative_training_tokens=100,
        dataset_revision_count=1,
        parent_model_digest="a" * 64,
        candidate_model_digest="b" * 64,
        probe_recipe_digest="c" * 64,
    )
    path = tmp_path / "autoresearch_diagnostic_sources.json"
    path.write_text(FirstPartyDiagnosticSourceBundle(sources=(source,)).model_dump_json())
    executor = dict(
        diagnostic_recipe=dict(probe_family="plasticity_probe", data_scope_ids=["scope-a"]),
        recipe_digest="c" * 64,
        input_files=[str(path)],
        expected_input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    binding = DiagnosticInputBinding.model_validate(
        diagnostic_input_binding_from_executor(executor)
    )
    assert binding.parent_model_digest == "a" * 64
    assert binding.candidate_model_digest == "b" * 64
    request = SimpleNamespace(
        input_binding=binding,
        recipe=SimpleNamespace(
            parameters={"metric_direction": "maximize", "fixed_step_budget": 20},
            seed=17,
            sample_limit=96,
        ),
    )
    assert _plasticity_values(request, source)["final_probe_metric"] == (0.5, 96)
    for field in ("parent_model_digest", "candidate_model_digest", "probe_recipe_digest"):
        with pytest.raises(ValueError, match="binding"):
            _plasticity_values(request, source.model_copy(update={field: "d" * 64}))
    with pytest.raises(ValueError, match="recipe mismatch"):
        diagnostic_input_binding_from_executor({**executor, "recipe_digest": "d" * 64})
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="digest mismatch"):
        diagnostic_input_binding_from_executor(executor)


def test_parent_checkpoint_binding_uses_completed_campaign_ledger_not_recipe_claims():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE autoresearch_proposal_controls (
          workspace_id TEXT, campaign_id TEXT, proposal_id TEXT,
          parent_proposal_id TEXT, role TEXT);
        CREATE TABLE campaign_studies (
          workspace_id TEXT, campaign_id TEXT, proposal_id TEXT, study_id TEXT);
        CREATE TABLE campaign_actions (
          workspace_id TEXT, campaign_id TEXT, study_id TEXT, action_id TEXT,
          stage_kind TEXT, status TEXT);
        CREATE TABLE campaign_attempts (
          workspace_id TEXT, action_id TEXT, status TEXT, executor_json TEXT);
        INSERT INTO autoresearch_proposal_controls VALUES ('w', 'c', 'probe', 'parent', 'diagnostic');
        INSERT INTO campaign_studies VALUES ('w', 'c', 'parent', 'study');
        INSERT INTO campaign_actions VALUES ('w', 'c', 'study', 'action',
                                            'development_evaluation', 'completed');
    """)
    connection.execute(
        "INSERT INTO campaign_attempts VALUES (?, ?, ?, ?)",
        (
            "w",
            "action",
            "completed",
            json.dumps(
                {
                    "kind": "ssh_remote",
                    "stage": "development_evaluation",
                    "evaluated_model_digest": "a" * 64,
                }
            ),
        ),
    )

    @contextmanager
    def db():
        yield connection

    repository = SimpleNamespace(_connection=db)
    validate = CampaignRuntimeRepository.validate_diagnostic_parent_binding
    validate(repository, "w", "c", "probe", {"parent_model_digest": "a" * 64})
    for campaign, digest in (("other", "a" * 64), ("c", "b" * 64)):
        with pytest.raises(CampaignPersistenceError, match="parent_checkpoint_mismatch"):
            validate(repository, "w", campaign, "probe", {"parent_model_digest": digest})
    connection.execute("UPDATE campaign_attempts SET status = 'failed'")
    with pytest.raises(CampaignPersistenceError, match="parent_checkpoint_mismatch"):
        validate(repository, "w", "c", "probe", {"parent_model_digest": "a" * 64})
    connection.close()


def test_method_scope_comes_from_selected_proposal_and_rejects_conflicting_recipes():
    selected = SimpleNamespace(
        dataset_recipe={"data_scope_ids": ["scope-a"], "preference_dataset_digest": "a" * 64},
        training_recipe={"labeling_contract_digest": "b" * 64},
        evaluation_recipe={},
    )
    assert method_contracts_for_proposal(selected) == {
        "preference_integrity_probe": {
            "data_scope_ids": ["scope-a"],
            "preference_dataset_digest": "a" * 64,
            "labeling_contract_digest": "b" * 64,
        },
    }
    selected.evaluation_recipe["preference_dataset_digest"] = "c" * 64
    assert method_contracts_for_proposal(selected) == {}
