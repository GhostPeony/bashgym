"""Hash-reconciled campaign export and smoke/finding separation tests."""

from __future__ import annotations

import json

import pytest

from bashgym.campaigns.export import (
    CampaignExportError,
    CampaignExportSnapshot,
    export_campaign_evidence,
)


def snapshot(*, full: bool = False) -> CampaignExportSnapshot:
    stage = "full_training" if full else "smoke_training"
    return CampaignExportSnapshot(
        campaign={
            "campaign_id": "campaign-1",
            "objective": "Improve held-out retrieval",
            "status": "active",
            "champion_ref": "base-qwen",
        },
        attempts=(
            {
                "attempt_id": "attempt-1",
                "study_id": "study-1",
                "stage": stage,
                "status": "completed",
                "candidate_digest": "a" * 64,
                "created_at": "2026-07-13T00:00:00Z",
                "updated_at": "2026-07-13T00:01:00Z",
            },
        ),
        artifacts=(
            {
                "artifact_id": "artifact-1",
                "schema_name": "training_metrics_jsonl.v1",
                "sha256": "b" * 64,
                "size_bytes": 42,
                "sealed": True,
                "valid": True,
                "created_at": "2026-07-13T00:01:00Z",
            },
        ),
        comparisons=(
            {
                "comparison_digest": "c" * 64,
                "champion_digest": "d" * 64,
                "candidate_digest": "a" * 64,
                "sample_count": 300,
                "verdict": "pass",
                "blocking_reasons": [],
                "warnings": [],
                "created_at": "2026-07-13T00:02:00Z",
            },
        ),
        loss_by_attempt={
            "attempt-1": (
                {"step": 1, "source": "training_metrics.jsonl", "value": 1.0},
                {"step": 2, "source": "training_metrics.jsonl", "value": 0.5},
            )
        },
        flags=("protected evaluation not run",),
    )


def test_smoke_export_is_hash_reconciled_but_never_claims_quality_findings(tmp_path):
    manifest = export_campaign_evidence(snapshot(), tmp_path / "export")

    assert manifest["quality_findings_available"] is False
    report = (tmp_path / "export" / "campaign_report.md").read_text(encoding="utf-8")
    assert "No model-quality findings are claimed" in report
    assert "Smoke attempts: 1 (runtime/semantics/memory evidence only)" in report
    assert "protected evaluation not run" in report
    assert {item["name"] for item in manifest["files"]} == {
        "artifacts.csv",
        "attempts.csv",
        "campaign_evidence.json",
        "campaign_report.docx",
        "campaign_report.md",
        "campaign_report.pdf",
        "comparisons.csv",
        "training_loss.png",
        "training_loss.svg",
    }
    assert "Dashed = smoke engineering evidence" in (
        tmp_path / "export" / "training_loss.svg"
    ).read_text(encoding="utf-8")


def test_full_run_plus_comparison_enables_quality_findings_and_is_deterministic(tmp_path):
    first = export_campaign_evidence(snapshot(full=True), tmp_path / "first")
    second = export_campaign_evidence(snapshot(full=True), tmp_path / "second")

    assert first == second
    assert first["quality_findings_available"] is True
    assert (tmp_path / "first" / "campaign_report.md").read_bytes() == (
        tmp_path / "second" / "campaign_report.md"
    ).read_bytes()
    assert (tmp_path / "first" / "campaign_report.docx").read_bytes() == (
        tmp_path / "second" / "campaign_report.docx"
    ).read_bytes()
    assert (tmp_path / "first" / "campaign_report.pdf").read_bytes() == (
        tmp_path / "second" / "campaign_report.pdf"
    ).read_bytes()
    assert (tmp_path / "first" / "training_loss.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    persisted = json.loads((tmp_path / "first" / "export_manifest.json").read_text())
    assert persisted == first


def test_export_rejects_local_path_bearing_projections(tmp_path):
    unsafe = CampaignExportSnapshot(
        campaign={"campaign_id": "campaign-1"},
        attempts=({"attempt_id": "attempt-1", "sealed_result_uri": "C:/private/run"},),
    )
    export_campaign_evidence(unsafe, tmp_path / "export")
    evidence = json.loads((tmp_path / "export" / "campaign_evidence.json").read_text())
    assert evidence["attempts"] == [{"attempt_id": "attempt-1"}]
    assert "private" not in json.dumps(evidence)

    unsafe_history = CampaignExportSnapshot(
        campaign={"campaign_id": "campaign-1"},
        autoresearch_history={
            "schema_version": "bashgym.autoresearch_history.v1",
            "experiments": [{"evidence": {"path": "C:/private/run"}}],
        },
    )
    with pytest.raises(CampaignExportError, match="campaign_export_contains_local_path"):
        export_campaign_evidence(unsafe_history, tmp_path / "history-export")


def test_export_allowlists_attempts_and_rejects_nested_private_locations(tmp_path):
    value = CampaignExportSnapshot(
        campaign={"campaign_id": "campaign-1", "objective": "Public experiment"},
        attempts=(
            {
                "attempt_id": "attempt-1",
                "study_id": "study-1",
                "stage": "full_training",
                "status": "completed",
                "candidate_digest": "a" * 64,
                "recipe_digest": "b" * 64,
                "compute_profile_id": "registered-compute",
                "created_at": "2026-08-15T00:00:00Z",
                "updated_at": "2026-08-15T00:01:00Z",
                "executor": {
                    "input_files": [
                        {
                            "local_path": "C:\\Users\\operator\\private-runner.py",
                            "remote_name": "runner.py",
                        }
                    ]
                },
            },
        ),
    )

    export_campaign_evidence(value, tmp_path / "export")

    evidence = json.loads((tmp_path / "export" / "campaign_evidence.json").read_text())
    assert evidence["attempts"] == [
        {
            "attempt_id": "attempt-1",
            "candidate_digest": "a" * 64,
            "compute_profile_id": "registered-compute",
            "created_at": "2026-08-15T00:00:00Z",
            "recipe_digest": "b" * 64,
            "stage": "full_training",
            "status": "completed",
            "study_id": "study-1",
            "updated_at": "2026-08-15T00:01:00Z",
        }
    ]
    assert "private-runner" not in json.dumps(evidence)
    assert "executor" not in evidence["attempts"][0]
    for path in (tmp_path / "export").iterdir():
        assert b"private-runner" not in path.read_bytes()
        assert b"Users\\operator" not in path.read_bytes()

    private_metadata = CampaignExportSnapshot(
        campaign={"campaign_id": "campaign-1"},
        artifacts=(
            {
                "artifact_id": "artifact-1",
                "metadata": {"details": {"relative_path": "/home/operator/private.bin"}},
            },
        ),
    )
    export_campaign_evidence(private_metadata, tmp_path / "private-metadata")
    evidence = json.loads((tmp_path / "private-metadata" / "campaign_evidence.json").read_text())
    assert evidence["artifacts"] == [{"artifact_id": "artifact-1"}]
    assert "private.bin" not in json.dumps(evidence)


def test_autoresearch_export_reports_cumulative_fixed_suite_performance(tmp_path):
    history = {
        "schema_version": "bashgym.autoresearch_history.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "campaign-1",
        "ledger_project_id": "project-a",
        "objective": "Improve held-out retrieval",
        "evaluation_suite_id": "suite-heldout",
        "primary_metric": "task_success",
        "metric_direction": "maximize",
        "method_thresholds": {
            "schema_version": "autoresearch_method_thresholds.v1",
            "min_demonstration_examples": 64,
        },
        "total_experiments": 2,
        "returned_experiments": 2,
        "omitted_experiments": 0,
        "experiments": [
            {
                "proposal_id": "baseline",
                "study_id": "study-baseline",
                "result_id": "result-baseline",
                "role": "baseline",
                "parent_proposal_id": None,
                "proposal": {
                    "hypothesis": "Record starting performance.",
                    "changed_variable": "baseline",
                    "expected_outcome": "Establish the fixed-suite reference.",
                    "falsification_criterion": "The evaluator readiness checks fail.",
                },
                "performance": {
                    "evaluation_suite_id": "suite-heldout",
                    "primary": {
                        "metric_name": "task_success",
                        "direction": "maximize",
                        "reference_proposal_id": None,
                        "reference_value": None,
                        "candidate_value": 0.5,
                        "improvement": None,
                        "minimum_improvement": 0.02,
                        "passed": None,
                    },
                    "protected_metrics": [],
                    "metrics": {"task_success": 0.5},
                    "metrics_omitted": 0,
                },
                "result": {
                    "outcome": "completed",
                    "provenance": "real",
                    "actual_cost": 0.1,
                    "recorded_at": "2026-08-15T12:00:00+00:00",
                },
                "decision": {
                    "decision": "baseline",
                    "reason_code": "real_baseline_verified",
                    "eligible_for_best": True,
                },
                "learning": {
                    "status": "baseline_recorded",
                    "summary": "Starting performance was recorded on the fixed evaluation suite.",
                },
                "data": None,
                "attempt_ids": ["attempt-baseline"],
                "attempt_ids_omitted": 0,
                "evidence_references": ["evaluation-baseline"],
                "evidence_references_omitted": 0,
            },
            {
                "proposal_id": "candidate-kept",
                "study_id": "study-candidate",
                "result_id": "result-candidate",
                "role": "candidate",
                "parent_proposal_id": "baseline",
                "proposal": {
                    "hypothesis": "Verified data improves completion.",
                    "changed_variable": "dataset_recipe.verifier_filter",
                    "expected_outcome": "Task success improves.",
                    "falsification_criterion": "The primary or protected gate fails.",
                },
                "performance": {
                    "evaluation_suite_id": "suite-heldout",
                    "primary": {
                        "metric_name": "task_success",
                        "direction": "maximize",
                        "reference_proposal_id": "baseline",
                        "reference_value": 0.5,
                        "candidate_value": 0.62,
                        "improvement": 0.12,
                        "minimum_improvement": 0.02,
                        "passed": True,
                    },
                    "protected_metrics": [
                        {
                            "metric_name": "invalid_tool_calls",
                            "direction": "minimize",
                            "reference_value": 0.04,
                            "candidate_value": 0.03,
                            "signed_change": 0.01,
                            "observed_regression": 0.0,
                            "maximum_regression": 0.01,
                            "passed": True,
                        }
                    ],
                    "metrics": {"invalid_tool_calls": 0.03, "task_success": 0.62},
                    "metrics_omitted": 0,
                },
                "result": {
                    "outcome": "completed",
                    "provenance": "real",
                    "actual_cost": 1.0,
                    "recorded_at": "2026-08-15T13:00:00+00:00",
                },
                "decision": {
                    "decision": "keep",
                    "reason_code": "candidate_improved_primary_metric",
                    "eligible_for_best": True,
                },
                "learning": {
                    "status": "retained",
                    "summary": (
                        "The candidate cleared the configured primary and protected metric gates "
                        "and became the reference."
                    ),
                },
                "failure_analysis": {
                    "schema_version": "bashgym.research_failures.v1",
                    "campaign_id": "campaign-1",
                    "reference": None,
                    "candidate": None,
                    "comparison": [
                        {
                            "category": "format_failure",
                            "reference_count": 1,
                            "candidate_count": 2,
                            "delta": 1,
                            "status": "regressed",
                        },
                        {
                            "category": "task_failure",
                            "reference_count": 20,
                            "candidate_count": 7,
                            "delta": -13,
                            "status": "improved",
                        },
                    ],
                    "truncated": False,
                },
                "outcome_assessment": {
                    "schema_version": "bashgym.autoresearch_outcome_assessment.v1",
                    "classification": "acceptable_tradeoff",
                    "is_failure": False,
                    "failure_kind": None,
                    "decision": "keep",
                    "reason_code": "primary_gain_with_nonblocking_tradeoff",
                    "observed_tradeoffs": ["format_failure"],
                    "observed_improvements": ["task_failure"],
                    "evidence_strength": "single_observation",
                },
                "data": {
                    "dataset_version_id": "dataset-v2",
                    "content_digest": "e" * 64,
                    "quality": {
                        "generated_rows": 90,
                        "accepted_rows": 60,
                        "acceptance_rate": 2 / 3,
                        "verification_pass_rate": 0.8,
                    },
                },
                "attempt_ids": ["attempt-data", "attempt-train", "attempt-eval"],
                "attempt_ids_omitted": 0,
                "evidence_references": ["evaluation-candidate"],
                "evidence_references_omitted": 0,
            },
        ],
        "diagnostic_results": [
            {
                "schema_version": "bashgym.research_diagnostic_result.v1",
                "probe_family": "loss_landscape",
                "question": "Does held-out loss rise after the retained checkpoint?",
                "hypothesis": "Longer continuation moves beyond the useful region.",
                "status": "completed",
                "measurements": [
                    {
                        "name": "heldout_loss_slope",
                        "value": 0.031,
                        "sample_count": 64,
                        "unit": "loss_per_step",
                    }
                ],
                "observations": [],
                "unsupported_reason": None,
            },
            {
                "probe_family": "reward_integrity_probe",
                "question": "Can the exact decomposed reward safely support verifier RL?",
                "hypothesis": "Hard constraints and exploit canaries pass.",
                "status": "completed",
                "comparison_contract": {
                    "reward_spec_digest": "a" * 64,
                    "canary_suite_id": "reward-hacking-v1",
                },
                "measurements": [
                    {
                        "name": "reward_canary_failure_rate",
                        "value": 0.0,
                        "sample_count": 4,
                        "unit": "fraction",
                    }
                ],
                "observations": [],
                "unsupported_reason": None,
            },
            {
                "probe_family": "preference_integrity_probe",
                "question": "Are the exact preference labels stable enough for DPO?",
                "hypothesis": "Position swaps preserve the preferred response.",
                "status": "completed",
                "comparison_contract": {
                    "preference_dataset_digest": "b" * 64,
                    "labeling_contract_digest": "c" * 64,
                },
                "measurements": [
                    {
                        "name": "preference_position_bias_rate",
                        "value": 0.02,
                        "sample_count": 96,
                        "unit": "fraction",
                    }
                ],
                "observations": [],
                "unsupported_reason": None,
            },
        ],
    }
    history["experiments"][-1]["experiment_power"] = {
        "schema_version": "bashgym.autoresearch_experiment_power.v1",
        "evaluation": {
            "evaluation_result_id": "evaluation-candidate",
            "sample_count": 64,
            "sample_count_source": "slice_metrics.example_count",
            "comparison_design": "paired",
            "uncertainty": {
                "method": "paired_bootstrap",
                "confidence_level": 0.95,
                "interval_lower": 0.04,
                "interval_upper": 0.2,
                "minimum_detectable_effect": 0.03,
            },
            "sufficiency": {"status": "not_assessed", "criteria": []},
        },
        "seed_uncertainty": {
            "status": "single_observation",
            "completed_real_results": 1,
            "distinct_training_seeds": 1,
            "sample_standard_deviation": None,
            "standard_error": None,
            "uncertainty_method": None,
            "limitation": "Between-run variation is not a per-example confidence interval.",
        },
        "sequential_stopping": {"status": "not_predeclared", "evidence": None},
        "limitations": ["An observed sample count is not evidence of adequate power."],
    }
    history["hypothesis_families"] = [
        {
            "hypothesis_family_id": "family-longer-training",
            "status": "replicated",
            "completed_real_results": 3,
            "lifecycle": {
                "status": "exhausted",
                "conclusion": {
                    "summary": "Repeated continuations did not improve the fixed suite.",
                    "proposal_ids": ["candidate-1", "candidate-2", "candidate-3"],
                    "result_ids": ["result-1", "result-2", "result-3"],
                    "aggregate_version": 9,
                },
                "follow_up": {
                    "hypothesis_family_id": "family-data-coverage",
                    "hypothesis": "Increase coverage of residual failure clusters.",
                },
            },
        }
    ]
    value = CampaignExportSnapshot(
        campaign={
            "campaign_id": "campaign-1",
            "objective": "Improve held-out retrieval",
            "status": "exhausted",
        },
        autoresearch_history=history,
    )

    manifest = export_campaign_evidence(value, tmp_path / "export")

    evidence = json.loads((tmp_path / "export" / "campaign_evidence.json").read_text())
    report = (tmp_path / "export" / "campaign_report.md").read_text(encoding="utf-8")
    assert evidence["schema_version"] == "campaign_export_snapshot.v2"
    assert evidence["autoresearch_history"] == history
    assert manifest["quality_findings_available"] is True
    assert "## AutoResearch experiment history" in report
    assert "## Research diagnostics" in report
    assert "### loss_landscape" in report
    assert "`heldout_loss_slope` = 0.031 loss_per_step (n=64)" in report
    assert f"Reward spec: `{'a' * 64}`" in report
    assert "Canary suite: `reward-hacking-v1`" in report
    assert f"Preference dataset: `{'b' * 64}`" in report
    assert f"Labeling contract: `{'c' * 64}`" in report
    assert "Current AutoResearch reference: `candidate-kept`" in report
    assert "Promoted campaign champion: `not promoted`" in report
    assert "Recorded baseline cost: `0.1`" in report
    assert "Recorded candidate cost: `1`" in report
    assert "Recorded campaign total cost: `1.1`" in report
    assert "| 1 | baseline | baseline | 0.5 | — | baseline |" in report
    assert "| 2 | candidate | dataset_recipe.verifier_filter | 0.62 | +0.12 | keep |" in report
    assert "### 2. candidate-kept" in report
    assert "Fixed evaluation suite: `suite-heldout`" in report
    assert "`invalid_tool_calls`: 0.03 vs 0.04; regression 0; limit 0.01; pass" in report
    assert "Data quality: 60/90 accepted; verification pass rate 0.8." in report
    assert "## Experiment power" in report
    assert "## Hypothesis families" in report
    assert "`family-longer-training`: evidence `replicated`; lifecycle `exhausted`" in report
    assert "Repeated continuations did not improve the fixed suite." in report
    assert "Follow-up `family-data-coverage`" in report
    assert "Evaluation sample count: `64`" in report
    assert "Sample-size sufficiency: `not_assessed`" in report
    assert "Sequential stopping: `not_predeclared`" in report
    assert "Method-readiness thresholds: `min_demonstration_examples=64`." in report
    assert "`task_failure`: 20 to 7 (-13; improved)" in report
    assert "Outcome assessment: `acceptable_tradeoff`; not a failed experiment" in report
    assert "Observed tradeoffs: `format_failure`." in report
    assert "Observed improvements: `task_failure`." in report
    assert "Evidence strength: `single_observation`." in report
    assert "Behavioral error categories:" in report
    assert "Failure categories:" not in report
    assert (
        "Finding: The candidate cleared the configured primary and protected metric gates and "
        "became the reference."
    ) in report
    assert "Evidence: `evaluation-candidate`" in report
