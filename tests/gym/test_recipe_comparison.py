"""Offline comparison uses test fixtures, never benchmark measurements."""

import json
from copy import deepcopy

import pytest

from bashgym.gym.recipe_comparison import compare_recipe_run_files, compare_recipe_runs


def records():
    baseline = {
        "schema_version": "bashgym.recipe_run_record.v1",
        "run_id": "baseline-fixture",
        "role": "baseline",
        "provenance": "measured",
        "contract": {
            key: "a" * 64
            for key in (
                "model_digest",
                "data_digest",
                "evaluation_digest",
                "target_digest",
                "software_digest",
                "training_digest",
                "measurement_digest",
            )
        },
        "optimization": {"backend": "plain", "liger_kernel": False},
        "performance": {
            "tokens_per_second": 100,
            "gpu_memory_peak_gb": 20,
            "elapsed_seconds": 1000,
            "cost": 2,
            "cost_unit": "USD",
        },
        "quality_contract": {
            "primary_metric": {"name": "accuracy", "direction": "maximize", "max_regression": 0.01},
            "protected_metrics": [
                {"name": "error_rate", "direction": "minimize", "max_regression": 0.02}
            ],
        },
        "quality": {"accuracy": 0.8, "error_rate": 0.1},
        "evaluation_sample_count": 100,
    }
    candidate = deepcopy(baseline)
    candidate.update(run_id="candidate-fixture", role="candidate")
    candidate["optimization"]["liger_kernel"] = True
    candidate["performance"].update(
        tokens_per_second=200, gpu_memory_peak_gb=10, elapsed_seconds=500, cost=1
    )
    return [baseline, candidate]


def test_comparable_measured_records_report_ratios_without_selecting_winner():
    baseline, candidate = records()
    report = compare_recipe_runs([candidate, baseline])
    assert report["status"] == "comparable"
    assert report["performance"]["tokens_per_second"]["ratio"] == 2
    assert report["performance"]["gpu_memory_peak_gb"]["delta"] == -10
    assert report["performance"]["cost"]["ratio"] == 0.5
    assert report["quality_guard_breached"] is False
    assert report["evidence_strength"] == "exploratory"
    assert report["selection_authority"] == "host_agent"
    assert "winner" not in report


def test_faster_candidate_can_breach_both_quality_guards():
    values = records()
    values[1]["quality"].update(accuracy=0.7, error_rate=0.2)
    report = compare_recipe_runs(values)
    assert report["performance"]["tokens_per_second"]["delta"] > 0
    assert report["quality_guard_breached"] is True
    assert all(item["guard_breached"] for item in report["quality"])


@pytest.mark.parametrize(
    "key",
    [
        "model_digest",
        "data_digest",
        "evaluation_digest",
        "target_digest",
        "software_digest",
        "training_digest",
        "measurement_digest",
    ],
)
def test_changed_scientific_or_execution_contract_is_incomparable(key):
    values = records()
    values[1]["contract"][key] = "b" * 64
    with pytest.raises(ValueError, match="incomparable"):
        compare_recipe_runs(values)


def test_absent_peak_memory_is_not_replaced_with_zero():
    values = records()
    del values[1]["performance"]["gpu_memory_peak_gb"]
    with pytest.raises(ValueError, match="gpu_memory_peak_gb"):
        compare_recipe_runs(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "200"])
def test_nonfinite_or_coerced_metrics_are_rejected(value):
    values = records()
    values[1]["performance"]["tokens_per_second"] = value
    with pytest.raises(ValueError):
        compare_recipe_runs(values)


def test_record_files_preserve_explicit_roles_and_source_hashes(tmp_path):
    baseline, candidate = records()
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    report = compare_recipe_run_files(baseline_path, candidate_path)
    assert set(report["source_file_digests"]) == {"baseline", "candidate"}
    with pytest.raises(ValueError, match="role"):
        compare_recipe_run_files(candidate_path, baseline_path)


def test_unsloth_is_an_explicit_comparison_variant_but_not_combined_with_liger():
    values = records()
    values[1]["optimization"] = {"backend": "unsloth", "liger_kernel": False}
    assert compare_recipe_runs(values)["status"] == "comparable"
    values[1]["optimization"]["liger_kernel"] = True
    with pytest.raises(ValueError, match="cannot be combined"):
        compare_recipe_runs(values)


def test_explicit_zero_cost_remains_zero_with_undefined_ratio():
    values = records()
    values[0]["performance"]["cost"] = 0
    report = compare_recipe_runs(values)
    assert report["performance"]["cost"] == {
        "baseline": 0,
        "candidate": 1,
        "delta": 1,
        "ratio": None,
    }


def test_equal_quality_guard_boundary_does_not_report_false_regression():
    values = records()
    values[1]["quality"]["accuracy"] = 0.79
    report = compare_recipe_runs(values)
    assert report["quality_guard_breached"] is False


@pytest.mark.parametrize("change", ["cost_unit", "sample_count", "quality_guard", "missing_guard"])
def test_fixed_evaluation_and_accounting_contracts_are_required(change):
    values = records()
    if change == "cost_unit":
        values[1]["performance"]["cost_unit"] = "GPU_hours"
    elif change == "sample_count":
        values[1]["evaluation_sample_count"] = 10
    elif change == "quality_guard":
        values[1]["quality_contract"]["primary_metric"]["max_regression"] = 0.5
    else:
        del values[1]["quality"]["error_rate"]
    with pytest.raises(ValueError):
        compare_recipe_runs(values)


def test_unmeasured_or_duplicate_run_records_cannot_be_compared():
    values = records()
    values[1]["provenance"] = "estimated"
    with pytest.raises(ValueError):
        compare_recipe_runs(values)
    values = records()
    values[1]["run_id"] = values[0]["run_id"]
    with pytest.raises(ValueError, match="distinct run identities"):
        compare_recipe_runs(values)


def test_duplicate_json_fields_are_rejected(tmp_path):
    values = records()
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(
        json.dumps(values[0]).replace(
            '"role": "baseline"', '"role": "candidate", "role": "baseline"'
        )
    )
    second.write_text(json.dumps(values[1]))
    with pytest.raises(ValueError, match="duplicate JSON field"):
        compare_recipe_run_files(first, second)
