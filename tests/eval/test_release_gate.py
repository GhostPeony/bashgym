"""Tests for unified release verdicts that include environment gate evidence."""

from __future__ import annotations

import pytest

from bashgym.eval.release_gate import combine_release_gate_evidence


def _heldout_report(ship: bool = True) -> dict:
    return {
        "ship": ship,
        "reasons": [] if ship else ["trace delta 0.000 < required 0.050"],
        "trace_delta": 0.12 if ship else 0.0,
    }


def test_combine_release_gate_preserves_clear_environment_evidence():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "required": True,
            "holdout_gate": {"gate": {"ship": True, "reasons": []}},
            "spurious_reward_control": {"result": {"gate": {"ship": True, "reasons": []}}},
        },
    )

    assert report["ship"] is True
    assert report["reasons"] == []
    assert report["release_gate"]["trace_ship"] is True
    assert report["release_gate"]["environment_ship"] is True
    assert report["release_gate"]["environment_sections"] == [
        "holdout_gate",
        "spurious_reward_control",
    ]


def test_combine_release_gate_blocks_on_environment_gate_failure():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "holdout_comparison": {
                "gate": {
                    "ship": False,
                    "reasons": ["pass@1 CI [0.000, 0.100] does not clear zero"],
                }
            }
        },
    )

    assert report["ship"] is False
    assert report["release_gate"]["trace_ship"] is True
    assert report["release_gate"]["environment_ship"] is False
    assert report["release_gate"]["blocking_environment_sections"] == ["holdout_comparison"]
    assert report["reasons"] == [
        "environment holdout comparison: pass@1 CI [0.000, 0.100] does not clear zero"
    ]


def test_combine_release_gate_required_blocks_missing_environment_gate():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {"required": True, "passk": {"pass_at_k": {"pass@1": 1.0}}},
    )

    assert report["ship"] is False
    assert report["release_gate"]["environment_required"] is True
    assert report["release_gate"]["environment_sections"] == []
    assert report["reasons"] == ["environment gate evidence required but missing"]


def test_combine_release_gate_keeps_trace_failure_with_clear_environment_gate():
    report = combine_release_gate_evidence(
        _heldout_report(ship=False),
        {"holdout_gate": {"gate": {"ship": True, "reasons": []}}},
    )

    assert report["ship"] is False
    assert report["release_gate"]["trace_ship"] is False
    assert report["release_gate"]["environment_ship"] is True
    assert report["reasons"] == ["trace delta 0.000 < required 0.050"]


def test_combine_release_gate_carries_clear_external_benchmarks():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "holdout_gate": {"gate": {"ship": True, "reasons": []}},
            "external_benchmarks": {
                "report": {
                    "scores": {"harbor_terminal_bench": 0.67},
                    "failures": [],
                    "results": [],
                },
                "manifest": {"dataset": "terminal-bench@2.0"},
            },
        },
    )

    assert report["ship"] is True
    assert report["release_gate"]["external_benchmark_ship"] is True
    assert report["release_gate"]["external_benchmark_sections"] == ["external_benchmarks"]
    assert report["environment_evidence"]["external_benchmarks"]["manifest"] == {
        "dataset": "terminal-bench@2.0"
    }


def test_combine_release_gate_blocks_on_external_benchmark_failure():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "external_benchmarks": {
                "scores": {"bfcl_v4": 0.81},
                "failures": ["swebench_verified_lite"],
                "results": [],
            }
        },
    )

    assert report["ship"] is False
    assert report["release_gate"]["external_benchmark_ship"] is False
    assert report["release_gate"]["blocking_external_benchmark_sections"] == [
        "external_benchmarks"
    ]
    assert report["reasons"] == [
        "external benchmark swebench_verified_lite: failed or missing score"
    ]


def test_combine_release_gate_blocks_on_external_benchmark_threshold():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "external_benchmarks": {
                "report": {
                    "scores": {"harbor_terminal_bench": 0.42},
                    "failures": [],
                    "results": [],
                },
                "min_scores": {"harbor_terminal_bench": 0.5},
            }
        },
    )

    assert report["ship"] is False
    assert report["release_gate"]["external_benchmark_ship"] is False
    assert report["reasons"] == [
        "external benchmark harbor_terminal_bench: score 0.420 < required 0.500"
    ]


def test_combine_release_gate_carries_world_model_quality_diagnostics():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "holdout_gate": {"gate": {"ship": True, "reasons": []}},
            "world_model_quality": {
                "metrics": {
                    "echo_loss": {"first": 1.2, "last": 0.7},
                    "rwml_pass_rate": 0.72,
                    "embedding_distance_mean": 0.12,
                    "exit_code_accuracy": 0.9,
                },
                "coverage": {"world_model_records": 16, "rwml_transitions": 42},
            },
        },
    )

    quality = report["release_gate"]["world_model_quality"]
    assert report["ship"] is True
    assert report["reasons"] == []
    assert report["release_gate"]["world_model_quality_present"] is True
    assert report["release_gate"]["world_model_quality_sections"] == ["world_model_quality"]
    assert quality["diagnostic_only"] is True
    assert quality["signal"] == "improving"
    assert quality["metrics"]["echo_loss"] == 0.7
    assert quality["metrics"]["echo_loss_delta"] == pytest.approx(-0.5)
    assert quality["coverage"]["rwml_transitions"] == 42


def test_combine_release_gate_world_model_quality_warning_is_not_a_ship_blocker():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "world_model_quality": {
                "training_metrics": {
                    "echo_loss": {"first": 0.8, "last": 1.1},
                    "rwml_pass_rate": {"last": 0.2},
                    "embedding_distance_mean": {"last": 0.31},
                }
            }
        },
    )

    quality = report["release_gate"]["world_model_quality"]
    assert report["ship"] is True
    assert report["reasons"] == []
    assert quality["signal"] == "needs_attention"
    assert quality["findings"] == [
        "ECHO loss increased across the observed window",
        "RWML pass rate is below the suggested smoke threshold",
        "mean RWML embedding distance is above the starter threshold",
    ]


def test_combine_release_gate_carries_learned_reward_evidence_diagnostics():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "learned_reward_evidence": {
                "schema_version": "bashgym.reward_model_eval.v1",
                "ok": True,
                "metrics": {
                    "heldout_pair_accuracy": 0.82,
                    "calibration_error": 0.08,
                    "reward_margin": 0.4,
                    "reward_variance": 0.05,
                    "eval_only_leakage_count": 0,
                    "pair_count": 12,
                },
                "findings": [],
            }
        },
    )

    reward = report["release_gate"]["learned_reward_evidence"]
    assert report["ship"] is True
    assert report["reasons"] == []
    assert report["release_gate"]["learned_reward_evidence_present"] is True
    assert report["release_gate"]["learned_reward_evidence_sections"] == [
        "learned_reward_evidence"
    ]
    assert reward["diagnostic_only"] is True
    assert reward["signal"] == "healthy"
    assert reward["metrics"]["heldout_pair_accuracy"] == 0.82


def test_combine_release_gate_learned_reward_warning_is_not_a_ship_blocker():
    report = combine_release_gate_evidence(
        _heldout_report(ship=True),
        {
            "learned_reward_evidence": {
                "ok": False,
                "metrics": {
                    "heldout_pair_accuracy": 0.5,
                    "calibration_error": 0.31,
                    "eval_only_leakage_count": 1,
                    "reward_variance": 0.0,
                },
                "findings": [
                    {
                        "code": "eval_only_leakage",
                        "message": "eval-only sources appear in reward eval",
                    }
                ],
            }
        },
    )

    reward = report["release_gate"]["learned_reward_evidence"]
    assert report["ship"] is True
    assert report["reasons"] == []
    assert reward["signal"] == "needs_attention"
    assert "learned reward evidence reports ok=false" in reward["findings"]
    assert "learned reward evidence reports eval-only leakage" in reward["findings"]
    assert "learned reward heldout pair accuracy is below the starter threshold" in reward[
        "findings"
    ]
