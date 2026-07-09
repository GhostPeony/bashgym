import json

from bashgym.factory.session_distillation import build_session_distillation_records
from bashgym.run_cards import (
    attach_run_card_evidence,
    create_run_card,
    read_run_card,
    validate_run_card_file,
    write_run_card,
)


def _strict_pair():
    return {
        "id": "pair-1",
        "prompt": "Fix the failing test",
        "chosen_response": "Run pytest, inspect the failure, patch the function.",
        "rejected_response": "Ignore the failure and claim success.",
        "metadata": {
            "pair_id": "pair-1",
            "prompt_hash": "abc123",
            "chosen_trace_id": "gold-1",
            "rejected_trace_id": "failed-1",
            "pair_generation_method": "trace_pair",
            "label_strength": "verified_success_vs_failure",
            "label_source": "trace_verifier",
            "chosen_quality_score": 0.95,
            "rejected_quality_score": 0.25,
            "domain": "terminal_agent",
            "task_family": "test_fix",
            "split": "train",
            "decontamination_status": "checked",
        },
    }


def _strict_reward_example():
    return {
        "id": "reward-1",
        "reward_type": "outcome_reward",
        "prompt": "Fix the failing test",
        "response": "Run pytest, inspect the failure, patch the function.",
        "score": 0.9,
        "metadata": {
            "reward_example_id": "reward-1",
            "reward_scale": "0_to_1",
            "label_source": "trace_verifier",
            "source_id": "helpsteer2",
            "quality_score": 0.95,
            "domain": "terminal_agent",
            "task_family": "test_fix",
            "split": "train",
            "decontamination_status": "checked",
        },
    }


def _reward_eval_artifact(*, ok=True):
    return {
        "schema_version": "bashgym.reward_model_eval.v1",
        "ok": ok,
        "split": "eval",
        "total_records": 2,
        "evaluated_records": 2,
        "prediction_records": 2,
        "metrics": {
            "heldout_pair_accuracy": 1.0,
            "pair_count": 1,
            "calibration_error": 0.1,
            "reward_margin": 0.6,
            "length_bias": 0.0,
            "reward_variance": 0.08,
            "eval_only_leakage_count": 0,
            "eval_only_leakage_rate": 0.0,
        },
        "task_family_breakdown": [],
        "eval_only_source_ids": [],
        "findings": [],
        "fail_count": 0 if ok else 1,
        "warn_count": 0,
    }


def _session_distillation_record():
    records = build_session_distillation_records(
        [
            {
                "tool": "Bash",
                "command": "pytest tests/missing.py",
                "output": "ERROR: file or directory not found: tests/missing.py",
                "success": False,
                "exit_code": 4,
            }
        ],
        task_prompt="Run focused tests.",
        trace_id="trace-session-distill",
        session_id="session-session-distill",
        source_metadata={"split": "train"},
    )
    return records[0].to_dict()


def _write_promotion_artifacts(
    tmp_path,
    *,
    ship=True,
    world_model=False,
    dpo_pairs=False,
    reward_examples=False,
    reward_eval=False,
    session_distillation_records=False,
):
    tmp_path.joinpath("plan.json").write_text("{}", encoding="utf-8")
    tmp_path.joinpath("source_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bashgym.source_manifest.v1",
                "source": {"id": "helpsteer2"},
                "goal": "reward_model",
                "use_verdict": {
                    "ok": True,
                    "blocking_codes": [],
                    "warnings": [],
                    "requires_override_reason": False,
                },
            }
        ),
        encoding="utf-8",
    )
    tmp_path.joinpath("metrics.jsonl").write_text(
        '{"step": 1, "eval_loss": 0.4}\n', encoding="utf-8"
    )
    release_gate = {"ship": ship}
    if world_model:
        release_gate["world_model_quality"] = {
            "present": True,
            "diagnostic_only": True,
            "signal": "present",
            "findings": [],
        }
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps(
            {
                "ship": ship,
                "reasons": [] if ship else ["heldout gate failed"],
                "release_gate": release_gate,
            }
        ),
        encoding="utf-8",
    )
    if dpo_pairs:
        tmp_path.joinpath("dpo_pairs.jsonl").write_text(
            json.dumps(_strict_pair()) + "\n",
            encoding="utf-8",
        )
    if reward_examples:
        tmp_path.joinpath("reward_examples.jsonl").write_text(
            json.dumps(_strict_reward_example()) + "\n",
            encoding="utf-8",
        )
    if reward_eval:
        tmp_path.joinpath("reward_eval.json").write_text(
            json.dumps(_reward_eval_artifact()),
            encoding="utf-8",
        )
    if session_distillation_records:
        tmp_path.joinpath("session_distillation_records.jsonl").write_text(
            json.dumps(_session_distillation_record()) + "\n",
            encoding="utf-8",
        )


def test_run_card_create_validate_and_read(tmp_path):
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plans/sft.json",
        source_manifest_path="data/source_manifest.json",
        include_git=False,
    )

    assert card.validation_findings() == []
    assert {finding["code"] for finding in card.validation_findings(promotion=True)} == {
        "missing_metrics_path",
        "missing_release_evidence_path",
    }

    path = tmp_path / "run_card.json"
    write_run_card(card, path)
    loaded = read_run_card(path)
    assert loaded.run_id == "run-demo"
    assert loaded.compute_target_id == "local_cpu_or_gpu"


def test_attach_run_card_evidence(tmp_path):
    card = create_run_card(
        run_id="run-demo",
        training_method="dpo",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="private_gpu",
        training_plan_path="plans/dpo.json",
        source_manifest_path="data/source_manifest.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    attach_run_card_evidence(
        path,
        metrics_path="runs/demo/metrics.jsonl",
        release_evidence_path="runs/demo/release_evidence.json",
        preference_pairs_path="runs/demo/dpo_pairs.jsonl",
    )

    loaded = read_run_card(path)
    assert loaded.metrics_path == "runs/demo/metrics.jsonl"
    assert loaded.release_evidence_path == "runs/demo/release_evidence.json"
    assert loaded.preference_pairs_path == "runs/demo/dpo_pairs.jsonl"
    assert loaded.validation_findings(promotion=True) == []


def test_validate_run_card_file_fails_closed_on_missing_promotion_artifacts(tmp_path):
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert {
        "missing_training_plan_path_file",
        "missing_source_manifest_path_file",
        "missing_metrics_path_file",
        "missing_release_evidence_path_file",
    } <= codes
    explanation = validation["promotion_explanation"]
    assert explanation["ok"] is False
    assert "run-demo is not promotable" in explanation["headline"]
    assert explanation["blocker_count"] >= 4
    gates = {gate["gate"] for gate in explanation["failed_gates"]}
    assert "training plan evidence" in gates
    assert "metrics evidence" in gates
    assert explanation["next_actions"]


def test_validate_run_card_file_blocks_non_shipping_release_evidence(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=False)
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "release_evidence_not_shippable" in codes


def test_validate_run_card_file_marks_world_model_quality_diagnostic_only(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, world_model=True)
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True
    assert any(
        finding["code"] == "world_model_quality_diagnostic_only"
        and finding["level"] == "diagnostic"
        for finding in validation["findings"]
    )


def test_validate_run_card_file_narrow_routing_requires_environment_evidence(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        claim_tier="narrow_routing",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "claim_tier_trace_gate_missing_or_failed" in codes
    assert "claim_tier_environment_evidence_missing_or_failed" in codes


def test_validate_run_card_file_narrow_routing_accepts_environment_evidence(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "release_gate": {
                    "ship": True,
                    "trace_ship": True,
                    "environment_ship": True,
                    "environment_sections": ["holdout_gate"],
                    "external_benchmark_ship": True,
                    "external_benchmark_sections": [],
                },
            }
        ),
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        claim_tier="narrow_routing",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True


def test_validate_run_card_file_broad_claim_requires_external_and_manifests(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "release_gate": {
                    "ship": True,
                    "trace_ship": True,
                    "environment_ship": True,
                    "environment_sections": ["holdout_gate"],
                    "external_benchmark_ship": True,
                    "external_benchmark_sections": [],
                },
            }
        ),
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        claim_tier="broad_public_claim",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "claim_tier_external_benchmark_missing_or_failed" in codes
    assert "claim_tier_split_manifest_missing" in codes
    assert "claim_tier_decontamination_manifest_missing" in codes


def test_validate_run_card_file_broad_claim_accepts_external_and_manifests(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    tmp_path.joinpath("release_evidence.json").write_text(
        json.dumps(
            {
                "ship": True,
                "reasons": [],
                "split_manifest": {"path": "splits/holdout.json"},
                "decontamination_manifest": {"path": "splits/decontam.json"},
                "release_gate": {
                    "ship": True,
                    "trace_ship": True,
                    "environment_ship": True,
                    "environment_sections": ["holdout_gate"],
                    "external_benchmark_ship": True,
                    "external_benchmark_sections": ["external_benchmarks"],
                },
            }
        ),
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-demo",
        training_method="sft",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        claim_tier="broad_public_claim",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True


def test_validate_run_card_file_requires_dpo_preference_pairs_for_promotion(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    card = create_run_card(
        run_id="run-demo",
        training_method="dpo",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "missing_preference_pairs_path" in codes


def test_validate_run_card_file_strict_validates_dpo_preference_pairs(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, dpo_pairs=True)
    card = create_run_card(
        run_id="run-demo",
        training_method="dpo",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        preference_pairs_path="dpo_pairs.jsonl",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True


def test_validate_run_card_file_blocks_weak_dpo_preference_pairs(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    tmp_path.joinpath("dpo_pairs.jsonl").write_text(
        json.dumps(
            {
                "id": "pair-1",
                "prompt": "Fix it",
                "chosen_response": "Same",
                "rejected_response": "Same",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-demo",
        training_method="dpo",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        preference_pairs_path="dpo_pairs.jsonl",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "preference_pairs_identical_chosen_rejected" in codes
    assert "preference_pairs_missing_decontamination_metadata" in codes


def test_validate_run_card_file_requires_session_distillation_evidence_for_promotion(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    card = create_run_card(
        run_id="run-session-distill",
        training_method="session_distillation",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "missing_session_distillation_records_path" in codes
    assert "missing_session_distillation_reader_model" in codes
    assert "missing_session_distillation_confidence_threshold" in codes
    assert "missing_session_distillation_mask_policy" in codes
    assert "session_distillation_metrics_missing_masked_loss" in codes


def test_validate_run_card_file_accepts_session_distillation_evidence(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, session_distillation_records=True)
    tmp_path.joinpath("metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "session_distillation_loss": 0.31,
                "session_distillation_kl": 0.2,
                "session_distillation_ce": 0.56,
                "session_distillation_masked_tokens": 14,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-session-distill",
        training_method="session_distillation",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        session_distillation_records_path="session_distillation_records.jsonl",
        session_distillation_reader_model="heuristic-session-distillation-reader-v1",
        session_distillation_confidence_threshold=0.6,
        session_distillation_hint_policy="hint_injected",
        session_distillation_mask_policy="target_span_only",
        session_distillation_target_token_count=14,
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True


def test_validate_run_card_file_requires_reward_examples_for_reward_model_promotion(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True)
    card = create_run_card(
        run_id="run-reward",
        training_method="reward_model",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "missing_reward_examples_path" in codes


def test_validate_run_card_file_strict_validates_reward_examples(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, reward_examples=True, reward_eval=True)
    card = create_run_card(
        run_id="run-reward",
        training_method="reward_model",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        reward_examples_path="reward_examples.jsonl",
        reward_eval_path="reward_eval.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is True


def test_validate_run_card_file_blocks_weak_reward_examples(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, reward_eval=True)
    tmp_path.joinpath("reward_examples.jsonl").write_text(
        json.dumps(
            {
                "id": "reward-1",
                "reward_type": "process_reward",
                "prompt": "Fix it",
                "response": "Good fix",
                "score": 1.0,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    card = create_run_card(
        run_id="run-reward",
        training_method="reward_model",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        reward_examples_path="reward_examples.jsonl",
        reward_eval_path="reward_eval.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "reward_examples_missing_process_reward_steps" in codes
    assert "reward_examples_missing_decontamination_metadata" in codes


def test_validate_run_card_file_requires_reward_eval_for_reward_model_promotion(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, reward_examples=True)
    card = create_run_card(
        run_id="run-reward",
        training_method="reward_model",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        reward_examples_path="reward_examples.jsonl",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "missing_reward_eval_path" in codes


def test_validate_run_card_file_blocks_failed_reward_eval(tmp_path):
    _write_promotion_artifacts(tmp_path, ship=True, reward_examples=True, reward_eval=True)
    failed_eval = _reward_eval_artifact(ok=False)
    failed_eval["metrics"]["eval_only_leakage_count"] = 1
    failed_eval["findings"] = [
        {
            "code": "eval_only_leakage",
            "level": "fail",
            "message": "eval-only sources appear in reward eval",
        }
    ]
    tmp_path.joinpath("reward_eval.json").write_text(json.dumps(failed_eval), encoding="utf-8")
    card = create_run_card(
        run_id="run-reward",
        training_method="reward_model",
        base_model="Qwen/Qwen3-Coder",
        compute_target_id="local_cpu_or_gpu",
        training_plan_path="plan.json",
        source_manifest_path="source_manifest.json",
        reward_examples_path="reward_examples.jsonl",
        reward_eval_path="reward_eval.json",
        metrics_path="metrics.jsonl",
        release_evidence_path="release_evidence.json",
        include_git=False,
    )
    path = tmp_path / "run_card.json"
    write_run_card(card, path)

    validation = validate_run_card_file(path, promotion=True)

    assert validation["ok"] is False
    codes = {finding["code"] for finding in validation["findings"]}
    assert "reward_eval_not_ok" in codes
    assert "reward_eval_eval_only_leakage" in codes
