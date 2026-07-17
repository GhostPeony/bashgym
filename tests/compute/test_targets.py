import pytest

from bashgym.compute import (
    get_compute_target,
    launch_plan,
    normalize_training_target_payload,
    preflight_compute_target,
)


def test_private_gpu_preflight_reports_missing_host_env(monkeypatch):
    monkeypatch.delenv("BASHGYM_PRIVATE_GPU_HOST", raising=False)
    target = get_compute_target("private_gpu")

    report = preflight_compute_target(target)

    assert report["schema_version"] == "bashgym.compute_preflight.v1"
    assert report["level"] == "needs_setup"
    assert any(check["code"] == "private_compute_target_configured" for check in report["checks"])


def test_private_gpu_uses_only_machine_neutral_configuration_names():
    target = get_compute_target("private_gpu")

    assert target.env_vars == ("BASHGYM_PRIVATE_GPU_HOST", "BASHGYM_PRIVATE_GPU_WORKDIR")
    assert set(target.metadata) == {"description", "host_env", "workdir_env"}


def test_skypilot_launch_plan_generates_yaml_without_secrets():
    target = get_compute_target("skypilot_a10g")

    plan = launch_plan(target, plan_path="runs/demo/plan.json")

    assert plan["schema_version"] == "bashgym.compute_launch_plan.v1"
    assert plan["dry_run"] is True
    assert plan["approval_required"] is True
    assert plan["provider_config"]["filename"] == "sky.yaml"
    assert "A10G:1" in plan["provider_config"]["content"]
    assert "HF_TOKEN" not in plan["provider_config"]["content"]


def test_dstack_launch_plan_generates_task_yaml():
    target = get_compute_target("dstack_a10g")

    plan = launch_plan(target, plan_path="runs/demo/plan.json")

    assert plan["provider_config"]["filename"] == ".dstack.yml"
    assert "gpu: A10G:1" in plan["provider_config"]["content"]
    assert (
        "python scripts/train_model.py --config runs/demo/plan.json"
        in plan["provider_config"]["content"]
    )


def test_ssh_target_activates_remote_training_flags():
    payload = normalize_training_target_payload(
        {"compute_target": "ssh:lab-box", "strategy": "sft"}
    )

    assert payload["use_remote_ssh"] is True
    assert payload["device_id"] == "lab-box"
    assert payload["compute_target"] == "ssh:lab-box"


def test_private_alias_activates_default_remote_target():
    payload = normalize_training_target_payload({"compute_target": "private"})

    assert payload["use_remote_ssh"] is True
    assert payload["compute_target"] == "ssh:remote"


def test_cloud_target_cannot_masquerade_as_local_training():
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_training_target_payload({"compute_target": "cloud"})

    with pytest.raises(ValueError, match="not launched by /api/training/start"):
        normalize_training_target_payload({"compute_target": "hf-jobs"})


def test_nemo_customizer_target_activates_only_customizer_backend():
    payload = normalize_training_target_payload({"compute_target": "cloud:nemo-customizer"})

    assert payload["use_nemo_customizer"] is True
    assert payload.get("use_nemo_gym", False) is False
    assert payload["compute_target"] == "cloud:nemo-customizer"


def test_legacy_nemo_target_is_canonicalized_without_claiming_gym():
    payload = normalize_training_target_payload({"compute_target": "cloud:nemo"})

    assert payload["use_nemo_customizer"] is True
    assert payload["compute_target"] == "cloud:nemo-customizer"


def test_deprecated_nemo_gym_flag_remains_a_customizer_alias():
    payload = normalize_training_target_payload({"use_nemo_gym": True})

    assert payload["use_nemo_customizer"] is True
    assert payload["compute_target"] == "cloud:nemo-customizer"
