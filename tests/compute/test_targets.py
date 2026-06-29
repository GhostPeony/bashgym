from bashgym.compute import get_compute_target, launch_plan, preflight_compute_target


def test_gx10_preflight_reports_missing_host_env(monkeypatch):
    monkeypatch.delenv("BASHGYM_GX10_HOST", raising=False)
    target = get_compute_target("gx10_ssh")

    report = preflight_compute_target(target)

    assert report["schema_version"] == "bashgym.compute_preflight.v1"
    assert report["level"] == "needs_setup"
    assert any(check["code"] == "ssh_host_configured" for check in report["checks"])


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
    assert "python scripts/train_model.py --config runs/demo/plan.json" in plan["provider_config"][
        "content"
    ]
