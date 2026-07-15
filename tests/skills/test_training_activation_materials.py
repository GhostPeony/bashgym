from pathlib import Path

ROOT = Path(__file__).parents[2]
SKILLS = ROOT / "assistant" / "workspace" / "skills"


def test_training_skill_routes_agents_through_compute_activation_contract():
    training = (SKILLS / "training" / "SKILL.md").read_text(encoding="utf-8")
    activation = (
        SKILLS / "training" / "references" / "compute-target-activation.md"
    ).read_text(encoding="utf-8")

    assert "references/compute-target-activation.md" in training
    assert "ssh:<device_id>" in activation
    assert "cloud:nemo-customizer" in activation
    assert "NeMo RL" in activation
    assert "hf jobs uv run" in activation
    assert "SkyPilot/dstack" in activation
    assert "plan only" in activation


def test_operator_and_router_require_target_specific_activation():
    operator = (SKILLS / "bashgym-operator" / "SKILL.md").read_text(encoding="utf-8")
    router = (SKILLS / "bashgym" / "SKILL.md").read_text(encoding="utf-8")

    assert "compute-target activation contract" in operator
    assert "doctor-verified activation lane" in operator
    assert "compute-target-activation.md" in router
    assert "not just label provenance" in router
