from pathlib import Path

ROOT = Path(__file__).parents[2]
SKILLS = ROOT / "assistant" / "workspace" / "skills"


def test_training_skill_routes_agents_through_compute_activation_contract():
    training = (SKILLS / "training" / "SKILL.md").read_text(encoding="utf-8")
    activation = (SKILLS / "training" / "references" / "compute-target-activation.md").read_text(
        encoding="utf-8"
    )

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


def test_operator_skill_is_agent_agnostic_and_uses_guided_autoresearch_setup():
    operator = (SKILLS / "bashgym-operator" / "SKILL.md").read_text(encoding="utf-8")

    assert "Codex, Claude Code, Hermes, or another compatible agent" in operator
    assert "from Hermes on Discord" not in operator
    assert "bashgym campaign setup-context" in operator
    assert "bashgym campaign setup-step" in operator
    assert "bashgym campaign setup-doctor" in operator
    assert "bashgym campaign setup-validate" in operator
    assert "bashgym campaign setup-create" in operator
    assert "zero registered choices" in operator
    assert "multiple registered choices" in operator


def test_operator_skill_requires_separate_explicit_start_confirmation():
    operator = (SKILLS / "bashgym-operator" / "SKILL.md").read_text(encoding="utf-8")

    assert "authorizes discovery and preparation only" in operator
    assert "campaign ID, model, data, evaluation, compute, budget, and stop rules" in operator
    assert "STOP and wait for a subsequent explicit Start confirmation" in operator
    assert "Never run `bashgym campaign start` in the preparation turn" in operator


def test_source_clone_agent_guides_route_autoresearch_through_the_shared_skill():
    codex = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "bashgym operator skills install --host codex" in codex
    assert "assistant/workspace/skills/bashgym-operator/SKILL.md" in codex
    assert "campaign setup-context" in codex
    assert "explicit Start" in codex
    assert "bashgym operator skills install --host claude" in claude
    assert "assistant/workspace/skills/bashgym-operator/SKILL.md" in claude
    assert "campaign setup-context" in claude
    assert "explicit Start" in claude
