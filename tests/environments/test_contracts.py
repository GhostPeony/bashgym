"""Tests for EnvironmentSpec serialization and validation."""

from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentAxis,
    EnvironmentSpec,
    RewardComponentSpec,
    RolloutSpec,
    VerifierSpec,
)


def test_environment_spec_round_trips():
    spec = EnvironmentSpec(
        id="env_demo",
        instruction="Fix the failing parser test.",
        source="unit",
        domain="software_engineering",
        skills=["debugging", "testing"],
        axes=[
            EnvironmentAxis(name="task_complexity", value="moderate"),
            EnvironmentAxis(name="command_complexity", value="bash+python"),
        ],
        verifier=VerifierSpec(
            kind="pytest",
            command="pytest tests",
            path="tests/test_parser.py",
            reward_type="components",
            reward_components=[
                RewardComponentSpec(name="correctness"),
                RewardComponentSpec(name="format", weight=0.25),
            ],
        ),
        build=BuildSpec(base_image="python:3.11-slim"),
        rollout=RolloutSpec(max_tool_calls=32),
        files={"tests/test_parser.py": "def test_parser():\n    assert True\n"},
        metadata={"pass@1": 0.25},
    )

    restored = EnvironmentSpec.from_dict(spec.to_dict())

    assert restored.id == "env_demo"
    assert restored.axis_value("task_complexity") == "moderate"
    assert restored.axis_value("skills") == "debugging,testing"
    assert restored.verifier.command == "pytest tests"
    assert restored.verifier.is_multi_reward is True
    assert restored.verifier.reward_components[1].weight == 0.25
    assert restored.verifier.combine_reward_components({"correctness": 1.0, "format": 0.5}) == 1.125
    assert restored.build.base_image == "python:3.11-slim"
    assert restored.validation_errors() == []


def test_environment_spec_reports_missing_required_fields():
    spec = EnvironmentSpec(
        id="",
        instruction="",
        verifier=VerifierSpec(command="", path=None, timeout_sec=0),
        rollout=RolloutSpec(max_tool_calls=0),
    )

    errors = spec.validation_errors()

    assert "missing id" in errors
    assert "missing instruction" in errors
    assert "missing verifier command/path" in errors
    assert "rollout.max_tool_calls must be positive" in errors
    assert "verifier.timeout_sec must be positive" in errors


def test_environment_spec_rejects_ambiguous_reward_components():
    spec = EnvironmentSpec(
        id="env_rewards",
        instruction="Return component rewards.",
        verifier=VerifierSpec(
            command="./verify.sh",
            reward_components=[
                RewardComponentSpec(name="correctness"),
                RewardComponentSpec(name="correctness"),
                RewardComponentSpec(name="", weight=float("inf")),
            ],
        ),
    )

    errors = spec.validation_errors()

    assert "verifier.reward_components names must be non-empty" in errors
    assert "verifier.reward_components names must be unique" in errors
    assert "verifier.reward_components weights must be finite" in errors
