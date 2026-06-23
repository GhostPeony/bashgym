"""Tests for EnvironmentSpec serialization and validation."""

from bashgym.environments.contracts import (
    BuildSpec,
    EnvironmentAxis,
    EnvironmentSpec,
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
        verifier=VerifierSpec(kind="pytest", command="pytest tests", path="tests/test_parser.py"),
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
