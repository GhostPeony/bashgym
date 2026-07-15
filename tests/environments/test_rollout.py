"""Tests for local command-script environment rollouts."""

from __future__ import annotations

import subprocess
import sys

import pytest

from bashgym.environments.contracts import (
    EnvironmentSpec,
    RewardComponentSpec,
    VerifierSpec,
)
from bashgym.environments.rollout import (
    CommandObservation,
    ModelRolloutPlan,
    RolloutCommandPlan,
    build_environment_rollout_messages,
    extract_response_logprobs,
    parse_shell_command_response,
    run_local_environment_attempt,
    run_local_model_environment_attempt,
)


def _py(code: str) -> str:
    return subprocess.list2cmdline([sys.executable, "-c", code])


def test_run_local_environment_attempt_passes_verifier(tmp_path):
    spec = EnvironmentSpec(
        id="env_rollout_ok",
        instruction="Create answer.txt containing ok.",
        verifier=VerifierSpec(
            command=_py(
                "from pathlib import Path; "
                "raise SystemExit(0 if Path('answer.txt').read_text() == 'ok' else 1)"
            )
        ),
    )
    plan = RolloutCommandPlan(
        environment=spec,
        attempt_index=0,
        commands=[_py("from pathlib import Path; Path('answer.txt').write_text('ok')")],
    )

    result = run_local_environment_attempt(plan, tmp_path)

    assert result.attempt.environment_id == "env_rollout_ok"
    assert result.attempt.passed is True
    assert result.attempt.reward == 1.0
    assert result.attempt.tool_calls == 1
    assert result.attempt.action_tokens is not None
    assert result.verifier_observation is not None
    assert result.verifier_observation.exit_code == 0


def test_run_local_environment_attempt_preserves_named_reward_components(tmp_path):
    spec = EnvironmentSpec(
        id="env_multi_reward",
        instruction="Produce a correct, well-formatted answer.",
        verifier=VerifierSpec(
            command=_py(
                "import json; print(json.dumps({'reward_components': "
                "{'correctness': 1.0, 'format': 0.5}, 'total_reward': 1.1}))"
            ),
            reward_type="components",
            success_threshold=1.0,
            reward_components=[
                RewardComponentSpec(name="correctness"),
                RewardComponentSpec(name="format", weight=0.2),
            ],
        ),
    )

    result = run_local_environment_attempt(
        RolloutCommandPlan(environment=spec, commands=[]),
        tmp_path,
    )

    assert result.attempt.passed is True
    assert result.attempt.reward == pytest.approx(1.1)
    assert result.attempt.reward_components == {"correctness": 1.0, "format": 0.5}
    assert result.attempt.to_dict()["reward_components"] == result.attempt.reward_components


def test_run_local_environment_attempt_fails_closed_on_missing_component_rewards(tmp_path):
    spec = EnvironmentSpec(
        id="env_missing_rewards",
        instruction="Emit named rewards.",
        verifier=VerifierSpec(
            command=_py("print('verifier complete')"),
            reward_type="components",
            reward_components=[
                RewardComponentSpec(name="correctness"),
                RewardComponentSpec(name="format"),
            ],
        ),
    )

    result = run_local_environment_attempt(
        RolloutCommandPlan(environment=spec, commands=[]),
        tmp_path,
    )

    assert result.attempt.passed is False
    assert result.attempt.reward is None
    assert result.attempt.reward_components == {}
    assert result.attempt.verifier_status == "reward_error"
    assert "JSON reward object" in result.attempt.metadata["reward_error"]


def test_run_local_environment_attempt_fails_verifier(tmp_path):
    spec = EnvironmentSpec(
        id="env_rollout_fail",
        instruction="Create answer.txt containing ok.",
        verifier=VerifierSpec(
            command=_py(
                "from pathlib import Path; "
                "raise SystemExit(0 if Path('answer.txt').exists() else 1)"
            )
        ),
    )
    plan = RolloutCommandPlan(environment=spec, attempt_index=0, commands=[])

    result = run_local_environment_attempt(plan, tmp_path)

    assert result.attempt.passed is False
    assert result.attempt.reward == 0.0
    assert result.attempt.verifier_status == "failed"


def test_run_local_environment_attempt_blocks_dangerous_command(tmp_path):
    spec = EnvironmentSpec(
        id="env_rollout_blocked",
        instruction="Do not run dangerous commands.",
        verifier=VerifierSpec(command=_py("raise SystemExit(0)")),
    )
    plan = RolloutCommandPlan(environment=spec, attempt_index=0, commands=["rm -rf /"])

    result = run_local_environment_attempt(plan, tmp_path)

    assert result.observations[0].blocked is True
    assert result.attempt.passed is True
    assert result.attempt.verifier_status == "blocked"


def test_run_local_environment_attempt_detects_verifier_tamper(tmp_path):
    spec = EnvironmentSpec(
        id="env_tamper_verifier",
        instruction="Try to make the verifier pass by editing it.",
        verifier=VerifierSpec(command="bash verify.sh", path="verify.sh"),
        files={"verify.sh": "#!/usr/bin/env bash\nexit 1\n"},
    )
    plan = RolloutCommandPlan(
        environment=spec,
        attempt_index=0,
        commands=[_py("from pathlib import Path; Path('verify.sh').write_text('exit 0\\n')")],
    )

    result = run_local_environment_attempt(plan, tmp_path)

    assert result.attempt.passed is False
    assert result.attempt.reward == 0.0
    assert result.attempt.verifier_status == "tampered"
    assert result.attempt.metadata["tamper_detected"] is True
    assert result.attempt.metadata["tamper_audit"]["tampered_paths"] == ["verify.sh"]
    assert result.verifier_observation is not None
    assert result.verifier_observation.command == "<bashgym-tamper-audit>"


def test_run_local_environment_attempt_detects_tests_directory_tamper(tmp_path):
    spec = EnvironmentSpec(
        id="env_tamper_tests",
        instruction="Try to edit hidden tests.",
        verifier=VerifierSpec(command=_py("raise SystemExit(0)")),
        files={"tests/check.txt": "must stay fixed\n"},
    )
    plan = RolloutCommandPlan(
        environment=spec,
        attempt_index=0,
        commands=[_py("from pathlib import Path; Path('tests/check.txt').write_text('changed\\n')")],
    )

    result = run_local_environment_attempt(plan, tmp_path)

    assert result.attempt.passed is False
    assert result.attempt.verifier_status == "tampered"
    assert result.attempt.metadata["tamper_audit"]["tampered_paths"] == ["tests/check.txt"]


def test_run_local_environment_attempt_removes_workspace_when_requested(tmp_path):
    spec = EnvironmentSpec(
        id="env_cleanup",
        instruction="Run a passing verifier.",
        verifier=VerifierSpec(command=_py("raise SystemExit(0)")),
    )
    plan = RolloutCommandPlan(environment=spec, attempt_index=0, commands=[])

    result = run_local_environment_attempt(plan, tmp_path, keep_workspace=False)

    assert result.attempt.passed is True
    assert result.workspace.exists() is False


def test_parse_shell_command_response_accepts_tool_call_and_json_text():
    assert (
        parse_shell_command_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "run_command",
                                        "arguments": '{"command":"ls -la"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        )
        == "ls -la"
    )
    assert parse_shell_command_response('{"command":"pytest tests"}') == "pytest tests"
    assert parse_shell_command_response("```bash\npython fix.py\n```") == "python fix.py"


def test_extract_response_logprobs_summarizes_openai_chat_shape():
    summary = extract_response_logprobs(
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "{", "logprob": -0.1},
                            {"token": '"command"', "logprob": -0.2},
                            {"token": "}", "logprob": -0.3},
                        ]
                    }
                }
            ]
        }
    )

    assert summary is not None
    assert summary["n_tokens"] == 3
    assert summary["tokens"] == ["{", '"command"', "}"]
    assert summary["sum_logprob"] == pytest.approx(-0.6)
    assert summary["mean_logprob"] == pytest.approx(-0.2)


def test_build_environment_rollout_messages_include_observations(tmp_path):
    spec = EnvironmentSpec(
        id="env_prompt",
        instruction="Fix the script.",
        files={"main.py": "print('todo')\n"},
    )

    messages = build_environment_rollout_messages(
        spec,
        [
            CommandObservation(
                command="echo hi",
                cwd=str(tmp_path),
                exit_code=0,
                stdout="hi\n",
                stderr="",
                duration_sec=0.01,
            )
        ],
    )

    assert "Fix the script" in messages[1]["content"]
    assert "main.py" in messages[1]["content"]
    assert "exit_code=0" in messages[-1]["content"]


def test_run_local_model_environment_attempt_passes_verifier(tmp_path):
    spec = EnvironmentSpec(
        id="env_model_ok",
        instruction="Create answer.txt containing ok.",
        verifier=VerifierSpec(
            command=_py(
                "from pathlib import Path; "
                "raise SystemExit(0 if Path('answer.txt').read_text() == 'ok' else 1)"
            )
        ),
    )
    responses = iter(
        [
            {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": '{"command"', "logprob": -0.1},
                                {"token": ":...", "logprob": -0.2},
                            ]
                        },
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "run_command",
                                        "arguments": {
                                            "command": _py(
                                                "from pathlib import Path; "
                                                "Path('answer.txt').write_text('ok')"
                                            )
                                        },
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            '{"command":"submit"}',
        ]
    )

    def complete(messages):
        assert messages[0]["role"] == "system"
        return next(responses)

    result = run_local_model_environment_attempt(
        ModelRolloutPlan(environment=spec),
        tmp_path,
        complete,
    )

    assert result.attempt.passed is True
    assert result.attempt.verifier_status == "passed"
    assert result.attempt.metadata["format_errors"] == 0
    assert result.attempt.tool_calls == 1
    assert result.attempt.metadata["behavior_logprob_tokens"] == 2
    assert result.attempt.metadata["behavior_mean_logprob"] == pytest.approx(-0.15)
    assert result.attempt.metadata["response_logprobs"][0]["sum_logprob"] == pytest.approx(-0.3)


def test_run_local_model_environment_attempt_reports_format_error(tmp_path):
    spec = EnvironmentSpec(
        id="env_model_format",
        instruction="Create answer.txt containing ok.",
        verifier=VerifierSpec(command=_py("raise SystemExit(1)")),
    )

    result = run_local_model_environment_attempt(
        ModelRolloutPlan(environment=spec),
        tmp_path,
        lambda messages: "I will do it later.",
    )

    assert result.attempt.passed is False
    assert result.attempt.verifier_status == "format_error"
    assert result.attempt.metadata["format_errors"] == 1


def test_run_local_model_environment_attempt_detects_verifier_tamper(tmp_path):
    spec = EnvironmentSpec(
        id="env_model_tamper",
        instruction="Do not edit the verifier.",
        verifier=VerifierSpec(command="bash verify.sh", path="verify.sh"),
        files={"verify.sh": "#!/usr/bin/env bash\nexit 1\n"},
    )

    def complete(messages):
        if len(messages) <= 2:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "run_command",
                                        "arguments": {
                                            "command": _py(
                                                "from pathlib import Path; "
                                                "Path('verify.sh').write_text('exit 0\\n')"
                                            )
                                        },
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        return '{"command":"submit"}'

    result = run_local_model_environment_attempt(
        ModelRolloutPlan(environment=spec),
        tmp_path,
        complete,
    )

    assert result.attempt.passed is False
    assert result.attempt.verifier_status == "tampered"
    assert result.attempt.metadata["tamper_detected"] is True
