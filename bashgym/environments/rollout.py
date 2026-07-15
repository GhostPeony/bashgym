"""Local persistent-session rollouts for executable environment specs.

This is the first runtime bridge between ``EnvironmentSpec`` and pass@k: a
caller supplies the commands for one attempt, BashGym materializes the
environment, executes those commands in a workspace with persistent cwd/env
bookkeeping, runs the verifier, and returns an ``EnvironmentAttempt`` record.

The implementation is intentionally local and hermetic. Docker/Harbor pools can
plug in later behind the same result shape.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bashgym.arena.sandbox import is_dangerous_command
from bashgym.environments.builder import audit_environment_manifest, materialize_environment
from bashgym.environments.contracts import EnvironmentSpec, VerifierSpec

ModelCompleter = Callable[[list[dict[str, str]]], Any]


def _approx_tokens(text: str) -> int:
    """Cheap telemetry token estimate for rollout accounting."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _safe_run_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._")[:80] or "env"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _extract_message(response: Any) -> dict[str, Any]:
    if isinstance(response, str):
        return {"content": response}
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    if isinstance(response.get("message"), dict):
        return response["message"]
    return response


def _json_from_text(text: str) -> Any | None:
    import re

    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_verifier_rewards(
    stdout: str,
    verifier: VerifierSpec,
) -> tuple[float, dict[str, float]]:
    """Read a declared multi-reward verifier result from JSON output.

    A verifier with ``reward_components`` must emit a JSON object on stdout:
    ``{"reward_components": {"correctness": 1.0, "format": 0.5}}``.
    It may optionally include ``total_reward``; when present, the value must
    match the component weights declared by the environment contract.
    """

    if not verifier.reward_components:
        raise ValueError("verifier does not declare reward_components")

    payload = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        candidate = _json_from_text(stdout)
        if isinstance(candidate, dict):
            payload = candidate
    if payload is None:
        raise ValueError("verifier stdout must contain a JSON reward object")

    if isinstance(payload.get("bashgym_reward"), dict):
        payload = payload["bashgym_reward"]
    raw_components = payload.get("reward_components", payload.get("components"))
    if not isinstance(raw_components, dict):
        raise ValueError("verifier JSON must contain a reward_components object")
    components: dict[str, float] = {}
    for raw_name, raw_value in raw_components.items():
        try:
            components[str(raw_name)] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward component {raw_name!r} must be numeric") from exc

    computed_total = verifier.combine_reward_components(components)
    declared_total = payload.get("total_reward", payload.get("reward"))
    if declared_total is not None:
        try:
            declared_total_value = float(declared_total)
        except (TypeError, ValueError) as exc:
            raise ValueError("declared total_reward must be numeric") from exc
        if not math.isclose(declared_total_value, computed_total, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "declared total_reward does not match the weighted reward components"
            )
    return computed_total, components


def _command_from_args(arguments: Any) -> str | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip() or None
    if isinstance(arguments, dict):
        command = arguments.get("command") or arguments.get("cmd")
        return str(command).strip() if command else None
    return None


def parse_shell_command_response(response: Any) -> str | None:
    """Extract one shell command from an OpenAI-like response or plain text."""
    message = _extract_message(response)
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            command = _command_from_args(function.get("arguments"))
            if command:
                return command

    content = str(message.get("content", "") if isinstance(message, dict) else "")
    payload = _json_from_text(content)
    if isinstance(payload, dict):
        command = payload.get("command") or payload.get("cmd")
        if command:
            return str(command).strip()

    import re

    shell_fence = re.search(r"```(?:bash|sh|shell)\s*(.*?)\s*```", content, re.DOTALL)
    if shell_fence:
        return shell_fence.group(1).strip() or None
    stripped = content.strip()
    if "\n" not in stripped and stripped.startswith(("$ ", "> ")):
        return stripped[2:].strip() or None
    return None


def extract_response_logprobs(response: Any) -> dict[str, Any] | None:
    """Extract OpenAI-compatible chat token logprobs from a response.

    vLLM/OpenAI-compatible endpoints return chat logprobs under
    ``choices[0].logprobs.content``. We persist the action-side tokens and
    logprobs as behavior-policy telemetry for later DPPO validation.
    """

    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list):
        return None

    tokens: list[str] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        raw_logprob = item.get("logprob")
        if raw_logprob is None:
            continue
        try:
            logprob = float(raw_logprob)
        except (TypeError, ValueError):
            continue
        tokens.append(str(item.get("token", "")))
        token_logprobs.append(logprob)

    if not token_logprobs:
        return None

    return {
        "n_tokens": len(token_logprobs),
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "sum_logprob": sum(token_logprobs),
        "mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "min_logprob": min(token_logprobs),
        "max_logprob": max(token_logprobs),
    }


def is_submit_command(command: str) -> bool:
    return command.strip().lower() in {"submit", "bashgym_submit", "done", "final"}


def _tamper_observation(workspace: Path, audit: dict[str, Any]) -> CommandObservation:
    paths = sorted(set(audit.get("tampered_paths", []) + audit.get("missing_paths", [])))
    detail = ", ".join(paths) if paths else "unknown protected file"
    return CommandObservation(
        command="<bashgym-tamper-audit>",
        cwd=str(workspace),
        exit_code=126,
        stdout="",
        stderr=f"Protected environment files changed before verification: {detail}",
        duration_sec=0.0,
        blocked=True,
    )


def build_environment_rollout_messages(
    environment: EnvironmentSpec,
    observations: list[CommandObservation],
    *,
    max_observation_chars: int = 6000,
) -> list[dict[str, str]]:
    """Prompt a served model to act as a one-command-at-a-time terminal agent."""
    file_list = "\n".join(f"- {path}" for path in sorted(environment.files)) or "- env.json"
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a terminal agent inside a local task workspace. "
                'Respond only as JSON: {"command":"<one shell command>"}. '
                'When the task is complete, respond with {"command":"submit"}. '
                "Do not explain your answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{environment.instruction}\n\n"
                f"Visible workspace files:\n{file_list}\n\n"
                "Issue the next shell command."
            ),
        },
    ]
    for observation in observations:
        messages.append(
            {"role": "assistant", "content": json.dumps({"command": observation.command})}
        )
        output = "\n".join(
            part
            for part in [
                f"exit_code={observation.exit_code}",
                f"stdout:\n{observation.stdout}",
                f"stderr:\n{observation.stderr}",
            ]
            if part
        )
        messages.append(
            {
                "role": "user",
                "content": _truncate(output, max_observation_chars)
                + "\n\nIssue the next shell command.",
            }
        )
    return messages


@dataclass
class CommandObservation:
    command: str
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timeout: bool = False
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_sec": self.duration_sec,
            "timeout": self.timeout,
            "blocked": self.blocked,
        }


@dataclass
class RolloutAttempt:
    """Verifier outcome for one local environment attempt."""

    environment_id: str
    attempt_index: int
    passed: bool
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    verifier_status: str | None = None
    timeout: bool = False
    tool_calls: int | None = None
    tokens: int | None = None
    action_tokens: int | None = None
    observation_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "environment_id": self.environment_id,
            "attempt_index": self.attempt_index,
            "passed": self.passed,
            "reward": self.reward,
            "verifier_status": self.verifier_status,
            "timeout": self.timeout,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "action_tokens": self.action_tokens,
            "observation_tokens": self.observation_tokens,
            "metadata": self.metadata,
        }
        if self.reward_components:
            payload["reward_components"] = self.reward_components
        return payload


@dataclass
class RolloutCommandPlan:
    environment: EnvironmentSpec
    commands: list[str]
    attempt_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRolloutPlan:
    environment: EnvironmentSpec
    attempt_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tool_calls: int | None = None


@dataclass
class EnvironmentRolloutResult:
    attempt: RolloutAttempt
    workspace: Path
    observations: list[CommandObservation]
    verifier_observation: CommandObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt.to_dict(),
            "workspace": str(self.workspace),
            "observations": [observation.to_dict() for observation in self.observations],
            "verifier_observation": (
                self.verifier_observation.to_dict() if self.verifier_observation else None
            ),
        }


def _resolve_verifier_reward(
    environment: EnvironmentSpec,
    verifier_observation: CommandObservation,
    *,
    verifier_passed: bool,
) -> tuple[float | None, dict[str, float], str | None]:
    """Resolve scalar and component rewards without making NeMo Gym mandatory."""

    if not verifier_passed:
        return 0.0, {}, None
    if not environment.verifier.reward_components:
        return 1.0, {}, None
    try:
        total, components = extract_verifier_rewards(
            verifier_observation.stdout,
            environment.verifier,
        )
    except ValueError as exc:
        return None, {}, str(exc)
    return total, components, None


class LocalPersistentShell:
    """Execute commands with persistent cwd and environment bookkeeping."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout_sec: int = 120,
        allow_dangerous_commands: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.cwd = self.workspace
        self.timeout_sec = timeout_sec
        self.allow_dangerous_commands = allow_dangerous_commands

    def run(self, command: str) -> CommandObservation:
        started = time.monotonic()
        if not self.allow_dangerous_commands and is_dangerous_command(command):
            return CommandObservation(
                command=command,
                cwd=str(self.cwd),
                exit_code=126,
                stdout="",
                stderr="Command blocked by BashGym safety policy",
                duration_sec=time.monotonic() - started,
                blocked=True,
            )

        cd_target = self._parse_cd(command)
        if cd_target is not None:
            return self._change_directory(command, cd_target, started)

        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            return CommandObservation(
                command=command,
                cwd=str(self.cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandObservation(
                command=command,
                cwd=str(self.cwd),
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Command timed out",
                duration_sec=time.monotonic() - started,
                timeout=True,
            )

    def _parse_cd(self, command: str) -> str | None:
        stripped = command.strip()
        if stripped == "cd":
            return "."
        if stripped.startswith("cd "):
            return stripped[3:].strip().strip('"').strip("'") or "."
        return None

    def _change_directory(self, command: str, target: str, started: float) -> CommandObservation:
        candidate = (self.cwd / target).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError:
            return CommandObservation(
                command=command,
                cwd=str(self.cwd),
                exit_code=1,
                stdout="",
                stderr="cd target escapes environment workspace",
                duration_sec=time.monotonic() - started,
                blocked=True,
            )
        if not candidate.is_dir():
            return CommandObservation(
                command=command,
                cwd=str(self.cwd),
                exit_code=1,
                stdout="",
                stderr=f"directory not found: {target}",
                duration_sec=time.monotonic() - started,
            )
        self.cwd = candidate
        return CommandObservation(
            command=command,
            cwd=str(self.cwd),
            exit_code=0,
            stdout="",
            stderr="",
            duration_sec=time.monotonic() - started,
        )


def run_local_environment_attempt(
    plan: RolloutCommandPlan,
    workspace_root: str | Path,
    *,
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = True,
) -> EnvironmentRolloutResult:
    """Run one scripted attempt and verifier in a materialized local workspace."""
    workspace_root_path = Path(workspace_root).resolve()
    run_name = f"{_safe_run_name(plan.environment.id)}_{plan.attempt_index}_{uuid.uuid4().hex[:8]}"
    run_root = workspace_root_path / run_name
    build = materialize_environment(plan.environment, run_root, overwrite=False)
    workspace = build.path
    shell = LocalPersistentShell(
        workspace,
        timeout_sec=plan.environment.rollout.bash_timeout_sec,
        allow_dangerous_commands=allow_dangerous_commands,
    )

    observations: list[CommandObservation] = []
    for command in plan.commands[: plan.environment.rollout.max_tool_calls]:
        observation = shell.run(command)
        observations.append(observation)
        if stop_on_error and observation.exit_code != 0:
            break

    tamper_audit = audit_environment_manifest(workspace)
    tamper_detected = not tamper_audit["ok"]
    if tamper_detected:
        verifier_observation = _tamper_observation(workspace, tamper_audit)
    else:
        verifier_observation = shell.run(plan.environment.verifier.command)

    verifier_passed = verifier_observation.exit_code == 0 and not tamper_detected
    reward, reward_components, reward_error = _resolve_verifier_reward(
        plan.environment,
        verifier_observation,
        verifier_passed=verifier_passed,
    )
    passed = verifier_passed and reward_error is None
    if passed and plan.environment.verifier.reward_components:
        passed = reward is not None and reward >= plan.environment.verifier.success_threshold
    timeout = (
        any(observation.timeout for observation in observations) or verifier_observation.timeout
    )
    blocked = (
        any(observation.blocked for observation in observations) or verifier_observation.blocked
    )
    action_text = "\n".join(observation.command for observation in observations)
    observation_text = "\n".join(
        f"{observation.stdout}\n{observation.stderr}" for observation in observations
    )
    verifier_status = "passed" if passed else "failed"
    if tamper_detected:
        verifier_status = "tampered"
    elif reward_error:
        verifier_status = "reward_error"
    elif blocked:
        verifier_status = "blocked"
    elif timeout:
        verifier_status = "timeout"

    attempt = RolloutAttempt(
        environment_id=plan.environment.id,
        attempt_index=plan.attempt_index,
        passed=passed,
        reward=reward,
        reward_components=reward_components,
        verifier_status=verifier_status,
        timeout=timeout,
        tool_calls=len(observations),
        tokens=_approx_tokens(action_text) + _approx_tokens(observation_text),
        action_tokens=_approx_tokens(action_text),
        observation_tokens=_approx_tokens(observation_text),
        metadata={
            **plan.metadata,
            "workspace": str(workspace),
            "keep_workspace": keep_workspace,
            "verifier_exit_code": verifier_observation.exit_code,
            "tamper_detected": tamper_detected,
            "tamper_audit": tamper_audit,
            **({"reward_error": reward_error} if reward_error else {}),
        },
    )
    result = EnvironmentRolloutResult(
        attempt=attempt,
        workspace=workspace,
        observations=observations,
        verifier_observation=verifier_observation,
    )
    if not keep_workspace:
        shutil.rmtree(run_root, ignore_errors=True)
    return result


def run_local_model_environment_attempt(
    plan: ModelRolloutPlan,
    workspace_root: str | Path,
    complete: ModelCompleter,
    *,
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = False,
    max_observation_chars: int = 6000,
) -> EnvironmentRolloutResult:
    """Run one served-model attempt in a local persistent shell, then verify it."""
    workspace_root_path = Path(workspace_root).resolve()
    run_name = f"{_safe_run_name(plan.environment.id)}_{plan.attempt_index}_{uuid.uuid4().hex[:8]}"
    run_root = workspace_root_path / run_name
    build = materialize_environment(plan.environment, run_root, overwrite=False)
    workspace = build.path
    shell = LocalPersistentShell(
        workspace,
        timeout_sec=plan.environment.rollout.bash_timeout_sec,
        allow_dangerous_commands=allow_dangerous_commands,
    )

    observations: list[CommandObservation] = []
    raw_responses: list[str] = []
    response_logprobs: list[dict[str, Any]] = []
    format_errors = 0
    max_tool_calls = plan.max_tool_calls or plan.environment.rollout.max_tool_calls
    for _ in range(max_tool_calls):
        messages = build_environment_rollout_messages(
            plan.environment,
            observations,
            max_observation_chars=max_observation_chars,
        )
        started = time.monotonic()
        try:
            response = complete(messages)
        except Exception as exc:  # noqa: BLE001 - request failure should fail one attempt
            observations.append(
                CommandObservation(
                    command="",
                    cwd=str(shell.cwd),
                    exit_code=502,
                    stdout="",
                    stderr=f"model request failed: {exc}",
                    duration_sec=time.monotonic() - started,
                    blocked=True,
                )
            )
            break
        raw_responses.append(_truncate(str(response), 2000))
        logprob_summary = extract_response_logprobs(response)
        if logprob_summary is not None:
            response_logprobs.append(logprob_summary)
        command = parse_shell_command_response(response)
        if not command:
            format_errors += 1
            observations.append(
                CommandObservation(
                    command="",
                    cwd=str(shell.cwd),
                    exit_code=2,
                    stdout="",
                    stderr="could not parse a shell command from model response",
                    duration_sec=time.monotonic() - started,
                    blocked=True,
                )
            )
            break
        if is_submit_command(command):
            break
        observation = shell.run(command)
        observations.append(observation)
        if stop_on_error and observation.exit_code != 0:
            break

    tamper_audit = audit_environment_manifest(workspace)
    tamper_detected = not tamper_audit["ok"]
    if tamper_detected:
        verifier_observation = _tamper_observation(workspace, tamper_audit)
    else:
        verifier_observation = shell.run(plan.environment.verifier.command)

    verifier_passed = verifier_observation.exit_code == 0 and not tamper_detected
    reward, reward_components, reward_error = _resolve_verifier_reward(
        plan.environment,
        verifier_observation,
        verifier_passed=verifier_passed,
    )
    passed = verifier_passed and reward_error is None
    if passed and plan.environment.verifier.reward_components:
        passed = reward is not None and reward >= plan.environment.verifier.success_threshold
    timeout = (
        any(observation.timeout for observation in observations) or verifier_observation.timeout
    )
    blocked = (
        any(observation.blocked for observation in observations) or verifier_observation.blocked
    )
    action_text = "\n".join(observation.command for observation in observations)
    observation_text = "\n".join(
        f"{observation.stdout}\n{observation.stderr}" for observation in observations
    )
    verifier_status = "passed" if passed else "failed"
    if tamper_detected:
        verifier_status = "tampered"
    elif reward_error:
        verifier_status = "reward_error"
    elif format_errors:
        verifier_status = "format_error"
    elif blocked:
        verifier_status = "blocked"
    elif timeout:
        verifier_status = "timeout"
    token_logprobs = [
        logprob
        for response_logprob in response_logprobs
        for logprob in response_logprob["token_logprobs"]
    ]

    attempt = RolloutAttempt(
        environment_id=plan.environment.id,
        attempt_index=plan.attempt_index,
        passed=passed,
        reward=reward,
        reward_components=reward_components,
        verifier_status=verifier_status,
        timeout=timeout,
        tool_calls=len([observation for observation in observations if observation.command]),
        tokens=_approx_tokens(action_text) + _approx_tokens(observation_text),
        action_tokens=_approx_tokens(action_text),
        observation_tokens=_approx_tokens(observation_text),
        metadata={
            **plan.metadata,
            "workspace": str(workspace),
            "keep_workspace": keep_workspace,
            "verifier_exit_code": verifier_observation.exit_code,
            "format_errors": format_errors,
            "model_responses": raw_responses,
            "response_logprobs": response_logprobs,
            "behavior_logprob_tokens": len(token_logprobs),
            "behavior_logprob_sum": sum(token_logprobs) if token_logprobs else None,
            "behavior_mean_logprob": (
                sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
            ),
            "tamper_detected": tamper_detected,
            "tamper_audit": tamper_audit,
            **({"reward_error": reward_error} if reward_error else {}),
        },
    )
    result = EnvironmentRolloutResult(
        attempt=attempt,
        workspace=workspace,
        observations=observations,
        verifier_observation=verifier_observation,
    )
    if not keep_workspace:
        shutil.rmtree(run_root, ignore_errors=True)
    return result


def run_local_environment_rollouts(
    plans: list[RolloutCommandPlan],
    workspace_root: str | Path,
    *,
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = True,
) -> list[EnvironmentRolloutResult]:
    """Run a batch of local command-script attempts."""
    return [
        run_local_environment_attempt(
            plan,
            workspace_root,
            keep_workspace=keep_workspace,
            allow_dangerous_commands=allow_dangerous_commands,
            stop_on_error=stop_on_error,
        )
        for plan in plans
    ]


def run_local_model_environment_rollouts(
    plans: list[ModelRolloutPlan],
    workspace_root: str | Path,
    complete: ModelCompleter,
    *,
    keep_workspace: bool = True,
    allow_dangerous_commands: bool = False,
    stop_on_error: bool = False,
    max_observation_chars: int = 6000,
) -> list[EnvironmentRolloutResult]:
    """Run a batch of served-model local environment attempts."""
    return [
        run_local_model_environment_attempt(
            plan,
            workspace_root,
            complete,
            keep_workspace=keep_workspace,
            allow_dangerous_commands=allow_dangerous_commands,
            stop_on_error=stop_on_error,
            max_observation_chars=max_observation_chars,
        )
        for plan in plans
    ]
