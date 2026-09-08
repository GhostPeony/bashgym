"""Fail-closed Docker episodes for coding evaluation, using Arena sandbox settings.

Unlike Arena's compatibility manager, this adapter never pulls images and
enforces execution deadlines. It shares the environment/rollout contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Callable

from bashgym.arena.sandbox import SandboxConfig, is_dangerous_command
from bashgym.environments.builder import (
    PROTECTED_MANIFEST_NAME,
    materialize_environment,
    protected_environment_paths,
)
from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.personal_coding import safe_task_path
from bashgym.environments.rollout import (
    CommandObservation,
    EnvironmentRolloutResult,
    ModelRolloutPlan,
    RolloutAttempt,
    RolloutCommandPlan,
    build_environment_rollout_messages,
    is_submit_command,
    parse_shell_command_response,
)


def validate_pinned_image(image: str) -> str:
    if not isinstance(image, str) or not re.fullmatch(
        r"(?:sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64})", image
    ):
        raise ValueError("Docker execution requires an explicitly pinned image digest")
    return image


def _validate_environment(spec: EnvironmentSpec) -> None:
    errors = spec.validation_errors()
    if errors:
        raise ValueError("invalid coding environment: " + "; ".join(errors))
    safe_task_path(spec.id)
    if "/" in spec.id:
        raise ValueError("environment ID must be a single safe path component")
    for path in spec.files:
        safe_task_path(path)
        if path in {"env.json", PROTECTED_MANIFEST_NAME}:
            raise ValueError("reserved environment file path")
    if len({path.casefold() for path in spec.files}) != len(spec.files):
        raise ValueError("case-colliding task file paths")
    if not spec.build.network_disabled or spec.build.setup_commands:
        raise ValueError(
            "coding episodes require an offline prebuilt environment without setup commands"
        )
    if spec.metadata.get("allow_protected_file_edits") or spec.rollout.metadata.get(
        "allow_protected_file_edits"
    ):
        raise ValueError("coding evaluation cannot allow protected test edits")
    for path in protected_environment_paths(spec):
        safe_task_path(path)
    if spec.verifier.kind != "coding_unittest" or spec.verifier.path != "verify.py":
        raise ValueError("Docker coding adapter requires the coding_unittest verifier")
    if (
        type(spec.verifier.metadata.get("test_count")) is not int
        or spec.verifier.metadata["test_count"] <= 0
    ):
        raise ValueError("coding verifier requires a positive test count")
    if min(spec.rollout.timeout_sec, spec.rollout.bash_timeout_sec, spec.verifier.timeout_sec) <= 0:
        raise ValueError("coding episode timeouts must be positive")


class DockerCodingEpisode:
    """One container and fresh workspace; every exit removes both."""

    def __init__(
        self, spec: EnvironmentSpec, *, image: str, client=None, max_output_bytes: int = 16384
    ):
        _validate_environment(spec)
        self.image = validate_pinned_image(image)
        if not 64 <= max_output_bytes <= 1024 * 1024:
            raise ValueError("output byte limit must be between 64 and 1048576")
        self.spec = spec
        self.max_output_bytes = max_output_bytes
        self.client = client
        self.owns_client = client is None
        self.container = None
        self.temporary = None
        self.dead = False
        self.config = SandboxConfig(
            image=image,
            memory_limit="512m",
            cpu_limit=1.0,
            network_mode="none",
            read_only_root=True,
            cap_add=[],
            mount_hooks=False,
        )

    def __enter__(self):
        try:
            if self.client is None:
                import docker

                self.client = docker.from_env(timeout=5)
            self.client.ping()
            local_image = self.client.images.get(self.image)
        except Exception as exc:
            if self.owns_client and self.client is not None:
                self.client.close()
            raise RuntimeError(
                "Docker and the pinned local image must be available; no image is pulled"
            ) from exc
        try:
            from docker.types import Mount

            self.temporary = tempfile.TemporaryDirectory(prefix="bashgym-coding-")
            build = materialize_environment(self.spec, self.temporary.name)
            self.workspace = build.path
            self.expected = {}
            protected = set(protected_environment_paths(self.spec)) | {PROTECTED_MANIFEST_NAME}
            for path in self.workspace.rglob("*"):
                if path.is_dir():
                    path.chmod(0o777)
                else:
                    relative = path.relative_to(self.workspace).as_posix()
                    path.chmod(0o444 if relative in protected else 0o666)
            self.workspace.chmod(0o777)
            mounts = [
                Mount(target="/workspace", source=str(self.workspace), type="bind", read_only=False)
            ]
            tests_root = self.workspace / "tests"
            if tests_root.is_dir():
                mounts.append(
                    Mount(
                        target="/workspace/tests",
                        source=str(tests_root),
                        type="bind",
                        read_only=True,
                    )
                )
            for relative in sorted(protected):
                path = self.workspace / relative
                if not path.is_file():
                    raise ValueError("protected coding file is missing")
                self.expected[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                if not relative.startswith("tests/"):
                    mounts.append(
                        Mount(
                            target=f"/workspace/{relative}",
                            source=str(path),
                            type="bind",
                            read_only=True,
                        )
                    )
            self.container = self.client.containers.create(
                image=local_image.id,
                name="bashgym-coding-" + uuid.uuid4().hex,
                mounts=mounts,
                working_dir="/workspace",
                command=["sleep", "infinity"],
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "HOME": "/tmp",
                },
                user="65534:65534",
                network_mode=self.config.network_mode,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                cap_drop=self.config.cap_drop,
                cap_add=[],
                read_only=self.config.read_only_root,
                privileged=False,
                security_opt=["no-new-privileges:true"],
                pids_limit=64,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m,mode=1777"},
                detach=True,
                tty=False,
                stdin_open=False,
            )
            self.container.start()
            self.deadline = time.monotonic() + self.spec.rollout.timeout_sec
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        try:
            if self.container is not None:
                self.container.remove(force=True, v=True)
        finally:
            if self.temporary is not None:
                self.temporary.cleanup()
            if self.owns_client and self.client is not None:
                self.client.close()

    def tampered(self) -> bool:
        root = self.workspace.resolve()
        for relative, expected in self.expected.items():
            path = self.workspace / relative
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                return True
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return True
        return False

    def kill(self):
        self.dead = True
        self.container.kill()

    def run(self, command: str, *, verifier: bool = False) -> CommandObservation:
        started = time.monotonic()
        if self.dead or started >= self.deadline:
            if not self.dead:
                self.kill()
            return CommandObservation(
                command, "/workspace", 124, "", "Episode timed out", 0, timeout=True
            )
        if not verifier and is_dangerous_command(command):
            return CommandObservation(
                command, "/workspace", 126, "", "Command blocked", 0, blocked=True
            )
        timeout = min(
            self.deadline - started,
            self.spec.verifier.timeout_sec if verifier else self.spec.rollout.bash_timeout_sec,
        )
        state = {"stdout": bytearray(), "stderr": bytearray(), "error": None, "exit_code": -1}

        def execute():
            try:
                argv = (
                    ["python", "-I", "/workspace/verify.py"]
                    if verifier
                    else ["/bin/sh", "-lc", command]
                )
                execution = self.client.api.exec_create(
                    self.container.id,
                    argv,
                    workdir="/workspace",
                    user="65534:65534",
                    stdout=True,
                    stderr=True,
                )
                stream = self.client.api.exec_start(execution["Id"], stream=True, demux=True)
                for stdout, stderr in stream:
                    for key, chunk in (("stdout", stdout), ("stderr", stderr)):
                        remaining = (
                            self.max_output_bytes - len(state["stdout"]) - len(state["stderr"])
                        )
                        if chunk and remaining > 0:
                            state[key].extend(chunk[:remaining])
                result = self.client.api.exec_inspect(execution["Id"])
                if result.get("Running") or result.get("ExitCode") is None:
                    raise RuntimeError("Docker execution has no terminal exit status")
                state["exit_code"] = int(result["ExitCode"])
            except Exception as exc:
                state["error"] = str(exc)

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        thread.join(timeout)
        timed_out = thread.is_alive()
        if timed_out:
            self.kill()
            thread.join(1)

        def bounded_text(data, limit):
            rendered = bytes(data).decode("utf-8", errors="replace").encode("utf-8")
            return rendered[:limit].decode("utf-8", errors="ignore")

        stdout = bounded_text(state["stdout"], self.max_output_bytes)
        stderr_data = str(state["error"]).encode("utf-8") if state["error"] else state["stderr"]
        stderr = bounded_text(stderr_data, self.max_output_bytes - len(stdout.encode("utf-8")))
        return CommandObservation(
            command,
            "/workspace",
            124 if timed_out else state["exit_code"],
            stdout,
            stderr,
            time.monotonic() - started,
            timeout=timed_out,
            blocked=state["error"] is not None,
        )


def _verified_result(observation: CommandObservation, expected_tests: int) -> tuple[bool, str]:
    if observation.timeout:
        return False, "timeout"
    try:
        report = json.loads(observation.stdout.splitlines()[-1])
        keys = ("tests_run", "failures", "errors", "skipped")
        if report.get("schema_version") != "bashgym.coding_tests.v1" or any(
            type(report.get(key)) is not int or report[key] < 0 for key in keys
        ):
            raise ValueError("invalid test report")
        if report["tests_run"] != expected_tests:
            raise ValueError("test count mismatch")
        passed = (
            observation.exit_code == 0
            and not observation.blocked
            and not any(report[key] for key in keys[1:])
        )
        return passed, "passed" if passed else "failed"
    except (IndexError, TypeError, ValueError, AttributeError):
        return False, "invalid_report"


def _run_episode(
    plan, command_provider: Callable, *, image: str, client=None, max_output_bytes=16384
) -> EnvironmentRolloutResult:
    spec = plan.environment
    observations = []
    verifier_observation = None
    passed, status = False, "failed"
    with DockerCodingEpisode(
        spec, image=image, client=client, max_output_bytes=max_output_bytes
    ) as episode:
        initial = spec.metadata.get("initial_command")
        if initial:
            observations.append(episode.run(str(initial)))
        budget = min(
            getattr(plan, "max_tool_calls", None) or spec.rollout.max_tool_calls,
            spec.rollout.max_tool_calls,
            spec.rollout.max_steps,
        ) - len(observations)
        for index in range(budget):
            if episode.dead or any(item.blocked for item in observations):
                break
            if episode.tampered():
                break
            provider_state = {}

            def get_command():
                try:
                    provider_state["command"] = command_provider(observations, index)
                except Exception as exc:
                    provider_state["error"] = exc

            provider_thread = threading.Thread(target=get_command, daemon=True)
            provider_thread.start()
            provider_thread.join(max(0, episode.deadline - time.monotonic()))
            if provider_thread.is_alive():
                episode.kill()
                observations.append(
                    CommandObservation(
                        "", "/workspace", 124, "", "Command provider timed out", 0, timeout=True
                    )
                )
                break
            if "error" in provider_state:
                raise provider_state["error"]
            command = provider_state["command"]
            if command is None or is_submit_command(command):
                break
            observations.append(episode.run(command))
        if episode.tampered():
            status = "tampered"
        elif any(item.timeout for item in observations):
            status = "timeout"
        elif any(item.blocked for item in observations):
            status = "blocked"
        else:
            verifier_observation = episode.run(spec.verifier.command, verifier=True)
            passed, status = _verified_result(
                verifier_observation, spec.verifier.metadata["test_count"]
            )
            if episode.tampered():
                passed, status = False, "tampered"
        workspace = episode.workspace
    action_chars = sum(len(item.command) for item in observations)
    observation_chars = sum(len(item.stdout) + len(item.stderr) for item in observations)
    attempt = RolloutAttempt(
        environment_id=spec.id,
        attempt_index=plan.attempt_index,
        passed=passed,
        reward=1.0 if passed else 0.0,
        verifier_status=status,
        timeout=status == "timeout",
        tool_calls=len(observations),
        action_tokens=action_chars // 4,
        observation_tokens=observation_chars // 4,
        tokens=(action_chars + observation_chars) // 4,
        metadata={
            **plan.metadata,
            "sandbox": "docker",
            "image": image,
            "split": spec.metadata.get("split"),
            "content_sha256": spec.metadata.get("content_sha256"),
            "workspace_retained": False,
            "token_evidence": "estimated_not_on_policy",
        },
    )
    return EnvironmentRolloutResult(attempt, workspace, observations, verifier_observation)


def run_docker_environment_attempt(
    plan: RolloutCommandPlan, *, image: str, client=None, max_output_bytes: int = 16384
) -> EnvironmentRolloutResult:
    return _run_episode(
        plan,
        lambda observations, index: plan.commands[index] if index < len(plan.commands) else None,
        image=image,
        client=client,
        max_output_bytes=max_output_bytes,
    )


def run_docker_model_environment_attempt(
    plan: ModelRolloutPlan,
    complete: Callable,
    *,
    image: str,
    client=None,
    max_output_bytes: int = 16384,
) -> EnvironmentRolloutResult:
    def next_command(observations, index):
        messages = build_environment_rollout_messages(plan.environment, observations)
        messages[0][
            "content"
        ] += " Each command starts in /workspace; filesystem changes persist between commands."
        response = complete(messages)
        command = parse_shell_command_response(response)
        if command is None:
            raise ValueError("model response must contain a shell command")
        return command

    return _run_episode(
        plan, next_command, image=image, client=client, max_output_bytes=max_output_bytes
    )
