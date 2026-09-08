"""Docker boundary behavior without launching containers or downloading images."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import bashgym.environments as environments
from bashgym.environments.rollout import RolloutCommandPlan

IMAGE = "python@sha256:" + "a" * 64
IMAGE_ID = "sha256:" + "b" * 64


def run_api():
    function = getattr(environments, "run_docker_environment_attempt", None)
    assert callable(function), "Missing fail-closed Docker environment adapter"
    return function


def spec():
    return next(
        task
        for task in environments.personal_coding_environment_specs(split="train")
        if task.metadata["task_family"] == "repository_repair"
    )


def report(*, tests=4, failures=0):
    return json.dumps(
        {
            "schema_version": "bashgym.coding_tests.v1",
            "tests_run": tests,
            "failures": failures,
            "errors": 0,
            "skipped": 0,
        }
    ).encode()


class FakeClient:
    def __init__(self, outputs=(), *, unavailable=False, missing_image=False):
        self.outputs = list(outputs)
        self.unavailable = unavailable
        self.missing_image = missing_image
        self.created = []
        self.execs = {}
        self.killed = []
        self.removed = []
        self.release = threading.Event()
        self.images = SimpleNamespace(get=self.image)
        self.containers = SimpleNamespace(create=self.create)
        self.api = self

    def ping(self):
        if self.unavailable:
            raise RuntimeError("Docker unavailable")

    def image(self, image):
        if self.missing_image:
            raise RuntimeError("Image not installed")
        return SimpleNamespace(id=IMAGE_ID)

    def create(self, **kwargs):
        self.created.append(kwargs)
        name = kwargs["name"]
        self.workspace = Path(
            next(mount["Source"] for mount in kwargs["mounts"] if mount["Target"] == "/workspace")
        )
        return SimpleNamespace(
            id=name,
            start=lambda: None,
            kill=lambda: (self.killed.append(name), self.release.set()),
            remove=lambda **kw: self.removed.append((name, kw)),
        )

    def exec_create(self, container, cmd, **kwargs):
        key = str(len(self.execs))
        self.execs[key] = {"command": cmd, "result": self.outputs.pop(0)}
        return {"Id": key}

    def exec_start(self, key, **kwargs):
        code, output = self.execs[key]["result"]
        if callable(output):
            yield from output(self)
        else:
            yield output, b""

    def exec_inspect(self, key):
        return {"Running": False, "ExitCode": self.execs[key]["result"][0]}


def test_docker_episode_is_isolated_readonly_and_cleanup_is_unconditional():
    run = run_api()
    task = spec()
    assert task.metadata["task_family"] == "repository_repair"
    client = FakeClient([(0, b"edited"), (0, report())] * 2)
    plan = RolloutCommandPlan(task, ["echo edited"])
    first = run(plan, image=IMAGE, client=client)
    second = run(plan, image=IMAGE, client=client)
    assert first.attempt.passed and second.attempt.passed
    assert len(client.created) == len(client.removed) == 2
    assert client.created[0]["name"] != client.created[1]["name"]
    for created in client.created:
        assert created["image"] == IMAGE_ID
        assert created["network_mode"] == "none"
        assert created["read_only"] is True
        assert created["cap_drop"] == ["ALL"]
        assert created["security_opt"] == ["no-new-privileges:true"]
        assert created["pids_limit"] > 0
        protected = {m["Target"] for m in created["mounts"] if m["ReadOnly"]}
        assert {"/workspace/verify.py", "/workspace/tests"} <= protected
    assert not first.workspace.exists()


@pytest.mark.parametrize("case", ["unavailable", "missing_image", "mutable"])
def test_docker_fails_closed_before_execution_without_pinned_local_image(case, monkeypatch):
    run = run_api()
    client = FakeClient(unavailable=case == "unavailable", missing_image=case == "missing_image")
    import subprocess

    def no_local_execution(*args, **kwargs):
        pytest.fail("Docker adapter must never fall back to a local subprocess")

    monkeypatch.setattr(subprocess, "run", no_local_execution)
    with pytest.raises((ValueError, RuntimeError), match="Docker|image|Image|pinned"):
        run(
            RolloutCommandPlan(spec(), []),
            image="python:latest" if case == "mutable" else IMAGE,
            client=client,
        )
    assert not client.created


def test_docker_timeout_kills_container_and_does_not_verify():
    run = run_api()
    task = spec()
    task.rollout.bash_timeout_sec = 0.02

    def blocked_output(client):
        client.release.wait(2)
        yield b"late", b""

    client = FakeClient([(0, blocked_output)])
    result = run(RolloutCommandPlan(task, ["sleep 20"]), image=IMAGE, client=client)
    assert result.attempt.verifier_status == "timeout"
    assert not result.attempt.passed and result.attempt.reward == 0
    assert len(client.execs) == len(client.killed) == len(client.removed) == 1


@pytest.mark.parametrize("output", [b"x" * 10000, b"\xff" * 10000], ids=["text", "invalid_utf8"])
def test_docker_output_is_bounded_and_unverified_success_exit_scores_zero(output):
    run = run_api()
    client = FakeClient([(0, output), (0, b"looks good")])
    result = run(
        RolloutCommandPlan(spec(), ["echo noisy"]), image=IMAGE, client=client, max_output_bytes=128
    )
    assert len(result.observations[0].stdout.encode()) <= 128
    assert result.attempt.reward == 0 and not result.attempt.passed
    assert result.attempt.verifier_status == "invalid_report"


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.py",
        "/tmp/escape.py",
        "C:/escape.py",
        "tests/../escape.py",
        "a\\b.py",
        "CON.txt",
        "alias.",
    ],
)
def test_docker_rejects_unsafe_fixture_paths_before_creating_workspace(unsafe):
    run = run_api()
    task = spec()
    task.files[unsafe] = "bad"
    client = FakeClient()
    with pytest.raises(ValueError, match="path"):
        run(RolloutCommandPlan(task, []), image=IMAGE, client=client)
    assert not client.created


def test_docker_protected_test_tampering_cannot_gain_reward():
    run = run_api()

    def tamper(client):
        target = client.workspace / "tests/test_solution.py"
        target.chmod(0o666)
        target.write_text("# erased tests", encoding="utf-8")
        yield b"", b""

    client = FakeClient([(0, tamper)])
    result = run(RolloutCommandPlan(spec(), ["attempt tamper"]), image=IMAGE, client=client)
    assert result.attempt.verifier_status == "tampered"
    assert not result.attempt.passed and result.attempt.reward == 0
    assert len(client.execs) == 1


def test_model_response_deadline_also_kills_and_cleans_episode():
    from bashgym.environments.rollout import ModelRolloutPlan

    run = getattr(environments, "run_docker_model_environment_attempt")
    task = spec()
    task.rollout.timeout_sec = 0.02
    client = FakeClient()

    def stalled_model(messages):
        client.release.wait(2)
        return '{"command":"submit"}'

    started = time.monotonic()
    result = run(ModelRolloutPlan(task), stalled_model, image=IMAGE, client=client)
    assert time.monotonic() - started < 1
    assert result.attempt.verifier_status == "timeout"
    assert client.killed and client.removed


def test_interactive_recovery_observes_failure_then_verifies_repair():
    from bashgym.environments.rollout import ModelRolloutPlan

    task = next(
        task
        for task in environments.personal_coding_environment_specs(split="train")
        if task.metadata["task_family"] == "tool_recovery"
    )
    client = FakeClient([(1, b"settings.json missing"), (0, b"created"), (0, report(tests=3))])
    prompts = []

    def model(messages):
        prompts.append(messages)
        return '{"command":"echo fixed"}' if len(prompts) == 1 else '{"command":"submit"}'

    result = environments.run_docker_model_environment_attempt(
        ModelRolloutPlan(task), model, image=IMAGE, client=client
    )
    assert "settings.json missing" in prompts[0][-1]["content"]
    assert result.attempt.passed and result.attempt.tool_calls == 2
    assert result.attempt.metadata["token_evidence"] == "estimated_not_on_policy"


@pytest.mark.parametrize("tamper", ["tests", "digest"])
def test_nemo_scorer_runs_the_same_docker_verifier_and_rejects_modified_contract(tamper):
    task = spec()
    response = {"output_text": json.dumps({"commands": ["echo repaired"]})}
    client = FakeClient([(0, b"repaired"), (0, report())])
    score = environments.score_personal_coding_nemo_response(
        response, task.to_dict(), IMAGE, client=client
    )
    assert score["correct"] and score["reward_components"] == {"verified_tests": 1.0}
    modified = task.to_dict()
    if tamper == "tests":
        modified["files"] = {**modified["files"], "tests/test_solution.py": "# no tests"}
    else:
        modified["metadata"] = {**modified["metadata"], "content_sha256": "0" * 64}
    with pytest.raises(ValueError, match="modified"):
        environments.score_personal_coding_nemo_response(response, modified, IMAGE, client=client)
