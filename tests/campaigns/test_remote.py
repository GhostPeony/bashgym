"""Typed SSH launch, reconciliation, output, and signal tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bashgym.campaigns.contracts import utc_now
from bashgym.campaigns.remote import (
    CodeLineageLaunchSnapshot,
    RemoteCapacityPolicy,
    RemoteCommandResult,
    RemoteLaunchRequest,
    RemoteObservation,
    RemoteRunIdentity,
    RemoteRunState,
    RemoteStreamCursor,
    RemoteTrainingAdapter,
)
from bashgym.gym.remote_trainer import SSHConfig


class MockSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []
        self.uploads = []
        self.downloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def run(self, command, *, timeout=None):
        self.commands.append((command, timeout))
        if not self.responses:
            raise AssertionError(f"unexpected remote command: {command}")
        return self.responses.pop(0)

    async def upload(self, local_path, remote_path):
        self.uploads.append((Path(local_path), remote_path))

    async def download(self, remote_path, local_path):
        local = Path(local_path)
        self.downloads.append((remote_path, local))
        if remote_path.endswith("/final"):
            local.mkdir(parents=True, exist_ok=True)
            (local / "config.json").write_text("{}", encoding="utf-8")
        else:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text("fixture", encoding="utf-8")
        return True


def result(stdout="", *, status=0, stderr=""):
    return RemoteCommandResult(stdout=stdout, stderr=stderr, exit_status=status)


def config():
    return SSHConfig(
        host="192.0.2.10",
        username="trainer",
        key_path="~/.ssh/id_ed25519",
        remote_work_dir="~/bashgym-training",
    )


def launch_request(tmp_path):
    script = tmp_path / "train.py"
    dataset = tmp_path / "train.jsonl"
    script.write_text("print('fixture')\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    return RemoteLaunchRequest(
        compute_profile_id="ssh-gpu-lab",
        run_id="campaign-action-1",
        script_path=script,
        input_files=(dataset,),
        script_args=("--grouped-jsonl", "train.jsonl", "--output-dir", "."),
        recipe_digest="e" * 64,
        output_paths=("final", "training_manifest.json", "training_metrics.jsonl"),
    )


def lineage_launch_request(tmp_path):
    request = launch_request(tmp_path)
    archive = (tmp_path / ("d" * 64)).with_suffix(".tar")
    archive.write_bytes(b"deterministic source snapshot")
    snapshot = CodeLineageLaunchSnapshot(
        binding_id="bashgym-trainer-entrypoint-v1",
        binding_revision=1,
        binding_digest="a" * 64,
        source_repository_profile_id="bashgym-source-v1",
        lineage_id="lineage-candidate-1",
        record_digest="b" * 64,
        commit_sha="c" * 40,
        patch_sha256="d" * 64,
        entrypoint_path="bashgym/gym/trainer.py",
        working_directory="source",
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        archive_size_bytes=archive.stat().st_size,
    )
    return request.model_copy(update={"source_snapshot": snapshot})


def identity():
    return RemoteRunIdentity(
        compute_profile_id="ssh-gpu-lab",
        run_id="campaign-action-1",
        remote_run_directory="/home/trainer/bashgym-training/campaign-action-1",
        remote_pid=4242,
        process_group_id=4242,
        process_start_ticks=9001,
        boot_id="boot-1",
        command_hash="a" * 64,
        launch_manifest_sha256="b" * 64,
        launched_at=utc_now(),
    )


def test_launch_request_pins_exact_python_executable(tmp_path):
    request = launch_request(tmp_path).model_copy(
        update={"python_executable": "/opt/memexai/.venv/bin/python"}
    )

    assert RemoteTrainingAdapter._argv(request, "/remote/run")[0] == (
        "/opt/memexai/.venv/bin/python"
    )
    with pytest.raises(ValidationError, match="exact executable path"):
        RemoteLaunchRequest(
            **launch_request(tmp_path).model_dump(exclude={"python_executable"}),
            python_executable="python3; touch /tmp/unsafe",
        )


@pytest.mark.asyncio
async def test_launch_executes_verified_captured_source_snapshot(tmp_path):
    request = lineage_launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    remote_directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, remote_directory)
    launched_identity = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )
    session = MockSession(
        [result("/home/trainer"), result(), result(), result(supervisor_json(launched_identity))]
    )
    adapter._session_factory = lambda: session

    recovered = await adapter.launch(request)

    assert recovered == launched_identity
    assert manifest["code_lineage"]["commit_sha"] == "c" * 40
    assert manifest["execution_context"] == {
        "entrypoint_kind": "captured_source_snapshot",
        "working_directory": f"{remote_directory}/source",
        "python_path": f"{remote_directory}/source",
    }
    assert manifest["argv"][1] == "-c"
    assert "sys.path.insert(0,source)" in manifest["argv"][2]
    assert manifest["argv"][3] == f"{remote_directory}/source"
    assert manifest["argv"][4] == f"{remote_directory}/source/bashgym/gym/trainer.py"
    assert [local for local, _remote in session.uploads] == [
        request.source_snapshot.archive_path,
        request.input_files[0],
    ]
    assert "tar --extract" in session.commands[2][0]
    assert "test ! -L source/bashgym/gym/trainer.py" in session.commands[2][0]
    assert f"PYTHONPATH={remote_directory}/source" in session.commands[3][0]


def test_launch_rechecks_snapshot_after_request_construction(tmp_path):
    request = lineage_launch_request(tmp_path)
    request.source_snapshot.archive_path.write_bytes(b"changed after request validation")

    with pytest.raises(ValueError, match="snapshot changed before launch"):
        RemoteTrainingAdapter._launch_manifest(request, "/remote/run")


def supervisor_json(value: RemoteRunIdentity) -> str:
    payload = value.model_dump(mode="json", exclude={"schema_version"})
    payload["schema_version"] = "campaign_remote_supervisor_state.v1"
    return json.dumps(payload)


@pytest.mark.asyncio
async def test_launch_exclusively_creates_verifies_and_returns_server_neutral_identity(tmp_path):
    request = launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    remote_directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, remote_directory)
    launched_identity = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": __import__("hashlib")
            .sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
            .hexdigest(),
        }
    )
    session = MockSession(
        [result("/home/trainer"), result(), result(), result(supervisor_json(launched_identity))]
    )
    adapter._session_factory = lambda: session

    recovered = await adapter.launch(request)

    assert recovered == launched_identity
    assert "host" not in type(recovered).model_fields
    assert "username" not in type(recovered).model_fields
    assert [remote for _local, remote in session.uploads] == [
        f"{remote_directory}/train.py",
        f"{remote_directory}/train.jsonl",
    ]
    assert "mkdir /home/trainer/bashgym-training/campaign-action-1" in session.commands[1][0]
    assert "sha256sum -c" in session.commands[2][0]
    assert "setsid" in session.commands[3][0]
    assert "remote_run_state.v1.json.tmp" in session.commands[3][0]


@pytest.mark.asyncio
async def test_discover_recovers_exact_manifest_without_starting_a_second_process(tmp_path):
    request = launch_request(tmp_path)
    adapter = RemoteTrainingAdapter(config(), compute_profile_id="ssh-gpu-lab")
    directory = "/home/trainer/bashgym-training/campaign-action-1"
    manifest = adapter._launch_manifest(request, directory)
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    expected = identity().model_copy(
        update={
            "command_hash": manifest["command_hash"],
            "launch_manifest_sha256": __import__("hashlib")
            .sha256(manifest_json.encode())
            .hexdigest(),
        }
    )
    session = MockSession([result("/home/trainer"), result(supervisor_json(expected))])
    adapter._session_factory = lambda: session

    recovered = await adapter.discover(request)

    assert recovered == expected
    assert not session.uploads
    assert not any("nohup" in command for command, _timeout in session.commands)


@pytest.mark.asyncio
async def test_discover_returns_none_only_for_absent_state(tmp_path):
    session = MockSession([result("/home/trainer"), result(status=1)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    assert await adapter.discover(launch_request(tmp_path)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "expected_state", "expected_reason"),
    [
        (f"boot-1\t9001\t4242\tS\t{'b' * 64}\t\n", RemoteRunState.RUNNING, "remote_process_alive"),
        (f"boot-1\t9001\t4242\tT\t{'b' * 64}\t\n", RemoteRunState.PAUSED, "remote_process_paused"),
        (f"boot-1\t\t\t\t{'b' * 64}\t0\n", RemoteRunState.COMPLETED, "remote_exit_code_recorded"),
        (f"boot-1\t\t\t\t{'b' * 64}\t7\n", RemoteRunState.FAILED, "remote_exit_code_recorded"),
        (
            f"boot-1\tbad\t4242\tS\t{'b' * 64}\t\n",
            RemoteRunState.UNKNOWN,
            "remote_observation_malformed",
        ),
        (
            f"boot-2\t9001\t4242\tS\t{'b' * 64}\t\n",
            RemoteRunState.UNKNOWN,
            "remote_process_identity_mismatch",
        ),
        (f"boot-1\t9001\t4242\tZ\t{'b' * 64}\t\n", RemoteRunState.UNKNOWN, "remote_exit_unproven"),
    ],
)
async def test_observe_handles_running_paused_exit_malformed_and_zombie_states(
    stdout, expected_state, expected_reason
):
    session = MockSession([result(stdout)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    observation = await adapter.observe(identity())
    assert observation.state == expected_state
    assert observation.safe_reason == expected_reason


@pytest.mark.asyncio
async def test_controls_validate_identity_and_signal_process_group_in_one_command():
    session = MockSession([result(), result(status=42)])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    assert await adapter.terminate(identity()) is True
    assert await adapter.force_stop(identity()) is False
    first = session.commands[0][0]
    assert "expected_start=9001" in first
    assert "expected_pgid=4242" in first
    assert 'kill -TERM -- "-$pgid"' in first
    assert 'kill -KILL -- "-$pgid"' in session.commands[1][0]


@pytest.mark.asyncio
async def test_collect_requires_every_non_symlink_output(tmp_path):
    request = launch_request(tmp_path)
    session = MockSession([result(f"boot-1\t\t\t\t{'b' * 64}\t0\n"), result()])
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    downloaded = await adapter.collect_outputs(identity(), request, tmp_path / "download")
    assert len(downloaded) == len(request.output_paths) + 3
    assert all(path.exists() for path in downloaded)
    assert "find" in session.commands[1][0]


@pytest.mark.asyncio
async def test_collect_terminal_evidence_requires_closed_supervisor_files(tmp_path):
    session = MockSession(
        [result("effective_config.json\ntraining_manifest.json\ntraining_metrics.jsonl\n")]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    remote_identity = identity()
    observation = RemoteObservation(
        identity=remote_identity,
        state=RemoteRunState.FAILED,
        observed_at=utc_now(),
        exit_code=7,
        safe_reason="remote_exit_code_recorded",
    )

    downloaded = await adapter.collect_terminal_evidence(
        remote_identity, tmp_path / "terminal", observation=observation
    )

    assert {path.name for path in downloaded} == {
        "training.log",
        "exit_code",
        "launch_manifest.json",
        "effective_config.json",
        "training_manifest.json",
        "training_metrics.jsonl",
    }
    assert "test -f" in session.commands[0][0]


@pytest.mark.asyncio
async def test_capacity_preflight_blocks_hermes_occupancy_and_low_memory():
    session = MockSession(
        [result("/home/trainer"), result("42.125\t167\t111, llama-server;222, llama-server;\n")]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    snapshot = await adapter.capacity_preflight(RemoteCapacityPolicy())
    assert snapshot.admitted is False
    assert snapshot.available_memory_gib == 42.125
    assert snapshot.blocking_reasons == (
        "available_memory_below_minimum",
        "external_gpu_process_limit_exceeded",
    )
    capacity_command = session.commands[1][0]
    assert 'while [ ! -e "$probe" ]' in capacity_command
    assert 'df -BG --output=avail "$probe"' in capacity_command


def test_supervisor_state_writer_uses_typed_json_instead_of_printf_placeholders(tmp_path):
    request = launch_request(tmp_path)
    remote_directory = f"/home/trainer/bashgym-training/{request.run_id}"
    manifest = RemoteTrainingAdapter._launch_manifest(request, remote_directory)
    session = MockSession(
        [
            result("/home/trainer"),
            result(),
            result(),
            result(
                json.dumps(
                    {
                        "schema_version": "campaign_remote_supervisor_state.v1",
                        "compute_profile_id": request.compute_profile_id,
                        "run_id": request.run_id,
                        "remote_run_directory": remote_directory,
                        "remote_pid": 123,
                        "process_group_id": 123,
                        "process_start_ticks": 456,
                        "boot_id": "boot-proof",
                        "command_hash": manifest["command_hash"],
                        "launch_manifest_sha256": hashlib.sha256(
                            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "launched_at": "2026-07-13T21:00:00Z",
                    }
                )
            ),
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )

    identity = asyncio.run(adapter.launch(request))

    assert identity.remote_pid == 123
    launch_command = session.commands[-1][0]
    assert "python3 -c" in launch_command
    assert "int(sys.argv[4])" in launch_command
    assert "remote_run_state.v1.json.tmp" in launch_command


@pytest.mark.asyncio
async def test_stream_cursor_preserves_partial_lines_across_reads():
    first_bytes = b'{"step":1}\n{"step"'
    second_bytes = b":2}\n"
    session = MockSession(
        [
            result(
                json.dumps(
                    {
                        "end_offset": len(first_bytes),
                        "data": base64.b64encode(first_bytes).decode(),
                    }
                )
            ),
            result(
                json.dumps(
                    {
                        "end_offset": len(first_bytes) + len(second_bytes),
                        "data": base64.b64encode(second_bytes).decode(),
                    }
                )
            ),
        ]
    )
    adapter = RemoteTrainingAdapter(
        config(), compute_profile_id="ssh-gpu-lab", session_factory=lambda: session
    )
    first = await adapter.read_stream(identity(), "training_metrics.jsonl")
    assert first.complete_lines == ('{"step":1}',)
    assert first.next_cursor.partial_line == '{"step"'
    second = await adapter.read_stream(identity(), "training_metrics.jsonl", first.next_cursor)
    assert second.complete_lines == ('{"step":2}',)
    assert second.next_cursor == RemoteStreamCursor(
        byte_offset=len(first_bytes) + len(second_bytes), partial_line=""
    )


def test_launch_contract_rejects_secret_arguments_and_path_escape(tmp_path):
    request = launch_request(tmp_path)
    with pytest.raises(ValidationError, match="credentials"):
        RemoteLaunchRequest(
            **request.model_dump(exclude={"script_args"}),
            script_args=("--api-key=raw-secret",),
        )
    with pytest.raises(ValidationError, match="inside"):
        RemoteLaunchRequest(
            **request.model_dump(exclude={"output_paths"}),
            output_paths=("../escape",),
        )
