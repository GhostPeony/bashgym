"""Tests for RemoteTrainer SSH execution."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bashgym.gym.remote_trainer import PreflightResult, RemoteTrainer, SSHConfig
from bashgym.gym.trainer import Trainer, TrainerConfig, TrainingRun, TrainingStrategy


class TestSSHConfig:
    def test_from_settings(self):
        from bashgym.config import SSHSettings

        settings = SSHSettings()
        config = SSHConfig.from_settings(settings)
        assert config.port == 22
        assert config.remote_work_dir == "~/bashgym-training"


class TestPreflight:
    @pytest.fixture
    def trainer(self):
        config = SSHConfig(
            host="192.0.2.10",
            username="remote-user",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
        return RemoteTrainer(config)

    def test_preflight_success(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            side_effect=[
                MagicMock(stdout="Python 3.12.0\n", exit_status=0),
                MagicMock(stdout="", exit_status=0),
                MagicMock(stdout="50G\n", exit_status=0),
            ]
        )

        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is True
            assert result.python_version == "Python 3.12.0"

    def test_preflight_no_unsloth(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            side_effect=[
                MagicMock(stdout="Python 3.12.0\n", exit_status=0),
                MagicMock(stdout="ModuleNotFoundError", exit_status=1),
            ]
        )

        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is False
            assert "unsloth" in result.error.lower()

    def test_preflight_connection_failed(self, trainer):
        with patch.object(trainer, "_connect", side_effect=OSError("Connection refused")):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is False
            assert "connect" in result.error.lower()


class TestUploadAndExecute:
    @pytest.fixture
    def trainer(self):
        config = SSHConfig(
            host="192.0.2.10",
            username="remote-user",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
        return RemoteTrainer(config)

    def test_remote_run_dir(self, trainer):
        run_dir = trainer._remote_run_dir("run_123")
        assert "bashgym-training" in str(run_dir)
        assert "run_123" in str(run_dir)

    def test_upload_files(self, trainer):
        mock_conn = AsyncMock()
        mock_sftp = AsyncMock()
        mock_conn.start_sftp_client = AsyncMock(return_value=mock_sftp)
        mock_sftp.makedirs = AsyncMock()
        mock_sftp.put = AsyncMock()

        with patch.object(trainer, "_connect", return_value=mock_conn):
            asyncio.run(
                trainer._upload_files(
                    mock_conn,
                    run_id="run_123",
                    script_path=Path("/tmp/train.py"),
                    dataset_path=Path("/tmp/train.jsonl"),
                )
            )
            mock_sftp.makedirs.assert_called_once()
            assert mock_sftp.put.call_count == 2

    def test_execute_returns_remote_pid(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(
            return_value=MagicMock(
                stdout="12345\n",
                exit_status=0,
            )
        )

        with patch.object(trainer, "_connect", return_value=mock_conn):
            pid = asyncio.run(
                trainer._start_remote_training(
                    mock_conn,
                    "run_123",
                )
            )
            assert pid == 12345

    def test_wait_for_remote_training_streams_bounded_progress_without_downloading_logs(
        self, trainer
    ):
        mock_conn = AsyncMock()
        commands = []
        observed = []

        async def mock_run(cmd, check=False):
            commands.append(cmd)
            if "remote_log_chunk" in cmd:
                return MagicMock(
                    stdout='{"data":"ZXBvY2ggMSBsb3NzIDAuNQo=","offset":17}',
                    exit_status=0,
                )
            if "kill -0" in cmd:
                return MagicMock(exit_status=1)
            if "test -f" in cmd:
                return MagicMock(stdout="", exit_status=0)
            return MagicMock(stdout='{"data":"","offset":17}', exit_status=0)

        mock_conn.run = mock_run

        with patch.object(trainer, "_connect", return_value=mock_conn):
            asyncio.run(
                trainer._stream_logs(
                    mock_conn,
                    "run_123",
                    12345,
                    log_callback=observed.append,
                )
            )
            assert observed == ["epoch 1 loss 0.5"]
            assert all("scp" not in command and "sftp" not in command for command in commands)
            assert any("65536" in command for command in commands)


class TestTrainRemote:
    @pytest.fixture
    def trainer(self):
        config = SSHConfig(
            host="192.0.2.10",
            username="remote-user",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
        return RemoteTrainer(config)

    def _wire_mocks(self, trainer, calls, exit_code="0"):
        """Wire stage mocks onto the trainer, recording call order in `calls`."""

        async def mock_preflight(require_unsloth=True):
            calls.append("preflight")
            return PreflightResult(ok=True, python_version="3.12")

        async def mock_upload(conn, run_id, script_path, dataset_path, work_dir=None):
            calls.append("upload")

        async def mock_start(conn, run_id, script_name="train_sft.py", work_dir=None):
            calls.append("start")
            return 99999

        async def mock_stream(
            conn, run_id, pid, log_callback=None, poll_interval=None, work_dir=None
        ):
            calls.append("stream")

        async def mock_download(conn, run_id, local_dir, work_dir=None):
            calls.append("download")

        trainer.preflight_check = mock_preflight
        trainer._upload_files = mock_upload
        trainer._start_remote_training = mock_start
        trainer._stream_logs = mock_stream
        trainer._download_artifacts = mock_download

        conn = AsyncMock()

        async def run(command, check=False):
            if "/exit_code" in command:
                return MagicMock(stdout=f"{exit_code}\n", exit_status=0)
            if "artifact_manifest" in command:
                return MagicMock(stdout="final\nmerged\n", exit_status=0)
            return MagicMock(stdout="", exit_status=0)

        conn.run = AsyncMock(side_effect=run)
        trainer._connect = AsyncMock(return_value=conn)

    def test_train_remote_keeps_model_artifacts_on_the_compute_target(self, trainer, tmp_path):
        """A private SSH run returns an opaque reference instead of downloading weights."""
        calls = []
        self._wire_mocks(trainer, calls, exit_code="0")

        script = tmp_path / "train_sft.py"
        script.write_text("print('hello')")
        dataset = tmp_path / "train.jsonl"
        dataset.write_text("{}")

        result = asyncio.run(
            trainer.train_remote(
                run_id="run_test",
                script_path=script,
                dataset_path=dataset,
                local_output_dir=tmp_path / "output",
            )
        )

        assert result["success"] is True
        assert result["remote_pid"] == 99999
        assert result["remote_run_ref"] == "ssh-run://run_test"
        assert result["artifact_refs"] == [
            "ssh-run://run_test/final",
            "ssh-run://run_test/merged",
        ]
        assert calls == ["preflight", "upload", "start", "stream"]
        assert not (tmp_path / "output" / "final").exists()
        assert not (tmp_path / "output" / "merged").exists()

    def test_train_remote_rejects_success_without_a_model_artifact(self, trainer, tmp_path):
        calls = []
        self._wire_mocks(trainer, calls, exit_code="0")
        conn = trainer._connect.return_value

        async def run(command, check=False):
            if "/exit_code" in command:
                return MagicMock(stdout="0\n", exit_status=0)
            if "artifact_manifest" in command:
                return MagicMock(stdout="", exit_status=0)
            return MagicMock(stdout="", exit_status=0)

        conn.run = AsyncMock(side_effect=run)
        script = tmp_path / "train_sft.py"
        dataset = tmp_path / "train.jsonl"
        script.write_text("print('hello')")
        dataset.write_text("{}")

        result = asyncio.run(
            trainer.train_remote(
                run_id="run_missing_model",
                script_path=script,
                dataset_path=dataset,
                local_output_dir=tmp_path / "output",
            )
        )

        assert result["success"] is False
        assert result["error"] == "Remote training produced no model artifact."

    def test_train_remote_fails_on_nonzero_exit_code(self, trainer, tmp_path):
        """A remote script that crashed must be reported as a failure, not success."""
        calls = []
        self._wire_mocks(trainer, calls, exit_code="1")

        script = tmp_path / "train_sft.py"
        script.write_text("print('hello')")
        dataset = tmp_path / "train.jsonl"
        dataset.write_text("{}")

        result = asyncio.run(
            trainer.train_remote(
                run_id="run_crash",
                script_path=script,
                dataset_path=dataset,
                local_output_dir=tmp_path / "output",
            )
        )

        assert result["success"] is False
        assert "exited with code 1" in result["error"]
        assert "Last log lines" not in result["error"]
        assert result["log_ref"] == "ssh-run://run_crash/training.log"
        assert "download" not in calls

    def test_resolve_work_dir_expands_tilde(self, trainer):
        """SFTP treats ~ literally, so the work dir must expand to $HOME."""
        conn = AsyncMock()
        conn.run = AsyncMock(return_value=MagicMock(stdout="/home/remote-user"))

        resolved = asyncio.run(trainer._resolve_work_dir(conn))

        assert resolved == "/home/remote-user/bashgym-training"

    def test_resolve_work_dir_keeps_absolute_paths(self, trainer):
        trainer.config.remote_work_dir = "/data/training"
        conn = AsyncMock()

        resolved = asyncio.run(trainer._resolve_work_dir(conn))

        assert resolved == "/data/training"
        conn.run.assert_not_called()

    def test_train_remote_fails_on_preflight(self, trainer, tmp_path):
        async def mock_preflight(require_unsloth=True):
            return PreflightResult(ok=False, error="no unsloth")

        trainer.preflight_check = mock_preflight

        result = asyncio.run(
            trainer.train_remote(
                run_id="run_fail",
                script_path=tmp_path / "x.py",
                dataset_path=tmp_path / "x.jsonl",
                local_output_dir=tmp_path / "output",
            )
        )

        assert result["success"] is False
        assert "unsloth" in result["error"]

    def test_train_remote_forwards_require_unsloth_to_preflight(self, trainer, tmp_path):
        seen = {}

        async def mock_preflight(require_unsloth=True):
            seen["require_unsloth"] = require_unsloth
            return PreflightResult(ok=False, error="stop after preflight")

        trainer.preflight_check = mock_preflight

        asyncio.run(
            trainer.train_remote(
                run_id="run_flag",
                script_path=tmp_path / "x.py",
                dataset_path=tmp_path / "x.jsonl",
                local_output_dir=tmp_path / "output",
                require_unsloth=False,
            )
        )

        assert seen["require_unsloth"] is False


def test_trainer_records_only_opaque_remote_artifact_references(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    run = TrainingRun(
        run_id="run_remote_resident",
        strategy=TrainingStrategy.SFT,
        base_model="registered-base-model",
        dataset_path=dataset,
        output_path=tmp_path / "controller-run",
    )
    trainer = Trainer(
        TrainerConfig(
            base_model="registered-base-model",
            output_dir=str(tmp_path / "controller-runs"),
        )
    )
    trainer.ssh_config = SSHConfig(host="research-host", username="research-user")

    class FakeRemoteTrainer:
        def __init__(self, _config):
            pass

        async def train_remote(self, **_kwargs):
            return {
                "success": True,
                "remote_pid": 123,
                "run_id": run.run_id,
                "remote_run_ref": f"ssh-run://{run.run_id}",
                "artifact_refs": [
                    f"ssh-run://{run.run_id}/final",
                    f"ssh-run://{run.run_id}/merged",
                ],
            }

    with patch("bashgym.gym.remote_trainer.RemoteTrainer", FakeRemoteTrainer):
        trainer._train_with_remote_ssh(run, None, None, None)

    assert run.training_metadata["remote_execution"] == {
        "schema_version": "bashgym.remote_training_reference.v1",
        "run_ref": "ssh-run://run_remote_resident",
        "artifact_refs": [
            "ssh-run://run_remote_resident/final",
            "ssh-run://run_remote_resident/merged",
        ],
    }
    assert "research-host" not in str(run.training_metadata)
    assert not (run.output_path / "final").exists()
    assert not (run.output_path / "merged").exists()


class TestRemoteProcessControl:
    @pytest.fixture
    def trainer(self):
        config = SSHConfig(
            host="192.0.2.10",
            username="remote-user",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
        return RemoteTrainer(config)

    def test_pause_sends_sigstop(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=MagicMock(exit_status=0))
        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.pause_remote(12345))
            assert result is True
            mock_conn.run.assert_called_once_with("kill -STOP 12345", check=False)

    def test_resume_sends_sigcont(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=MagicMock(exit_status=0))
        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.resume_remote(12345))
            assert result is True
            mock_conn.run.assert_called_once_with("kill -CONT 12345", check=False)

    def test_cancel_sends_sigterm(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=MagicMock(exit_status=0))
        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.cancel_remote(12345))
            assert result is True
            mock_conn.run.assert_called_once_with("kill -TERM 12345", check=False)

    def test_pause_returns_false_on_failure(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=MagicMock(exit_status=1))
        with patch.object(trainer, "_connect", return_value=mock_conn):
            result = asyncio.run(trainer.pause_remote(12345))
            assert result is False

    def test_cancel_returns_false_on_connection_error(self, trainer):
        with patch.object(trainer, "_connect", side_effect=OSError("Connection refused")):
            result = asyncio.run(trainer.cancel_remote(12345))
            assert result is False
