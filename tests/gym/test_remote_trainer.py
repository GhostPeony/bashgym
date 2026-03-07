"""Tests for RemoteTrainer SSH execution."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig, PreflightResult


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
            host="192.168.1.100",
            username="ponyo",
            port=22,
            key_path="~/.ssh/id_rsa",
            remote_work_dir="~/bashgym-training",
        )
        return RemoteTrainer(config)

    def test_preflight_success(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(side_effect=[
            MagicMock(stdout="Python 3.12.0\n", exit_status=0),
            MagicMock(stdout="", exit_status=0),
            MagicMock(stdout="50G\n", exit_status=0),
        ])

        with patch.object(trainer, '_connect', return_value=mock_conn):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is True
            assert result.python_version == "Python 3.12.0"

    def test_preflight_no_unsloth(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(side_effect=[
            MagicMock(stdout="Python 3.12.0\n", exit_status=0),
            MagicMock(stdout="ModuleNotFoundError", exit_status=1),
        ])

        with patch.object(trainer, '_connect', return_value=mock_conn):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is False
            assert "unsloth" in result.error.lower()

    def test_preflight_connection_failed(self, trainer):
        with patch.object(trainer, '_connect', side_effect=OSError("Connection refused")):
            result = asyncio.run(trainer.preflight_check())
            assert result.ok is False
            assert "connect" in result.error.lower()


class TestUploadAndExecute:
    @pytest.fixture
    def trainer(self):
        config = SSHConfig(
            host="192.168.1.100",
            username="ponyo",
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

        with patch.object(trainer, '_connect', return_value=mock_conn):
            asyncio.run(trainer._upload_files(
                mock_conn,
                run_id="run_123",
                script_path=Path("/tmp/train.py"),
                dataset_path=Path("/tmp/train.jsonl"),
            ))
            mock_sftp.makedirs.assert_called_once()
            assert mock_sftp.put.call_count == 2

    def test_execute_returns_remote_pid(self, trainer):
        mock_conn = AsyncMock()
        mock_conn.run = AsyncMock(return_value=MagicMock(
            stdout="12345\n", exit_status=0,
        ))

        with patch.object(trainer, '_connect', return_value=mock_conn):
            pid = asyncio.run(trainer._start_remote_training(
                mock_conn, "run_123",
            ))
            assert pid == 12345

    def test_stream_logs_calls_callback(self, trainer):
        mock_conn = AsyncMock()
        log_lines = []

        async def mock_run(cmd, check=False):
            if "tail" in cmd:
                return MagicMock(stdout="epoch 1 loss 0.5\nepoch 2 loss 0.3\n", exit_status=0)
            elif "kill -0" in cmd:
                return MagicMock(exit_status=1)
            return MagicMock(stdout="", exit_status=0)

        mock_conn.run = mock_run

        with patch.object(trainer, '_connect', return_value=mock_conn):
            asyncio.run(trainer._stream_logs(
                mock_conn, "run_123", 12345,
                log_callback=lambda line: log_lines.append(line),
            ))
            assert len(log_lines) >= 2
