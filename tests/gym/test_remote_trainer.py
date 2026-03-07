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
