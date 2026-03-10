"""Integration test for remote SSH training — requires actual SSH access."""

import os
import pytest
import asyncio
from pathlib import Path

from bashgym.gym.remote_trainer import RemoteTrainer, SSHConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("SSH_REMOTE_HOST"),
    reason="SSH_REMOTE_HOST not set — skipping SSH integration tests",
)


class TestRemoteIntegration:
    @pytest.fixture
    def trainer(self):
        return RemoteTrainer(SSHConfig(
            host=os.environ["SSH_REMOTE_HOST"],
            username=os.environ.get("SSH_REMOTE_USER", "ponyo"),
            port=int(os.environ.get("SSH_REMOTE_PORT", "22")),
            key_path=os.environ.get("SSH_REMOTE_KEY_PATH", "~/.ssh/id_rsa"),
            remote_work_dir=os.environ.get("SSH_REMOTE_WORK_DIR", "~/bashgym-training"),
        ))

    def test_preflight_passes(self, trainer):
        result = asyncio.run(trainer.preflight_check())
        assert result.ok is True
        assert "python" in result.python_version.lower()
