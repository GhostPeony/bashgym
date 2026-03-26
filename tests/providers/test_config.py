import pytest
from bashgym.config import OllamaSettings, Settings, SSHSettings


class TestOllamaSettings:
    def test_defaults(self):
        settings = OllamaSettings()
        assert settings.enabled is True
        assert settings.base_url == "http://localhost:11434"
        assert settings.default_model == ""
        assert settings.auto_register is True
        assert settings.health_interval == 30
        assert settings.request_timeout == 120
        assert settings.prefer_code_models is True

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://dgx:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

        settings = OllamaSettings()
        assert settings.enabled is False
        assert settings.base_url == "http://dgx:11434"
        assert settings.default_model == "qwen2.5-coder:7b"


class TestSettingsIncludesOllama:
    def test_settings_has_ollama(self):
        settings = Settings()
        assert hasattr(settings, "ollama")
        assert isinstance(settings.ollama, OllamaSettings)


class TestSSHSettings:
    def test_defaults(self, monkeypatch):
        # Clear SSH env vars so we test actual defaults
        monkeypatch.delenv("SSH_REMOTE_ENABLED", raising=False)
        monkeypatch.delenv("SSH_REMOTE_HOST", raising=False)
        monkeypatch.delenv("SSH_REMOTE_PORT", raising=False)
        monkeypatch.delenv("SSH_REMOTE_USER", raising=False)
        monkeypatch.delenv("SSH_REMOTE_KEY_PATH", raising=False)
        monkeypatch.delenv("SSH_REMOTE_WORK_DIR", raising=False)
        s = SSHSettings()
        assert s.enabled is False
        assert s.host == ""
        assert s.port == 22
        assert s.username == ""
        assert s.remote_work_dir == "~/bashgym-training"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("SSH_REMOTE_ENABLED", "true")
        monkeypatch.setenv("SSH_REMOTE_HOST", "192.168.1.100")
        monkeypatch.setenv("SSH_REMOTE_USER", "ponyo")
        monkeypatch.setenv("SSH_REMOTE_PORT", "2222")
        monkeypatch.setenv("SSH_REMOTE_KEY_PATH", "~/.ssh/id_ed25519")
        monkeypatch.setenv("SSH_REMOTE_WORK_DIR", "~/training")
        s = SSHSettings()
        assert s.enabled is True
        assert s.host == "192.168.1.100"
        assert s.username == "ponyo"
        assert s.port == 2222
        assert s.key_path == "~/.ssh/id_ed25519"
        assert s.remote_work_dir == "~/training"

    def test_settings_has_ssh(self):
        s = Settings()
        assert hasattr(s, 'ssh')
