import pytest
from bashgym.config import OllamaSettings, Settings


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
