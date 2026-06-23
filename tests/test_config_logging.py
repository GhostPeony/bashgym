import logging

from bashgym.config import LoggingSettings, Settings


def test_json_log_format_preset_resolves_to_percent_style_format():
    settings = LoggingSettings(log_format="json")

    resolved = settings.resolved_log_format()

    assert resolved.startswith('{"timestamp":"%(asctime)s"')
    logging.Formatter(resolved)


def test_setup_logging_accepts_json_log_format(monkeypatch):
    captured = {}

    def fake_basic_config(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    settings = Settings(logging=LoggingSettings(log_format="json"))

    settings._setup_logging()

    assert captured["format"].startswith('{"timestamp":"%(asctime)s"')
    assert captured["level"] == logging.INFO
