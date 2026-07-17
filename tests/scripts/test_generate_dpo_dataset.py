import importlib.util
import logging
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_dpo_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_dpo_dataset_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module():
    return _load_script_module()


def test_dpo_dataset_loader_has_no_private_checkout_env_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert not re.search(
        r'Path\.home\(\)\s*/\s*"desktop-home"\s*/\s*"Projects"\s*/\s*"ghostwork"\s*/\s*"\.env"',
        source,
    )
    assert "desktop-home" not in source
    assert 'Path.home() / "bashgym"' not in source
    assert 'Path.home() / ".bashgym" / ".env"' in source
    assert 'default=Path.home() / ".bashgym" / "gold_traces"' in source
    assert 'default=Path.home() / ".bashgym" / "data" / "dpo_synthetic"' in source


def test_key_loader_prefers_explicit_environment_over_bashgym_env(
    script_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bashgym_dir = tmp_path / ".bashgym"
    bashgym_dir.mkdir()
    (bashgym_dir / ".env").write_text("NVIDIA_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setattr(script_module.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "environment-secret")

    assert script_module._load_nvidia_key() == "environment-secret"


def test_key_loader_reads_platform_neutral_bashgym_env(
    script_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bashgym_dir = tmp_path / ".bashgym"
    bashgym_dir.mkdir()
    (bashgym_dir / ".env").write_text("ANTHROPIC_API_KEY=file-secret\n", encoding="utf-8")
    monkeypatch.setattr(script_module.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert script_module._load_anthropic_key() == "file-secret"


def test_main_logs_resolution_without_credential_values(
    script_module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    nvidia_secret = "7F3KQ9PX2MVD-nv-canary-secret"
    anthropic_secret = "4Z8MTRD6YQWBCA-an-canary-secret"
    monkeypatch.setenv("NVIDIA_API_KEY", nvidia_secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_secret)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--gold-traces",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with caplog.at_level(logging.INFO, logger=script_module.logger.name):
        with pytest.raises(SystemExit):
            script_module.main()

    emitted_messages = "\n".join(caplog.messages)
    assert "NVIDIA_API_KEY resolved" in emitted_messages
    assert "ANTHROPIC_API_KEY resolved" in emitted_messages

    secret_fragments = {
        nvidia_secret[:8],
        nvidia_secret[:12],  # Former NVIDIA prefix slice.
        nvidia_secret[:20],
        anthropic_secret[:8],
        anthropic_secret[:14],  # Former Anthropic prefix slice.
        anthropic_secret[:22],
        nvidia_secret,
        anthropic_secret,
    }
    for fragment in secret_fragments:
        assert fragment not in emitted_messages
