from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from bashgym.models.deployment import deploy_gguf_to_ollama


def test_deploy_gguf_to_ollama_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.gguf"

    result = deploy_gguf_to_ollama(str(missing), "missing-model")

    assert result == {"success": False, "error": f"GGUF file not found: {missing}"}


def test_deploy_gguf_to_ollama_invokes_ollama_create(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    invocation = {}

    def fake_run(command, **kwargs):
        invocation["command"] = command
        invocation["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("bashgym.models.deployment.subprocess.run", fake_run)

    result = deploy_gguf_to_ollama(str(gguf), "bashgym-test")

    assert result == {"success": True, "model_name": "bashgym-test"}
    assert invocation["command"][:3] == ["ollama", "create", "bashgym-test"]
    assert invocation["command"][3] == "-f"
    assert invocation["kwargs"] == {"capture_output": True, "text": True, "timeout": 300}


def test_deploy_gguf_to_ollama_cleans_modelfile_when_ollama_is_missing(tmp_path, monkeypatch):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    invocation = {}

    def missing_ollama(command, **_kwargs):
        invocation["modelfile"] = command[4]
        raise FileNotFoundError

    monkeypatch.setattr("bashgym.models.deployment.subprocess.run", missing_ollama)

    result = deploy_gguf_to_ollama(str(gguf), "bashgym-test")

    assert result == {"success": False, "error": "Ollama not installed"}
    assert not Path(invocation["modelfile"]).exists()
