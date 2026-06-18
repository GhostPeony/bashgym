"""Tests for the standalone merged-HF -> GGUF converter (llama.cpp shell-out).

Hermetic: ``run`` (subprocess) and ``converter`` (the convert script path) are
injected, so no llama.cpp install is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

from bashgym.export.gguf import convert_merged_to_gguf, find_gguf_converter


def _ok(stdout="", stderr=""):
    return lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)


def _merged(tmp_path):
    d = tmp_path / "merged"
    d.mkdir()
    (d / "config.json").write_text("{}")
    return str(d)


# ── converter discovery ──────────────────────────────────────────────────────


def test_find_converter_from_env(tmp_path, monkeypatch):
    (tmp_path / "convert_hf_to_gguf.py").write_text("# converter")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert find_gguf_converter() == str(tmp_path / "convert_hf_to_gguf.py")


def test_find_converter_missing(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr("bashgym.export.gguf.shutil.which", lambda name: None)
    assert find_gguf_converter() is None


# ── conversion ───────────────────────────────────────────────────────────────


def test_missing_merged_dir(tmp_path):
    out = convert_merged_to_gguf(str(tmp_path / "nope"), str(tmp_path / "out"), converter="c.py")
    assert out["success"] is False
    assert "not found" in out["error"]


def test_no_converter_available(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr("bashgym.export.gguf.shutil.which", lambda name: None)
    out = convert_merged_to_gguf(_merged(tmp_path), str(tmp_path / "out"))
    assert out["success"] is False
    assert "not found" in out["error"] and out["hint"]


def test_f16_skips_quantize(tmp_path):
    calls = []

    def run(argv, **k):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    out = convert_merged_to_gguf(
        _merged(tmp_path), str(tmp_path / "out"), quantization="F16", converter="c.py", run=run
    )
    assert out["success"] is True
    assert out["quantization"] == "F16"
    assert out["path"].endswith("model-f16.gguf")
    assert len(calls) == 1  # convert only, no quantize


def test_quantized_runs_convert_then_quantize(tmp_path):
    calls = []

    def run(argv, **k):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    out = convert_merged_to_gguf(
        _merged(tmp_path),
        str(tmp_path / "out"),
        quantization="Q4_K_M",
        converter="c.py",
        run=run,
    )
    assert out["success"] is True
    assert out["quantization"] == "Q4_K_M"
    assert out["path"].endswith("model-Q4_K_M.gguf")
    assert len(calls) == 2  # convert + quantize
    assert "--outtype" in calls[0]
    assert calls[1][0] == "llama-quantize"


def test_convert_failure(tmp_path):
    def run(argv, **k):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad arch")

    out = convert_merged_to_gguf(
        _merged(tmp_path), str(tmp_path / "out"), converter="c.py", run=run
    )
    assert out["success"] is False
    assert "convert failed" in out["error"]
    assert "bad arch" in out["error"]


def test_quantize_failure_falls_back_to_f16(tmp_path):
    def run(argv, **k):
        # convert ok, quantize fails
        rc = 0 if argv[0] != "llama-quantize" else 1
        return SimpleNamespace(returncode=rc, stdout="", stderr="no quant tool")

    out = convert_merged_to_gguf(
        _merged(tmp_path),
        str(tmp_path / "out"),
        quantization="Q4_K_M",
        converter="c.py",
        run=run,
    )
    assert out["success"] is True
    assert out["quantization"] == "F16"  # degraded
    assert "note" in out and "quantize" in out["note"]
