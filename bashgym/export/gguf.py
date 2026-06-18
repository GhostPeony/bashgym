"""GGUF export + Ollama Modelfile generation with train-vs-serve template verification.

Replicates Unsloth's ``save_pretrained_gguf`` + a *correct* Ollama Modelfile — the
single most valuable export feature we lost, and the fix for the #1 Gemma 4 deploy
bug: a Modelfile with no ``TEMPLATE`` makes Ollama infer a wrong Go template,
producing double-BOS, leaked thinking-channel text, and broken tool calls.

The serve template is reused from the base model's known-good Ollama Modelfile
(``ollama show <base> --modelfile``) — recommended over hand-writing a Go template
per family — and :func:`check_template_roundtrip` guards train↔serve consistency.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelfileSpec:
    """Inputs for an Ollama Modelfile."""

    from_path: str  # GGUF path or a base model tag
    template: str = ""  # Ollama Go template; empty => Ollama infers one (usually wrong!)
    system: str | None = None
    stop_tokens: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)


def build_modelfile(spec: ModelfileSpec) -> str:
    """Render an Ollama Modelfile string. Emits ``TEMPLATE`` whenever one is provided.

    Emitting the right TEMPLATE (rather than letting Ollama infer it) is the whole
    point — that inference is what breaks Gemma 4 tool calls when omitted.
    """
    lines = [f"FROM {spec.from_path}", ""]
    if spec.template:
        lines.append(f'TEMPLATE """{spec.template}"""')
    if spec.system is not None:
        lines.append(f'SYSTEM """{spec.system}"""')
    for key, val in spec.parameters.items():
        lines.append(f"PARAMETER {key} {val}")
    for tok in spec.stop_tokens:
        lines.append(f'PARAMETER stop "{tok}"')
    return "\n".join(lines).rstrip() + "\n"


def parse_ollama_modelfile(text: str) -> tuple[str, tuple[str, ...]]:
    """Extract ``(template, stop_tokens)`` from ``ollama show <model> --modelfile`` output."""
    template = ""
    m = re.search(r'TEMPLATE\s+"""(.*?)"""', text, re.DOTALL)
    if m:
        template = m.group(1)
    else:
        m = re.search(r"TEMPLATE\s+(.+)", text)
        if m:
            template = m.group(1).strip().strip('"')
    stops = tuple(re.findall(r'PARAMETER\s+stop\s+"([^"]*)"', text))
    return template, stops


def template_from_ollama_base(base_tag: str) -> tuple[str, tuple[str, ...]]:
    """``ollama show <base_tag> --modelfile`` -> ``(template, stops)``. Runtime (needs Ollama)."""
    proc = subprocess.run(
        ["ollama", "show", base_tag, "--modelfile"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ollama show {base_tag} failed: {proc.stderr.strip()}")
    return parse_ollama_modelfile(proc.stdout)


@dataclass
class RoundtripVerdict:
    ok: bool
    issues: list[str] = field(default_factory=list)


def check_template_roundtrip(
    train_render: str,
    serve_render: str,
    *,
    bos_token: str = "<bos>",
    required_tokens: tuple[str, ...] = (),
) -> RoundtripVerdict:
    """Compare the training-template render vs the serve (Ollama) render of the same
    conversation and flag the known Gemma 4 deploy footguns.

    The renders come from the runtime (HF ``tokenizer.apply_chat_template`` for train;
    Ollama for serve); this validates their consistency before a deploy is trusted.
    """
    issues: list[str] = []
    if bos_token:
        t, s = train_render.count(bos_token), serve_render.count(bos_token)
        if s > t:
            issues.append(f"double-BOS: serve has {s}x {bos_token} vs train {t}x")
    for tok in required_tokens:
        if tok in train_render and tok not in serve_render:
            issues.append(f"serve render dropped required token {tok!r}")
    if train_render.strip() != serve_render.strip():
        issues.append("train and serve renders differ")
    return RoundtripVerdict(ok=not issues, issues=issues)


def build_finetuned_modelfile(
    gguf_path: str,
    *,
    profile=None,
    base_ollama_tag: str | None = None,
    system: str | None = None,
    parameters: dict | None = None,
) -> str:
    """Build a Modelfile for a fine-tuned checkpoint, reusing the base family's
    known-good ``TEMPLATE`` + stops (via ``ollama show <base_ollama_tag>``) so the
    served chat/tool-call format matches training.

    When no ``base_ollama_tag`` is given the Modelfile carries the profile's stop
    tokens but no TEMPLATE — callers should treat that as a degraded deploy.
    """
    template = ""
    stops: tuple[str, ...] = tuple(profile.stop_tokens) if profile else ()
    if base_ollama_tag:
        template, base_stops = template_from_ollama_base(base_ollama_tag)
        stops = stops or base_stops
    params = parameters if parameters is not None else {"temperature": 0.7, "num_ctx": 8192}
    spec = ModelfileSpec(
        from_path=gguf_path,
        template=template,
        system=system,
        stop_tokens=stops,
        parameters=params,
    )
    return build_modelfile(spec)


# ── Merged HF checkpoint -> GGUF (llama.cpp) ─────────────────────────────────
#
# GGUF is normally produced inside the trainer via Unsloth's
# ``save_pretrained_gguf``. This standalone path converts an *already-merged* HF
# directory after the fact (e.g. when deploying a checkpoint that was never
# GGUF-exported) by shelling out to llama.cpp — no Unsloth/GPU needed.


def find_gguf_converter() -> str | None:
    """Locate llama.cpp's ``convert_hf_to_gguf.py``.

    Checks ``LLAMA_CPP_DIR`` (a llama.cpp checkout), then ``PATH``. Returns the
    script path or ``None`` when llama.cpp is not installed.
    """
    names = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        for name in names:
            p = Path(env_dir) / name
            if p.exists():
                return str(p)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


# Quantizations the converter emits directly; anything else needs llama-quantize.
_DIRECT_OUTTYPES = {"F16": "f16", "FP16": "f16", "F32": "f32", "FP32": "f32", "BF16": "bf16"}


def convert_merged_to_gguf(
    merged_path: str,
    out_dir: str,
    *,
    quantization: str = "Q4_K_M",
    python_exe: str = "python",
    converter: str | None = None,
    quantize_bin: str = "llama-quantize",
    run: Callable[..., Any] = subprocess.run,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Convert a merged HF model directory to a GGUF file via llama.cpp.

    Two stages: ``convert_hf_to_gguf.py`` produces an f16 GGUF, then (for a
    k-quant like ``Q4_K_M``) ``llama-quantize`` quantizes it. ``run`` and
    ``converter`` are injected so this is unit-testable without llama.cpp.

    Returns ``{"success": True, "path", "quantization"}`` or
    ``{"success": False, "error", "hint"?}``. A missing quantizer degrades to the
    f16 GGUF with a ``note`` rather than failing the whole export.
    """
    merged = Path(merged_path)
    if not merged.exists():
        return {"success": False, "error": f"merged model not found: {merged}"}

    conv = converter or find_gguf_converter()
    if not conv:
        return {
            "success": False,
            "error": "llama.cpp GGUF converter not found",
            "hint": "Set LLAMA_CPP_DIR to a llama.cpp checkout, or put "
            "convert_hf_to_gguf.py on PATH (pip install gguf + clone ggml-org/llama.cpp).",
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quant = (quantization or "F16").upper()
    f16_path = out / "model-f16.gguf"

    r1 = run(
        [python_exe, conv, str(merged), "--outfile", str(f16_path), "--outtype", "f16"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if getattr(r1, "returncode", 1) != 0:
        return {"success": False, "error": f"convert failed: {(r1.stderr or '').strip()[:500]}"}

    if quant in _DIRECT_OUTTYPES or quant in ("", "F16"):
        return {"success": True, "path": str(f16_path), "quantization": "F16"}

    quant_path = out / f"model-{quant}.gguf"
    r2 = run(
        [quantize_bin, str(f16_path), str(quant_path), quant],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if getattr(r2, "returncode", 1) != 0:
        return {
            "success": True,
            "path": str(f16_path),
            "quantization": "F16",
            "note": f"quantize to {quant} failed ({(r2.stderr or '').strip()[:200]}); kept F16",
        }
    return {"success": True, "path": str(quant_path), "quantization": quant}
