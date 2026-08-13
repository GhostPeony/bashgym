"""Model deployment operations shared by API and training workflows."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from bashgym.export.gguf import ModelfileSpec, build_modelfile, template_from_ollama_base

logger = logging.getLogger(__name__)


def deploy_gguf_to_ollama(
    gguf_path: str,
    model_name: str,
    *,
    base_ollama_tag: str | None = None,
    template: str | None = None,
    stop_tokens: tuple[str, ...] = (),
    system: str | None = "You are a helpful coding assistant trained with Bash Gym.",
) -> dict[str, Any]:
    """Register a GGUF file with Ollama using an explicit Modelfile."""
    gguf_file = Path(gguf_path)
    if not gguf_file.exists():
        return {"success": False, "error": f"GGUF file not found: {gguf_path}"}

    resolved_template = template or ""
    resolved_stops = tuple(stop_tokens)
    if base_ollama_tag and not resolved_template:
        try:
            resolved_template, base_stops = template_from_ollama_base(base_ollama_tag)
            resolved_stops = resolved_stops or base_stops
        except Exception as exc:  # degrade to template-less deploy rather than fail
            logger.warning("Could not reuse base template from %s: %s", base_ollama_tag, exc)

    modelfile_content = build_modelfile(
        ModelfileSpec(
            from_path=gguf_path,
            template=resolved_template,
            system=system,
            stop_tokens=resolved_stops,
            parameters={"temperature": 0.7, "num_ctx": 8192},
        )
    )

    modelfile_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".modelfile", delete=False) as handle:
            handle.write(modelfile_content)
            modelfile_path = handle.name

        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {"success": False, "error": f"Ollama create failed: {result.stderr}"}
        return {"success": True, "model_name": model_name}
    except FileNotFoundError:
        return {"success": False, "error": "Ollama not installed"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Ollama create timed out"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if modelfile_path is not None:
            Path(modelfile_path).unlink(missing_ok=True)
