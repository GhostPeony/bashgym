"""Keep public BashGym explanations neutral and experiment-first."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _tracked_public_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[Path] = []
    for relative in result.stdout.splitlines():
        path = ROOT / relative
        if not path.is_file():
            continue
        if relative in {"README.md", "AGENTS.md", "CLAUDE.md"}:
            paths.append(path)
            continue
        if relative.startswith("docs/") and path.suffix.lower() == ".md":
            paths.append(path)
            continue
        if relative.startswith("frontend/src/") and path.suffix.lower() in {
            ".css",
            ".html",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
        }:
            paths.append(path)
    return paths


def test_public_surfaces_do_not_encode_private_deployments_or_empty_positioning():
    forbidden = {
        "private hostname": "pon" + "yo",
        "device campaign label": "gx" + "10",
        "processor campaign label": "gb" + "10",
        "personal filesystem": "c:" + "\\users\\cade",
        "infrastructure-first phrase": "control" + " plane",
        "status-first phrase": "planned" + " not launched",
        "empty thesis label": "product" + " thesis",
        "rejected slogan": "the agent moves the science",
    }

    violations: list[str] = []
    for path in _tracked_public_text_files():
        text = path.read_text(encoding="utf-8").lower()
        for label, phrase in forbidden.items():
            if phrase in text:
                violations.append(f"{path.relative_to(ROOT)}: {label}")

    assert not violations, "public narrative boundary violations:\n" + "\n".join(violations)
