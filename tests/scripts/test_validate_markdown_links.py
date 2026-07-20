"""Tests for the offline repository Markdown link validator."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_markdown_links.py"
_SPEC = importlib.util.spec_from_file_location("validate_markdown_links", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
validate_markdown_links = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_markdown_links)


def test_validate_markdown_links_accepts_local_files_and_anchors(tmp_path):
    guide = tmp_path / "guide.md"
    guide.write_text("# Getting Started\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "[Guide](guide.md#getting-started)\n[Section](#overview)\n\n# Overview\n", encoding="utf-8"
    )

    assert validate_markdown_links.validate_markdown_links([readme]) == []


def test_validate_markdown_links_reports_missing_local_target(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("[Missing](missing.md)\n", encoding="utf-8")

    issues = validate_markdown_links.validate_markdown_links([readme])

    assert issues == [f"{readme}:1: missing local target: missing.md"]
