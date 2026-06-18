"""Implementation of the BashGym gold-trace seed reader plugin."""

from __future__ import annotations

import json
from typing import Any

from data_designer.engine.resources.seed_reader import (
    FileSystemSeedReader,
    SeedReaderFileSystemContext,
)

from bashgym.factory.dd_plugin.config import GoldTraceSeedSource

_EXT_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".sh": "bash",
    ".rb": "ruby",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
}


def _detect_language(trace_steps: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for step in trace_steps:
        command = step.get("command", "") or ""
        for ext, lang in _EXT_TO_LANG.items():
            if ext in command:
                counts[lang] = counts.get(lang, 0) + 1
    return max(counts, key=counts.get) if counts else "python"


class GoldTraceSeedReader(FileSystemSeedReader[GoldTraceSeedSource]):
    """Reads BashGym gold-trace JSON files into seed rows.

    ``build_manifest`` lists matching files; ``hydrate_row`` parses each trace and
    emits the seed fields (skipping traces with no user prompt).
    """

    output_columns = [
        "seed_task",
        "seed_tools",
        "seed_complexity",
        "seed_language",
        "seed_step_count",
        "source_path",
    ]

    def build_manifest(self, *, context: SeedReaderFileSystemContext) -> list[dict[str, Any]]:
        matched = self.get_matching_relative_paths(
            context=context,
            file_pattern=self.source.file_pattern,
            recursive=self.source.recursive,
        )
        return [{"relative_path": p} for p in matched]

    def hydrate_row(
        self, *, manifest_row: dict[str, Any], context: SeedReaderFileSystemContext
    ) -> list[dict[str, Any]]:
        relative_path = manifest_row["relative_path"]
        abs_path = context.root_path / relative_path
        try:
            data = json.loads(abs_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []

        prompt = (data.get("metadata") or {}).get("user_initial_prompt", "")
        if not prompt:
            return []

        steps = data.get("trace", []) or []
        tools = sorted({s.get("tool_name", "unknown") for s in steps})
        step_count = len(steps)
        complexity = "simple" if step_count <= 5 else "moderate" if step_count <= 15 else "complex"

        return [
            {
                "seed_task": prompt,
                "seed_tools": ", ".join(tools),
                "seed_complexity": complexity,
                "seed_language": _detect_language(steps),
                "seed_step_count": step_count,
                "source_path": str(abs_path),
            }
        ]
