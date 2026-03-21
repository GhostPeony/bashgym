"""SkillRegistry — loads JSON skill manifests and matches user messages by keyword overlap."""

from __future__ import annotations

import json
import re
from pathlib import Path


class SkillRegistry:
    """Loads JSON skill manifests and matches user messages to relevant skills.

    Manifests are JSON files containing at minimum "name" and "trigger_keywords"
    keys. The registry recursively discovers all such files under a base directory,
    then provides keyword-overlap matching against user messages.
    """

    def __init__(self, skills_base_dir: Path | None = None) -> None:
        if skills_base_dir is None:
            skills_base_dir = Path(__file__).parent
        self.skills_base_dir = skills_base_dir
        self.skills: list[dict] = []
        self._load_manifests()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_manifests(self) -> None:
        """Recursively load all *.json files that have 'name' and 'trigger_keywords' keys."""
        for json_path in self.skills_base_dir.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(data, dict) and "name" in data and "trigger_keywords" in data:
                self.skills.append(data)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase alphanumeric words (including hyphens)."""
        return re.findall(r"[a-z0-9-]+", text.lower())

    def match(self, user_message: str, top_n: int = 3) -> list[dict]:
        """Match user message to skills by keyword overlap.

        Returns the top *top_n* skills whose trigger_keywords have at least one
        token in common with the message, sorted by overlap count descending.
        """
        tokens = set(self._tokenize(user_message))
        scored: list[tuple[int, dict]] = []
        for skill in self.skills:
            keywords = set(skill.get("trigger_keywords", []))
            overlap = len(tokens & keywords)
            if overlap > 0:
                scored.append((overlap, skill))
        # Sort descending by overlap count
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [skill for _, skill in scored[:top_n]]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_tools(self, matches: list[dict]) -> list[dict]:
        """Extract and deduplicate tool definitions from matched skills."""
        seen_names: set[str] = set()
        tools: list[dict] = []
        for skill in matches:
            for tool in skill.get("tools", []):
                name = tool.get("name", "")
                if name not in seen_names:
                    seen_names.add(name)
                    tools.append(tool)
        return tools

    def get_knowledge(self, matches: list[dict]) -> str:
        """Join knowledge sections from matched skills with headers."""
        sections: list[str] = []
        for skill in matches:
            knowledge = skill.get("knowledge", "")
            if knowledge:
                sections.append(f"## {skill['name']}\n\n{knowledge}")
        return "\n\n".join(sections)

    def list_all(self) -> list[dict]:
        """Return a summary list of all loaded skills."""
        return [
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "tools": [t["name"] for t in skill.get("tools", [])],
            }
            for skill in self.skills
        ]
