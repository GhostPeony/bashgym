"""PeonyMemory — persistent memory for the Peony agent.

Four layers:
  - Profile: user identity and preferences
  - Facts: categorized knowledge snippets
  - Episodes: session summaries
  - Prompt builder: assembles all layers into a system prompt section
"""

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FACTS = 50

# Profile fields and their defaults
_PROFILE_DEFAULTS: dict[str, Any] = {
    "hf_username": None,
    "preferred_base_model": None,
    "preferred_strategy": None,
    "projects": [],
    "notes": "",
}


class PeonyMemory:
    """Persistent memory store for the Peony agent.

    Data is stored as JSON files under ``memory_dir``:
      - profile.json   — user profile
      - facts.json     — categorized facts
      - episodes/      — one JSON file per session episode
    """

    def __init__(self, memory_dir: Path | None = None) -> None:
        if memory_dir is None:
            from bashgym.config import get_bashgym_dir

            memory_dir = get_bashgym_dir() / "peony_memory"
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._episodes_dir = self._dir / "episodes"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Profile layer
    # ------------------------------------------------------------------

    @property
    def _profile_path(self) -> Path:
        return self._dir / "profile.json"

    def load_profile(self) -> dict[str, Any]:
        """Load the user profile from disk, or return defaults."""
        if self._profile_path.exists():
            try:
                data = json.loads(self._profile_path.read_text(encoding="utf-8"))
                defaults = copy.deepcopy(_PROFILE_DEFAULTS)
                return {**defaults, **data}
            except (json.JSONDecodeError, OSError):
                pass
        return copy.deepcopy(_PROFILE_DEFAULTS)

    def update_profile(self, field: str, value: Any) -> None:
        """Update a single profile field and persist to disk.

        Raises ``ValueError`` if *field* is not a known profile key.
        """
        if field not in _PROFILE_DEFAULTS:
            raise ValueError(f"Unknown profile field: {field}")
        profile = self.load_profile()
        profile[field] = value
        self._profile_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Facts layer
    # ------------------------------------------------------------------

    @property
    def _facts_path(self) -> Path:
        return self._dir / "facts.json"

    def _load_all_facts(self) -> list[dict[str, Any]]:
        """Load the raw facts list from disk."""
        if self._facts_path.exists():
            try:
                data = json.loads(self._facts_path.read_text(encoding="utf-8"))
                return data.get("facts", [])
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_all_facts(self, facts: list[dict[str, Any]]) -> None:
        """Persist the full facts list to disk, enforcing the cap."""
        # Sort by created_at descending, keep most recent MAX_FACTS
        facts = sorted(facts, key=lambda f: f.get("created_at", ""), reverse=True)
        facts = facts[:MAX_FACTS]
        self._facts_path.write_text(
            json.dumps({"facts": facts}, indent=2, default=str),
            encoding="utf-8",
        )

    def remember_fact(self, category: str, content: str) -> dict[str, Any]:
        """Create and persist a new fact. Returns the fact dict."""
        fact: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        facts = self._load_all_facts()
        facts.append(fact)
        self._save_all_facts(facts)
        return fact

    def recall_facts(
        self,
        category: str | None = None,
        keyword: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter facts by category and/or keyword (case-insensitive)."""
        facts = self._load_all_facts()
        if category is not None:
            facts = [f for f in facts if f["category"] == category]
        if keyword is not None:
            kw_lower = keyword.lower()
            facts = [f for f in facts if kw_lower in f["content"].lower()]
        return facts

    def load_facts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent facts, capped at *limit*."""
        facts = self._load_all_facts()
        # Already sorted descending by _save_all_facts
        return facts[:limit]

    def forget_fact(self, fact_id: str) -> None:
        """Remove a fact by id. Raises ``ValueError`` if not found."""
        facts = self._load_all_facts()
        new_facts = [f for f in facts if f["id"] != fact_id]
        if len(new_facts) == len(facts):
            raise ValueError(f"Fact not found: {fact_id}")
        self._save_all_facts(new_facts)

    # ------------------------------------------------------------------
    # Episodes layer
    # ------------------------------------------------------------------

    def save_episode(self, session_id: str, summary: str) -> dict[str, Any]:
        """Save an episode summary for a session. Returns the episode dict."""
        now = datetime.now(timezone.utc)
        episode: dict[str, Any] = {
            "session_id": session_id,
            "summary": summary,
            "created_at": now.isoformat(),
        }
        date_str = now.strftime("%Y%m%d%H%M%S%f")
        filename = f"{date_str}_{session_id}.json"
        path = self._episodes_dir / filename
        path.write_text(json.dumps(episode, indent=2, default=str), encoding="utf-8")
        return episode

    def load_recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        """Load all episodes, sorted by recency, return top *limit*."""
        episodes: list[dict[str, Any]] = []
        for path in self._episodes_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                episodes.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        episodes.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return episodes[:limit]

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_memory_prompt(self) -> str:
        """Assemble all memory layers into a formatted prompt string."""
        parts: list[str] = []

        # --- Profile ---
        profile = self.load_profile()
        parts.append("--- USER PROFILE ---")
        for key, value in profile.items():
            if value is not None and value != "" and value != []:
                parts.append(f"{key}: {value}")
        parts.append("--- END USER PROFILE ---")

        # --- Facts ---
        facts = self.load_facts()
        parts.append("")
        parts.append(f"--- KNOWN FACTS ({len(facts)} total) ---")
        for fact in facts:
            parts.append(f"- [{fact['category']}] {fact['content']}")
        parts.append("--- END KNOWN FACTS ---")

        # --- Episodes ---
        episodes = self.load_recent_episodes()
        parts.append("")
        parts.append("--- RECENT HISTORY ---")
        for ep in episodes:
            parts.append(f"- {ep['summary']}")
        parts.append("--- END RECENT HISTORY ---")

        return "\n".join(parts)
