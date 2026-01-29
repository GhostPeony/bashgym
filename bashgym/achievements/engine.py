"""
Achievement Engine

Evaluates achievements against lifetime stats, tracks earned status,
and persists to ~/.bashgym/achievements.json.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from bashgym.config import get_bashgym_dir
from bashgym.achievements.stats_engine import LifetimeStats
from bashgym.achievements.definitions import ACHIEVEMENTS, AchievementDef

logger = logging.getLogger(__name__)


@dataclass
class AchievementStatus:
    """Status of a single achievement."""
    id: str
    name: str
    description: str
    category: str
    rarity: str
    icon: str
    points: int
    earned: bool
    earned_at: Optional[str] = None
    progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "rarity": self.rarity,
            "icon": self.icon,
            "points": self.points,
            "earned": self.earned,
            "earned_at": self.earned_at,
            "progress": round(self.progress, 3),
        }


class AchievementEngine:
    """Evaluates and persists achievements."""

    def __init__(self):
        self._persistence_path = get_bashgym_dir() / "achievements.json"
        self._earned: Dict[str, str] = {}  # id -> ISO timestamp
        self._total_points: int = 0
        self._load()

    def _load(self) -> None:
        """Load persisted achievement state."""
        if self._persistence_path.exists():
            try:
                data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
                self._earned = data.get("earned", {})
                self._total_points = data.get("total_points", 0)
            except Exception:
                logger.warning("Failed to load achievements.json, starting fresh")
                self._earned = {}
                self._total_points = 0

    def _save(self) -> None:
        """Persist earned achievements."""
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "earned": self._earned,
            "total_points": self._total_points,
        }
        self._persistence_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def evaluate(self, stats: LifetimeStats) -> List[AchievementStatus]:
        """Check all achievements against current stats."""
        results: List[AchievementStatus] = []

        for defn in ACHIEVEMENTS:
            try:
                earned_now, progress = defn.check(stats)
            except Exception:
                earned_now, progress = False, 0.0

            # If already earned previously, keep that timestamp
            if defn.id in self._earned:
                results.append(AchievementStatus(
                    id=defn.id,
                    name=defn.name,
                    description=defn.description,
                    category=defn.category,
                    rarity=defn.rarity,
                    icon=defn.icon,
                    points=defn.points,
                    earned=True,
                    earned_at=self._earned[defn.id],
                    progress=1.0,
                ))
            else:
                results.append(AchievementStatus(
                    id=defn.id,
                    name=defn.name,
                    description=defn.description,
                    category=defn.category,
                    rarity=defn.rarity,
                    icon=defn.icon,
                    points=defn.points,
                    earned=earned_now,
                    earned_at=None,
                    progress=min(max(progress, 0.0), 1.0),
                ))

        return results

    def get_newly_earned(self, stats: LifetimeStats) -> List[AchievementStatus]:
        """Evaluate and persist any newly earned achievements. Returns new unlocks."""
        results = self.evaluate(stats)
        newly_earned: List[AchievementStatus] = []
        now = datetime.now(timezone.utc).isoformat()

        for status in results:
            if status.earned and status.id not in self._earned:
                self._earned[status.id] = now
                status.earned_at = now
                status.progress = 1.0
                newly_earned.append(status)

        if newly_earned:
            self._total_points = sum(
                defn.points for defn in ACHIEVEMENTS if defn.id in self._earned
            )
            self._save()

        return newly_earned

    def get_recent(self, limit: int = 5) -> List[AchievementStatus]:
        """Get the most recently earned achievements."""
        if not self._earned:
            return []

        # Sort by earned timestamp descending
        sorted_ids = sorted(
            self._earned.keys(),
            key=lambda aid: self._earned[aid],
            reverse=True,
        )[:limit]

        results = []
        from bashgym.achievements.definitions import ACHIEVEMENTS_BY_ID
        for aid in sorted_ids:
            defn = ACHIEVEMENTS_BY_ID.get(aid)
            if defn:
                results.append(AchievementStatus(
                    id=defn.id,
                    name=defn.name,
                    description=defn.description,
                    category=defn.category,
                    rarity=defn.rarity,
                    icon=defn.icon,
                    points=defn.points,
                    earned=True,
                    earned_at=self._earned[aid],
                    progress=1.0,
                ))

        return results

    @property
    def total_points(self) -> int:
        return self._total_points

    @property
    def earned_count(self) -> int:
        return len(self._earned)

    @property
    def total_count(self) -> int:
        return len(ACHIEVEMENTS)
