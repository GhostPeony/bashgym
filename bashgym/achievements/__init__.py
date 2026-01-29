"""
Bash Gym Achievements - Lifetime Stats & Achievement System

Skyrim-style lifetime statistics + Xbox/Steam-style unlockable achievements,
driven by existing trace/training/factory data.
"""

from bashgym.achievements.stats_engine import (
    StatsEngine,
    LifetimeStats,
    TraceStats,
    TrainingStats,
    FactoryStats,
    RouterStats,
)
from bashgym.achievements.engine import AchievementEngine, AchievementStatus
from bashgym.achievements.definitions import AchievementDef, ACHIEVEMENTS

__all__ = [
    "StatsEngine",
    "LifetimeStats",
    "TraceStats",
    "TrainingStats",
    "FactoryStats",
    "RouterStats",
    "AchievementEngine",
    "AchievementStatus",
    "AchievementDef",
    "ACHIEVEMENTS",
]
