"""
Bash Gym Achievements - Lifetime Stats & Achievement System

Skyrim-style lifetime statistics + Xbox/Steam-style unlockable achievements,
driven by existing trace/training/factory data.
"""

from bashgym.achievements.definitions import ACHIEVEMENTS, AchievementDef
from bashgym.achievements.engine import AchievementEngine, AchievementStatus
from bashgym.achievements.stats_engine import (
    FactoryStats,
    LifetimeStats,
    RouterStats,
    StatsEngine,
    TraceStats,
    TrainingStats,
)

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
