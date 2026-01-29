"""
Achievement API Routes

Endpoints for lifetime stats and achievements.
"""

from fastapi import APIRouter, Request
from typing import Dict, Any, List

from bashgym.achievements.stats_engine import StatsEngine
from bashgym.achievements.engine import AchievementEngine

router = APIRouter(prefix="/api/achievements", tags=["achievements"])

# Module-level singletons (initialized on first request)
_stats_engine: StatsEngine | None = None
_achievement_engine: AchievementEngine | None = None


def _get_engines(request: Request) -> tuple:
    """Lazy-init engines, passing app.state for in-memory data access."""
    global _stats_engine, _achievement_engine
    if _stats_engine is None:
        _stats_engine = StatsEngine(app_state=request.app.state)
    if _achievement_engine is None:
        _achievement_engine = AchievementEngine()
    return _stats_engine, _achievement_engine


@router.get("/stats")
async def get_stats(request: Request) -> Dict[str, Any]:
    """Get full lifetime statistics."""
    stats_engine, achievement_engine = _get_engines(request)
    stats = stats_engine.compute()
    stats.achievement_points = achievement_engine.total_points
    return stats.to_dict()


@router.get("")
async def get_achievements(request: Request) -> Dict[str, Any]:
    """Get all achievements with earned/progress status."""
    stats_engine, achievement_engine = _get_engines(request)
    stats = stats_engine.compute()
    all_achievements = achievement_engine.evaluate(stats)
    return {
        "achievements": [a.to_dict() for a in all_achievements],
        "earned_count": achievement_engine.earned_count,
        "total_count": achievement_engine.total_count,
        "total_points": achievement_engine.total_points,
    }


@router.get("/recent")
async def get_recent(request: Request) -> Dict[str, Any]:
    """Get recently earned achievements (for Home widget)."""
    stats_engine, achievement_engine = _get_engines(request)
    recent = achievement_engine.get_recent(limit=5)
    return {
        "recent": [a.to_dict() for a in recent],
        "earned_count": achievement_engine.earned_count,
        "total_count": achievement_engine.total_count,
        "total_points": achievement_engine.total_points,
    }


@router.post("/refresh")
async def refresh_achievements(request: Request) -> Dict[str, Any]:
    """Force re-evaluate achievements and return any newly earned."""
    stats_engine, achievement_engine = _get_engines(request)
    stats = stats_engine.compute(force=True)
    newly_earned = achievement_engine.get_newly_earned(stats)
    return {
        "newly_earned": [a.to_dict() for a in newly_earned],
        "earned_count": achievement_engine.earned_count,
        "total_count": achievement_engine.total_count,
        "total_points": achievement_engine.total_points,
    }
