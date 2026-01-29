"""
Achievement Definitions

All achievements with their check functions, organized by category.
"""

from dataclasses import dataclass
from typing import Callable, Tuple, List

from bashgym.achievements.stats_engine import LifetimeStats


@dataclass
class AchievementDef:
    id: str
    name: str
    description: str
    category: str       # collection, quality, training, factory, mastery
    rarity: str         # common, uncommon, rare, epic, legendary
    icon: str           # Lucide icon name
    points: int         # 10/25/50/100/250 by rarity
    check: Callable[[LifetimeStats], Tuple[bool, float]]  # (earned, progress 0-1)


def _threshold(value: float, target: float) -> Tuple[bool, float]:
    """Helper: check if value >= target, return (earned, progress)."""
    if target <= 0:
        return (True, 1.0)
    return (value >= target, min(value / target, 1.0))


# ── Collection (trace milestones) ────────────────────────────────────

_collection: List[AchievementDef] = [
    AchievementDef(
        id="first_steps",
        name="First Steps",
        description="Capture your first trace",
        category="collection",
        rarity="common",
        icon="Footprints",
        points=10,
        check=lambda s: _threshold(s.traces.total, 1),
    ),
    AchievementDef(
        id="novice_collector",
        name="Novice Collector",
        description="Capture 10 traces",
        category="collection",
        rarity="common",
        icon="FolderOpen",
        points=10,
        check=lambda s: _threshold(s.traces.total, 10),
    ),
    AchievementDef(
        id="seasoned_collector",
        name="Seasoned Collector",
        description="Capture 50 traces",
        category="collection",
        rarity="uncommon",
        icon="FolderArchive",
        points=25,
        check=lambda s: _threshold(s.traces.total, 50),
    ),
    AchievementDef(
        id="trace_hoarder",
        name="Trace Hoarder",
        description="Capture 100 traces",
        category="collection",
        rarity="rare",
        icon="Database",
        points=50,
        check=lambda s: _threshold(s.traces.total, 100),
    ),
    AchievementDef(
        id="data_magnate",
        name="Data Magnate",
        description="Capture 500 traces",
        category="collection",
        rarity="epic",
        icon="Crown",
        points=100,
        check=lambda s: _threshold(s.traces.total, 500),
    ),
    AchievementDef(
        id="thousand_traces",
        name="Thousand Traces",
        description="Capture 1,000 traces",
        category="collection",
        rarity="legendary",
        icon="Gem",
        points=250,
        check=lambda s: _threshold(s.traces.total, 1000),
    ),
]

# ── Quality (trace quality) ──────────────────────────────────────────

_quality: List[AchievementDef] = [
    AchievementDef(
        id="gold_standard",
        name="Gold Standard",
        description="Get your first gold-rated trace",
        category="quality",
        rarity="common",
        icon="Award",
        points=10,
        check=lambda s: _threshold(s.traces.gold, 1),
    ),
    AchievementDef(
        id="quality_control",
        name="Quality Control",
        description="Collect 10 gold traces",
        category="quality",
        rarity="uncommon",
        icon="BadgeCheck",
        points=25,
        check=lambda s: _threshold(s.traces.gold, 10),
    ),
    AchievementDef(
        id="gold_rush",
        name="Gold Rush",
        description="Collect 50 gold traces",
        category="quality",
        rarity="rare",
        icon="Medal",
        points=50,
        check=lambda s: _threshold(s.traces.gold, 50),
    ),
    AchievementDef(
        id="master_curator",
        name="Master Curator",
        description="Collect 100 gold traces",
        category="quality",
        rarity="epic",
        icon="ShieldCheck",
        points=100,
        check=lambda s: _threshold(s.traces.gold, 100),
    ),
    AchievementDef(
        id="perfectionist",
        name="Perfectionist",
        description="Achieve a perfect 1.0 quality score on a trace",
        category="quality",
        rarity="legendary",
        icon="Star",
        points=250,
        check=lambda s: _threshold(s.traces.highest_quality, 1.0),
    ),
]

# ── Training (model training) ────────────────────────────────────────

_training: List[AchievementDef] = [
    AchievementDef(
        id="first_training",
        name="First Training",
        description="Complete your first training run",
        category="training",
        rarity="common",
        icon="Dumbbell",
        points=10,
        check=lambda s: _threshold(s.training.runs_completed, 1),
    ),
    AchievementDef(
        id="sft_graduate",
        name="SFT Graduate",
        description="Complete an SFT training run",
        category="training",
        rarity="common",
        icon="GraduationCap",
        points=10,
        check=lambda s: (s.training.runs_by_strategy.get("sft", 0) >= 1,
                         min(s.training.runs_by_strategy.get("sft", 0), 1.0)),
    ),
    AchievementDef(
        id="dpo_practitioner",
        name="DPO Practitioner",
        description="Complete a DPO training run",
        category="training",
        rarity="uncommon",
        icon="GitCompare",
        points=25,
        check=lambda s: (s.training.runs_by_strategy.get("dpo", 0) >= 1,
                         min(s.training.runs_by_strategy.get("dpo", 0), 1.0)),
    ),
    AchievementDef(
        id="grpo_explorer",
        name="GRPO Explorer",
        description="Complete a GRPO training run",
        category="training",
        rarity="uncommon",
        icon="Compass",
        points=25,
        check=lambda s: (s.training.runs_by_strategy.get("grpo", 0) >= 1,
                         min(s.training.runs_by_strategy.get("grpo", 0), 1.0)),
    ),
    AchievementDef(
        id="multi_strategist",
        name="Multi-Strategist",
        description="Use all training strategies (SFT, DPO, GRPO)",
        category="training",
        rarity="rare",
        icon="Layers",
        points=50,
        check=lambda s: (
            all(s.training.runs_by_strategy.get(k, 0) >= 1 for k in ("sft", "dpo", "grpo")),
            sum(1 for k in ("sft", "dpo", "grpo") if s.training.runs_by_strategy.get(k, 0) >= 1) / 3.0,
        ),
    ),
    AchievementDef(
        id="sub_one_loss",
        name="Sub-One Loss",
        description="Achieve a training loss below 1.0",
        category="training",
        rarity="rare",
        icon="TrendingDown",
        points=50,
        check=lambda s: (
            s.training.lowest_loss < 1.0 if s.training.lowest_loss != float("inf") else False,
            min(1.0 / max(s.training.lowest_loss, 0.001), 1.0) if s.training.lowest_loss != float("inf") else 0.0,
        ),
    ),
    AchievementDef(
        id="loss_crusher",
        name="Loss Crusher",
        description="Achieve a training loss below 0.5",
        category="training",
        rarity="epic",
        icon="Flame",
        points=100,
        check=lambda s: (
            s.training.lowest_loss < 0.5 if s.training.lowest_loss != float("inf") else False,
            min(0.5 / max(s.training.lowest_loss, 0.001), 1.0) if s.training.lowest_loss != float("inf") else 0.0,
        ),
    ),
]

# ── Factory (synthetic data) ─────────────────────────────────────────

_factory: List[AchievementDef] = [
    AchievementDef(
        id="first_synthesis",
        name="First Synthesis",
        description="Generate your first training example",
        category="factory",
        rarity="common",
        icon="Sparkles",
        points=10,
        check=lambda s: _threshold(s.training.total_examples_generated, 1),
    ),
    AchievementDef(
        id="data_manufacturer",
        name="Data Manufacturer",
        description="Generate 100 training examples",
        category="factory",
        rarity="uncommon",
        icon="Factory",
        points=25,
        check=lambda s: _threshold(s.training.total_examples_generated, 100),
    ),
    AchievementDef(
        id="factory_floor",
        name="Factory Floor",
        description="Generate 500 training examples",
        category="factory",
        rarity="rare",
        icon="Warehouse",
        points=50,
        check=lambda s: _threshold(s.training.total_examples_generated, 500),
    ),
    AchievementDef(
        id="industrial_scale",
        name="Industrial Scale",
        description="Generate 1,000 training examples",
        category="factory",
        rarity="epic",
        icon="Rocket",
        points=100,
        check=lambda s: _threshold(s.training.total_examples_generated, 1000),
    ),
]

# ── Mastery (compound) ───────────────────────────────────────────────

_mastery: List[AchievementDef] = [
    AchievementDef(
        id="full_pipeline",
        name="Full Pipeline",
        description="Collect traces, train a model, and export it",
        category="mastery",
        rarity="rare",
        icon="Workflow",
        points=50,
        check=lambda s: (
            s.traces.total >= 1 and s.training.runs_completed >= 1 and s.training.models_exported >= 1,
            sum([
                1.0 if s.traces.total >= 1 else 0.0,
                1.0 if s.training.runs_completed >= 1 else 0.0,
                1.0 if s.training.models_exported >= 1 else 0.0,
            ]) / 3.0,
        ),
    ),
    AchievementDef(
        id="self_improving",
        name="Self-Improving",
        description="Student model success rate above 70%",
        category="mastery",
        rarity="epic",
        icon="BrainCircuit",
        points=100,
        check=lambda s: (
            s.router.student_success_rate > 0.7 and s.router.total_routed >= 1,
            s.router.student_success_rate / 0.7 if s.router.total_routed >= 1 else 0.0,
        ),
    ),
    AchievementDef(
        id="the_ouroboros",
        name="The Ouroboros",
        description="Complete the full flywheel: collect, synthesize, train, deploy, and route successfully",
        category="mastery",
        rarity="legendary",
        icon="RefreshCcw",
        points=250,
        check=lambda s: (
            (s.traces.gold >= 1 and
             s.training.total_examples_generated >= 1 and
             s.training.runs_completed >= 1 and
             s.training.models_exported >= 1 and
             s.router.total_routed >= 1 and
             s.router.student_success_rate > 0.5),
            sum([
                1.0 if s.traces.gold >= 1 else 0.0,
                1.0 if s.training.total_examples_generated >= 1 else 0.0,
                1.0 if s.training.runs_completed >= 1 else 0.0,
                1.0 if s.training.models_exported >= 1 else 0.0,
                1.0 if s.router.total_routed >= 1 else 0.0,
                1.0 if s.router.student_success_rate > 0.5 else 0.0,
            ]) / 6.0,
        ),
    ),
]

# ── All achievements ─────────────────────────────────────────────────

ACHIEVEMENTS: List[AchievementDef] = (
    _collection + _quality + _training + _factory + _mastery
)

ACHIEVEMENTS_BY_ID = {a.id: a for a in ACHIEVEMENTS}
