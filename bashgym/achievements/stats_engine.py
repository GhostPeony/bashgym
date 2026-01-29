"""
Lifetime Stats Engine

Scans data directories to compute lifetime statistics across traces,
training runs, factory jobs, and router usage. Results cached with 60s TTL.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter
from datetime import datetime

from bashgym.config import get_bashgym_dir

logger = logging.getLogger(__name__)


@dataclass
class TraceStats:
    total: int = 0
    gold: int = 0
    silver: int = 0
    bronze: int = 0
    failed: int = 0
    pending: int = 0
    highest_quality: float = 0.0
    avg_quality: float = 0.0
    total_steps: int = 0
    most_used_tool: str = ""
    unique_repos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "gold": self.gold,
            "silver": self.silver,
            "bronze": self.bronze,
            "failed": self.failed,
            "pending": self.pending,
            "highest_quality": self.highest_quality,
            "avg_quality": round(self.avg_quality, 3),
            "total_steps": self.total_steps,
            "most_used_tool": self.most_used_tool,
            "unique_repos": self.unique_repos,
        }


@dataclass
class TrainingStats:
    runs_completed: int = 0
    runs_by_strategy: Dict[str, int] = field(default_factory=dict)
    lowest_loss: float = float("inf")
    models_finetuned: int = 0
    models_exported: int = 0
    total_examples_generated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runs_completed": self.runs_completed,
            "runs_by_strategy": self.runs_by_strategy,
            "lowest_loss": self.lowest_loss if self.lowest_loss != float("inf") else None,
            "models_finetuned": self.models_finetuned,
            "models_exported": self.models_exported,
            "total_examples_generated": self.total_examples_generated,
        }


@dataclass
class FactoryStats:
    jobs_completed: int = 0
    total_generated: int = 0
    total_valid: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jobs_completed": self.jobs_completed,
            "total_generated": self.total_generated,
            "total_valid": self.total_valid,
        }


@dataclass
class RouterStats:
    total_routed: int = 0
    student_success_rate: float = 0.0
    teacher_success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_routed": self.total_routed,
            "student_success_rate": round(self.student_success_rate, 3),
            "teacher_success_rate": round(self.teacher_success_rate, 3),
        }


@dataclass
class LifetimeStats:
    traces: TraceStats = field(default_factory=TraceStats)
    training: TrainingStats = field(default_factory=TrainingStats)
    factory: FactoryStats = field(default_factory=FactoryStats)
    router: RouterStats = field(default_factory=RouterStats)
    first_activity: str = ""
    days_active: int = 0
    achievement_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traces": self.traces.to_dict(),
            "training": self.training.to_dict(),
            "factory": self.factory.to_dict(),
            "router": self.router.to_dict(),
            "first_activity": self.first_activity,
            "days_active": self.days_active,
            "achievement_points": self.achievement_points,
        }


class StatsEngine:
    """Scans data directories to compute lifetime statistics."""

    def __init__(self, app_state: Optional[Any] = None):
        self._cache: Optional[LifetimeStats] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 60.0  # seconds
        self._app_state = app_state
        self._bashgym_dir = get_bashgym_dir()

    def compute(self, force: bool = False) -> LifetimeStats:
        """Compute lifetime stats. Returns cached value if fresh."""
        now = time.time()
        if not force and self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        stats = LifetimeStats()
        stats.traces = self._scan_traces()
        stats.training = self._scan_training()
        stats.factory = self._scan_factory()
        stats.router = self._scan_router()
        stats.first_activity, stats.days_active = self._compute_activity()

        self._cache = stats
        self._cache_time = now
        return stats

    def _scan_traces(self) -> TraceStats:
        """Scan trace directories for counts and quality metrics."""
        ts = TraceStats()
        quality_scores: list = []
        tool_counter: Counter = Counter()
        repos: set = set()
        tier_dirs = {
            "gold": self._bashgym_dir / "gold_traces",
            "silver": self._bashgym_dir / "silver_traces",
            "bronze": self._bashgym_dir / "bronze_traces",
            "failed": self._bashgym_dir / "failed_traces",
            "pending": self._bashgym_dir / "traces",
        }

        for tier, dir_path in tier_dirs.items():
            if not dir_path.exists():
                continue
            trace_files = list(dir_path.glob("*.json"))
            count = len(trace_files)

            if tier == "gold":
                ts.gold = count
            elif tier == "silver":
                ts.silver = count
            elif tier == "bronze":
                ts.bronze = count
            elif tier == "failed":
                ts.failed = count
            elif tier == "pending":
                ts.pending = count

            ts.total += count

            # Sample trace files for quality/tool/repo data
            for fp in trace_files:
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    # Quality score
                    quality = data.get("quality", {})
                    if isinstance(quality, dict):
                        score = quality.get("score") or quality.get("total_score", 0)
                    elif isinstance(quality, (int, float)):
                        score = float(quality)
                    else:
                        score = 0
                    if score:
                        quality_scores.append(float(score))

                    # Steps and tools
                    steps = data.get("steps", data.get("normalized_steps", []))
                    if isinstance(steps, list):
                        ts.total_steps += len(steps)
                        for step in steps:
                            if isinstance(step, dict):
                                tool = step.get("tool", step.get("type", ""))
                                if tool:
                                    tool_counter[tool] += 1

                    # Repos
                    repo = data.get("metadata", {}).get("primary_repo", {}).get("name", "")
                    if not repo:
                        repo = data.get("primary_repo", {}).get("name", "") if isinstance(data.get("primary_repo"), dict) else ""
                    if repo:
                        repos.add(repo)
                except Exception:
                    continue

        if quality_scores:
            ts.highest_quality = max(quality_scores)
            ts.avg_quality = sum(quality_scores) / len(quality_scores)
        if tool_counter:
            ts.most_used_tool = tool_counter.most_common(1)[0][0]
        ts.unique_repos = len(repos)

        return ts

    def _scan_training(self) -> TrainingStats:
        """Scan training run directories for metadata."""
        ts = TrainingStats()
        models_dir = self._bashgym_dir / "models"

        if not models_dir.exists():
            return ts

        for run_dir in models_dir.iterdir():
            if not run_dir.is_dir():
                continue

            ts.models_finetuned += 1

            # Check for metadata
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    status = meta.get("status", "")
                    if status == "completed":
                        ts.runs_completed += 1

                    strategy = meta.get("strategy", "unknown")
                    ts.runs_by_strategy[strategy] = ts.runs_by_strategy.get(strategy, 0) + 1

                    # Loss
                    loss = meta.get("metrics", {}).get("final_loss")
                    if loss is not None and loss < ts.lowest_loss:
                        ts.lowest_loss = float(loss)

                    # Check loss curve too
                    for point in meta.get("loss_curve", []):
                        if isinstance(point, dict) and "loss" in point:
                            val = float(point["loss"])
                            if val < ts.lowest_loss:
                                ts.lowest_loss = val
                except Exception:
                    continue
            else:
                # No metadata — count it as a completed run if final/ exists
                if (run_dir / "final").exists() or (run_dir / "merged").exists():
                    ts.runs_completed += 1

            # Check for GGUF exports
            gguf_files = list(run_dir.glob("*.gguf"))
            if gguf_files:
                ts.models_exported += len(gguf_files)

        # Also check integration manifest for exports
        manifest_path = self._bashgym_dir / "integration" / "model_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    exports = manifest.get("exports", [])
                    if isinstance(exports, list):
                        ts.models_exported = max(ts.models_exported, len(exports))
            except Exception:
                pass

        # Count training examples
        batches_dir = self._bashgym_dir / "training_batches"
        if batches_dir.exists():
            for jsonl_file in batches_dir.glob("*.jsonl"):
                try:
                    ts.total_examples_generated += sum(
                        1 for _ in jsonl_file.open(encoding="utf-8")
                    )
                except Exception:
                    continue

        return ts

    def _scan_factory(self) -> FactoryStats:
        """Get factory stats from app state if available."""
        fs = FactoryStats()

        if self._app_state:
            # Read from in-memory synthesis jobs
            jobs = getattr(self._app_state, "synthesis_jobs", {})
            if isinstance(jobs, dict):
                for job in jobs.values():
                    if isinstance(job, dict):
                        status = job.get("status", "")
                        if status in ("completed", "done"):
                            fs.jobs_completed += 1
                            fs.total_generated += job.get("total_generated", 0)
                            fs.total_valid += job.get("total_valid", 0)

        return fs

    def _scan_router(self) -> RouterStats:
        """Get router stats from app state if available."""
        rs = RouterStats()

        if self._app_state:
            router = getattr(self._app_state, "router", None)
            if router:
                history = getattr(router, "routing_history", [])
                rs.total_routed = len(history)

                models = getattr(router, "models", {})
                for name, model in models.items():
                    model_type = getattr(model, "model_type", None)
                    if model_type:
                        type_val = model_type.value if hasattr(model_type, "value") else str(model_type)
                        if "student" in type_val.lower():
                            rs.student_success_rate = getattr(model, "success_rate", 0.0)
                        elif "teacher" in type_val.lower():
                            rs.teacher_success_rate = getattr(model, "success_rate", 0.0)

        return rs

    def _compute_activity(self) -> tuple:
        """Compute first activity timestamp and days active."""
        first_ts: Optional[datetime] = None

        # Check trace file modification times
        for subdir in ("gold_traces", "silver_traces", "bronze_traces", "failed_traces", "traces"):
            dir_path = self._bashgym_dir / subdir
            if not dir_path.exists():
                continue
            for fp in dir_path.glob("*.json"):
                try:
                    mtime = datetime.fromtimestamp(fp.stat().st_mtime)
                    ctime = datetime.fromtimestamp(fp.stat().st_ctime)
                    earliest = min(mtime, ctime)
                    if first_ts is None or earliest < first_ts:
                        first_ts = earliest
                except Exception:
                    continue

        # Check model directories
        models_dir = self._bashgym_dir / "models"
        if models_dir.exists():
            for run_dir in models_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        ctime = datetime.fromtimestamp(run_dir.stat().st_ctime)
                        if first_ts is None or ctime < first_ts:
                            first_ts = ctime
                    except Exception:
                        continue

        if first_ts is None:
            return ("", 0)

        now = datetime.now()
        days = (now - first_ts).days + 1  # inclusive
        return (first_ts.isoformat(), days)
