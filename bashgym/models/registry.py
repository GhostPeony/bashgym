"""
Model Registry - Index and query trained models

Provides:
- Scanning existing model directories for profiles
- Creating profiles from training state files
- CRUD operations on model profiles
- Filtering, sorting, comparison queries
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from .profile import (
    ModelProfile,
    ModelArtifacts,
    CheckpointInfo,
    GGUFExport,
    BenchmarkResult,
)


class ModelRegistry:
    """
    Central registry for all trained models.

    Scans model directories, builds profiles, and provides query operations.
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the registry.

        Args:
            models_dir: Directory containing trained models.
                       Defaults to scanning both ~/.bashgym/models and data/models
        """
        if models_dir:
            self.models_dirs = [Path(models_dir)]
        else:
            # Scan both user home and project data directories
            self.models_dirs = [
                Path.home() / ".bashgym" / "models",
                Path("data/models"),
                Path.cwd() / "data" / "models",
            ]

        # Use first existing dir as primary for index storage
        self.models_dir = self.models_dirs[0]
        for d in self.models_dirs:
            if d.exists():
                self.models_dir = d
                break

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, ModelProfile] = {}
        self._index_path = self.models_dir / "registry_index.json"

    def scan(self, force_rescan: bool = False) -> int:
        """
        Scan model directories and build/update profiles.

        Args:
            force_rescan: If True, rescan even if profiles exist

        Returns:
            Number of models found
        """
        if not force_rescan and self._profiles:
            return len(self._profiles)

        self._profiles = {}
        seen_ids = set()

        # Find all run directories from all model directories
        for models_dir in self.models_dirs:
            if not models_dir.exists():
                continue

            for run_dir in models_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if run_dir.name.startswith(".") or run_dir.name == "registry_index.json":
                    continue
                # Skip duplicates (prefer first found)
                if run_dir.name in seen_ids:
                    continue
                seen_ids.add(run_dir.name)

                profile = self._load_or_create_profile(run_dir)
                if profile:
                    self._profiles[profile.model_id] = profile

        self._save_index()
        return len(self._profiles)

    def _load_or_create_profile(self, run_dir: Path) -> Optional[ModelProfile]:
        """Load existing profile or create from training artifacts."""
        profile_path = run_dir / "model_profile.json"

        # Try to load existing profile
        if profile_path.exists():
            try:
                return ModelProfile.load(profile_path)
            except Exception as e:
                print(f"Warning: Failed to load profile from {profile_path}: {e}")

        # Create profile from training artifacts
        return self._create_profile_from_artifacts(run_dir)

    def _create_profile_from_artifacts(self, run_dir: Path) -> Optional[ModelProfile]:
        """Create a ModelProfile from training artifacts in a run directory."""
        run_id = run_dir.name

        # Look for trainer_state.json (contains training progress)
        trainer_state = None
        trainer_state_paths = [
            run_dir / "trainer_state.json",
            run_dir / "final" / "trainer_state.json",
        ]
        for path in trainer_state_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        trainer_state = json.load(f)
                    break
                except Exception:
                    pass

        # Look for adapter_config.json (contains model info)
        adapter_config = None
        adapter_config_paths = [
            run_dir / "adapter_config.json",
            run_dir / "final" / "adapter_config.json",
        ]
        for path in adapter_config_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        adapter_config = json.load(f)
                    break
                except Exception:
                    pass

        # If we have neither, this might not be a valid model directory
        if not trainer_state and not adapter_config:
            # Check if there's any checkpoint directory
            has_checkpoint = any(
                d.name.startswith("checkpoint-") for d in run_dir.iterdir() if d.is_dir()
            )
            if not has_checkpoint:
                return None

        # Extract information
        base_model = ""
        if adapter_config:
            base_model = adapter_config.get("base_model_name_or_path", "")

        # Parse run_id for timestamp (format: run_YYYYMMDD_HHMMSS)
        created_at = datetime.now()
        if run_id.startswith("run_"):
            try:
                date_str = run_id[4:19]  # YYYYMMDD_HHMMSS
                created_at = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            except ValueError:
                pass

        # Build artifacts list
        artifacts = self._scan_artifacts(run_dir)

        # Calculate model size
        model_size = self._calculate_model_size(run_dir)

        # Extract loss curve from trainer_state if available
        loss_curve = []
        if trainer_state and "log_history" in trainer_state:
            for entry in trainer_state["log_history"]:
                if "loss" in entry:
                    loss_curve.append({
                        "step": entry.get("step", 0),
                        "loss": entry["loss"],
                        "epoch": entry.get("epoch"),
                        "learning_rate": entry.get("learning_rate"),
                    })

        # Determine training strategy from directory contents or config
        strategy = "sft"  # Default
        if (run_dir / "dpo_config.json").exists():
            strategy = "dpo"
        elif (run_dir / "grpo_config.json").exists():
            strategy = "grpo"

        # Generate a display name
        base_name = base_model.split("/")[-1] if base_model else "unknown"
        display_name = f"{base_name}-{strategy}-{run_id[-6:]}"

        # Determine status
        status = "ready"
        if (run_dir / "final").exists() or artifacts.merged_path:
            status = "ready"
        elif any(d.name.startswith("checkpoint-") for d in run_dir.iterdir() if d.is_dir()):
            status = "ready"
        else:
            status = "needs_eval"

        # Determine completion time from trainer_state or file modification
        completed_at = None
        if trainer_state:
            completed_at = created_at  # Approximate
        final_dir = run_dir / "final"
        if final_dir.exists():
            try:
                completed_at = datetime.fromtimestamp(final_dir.stat().st_mtime)
            except Exception:
                pass

        # Calculate duration
        duration = 0.0
        if trainer_state and "total_runtime" in trainer_state:
            duration = trainer_state["total_runtime"]

        profile = ModelProfile(
            model_id=run_id,
            run_id=run_id,
            display_name=display_name,
            description=f"Auto-generated profile for {run_id}",
            created_at=created_at,
            base_model=base_model,
            training_strategy=strategy,
            config=adapter_config or {},
            started_at=created_at,
            completed_at=completed_at,
            duration_seconds=duration,
            loss_curve=loss_curve,
            final_metrics=self._extract_final_metrics(trainer_state),
            artifacts=artifacts,
            model_dir=str(run_dir),
            model_size_bytes=model_size,
            status=status,
        )

        # Save the newly created profile
        profile.save()
        return profile

    def _scan_artifacts(self, run_dir: Path) -> ModelArtifacts:
        """Scan a run directory for model artifacts."""
        artifacts = ModelArtifacts()

        # Find checkpoints
        for item in run_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    artifacts.checkpoints.append(CheckpointInfo(
                        path=str(item),
                        step=step,
                    ))
                except (IndexError, ValueError):
                    pass

        # Sort checkpoints by step
        artifacts.checkpoints.sort(key=lambda c: c.step)

        # Final adapter
        final_dir = run_dir / "final"
        if final_dir.exists():
            artifacts.final_adapter_path = str(final_dir)

        # Merged model
        merged_dir = run_dir / "merged"
        if merged_dir.exists():
            artifacts.merged_path = str(merged_dir)

        # GGUF exports
        gguf_dir = run_dir / "exported_gguf"
        if gguf_dir.exists():
            for gguf_file in gguf_dir.glob("*.gguf"):
                # Parse quantization from filename (e.g., model-Q4_K_M.gguf)
                name = gguf_file.stem
                quant = "unknown"
                for q in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16", "F32"]:
                    if q in name:
                        quant = q
                        break

                artifacts.gguf_exports.append(GGUFExport(
                    path=str(gguf_file),
                    quantization=quant,
                    size_bytes=gguf_file.stat().st_size,
                    created_at=datetime.fromtimestamp(gguf_file.stat().st_mtime),
                ))

        return artifacts

    def _calculate_model_size(self, run_dir: Path) -> int:
        """Calculate total size of model files."""
        total_size = 0

        # Prioritize: merged > final > largest checkpoint
        for subdir in ["merged", "final"]:
            path = run_dir / subdir
            if path.exists():
                for f in path.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
                if total_size > 0:
                    return total_size

        # Fall back to checkpoints
        for checkpoint in sorted(run_dir.glob("checkpoint-*"), reverse=True):
            for f in checkpoint.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
            if total_size > 0:
                break

        return total_size

    def _extract_final_metrics(self, trainer_state: Optional[Dict]) -> Dict[str, float]:
        """Extract final training metrics from trainer state."""
        if not trainer_state:
            return {}

        metrics = {}

        # Get final loss from log history
        if "log_history" in trainer_state:
            log_history = trainer_state["log_history"]
            for entry in reversed(log_history):
                if "loss" in entry and "final_loss" not in metrics:
                    metrics["final_loss"] = entry["loss"]
                if "eval_loss" in entry and "eval_loss" not in metrics:
                    metrics["eval_loss"] = entry["eval_loss"]
                if "final_loss" in metrics:
                    break

        # Other metrics
        if "epoch" in trainer_state:
            metrics["epochs_completed"] = trainer_state["epoch"]
        if "global_step" in trainer_state:
            metrics["total_steps"] = trainer_state["global_step"]

        return metrics

    def _save_index(self):
        """Save a lightweight index of all models."""
        index = {
            "updated_at": datetime.now().isoformat(),
            "model_count": len(self._profiles),
            "models": [
                {
                    "model_id": p.model_id,
                    "display_name": p.display_name,
                    "base_model": p.base_model,
                    "strategy": p.training_strategy,
                    "status": p.status,
                    "starred": p.starred,
                    "created_at": p.created_at.isoformat(),
                }
                for p in self._profiles.values()
            ]
        }
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    # Query methods

    def list(
        self,
        strategy: Optional[str] = None,
        base_model: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        starred_only: bool = False,
        sort_by: str = "created_at",
        sort_order: Literal["asc", "desc"] = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ModelProfile]:
        """
        List models with filtering and sorting.

        Args:
            strategy: Filter by training strategy
            base_model: Filter by base model (partial match)
            status: Filter by status
            tags: Filter by tags (any match)
            starred_only: Only return starred models
            sort_by: Field to sort by
            sort_order: Sort direction
            limit: Max results to return
            offset: Results to skip

        Returns:
            List of matching ModelProfiles
        """
        results = list(self._profiles.values())

        # Apply filters
        if strategy:
            results = [p for p in results if p.training_strategy == strategy]
        if base_model:
            results = [p for p in results if base_model.lower() in p.base_model.lower()]
        if status:
            results = [p for p in results if p.status == status]
        if tags:
            results = [p for p in results if any(t in p.tags for t in tags)]
        if starred_only:
            results = [p for p in results if p.starred]

        # Sort
        reverse = sort_order == "desc"
        if sort_by == "created_at":
            results.sort(key=lambda p: p.created_at, reverse=reverse)
        elif sort_by == "display_name":
            results.sort(key=lambda p: p.display_name.lower(), reverse=reverse)
        elif sort_by == "custom_eval":
            results.sort(
                key=lambda p: p.custom_eval_pass_rate or 0,
                reverse=reverse
            )
        elif sort_by == "benchmark_avg":
            results.sort(
                key=lambda p: p.benchmark_avg_score or 0,
                reverse=reverse
            )
        elif sort_by == "model_size":
            results.sort(key=lambda p: p.model_size_bytes, reverse=reverse)

        # Starred items first (within sort)
        results.sort(key=lambda p: not p.starred)

        # Pagination
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    def get(self, model_id: str) -> Optional[ModelProfile]:
        """Get a model by ID."""
        return self._profiles.get(model_id)

    def update(self, model_id: str, updates: Dict[str, Any]) -> Optional[ModelProfile]:
        """
        Update a model's editable fields.

        Editable fields: display_name, description, tags, starred
        """
        profile = self._profiles.get(model_id)
        if not profile:
            return None

        editable = {"display_name", "description", "tags", "starred"}
        for key, value in updates.items():
            if key in editable:
                setattr(profile, key, value)

        profile.save()
        self._save_index()
        return profile

    def delete(self, model_id: str, archive: bool = True) -> bool:
        """
        Delete or archive a model.

        Args:
            model_id: Model to delete
            archive: If True, mark as archived instead of deleting files

        Returns:
            True if successful
        """
        profile = self._profiles.get(model_id)
        if not profile:
            return False

        if archive:
            profile.status = "archived"
            if "archived" not in profile.tags:
                profile.tags.append("archived")
            profile.save()
        else:
            # Actually remove from registry (files remain)
            del self._profiles[model_id]

        self._save_index()
        return True

    def star(self, model_id: str, starred: bool = True) -> Optional[ModelProfile]:
        """Star or unstar a model."""
        return self.update(model_id, {"starred": starred})

    def compare(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.

        Args:
            model_ids: Models to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Dict mapping model_id to metrics dict
        """
        result = {}

        for model_id in model_ids:
            profile = self._profiles.get(model_id)
            if not profile:
                continue

            model_metrics = {
                "display_name": profile.display_name,
                "base_model": profile.base_model,
                "strategy": profile.training_strategy,
                "created_at": profile.created_at.isoformat(),
                "custom_eval_pass_rate": profile.custom_eval_pass_rate,
                "benchmark_avg_score": profile.benchmark_avg_score,
                "model_size_bytes": profile.model_size_bytes,
                "inference_latency_ms": profile.inference_latency_ms,
                "final_loss": profile.final_metrics.get("final_loss"),
                "training_duration": profile.duration_seconds,
            }

            # Add individual benchmark scores
            for name, bench in profile.benchmarks.items():
                model_metrics[f"benchmark_{name}"] = bench.score

            # Filter to requested metrics
            if metrics:
                model_metrics = {k: v for k, v in model_metrics.items() if k in metrics}

            result[model_id] = model_metrics

        return result

    def leaderboard(
        self,
        metric: str = "custom_eval_pass_rate",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get ranked leaderboard by metric.

        Args:
            metric: Metric to rank by
            limit: Max results

        Returns:
            Ranked list of model summaries
        """
        models = list(self._profiles.values())

        # Get metric value for each model
        def get_metric(p: ModelProfile) -> float:
            if metric == "custom_eval_pass_rate":
                return p.custom_eval_pass_rate or 0
            elif metric == "benchmark_avg_score":
                return p.benchmark_avg_score or 0
            elif metric.startswith("benchmark_"):
                bench_name = metric[10:]
                if bench_name in p.benchmarks:
                    return p.benchmarks[bench_name].score
            elif metric == "inference_latency_ms":
                # Lower is better for latency
                return -(p.inference_latency_ms or float("inf"))
            return 0

        # Sort by metric (descending)
        models.sort(key=get_metric, reverse=True)
        models = models[:limit]

        # Build leaderboard
        return [
            {
                "rank": i + 1,
                "model_id": p.model_id,
                "display_name": p.display_name,
                "value": get_metric(p) if not metric == "inference_latency_ms" else (p.inference_latency_ms or 0),
                "base_model": p.base_model,
                "strategy": p.training_strategy,
            }
            for i, p in enumerate(models)
        ]

    def trends(
        self,
        metric: str = "benchmark_avg_score",
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get metric trends over time.

        Returns data points for charting metric changes across models over time.
        """
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        cutoff_dt = datetime.fromtimestamp(cutoff)

        data_points = []

        for profile in self._profiles.values():
            if profile.created_at < cutoff_dt:
                continue

            value = None
            if metric == "benchmark_avg_score":
                value = profile.benchmark_avg_score
            elif metric == "custom_eval_pass_rate":
                value = profile.custom_eval_pass_rate
            elif metric == "final_loss":
                value = profile.final_metrics.get("final_loss")

            if value is not None:
                data_points.append({
                    "timestamp": profile.created_at.isoformat(),
                    "model_id": profile.model_id,
                    "display_name": profile.display_name,
                    "value": value,
                })

        # Sort by timestamp
        data_points.sort(key=lambda d: d["timestamp"])
        return data_points

    def add_benchmark_result(
        self,
        model_id: str,
        benchmark_name: str,
        score: float,
        passed: int,
        total: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[ModelProfile]:
        """Add a benchmark result to a model."""
        profile = self._profiles.get(model_id)
        if not profile:
            return None

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            score=score,
            passed=passed,
            total=total,
            metrics=metrics or {},
        )
        profile.add_benchmark_result(result)
        profile.update_status()
        profile.save()
        self._save_index()
        return profile


# Singleton registry instance
_registry: Optional[ModelRegistry] = None


def get_registry(models_dir: Optional[str] = None) -> ModelRegistry:
    """Get the global ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(models_dir)
        _registry.scan()
    return _registry
