"""Pipeline orchestrator — connects watcher, importer, quality gate, thresholds.

Optionally wires a SemanticJudge into the QualityGate when the
`semantic_judge_enabled` flag is set (default: False). The judge is
constructed lazily so the pipeline never breaks if the anthropic package
isn't installed.
"""

import json
import logging
import os
import platform
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import PipelineConfig
from .quality_gate import QualityGate, Classification
from .threshold_monitor import ThresholdMonitor
from .watcher import ImportWatcher

from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter
from bashgym.trace_capture.core import TraceCapture

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the auto-import pipeline."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        bashgym_dir: Optional[Path] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        semantic_judge_enabled: bool = False,
        semantic_judge_model: str = "claude-haiku-4-5-20251001",
    ):
        self._bashgym_dir = bashgym_dir or self._default_bashgym_dir()
        self._config_path = config_path or (self._bashgym_dir / "pipeline_config.json")
        self.config = PipelineConfig.load(self._config_path)
        self._on_event = on_event

        self._semantic_judge_enabled = semantic_judge_enabled
        self._semantic_judge_model = semantic_judge_model

        self._trace_capture = TraceCapture()
        self._importer = ClaudeSessionImporter()
        self._gate = self._build_quality_gate()
        self._monitor = ThresholdMonitor(
            self.config,
            watermark_path=self._bashgym_dir / "pipeline_watermarks.json",
        )
        self._watcher: Optional[ImportWatcher] = None

    def _build_quality_gate(self) -> QualityGate:
        """Build QualityGate, optionally with a SemanticJudge."""
        semantic_judge = None
        if self._semantic_judge_enabled:
            try:
                from bashgym.judge.semantic_judge import SemanticJudge
                semantic_judge = SemanticJudge(
                    model=self._semantic_judge_model,
                    enabled=True,
                )
                logger.info(
                    "Semantic judge enabled (model=%s)", self._semantic_judge_model
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize semantic judge, continuing without: %s", exc
                )
        return QualityGate(self.config, semantic_judge=semantic_judge)

    @staticmethod
    def _default_bashgym_dir() -> Path:
        if platform.system() == "Windows":
            return Path(os.environ.get("USERPROFILE", "")) / ".bashgym"
        return Path.home() / ".bashgym"

    def start_watcher(self) -> None:
        claude_projects = self._importer.claude_dir / "projects"
        self._watcher = ImportWatcher(
            config=self.config,
            watch_dir=claude_projects,
            on_import=self.handle_session_file,
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def handle_session_file(self, session_file: Path) -> Optional[Dict[str, Any]]:
        """Full pipeline: import -> classify -> route -> check thresholds."""
        result = self._importer.import_session(session_file)
        if result.skipped or result.error or result.steps_imported == 0:
            return None

        self._emit("pipeline:import", {
            "session_id": result.session_id,
            "steps_imported": result.steps_imported,
            "source_file": str(session_file),
        })

        if not result.destination_file or not result.destination_file.exists():
            return None

        try:
            with open(result.destination_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
            summary = trace_data.get("summary", {})
            success_rate = summary.get("success_rate", 0)
            total_steps = summary.get("total_steps", 0)
        except (json.JSONDecodeError, IOError):
            success_rate = 0
            total_steps = 0

        classification = self._gate.classify(success_rate, total_steps)
        dest = self._gate.route_trace(
            result.destination_file,
            classification,
            self._trace_capture.gold_traces_dir,
            self._trace_capture.failed_traces_dir,
        )

        self._emit("pipeline:classified", {
            "session_id": result.session_id,
            "classification": classification.value,
            "success_rate": success_rate,
            "destination": str(dest),
        })

        # Emit JudgeVerdict event when semantic judge is available
        # Note: The async classify_with_semantics path is for callers that
        # operate in an async context. The watcher callback path stays sync
        # and uses structural classification only. Async callers (e.g. API
        # routes) can call _gate.classify_with_semantics() directly and
        # emit the verdict event.

        if self._monitor.should_generate(self._trace_capture.gold_traces_dir):
            self._emit("pipeline:threshold_reached", {
                "stage": "generate",
                "gold_count": len(list(self._trace_capture.gold_traces_dir.glob("*.json"))),
                "threshold": self.config.generate_gold_threshold,
            })
            self._monitor.mark_generate_triggered(self._trace_capture.gold_traces_dir)

        return {
            "session_id": result.session_id,
            "steps_imported": result.steps_imported,
            "classification": classification.value,
            "destination": str(dest),
        }

    async def handle_session_file_async(
        self, session_file: Path
    ) -> Optional[Dict[str, Any]]:
        """Async pipeline path with semantic judge support.

        Same as handle_session_file but uses classify_with_semantics()
        for GOLD traces when a semantic judge is configured. Emits a
        JudgeVerdict event on the event bus.
        """
        result = self._importer.import_session(session_file)
        if result.skipped or result.error or result.steps_imported == 0:
            return None

        self._emit("pipeline:import", {
            "session_id": result.session_id,
            "steps_imported": result.steps_imported,
            "source_file": str(session_file),
        })

        if not result.destination_file or not result.destination_file.exists():
            return None

        try:
            with open(result.destination_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
            summary = trace_data.get("summary", {})
            success_rate = summary.get("success_rate", 0)
            total_steps = summary.get("total_steps", 0)
        except (json.JSONDecodeError, IOError):
            success_rate = 0
            total_steps = 0

        # Build a lightweight ProcessedTrace for the semantic judge
        processed_trace = None
        if self._gate.semantic_judge is not None:
            processed_trace = self._build_processed_trace(
                trace_data, result.destination_file
            )

        classification, verdict = await self._gate.classify_with_semantics(
            success_rate, total_steps, trace=processed_trace
        )

        dest = self._gate.route_trace(
            result.destination_file,
            classification,
            self._trace_capture.gold_traces_dir,
            self._trace_capture.failed_traces_dir,
        )

        event_payload: Dict[str, Any] = {
            "session_id": result.session_id,
            "classification": classification.value,
            "success_rate": success_rate,
            "destination": str(dest),
        }

        # Attach semantic verdict if available
        if verdict is not None:
            event_payload["semantic_score"] = verdict.score
            event_payload["semantic_confidence"] = verdict.confidence
            event_payload["quality_flags"] = verdict.quality_flags

            # Emit JudgeVerdict event via event bus
            self._emit_judge_verdict(result.session_id, verdict)

        self._emit("pipeline:classified", event_payload)

        if self._monitor.should_generate(self._trace_capture.gold_traces_dir):
            self._emit("pipeline:threshold_reached", {
                "stage": "generate",
                "gold_count": len(list(
                    self._trace_capture.gold_traces_dir.glob("*.json")
                )),
                "threshold": self.config.generate_gold_threshold,
            })
            self._monitor.mark_generate_triggered(self._trace_capture.gold_traces_dir)

        return {
            "session_id": result.session_id,
            "steps_imported": result.steps_imported,
            "classification": classification.value,
            "destination": str(dest),
            **({"semantic_score": verdict.score} if verdict else {}),
        }

    def _build_processed_trace(
        self, trace_data: Dict[str, Any], trace_file: Path
    ) -> Any:
        """Build a minimal ProcessedTrace from raw trace data for semantic judging."""
        try:
            from bashgym.factory.trace_processor import ProcessedTrace, TraceQualityMetrics

            steps = trace_data.get("steps", [])
            summary = trace_data.get("summary", {})
            metadata = trace_data.get("metadata", {})

            metrics = TraceQualityMetrics(
                total_steps=summary.get("total_steps", len(steps)),
                successful_steps=summary.get("successful_steps", 0),
                failed_steps=summary.get("failed_steps", 0),
            )

            # Extract task prompt from metadata or first user message
            task_prompt = metadata.get("task", "")
            if not task_prompt:
                task_prompt = metadata.get("description", "")
            if not task_prompt and steps:
                # Use the first step's command as a rough proxy
                task_prompt = str(steps[0].get("command", ""))[:500]

            return ProcessedTrace(
                trace_id=trace_data.get("session_id", trace_file.stem),
                original_path=trace_file,
                task_prompt=task_prompt,
                normalized_steps=steps,
                quality_metrics=metrics,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to build ProcessedTrace: %s", exc)
            return None

    def _emit_judge_verdict(self, session_id: str, verdict: Any) -> None:
        """Emit a JudgeVerdict event on the typed event bus."""
        try:
            from bashgym.events.bus import event_bus
            from bashgym.events.types import JudgeVerdict

            event_bus.emit(JudgeVerdict(
                task_id=session_id,
                passed=verdict.score >= 0.6,
                confidence=verdict.confidence,
                reasoning=verdict.reasoning,
                details={
                    "score": verdict.score,
                    "quality_flags": verdict.quality_flags,
                    "issues": verdict.issues,
                },
            ))
        except Exception as exc:
            logger.debug("Failed to emit JudgeVerdict event: %s", exc)

    def reload_config(self) -> None:
        self.config = PipelineConfig.load(self._config_path)
        self._gate = self._build_quality_gate()
        self._monitor = ThresholdMonitor(
            self.config,
            watermark_path=self._bashgym_dir / "pipeline_watermarks.json",
        )
        if self._watcher:
            self._watcher.reload_config(self.config)

    def save_config(self, updates: Dict[str, Any]) -> PipelineConfig:
        merged = {**self.config.to_dict(), **updates}
        self.config = PipelineConfig.from_dict(merged)
        self.config.save(self._config_path)
        self.reload_config()
        return self.config

    def get_status(self) -> Dict[str, Any]:
        gold_count = len(list(self._trace_capture.gold_traces_dir.glob("*.json")))
        pending_count = len(list(self._trace_capture.traces_dir.glob("*.json")))
        failed_count = len(list(self._trace_capture.failed_traces_dir.glob("*.json")))

        status: Dict[str, Any] = {
            "watcher_running": self._watcher._running if self._watcher else False,
            "config": self.config.to_dict(),
            "gold_count": gold_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
        }

        # Include semantic judge status
        if self._gate.semantic_judge is not None:
            status["semantic_judge"] = {
                "enabled": self._gate.semantic_judge.enabled,
                "model": self._gate.semantic_judge.model,
            }

        return status

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(event_type, payload)
