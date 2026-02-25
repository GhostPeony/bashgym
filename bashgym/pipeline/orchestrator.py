"""Pipeline orchestrator — connects watcher, importer, quality gate, thresholds."""

import json
import platform
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import PipelineConfig
from .quality_gate import QualityGate, Classification
from .threshold_monitor import ThresholdMonitor
from .watcher import ImportWatcher

from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter
from bashgym.trace_capture.core import TraceCapture


class Pipeline:
    """Orchestrates the auto-import pipeline."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        bashgym_dir: Optional[Path] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self._bashgym_dir = bashgym_dir or self._default_bashgym_dir()
        self._config_path = config_path or (self._bashgym_dir / "pipeline_config.json")
        self.config = PipelineConfig.load(self._config_path)
        self._on_event = on_event

        self._trace_capture = TraceCapture()
        self._importer = ClaudeSessionImporter()
        self._gate = QualityGate(self.config)
        self._monitor = ThresholdMonitor(
            self.config,
            watermark_path=self._bashgym_dir / "pipeline_watermarks.json",
        )
        self._watcher: Optional[ImportWatcher] = None

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
            with open(result.destination_file, "r") as f:
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

    def reload_config(self) -> None:
        self.config = PipelineConfig.load(self._config_path)
        self._gate = QualityGate(self.config)
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

        return {
            "watcher_running": self._watcher._running if self._watcher else False,
            "config": self.config.to_dict(),
            "gold_count": gold_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
        }

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(event_type, payload)
