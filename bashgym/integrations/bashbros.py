"""
Bashbros Integration for Bash Gym

Provides integration between bashbros (security middleware + AI sidekick) and
bashgym (self-improving agent training) through a shared directory protocol.

Key features:
- Directory watcher for traces from bashbros
- Settings synchronization
- GGUF model export + Ollama registration
- Security delegation

Neither tool requires the other, but when both are present, they create
a continuous improvement loop.

Data Flow:
    bashbros captures traces -> bashgym trains -> GGUF to Ollama -> bashbros sidekick improves
"""

import os
import json
import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timezone
from enum import Enum
import threading
import platform

logger = logging.getLogger(__name__)


class CaptureMode(Enum):
    """Capture mode for trace collection."""
    EVERYTHING = "everything"           # Capture all sessions
    SUCCESSFUL_ONLY = "successful_only" # Only verified successful
    SIDEKICK_CURATED = "sidekick_curated"  # AI sidekick picks teachable moments


class TrainingTrigger(Enum):
    """When to trigger automatic training."""
    MANUAL = "manual"           # Only on user request
    QUALITY_BASED = "quality_based"   # When gold traces >= threshold
    SCHEDULED = "scheduled"     # On a schedule (daily, weekly)


@dataclass
class IntegrationSettings:
    """Shared settings between bashbros and bashgym."""

    version: str = "1.0"
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    # Integration state
    enabled: bool = False
    linked_at: Optional[str] = None

    # Capture settings
    capture_mode: CaptureMode = CaptureMode.SUCCESSFUL_ONLY
    auto_stream: bool = True

    # Training settings
    auto_training_enabled: bool = False
    quality_threshold: int = 50  # Gold traces needed to trigger training
    trigger: TrainingTrigger = TrainingTrigger.QUALITY_BASED

    # Security settings
    bashbros_primary: bool = True
    policy_path: Optional[str] = None

    # Model sync settings
    auto_export_ollama: bool = True
    ollama_model_name: str = "bashgym-sidekick"
    notify_on_update: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
            "integration": {
                "enabled": self.enabled,
                "linked_at": self.linked_at,
            },
            "capture": {
                "mode": self.capture_mode.value,
                "auto_stream": self.auto_stream,
            },
            "training": {
                "auto_enabled": self.auto_training_enabled,
                "quality_threshold": self.quality_threshold,
                "trigger": self.trigger.value,
            },
            "security": {
                "bashbros_primary": self.bashbros_primary,
                "policy_path": self.policy_path,
            },
            "model_sync": {
                "auto_export_ollama": self.auto_export_ollama,
                "ollama_model_name": self.ollama_model_name,
                "notify_on_update": self.notify_on_update,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationSettings":
        """Create from dictionary."""
        settings = cls()
        settings.version = data.get("version", "1.0")
        settings.updated_at = data.get("updated_at")
        settings.updated_by = data.get("updated_by")

        integration = data.get("integration", {})
        settings.enabled = integration.get("enabled", False)
        settings.linked_at = integration.get("linked_at")

        capture = data.get("capture", {})
        try:
            settings.capture_mode = CaptureMode(capture.get("mode", "successful_only"))
        except ValueError:
            settings.capture_mode = CaptureMode.SUCCESSFUL_ONLY
        settings.auto_stream = capture.get("auto_stream", True)

        training = data.get("training", {})
        settings.auto_training_enabled = training.get("auto_enabled", False)
        settings.quality_threshold = training.get("quality_threshold", 50)
        try:
            settings.trigger = TrainingTrigger(training.get("trigger", "quality_based"))
        except ValueError:
            settings.trigger = TrainingTrigger.QUALITY_BASED

        security = data.get("security", {})
        settings.bashbros_primary = security.get("bashbros_primary", True)
        settings.policy_path = security.get("policy_path")

        model_sync = data.get("model_sync", {})
        settings.auto_export_ollama = model_sync.get("auto_export_ollama", True)
        settings.ollama_model_name = model_sync.get("ollama_model_name", "bashgym-sidekick")
        settings.notify_on_update = model_sync.get("notify_on_update", True)

        return settings


@dataclass
class ModelVersion:
    """A version of the exported model."""
    version: str
    created: str
    traces_used: int
    quality_avg: float
    gguf_path: Optional[str] = None


@dataclass
class ModelManifest:
    """Manifest tracking model versions."""
    latest: str = "v0"
    versions: List[ModelVersion] = field(default_factory=list)
    rollback_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latest": self.latest,
            "versions": [
                {
                    "version": v.version,
                    "created": v.created,
                    "traces_used": v.traces_used,
                    "quality_avg": v.quality_avg,
                    "gguf_path": v.gguf_path,
                }
                for v in self.versions
            ],
            "rollback_available": self.rollback_available,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        manifest = cls()
        manifest.latest = data.get("latest", "v0")
        manifest.rollback_available = data.get("rollback_available", False)

        for v in data.get("versions", []):
            manifest.versions.append(ModelVersion(
                version=v.get("version", ""),
                created=v.get("created", ""),
                traces_used=v.get("traces_used", 0),
                quality_avg=v.get("quality_avg", 0.0),
                gguf_path=v.get("gguf_path"),
            ))

        return manifest


@dataclass
class IntegrationStatus:
    """Current status of the integration."""
    bashbros_connected: bool = False
    bashgym_connected: bool = False
    last_trace_received: Optional[str] = None
    pending_traces: int = 0
    processed_traces: int = 0
    current_model_version: Optional[str] = None
    training_in_progress: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bashbros_connected": self.bashbros_connected,
            "bashgym_connected": self.bashgym_connected,
            "last_trace_received": self.last_trace_received,
            "pending_traces": self.pending_traces,
            "processed_traces": self.processed_traces,
            "current_model_version": self.current_model_version,
            "training_in_progress": self.training_in_progress,
        }


class BashbrosIntegration:
    """
    Manages integration between bashbros and bashgym.

    Responsibilities:
    - Watch for traces from bashbros in shared directory
    - Process and ingest traces into bashgym
    - Sync settings between both tools
    - Export trained models to GGUF and register with Ollama
    """

    # Default integration directory
    DEFAULT_INTEGRATION_DIR = Path.home() / ".bashgym" / "integration"

    def __init__(
        self,
        integration_dir: Optional[Path] = None,
        trace_callback: Optional[Callable[[Path], None]] = None,
    ):
        """Initialize the integration.

        Args:
            integration_dir: Path to shared integration directory
            trace_callback: Callback when new trace is received
        """
        self.integration_dir = Path(integration_dir) if integration_dir else self.DEFAULT_INTEGRATION_DIR
        self.trace_callback = trace_callback

        # Directory structure
        self.traces_dir = self.integration_dir / "traces"
        self.pending_dir = self.traces_dir / "pending"
        self.processed_dir = self.traces_dir / "processed"
        self.failed_dir = self.traces_dir / "failed"
        self.models_dir = self.integration_dir / "models"
        self.config_dir = self.integration_dir / "config"
        self.status_dir = self.integration_dir / "status"

        # State
        self._settings: Optional[IntegrationSettings] = None
        self._manifest: Optional[ModelManifest] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._watching = False
        self._lock_file: Optional[Path] = None

    def setup(self) -> bool:
        """Initialize the integration directory structure.

        Returns:
            True if setup successful
        """
        try:
            # Create directory structure
            for directory in [
                self.pending_dir,
                self.processed_dir,
                self.failed_dir,
                self.models_dir / "latest",
                self.models_dir / "archive",
                self.config_dir,
                self.status_dir,
            ]:
                directory.mkdir(parents=True, exist_ok=True)

            # Initialize settings if not exists
            settings_path = self.config_dir / "settings.json"
            if not settings_path.exists():
                self._settings = IntegrationSettings()
                self._settings.updated_at = datetime.now(timezone.utc).isoformat()
                self._settings.updated_by = "bashgym"
                self._save_settings()
            else:
                self._settings = self._load_settings()

            # Initialize model manifest if not exists
            manifest_path = self.models_dir / "manifest.json"
            if not manifest_path.exists():
                self._manifest = ModelManifest()
                self._save_manifest()
            else:
                self._manifest = self._load_manifest()

            # Write bashgym status
            self._update_status()

            logger.info(f"Bashbros integration initialized at {self.integration_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to setup integration: {e}")
            return False

    def is_linked(self) -> bool:
        """Check if bashbros is linked."""
        settings = self.get_settings()
        return settings.enabled and settings.linked_at is not None

    def link(self) -> bool:
        """Link bashgym with bashbros.

        Called when bashbros runs `bashbros init` and selects to link.
        """
        settings = self.get_settings()
        settings.enabled = True
        settings.linked_at = datetime.now(timezone.utc).isoformat()
        settings.updated_by = "bashgym"
        self._settings = settings
        self._save_settings()

        # Broadcast via WebSocket
        self._broadcast_event("linked")

        logger.info("Bashbros integration linked")
        return True

    def unlink(self) -> bool:
        """Unlink bashgym from bashbros."""
        settings = self.get_settings()
        settings.enabled = False
        settings.linked_at = None
        settings.updated_by = "bashgym"
        self._settings = settings
        self._save_settings()

        # Broadcast via WebSocket
        self._broadcast_event("unlinked")

        logger.info("Bashbros integration unlinked")
        return True

    def _broadcast_event(self, event_type: str, **kwargs) -> None:
        """Broadcast an integration event via WebSocket (non-blocking)."""
        try:
            import asyncio
            from bashgym.api.websocket import (
                broadcast_integration_linked,
                broadcast_integration_unlinked,
                broadcast_integration_trace_received,
                broadcast_integration_trace_processed,
                broadcast_integration_model_exported,
                broadcast_integration_model_rollback,
                broadcast_integration_training_triggered,
            )

            coro = None
            if event_type == "linked":
                coro = broadcast_integration_linked()
            elif event_type == "unlinked":
                coro = broadcast_integration_unlinked()
            elif event_type == "trace_received":
                coro = broadcast_integration_trace_received(**kwargs)
            elif event_type == "trace_processed":
                coro = broadcast_integration_trace_processed(**kwargs)
            elif event_type == "model_exported":
                coro = broadcast_integration_model_exported(**kwargs)
            elif event_type == "model_rollback":
                coro = broadcast_integration_model_rollback(**kwargs)
            elif event_type == "training_triggered":
                coro = broadcast_integration_training_triggered(**kwargs)

            if coro:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(coro, loop=loop)
                    else:
                        asyncio.run(coro)
                except RuntimeError:
                    asyncio.run(coro)

        except ImportError:
            pass  # WebSocket module not available
        except Exception as e:
            logger.debug(f"Failed to broadcast event {event_type}: {e}")

    # =========================================================================
    # Settings Management
    # =========================================================================

    def get_settings(self) -> IntegrationSettings:
        """Get current integration settings."""
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings

    def update_settings(self, updates: Dict[str, Any]) -> IntegrationSettings:
        """Update integration settings.

        Args:
            updates: Dictionary of settings to update

        Returns:
            Updated settings
        """
        settings = self.get_settings()

        # Apply updates
        if "capture" in updates:
            capture = updates["capture"]
            if "mode" in capture:
                try:
                    settings.capture_mode = CaptureMode(capture["mode"])
                except ValueError:
                    pass
            if "auto_stream" in capture:
                settings.auto_stream = capture["auto_stream"]

        if "training" in updates:
            training = updates["training"]
            if "auto_enabled" in training:
                settings.auto_training_enabled = training["auto_enabled"]
            if "quality_threshold" in training:
                settings.quality_threshold = training["quality_threshold"]
            if "trigger" in training:
                try:
                    settings.trigger = TrainingTrigger(training["trigger"])
                except ValueError:
                    pass

        if "security" in updates:
            security = updates["security"]
            if "bashbros_primary" in security:
                settings.bashbros_primary = security["bashbros_primary"]
            if "policy_path" in security:
                settings.policy_path = security["policy_path"]

        if "model_sync" in updates:
            model_sync = updates["model_sync"]
            if "auto_export_ollama" in model_sync:
                settings.auto_export_ollama = model_sync["auto_export_ollama"]
            if "ollama_model_name" in model_sync:
                settings.ollama_model_name = model_sync["ollama_model_name"]
            if "notify_on_update" in model_sync:
                settings.notify_on_update = model_sync["notify_on_update"]

        settings.updated_at = datetime.now(timezone.utc).isoformat()
        settings.updated_by = "bashgym"
        self._settings = settings
        self._save_settings()

        return settings

    def _load_settings(self) -> IntegrationSettings:
        """Load settings from file."""
        settings_path = self.config_dir / "settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    data = json.load(f)
                return IntegrationSettings.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
        return IntegrationSettings()

    def _save_settings(self) -> None:
        """Save settings to file with locking."""
        settings_path = self.config_dir / "settings.json"
        lock_path = self.config_dir / "settings.lock"

        # Acquire lock
        self._acquire_lock(lock_path)
        try:
            with open(settings_path, 'w') as f:
                json.dump(self._settings.to_dict(), f, indent=2)
        finally:
            self._release_lock(lock_path)

    def _acquire_lock(self, lock_path: Path) -> None:
        """Acquire file lock (platform-specific)."""
        lock_path.touch()
        self._lock_file = lock_path

        if platform.system() == "Windows":
            import msvcrt
            self._lock_fd = open(lock_path, 'r+')
            msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            self._lock_fd = open(lock_path, 'r+')
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)

    def _release_lock(self, lock_path: Path) -> None:
        """Release file lock."""
        if hasattr(self, '_lock_fd') and self._lock_fd:
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None

    # =========================================================================
    # Trace Management
    # =========================================================================

    def start_watching(self) -> None:
        """Start watching for new traces from bashbros."""
        if self._watching:
            return

        self._watching = True
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()
        logger.info("Started watching for bashbros traces")

    def stop_watching(self) -> None:
        """Stop watching for traces."""
        self._watching = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
            self._watcher_thread = None
        logger.info("Stopped watching for bashbros traces")

    def _watch_loop(self) -> None:
        """Background loop to watch for new traces."""
        while self._watching:
            try:
                self._process_pending_traces()
            except Exception as e:
                logger.error(f"Error in trace watcher: {e}")

            # Sleep between checks
            for _ in range(50):  # 5 second check interval, but responsive stop
                if not self._watching:
                    break
                import time
                time.sleep(0.1)

    def _process_pending_traces(self) -> int:
        """Process all pending traces.

        Returns:
            Number of traces processed
        """
        if not self.pending_dir.exists():
            return 0

        processed = 0
        for trace_file in self.pending_dir.glob("*.json"):
            try:
                # Process the trace
                if self._process_trace(trace_file):
                    # Move to processed
                    dest = self.processed_dir / trace_file.name
                    shutil.move(str(trace_file), str(dest))
                    processed += 1
                else:
                    # Move to failed
                    dest = self.failed_dir / trace_file.name
                    shutil.move(str(trace_file), str(dest))

            except Exception as e:
                logger.error(f"Failed to process trace {trace_file}: {e}")
                # Move to failed
                dest = self.failed_dir / trace_file.name
                try:
                    shutil.move(str(trace_file), str(dest))
                except Exception:
                    pass

        if processed > 0:
            self._update_status()
            logger.info(f"Processed {processed} traces from bashbros")

        return processed

    def _process_trace(self, trace_path: Path) -> bool:
        """Process a single trace file.

        Args:
            trace_path: Path to trace JSON file

        Returns:
            True if processing successful
        """
        try:
            with open(trace_path, 'r') as f:
                trace_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load trace {trace_path}: {e}")
            self._broadcast_event("trace_processed",
                filename=trace_path.name,
                success=False,
                error=str(e)
            )
            return False

        # Validate trace format
        if not self._validate_trace(trace_data):
            logger.warning(f"Invalid trace format: {trace_path}")
            self._broadcast_event("trace_processed",
                filename=trace_path.name,
                success=False,
                error="Invalid trace format"
            )
            return False

        metadata = trace_data.get("metadata", {})
        task = metadata.get("user_initial_prompt", "")
        steps = len(trace_data.get("trace", []))
        verified = metadata.get("verification_passed", False)

        # Broadcast trace received
        self._broadcast_event("trace_received",
            filename=trace_path.name,
            task=task,
            source=metadata.get("source_tool", "bashbros"),
            steps=steps,
            verified=verified
        )

        # Check capture mode filter
        settings = self.get_settings()
        if settings.capture_mode == CaptureMode.SUCCESSFUL_ONLY:
            if not verified:
                logger.debug(f"Skipping unverified trace: {trace_path}")
                return True  # Still counts as processed (intentionally skipped)

        # Invoke callback if registered
        if self.trace_callback:
            try:
                self.trace_callback(trace_path)
            except Exception as e:
                logger.error(f"Trace callback error: {e}")

        # Broadcast processed
        self._broadcast_event("trace_processed",
            filename=trace_path.name,
            success=True
        )

        return True

    def _validate_trace(self, trace_data: Dict[str, Any]) -> bool:
        """Validate trace format is compatible with bashgym.

        Required fields:
        - metadata.user_initial_prompt (task prompt)
        - trace (list of tool calls)
        """
        metadata = trace_data.get("metadata", {})

        # user_initial_prompt is REQUIRED
        if not metadata.get("user_initial_prompt"):
            return False

        # trace should be a list
        trace = trace_data.get("trace", [])
        if not isinstance(trace, list):
            return False

        # Validate each step has required fields
        for step in trace:
            if not isinstance(step, dict):
                return False
            # tool_name or tool is required
            if not (step.get("tool_name") or step.get("tool")):
                return False

        return True

    def get_pending_count(self) -> int:
        """Get count of pending traces."""
        if not self.pending_dir.exists():
            return 0
        return len(list(self.pending_dir.glob("*.json")))

    def get_processed_count(self) -> int:
        """Get count of processed traces."""
        if not self.processed_dir.exists():
            return 0
        return len(list(self.processed_dir.glob("*.json")))

    def list_pending_traces(self) -> List[Dict[str, Any]]:
        """List pending traces with metadata."""
        traces = []
        if not self.pending_dir.exists():
            return traces

        for trace_file in self.pending_dir.glob("*.json"):
            try:
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                traces.append({
                    "filename": trace_file.name,
                    "task": metadata.get("user_initial_prompt", "")[:100],
                    "source": metadata.get("source_tool", "unknown"),
                    "verified": metadata.get("verification_passed", False),
                    "steps": len(data.get("trace", [])),
                })
            except Exception:
                pass

        return traces

    # =========================================================================
    # Model Export & Ollama Registration
    # =========================================================================

    def export_to_gguf(
        self,
        model_path: Path,
        quantization: str = "q4_k_m",
        traces_used: int = 0,
        quality_avg: float = 0.0,
    ) -> Optional[Path]:
        """Export a trained model to GGUF format and register with Ollama.

        Args:
            model_path: Path to the trained model (merged weights)
            quantization: GGUF quantization level
            traces_used: Number of traces used in training
            quality_avg: Average quality score of training traces

        Returns:
            Path to GGUF file, or None if export failed
        """
        settings = self.get_settings()
        manifest = self._load_manifest()

        # Determine version
        current_version_num = 0
        if manifest.latest and manifest.latest.startswith("v"):
            try:
                current_version_num = int(manifest.latest[1:])
            except ValueError:
                pass
        new_version = f"v{current_version_num + 1}"

        # Output path
        gguf_filename = f"sidekick-{new_version}.gguf"
        gguf_path = self.models_dir / gguf_filename

        logger.info(f"Exporting model to GGUF: {gguf_path}")

        # Generate and run export script
        try:
            # Try using llama-cpp-python for conversion
            export_success = self._run_gguf_export(model_path, gguf_path, quantization)

            if not export_success:
                logger.error("GGUF export failed")
                return None

        except Exception as e:
            logger.error(f"GGUF export error: {e}")
            return None

        # Update manifest
        new_model_version = ModelVersion(
            version=new_version,
            created=datetime.now(timezone.utc).isoformat(),
            traces_used=traces_used,
            quality_avg=quality_avg,
            gguf_path=str(gguf_path),
        )
        manifest.versions.append(new_model_version)
        manifest.latest = new_version
        manifest.rollback_available = len(manifest.versions) > 1
        self._manifest = manifest
        self._save_manifest()

        # Update latest symlink
        latest_link = self.models_dir / "latest" / "sidekick.gguf"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create symlink (or copy on Windows if symlinks not supported)
        try:
            latest_link.symlink_to(gguf_path)
        except OSError:
            # Fallback to copy on Windows
            shutil.copy2(gguf_path, latest_link)

        # Archive old versions (keep last 3)
        self._archive_old_versions(keep=3)

        # Register with Ollama if enabled
        ollama_registered = False
        if settings.auto_export_ollama:
            ollama_registered = self._register_with_ollama(gguf_path, settings.ollama_model_name, new_version)

        # Update status
        self._update_status()

        # Broadcast model exported event
        self._broadcast_event("model_exported",
            version=new_version,
            gguf_path=str(gguf_path),
            ollama_registered=ollama_registered,
            traces_used=traces_used,
            quality_avg=quality_avg
        )

        logger.info(f"Model exported to GGUF: {gguf_path}")
        return gguf_path

    def _run_gguf_export(self, model_path: Path, output_path: Path, quantization: str) -> bool:
        """Run the actual GGUF export.

        Returns:
            True if successful
        """
        # First, try llama.cpp's convert script
        try:
            # Convert to F16 GGUF first
            f16_path = output_path.parent / "temp_f16.gguf"

            result = subprocess.run(
                [
                    "python", "-m", "llama_cpp.llama_convert",
                    str(model_path),
                    "--outfile", str(f16_path),
                    "--outtype", "f16"
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                logger.warning(f"llama_cpp.llama_convert failed: {result.stderr}")
                raise FileNotFoundError("llama_cpp not available")

            # Then quantize
            result = subprocess.run(
                [
                    "llama-quantize",
                    str(f16_path),
                    str(output_path),
                    quantization
                ],
                capture_output=True,
                text=True,
                timeout=600
            )

            # Clean up temp file
            if f16_path.exists():
                f16_path.unlink()

            if result.returncode == 0:
                return True
            else:
                logger.warning(f"llama-quantize failed: {result.stderr}")

        except FileNotFoundError:
            logger.info("llama.cpp not found, trying alternative export...")
        except subprocess.TimeoutExpired:
            logger.error("GGUF export timed out")
            return False
        except Exception as e:
            logger.warning(f"llama.cpp export failed: {e}")

        # Fallback: Use unsloth's GGUF export if available
        try:
            export_script = f'''
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("{model_path}")
model.save_pretrained_gguf(
    "{output_path.parent}",
    tokenizer,
    quantization_method="{quantization}"
)
'''
            result = subprocess.run(
                ["python", "-c", export_script],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                # Rename output file if needed
                expected_output = output_path.parent / f"unsloth.{quantization.upper()}.gguf"
                if expected_output.exists() and not output_path.exists():
                    shutil.move(expected_output, output_path)
                return output_path.exists()

        except Exception as e:
            logger.warning(f"Unsloth GGUF export failed: {e}")

        logger.error("All GGUF export methods failed")
        return False

    def _register_with_ollama(self, gguf_path: Path, model_name: str, version: str) -> bool:
        """Register the GGUF model with Ollama.

        Args:
            gguf_path: Path to GGUF file
            model_name: Name to register with Ollama
            version: Version tag

        Returns:
            True if registration successful
        """
        # Check if Ollama is available
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("Ollama not available")
                return False
        except FileNotFoundError:
            logger.warning("Ollama not installed")
            return False
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")
            return False

        # Create Modelfile
        modelfile_path = gguf_path.parent / "Modelfile"
        modelfile_content = f'''FROM {gguf_path}

SYSTEM """You are a helpful AI coding assistant trained on successful coding sessions.
You help users write, debug, and understand code."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
'''
        modelfile_path.write_text(modelfile_content)

        # Register with Ollama
        try:
            full_name = f"{model_name}:{version}"
            result = subprocess.run(
                ["ollama", "create", full_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"Registered model with Ollama: {full_name}")

                # Also tag as 'latest'
                subprocess.run(
                    ["ollama", "cp", full_name, f"{model_name}:latest"],
                    capture_output=True,
                    timeout=30
                )
                return True
            else:
                logger.warning(f"Ollama registration failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Ollama registration timed out")
            return False
        except Exception as e:
            logger.error(f"Ollama registration error: {e}")
            return False

    def _archive_old_versions(self, keep: int = 3) -> None:
        """Archive old model versions, keeping only the most recent.

        Args:
            keep: Number of versions to keep in main directory
        """
        manifest = self._load_manifest()
        if len(manifest.versions) <= keep:
            return

        # Sort by version number (descending)
        sorted_versions = sorted(
            manifest.versions,
            key=lambda v: int(v.version[1:]) if v.version.startswith("v") else 0,
            reverse=True
        )

        # Archive older versions
        archive_dir = self.models_dir / "archive"
        for version in sorted_versions[keep:]:
            if version.gguf_path:
                gguf_file = Path(version.gguf_path)
                if gguf_file.exists():
                    archive_path = archive_dir / gguf_file.name
                    shutil.move(str(gguf_file), str(archive_path))
                    version.gguf_path = str(archive_path)

        self._manifest = manifest
        self._save_manifest()

    def get_model_versions(self) -> List[Dict[str, Any]]:
        """Get list of available model versions."""
        manifest = self._load_manifest()
        return [
            {
                "version": v.version,
                "created": v.created,
                "traces_used": v.traces_used,
                "quality_avg": v.quality_avg,
                "is_latest": v.version == manifest.latest,
                "gguf_available": v.gguf_path and Path(v.gguf_path).exists(),
            }
            for v in manifest.versions
        ]

    def rollback_model(self, version: str) -> bool:
        """Rollback to a previous model version.

        Args:
            version: Version to rollback to (e.g., "v2")

        Returns:
            True if rollback successful
        """
        manifest = self._load_manifest()

        # Find the version
        target_version = None
        for v in manifest.versions:
            if v.version == version:
                target_version = v
                break

        if not target_version:
            logger.error(f"Version {version} not found")
            return False

        if not target_version.gguf_path or not Path(target_version.gguf_path).exists():
            logger.error(f"GGUF file not found for version {version}")
            return False

        # Update latest symlink
        latest_link = self.models_dir / "latest" / "sidekick.gguf"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        try:
            latest_link.symlink_to(target_version.gguf_path)
        except OSError:
            shutil.copy2(target_version.gguf_path, latest_link)

        # Update manifest
        manifest.latest = version
        self._manifest = manifest
        self._save_manifest()

        # Re-register with Ollama
        settings = self.get_settings()
        if settings.auto_export_ollama:
            self._register_with_ollama(
                Path(target_version.gguf_path),
                settings.ollama_model_name,
                version
            )

        # Broadcast rollback event
        previous_version = manifest.latest if manifest.latest != version else None
        self._broadcast_event("model_rollback",
            version=version,
            previous_version=previous_version
        )

        logger.info(f"Rolled back to model version {version}")
        return True

    def _load_manifest(self) -> ModelManifest:
        """Load model manifest from file."""
        manifest_path = self.models_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                return ModelManifest.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
        return ModelManifest()

    def _save_manifest(self) -> None:
        """Save model manifest to file."""
        manifest_path = self.models_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self._manifest.to_dict(), f, indent=2)

    # =========================================================================
    # Status Management
    # =========================================================================

    def get_status(self) -> IntegrationStatus:
        """Get current integration status."""
        status = IntegrationStatus()

        # Check bashbros connection (presence of recent heartbeat)
        bashbros_status = self.status_dir / "bashbros.json"
        if bashbros_status.exists():
            try:
                with open(bashbros_status, 'r') as f:
                    data = json.load(f)
                # Check if heartbeat is recent (within 5 minutes)
                heartbeat = data.get("heartbeat")
                if heartbeat:
                    heartbeat_time = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
                    age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
                    status.bashbros_connected = age < 300
            except Exception:
                pass

        status.bashgym_connected = True  # We're running
        status.pending_traces = self.get_pending_count()
        status.processed_traces = self.get_processed_count()

        # Get current model version
        manifest = self._load_manifest()
        status.current_model_version = manifest.latest

        # Check training status
        training_status = self.status_dir / "training.json"
        if training_status.exists():
            try:
                with open(training_status, 'r') as f:
                    data = json.load(f)
                status.training_in_progress = data.get("state") == "running"
            except Exception:
                pass

        return status

    def _update_status(self) -> None:
        """Update bashgym status file."""
        status_path = self.status_dir / "bashgym.json"
        status_data = {
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "pending_traces": self.get_pending_count(),
            "processed_traces": self.get_processed_count(),
            "model_version": self._load_manifest().latest if self._manifest else None,
        }

        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)

    def update_training_status(self, state: str, run_id: Optional[str] = None, model: Optional[str] = None) -> None:
        """Update training status file.

        Args:
            state: Training state (running, complete, failed)
            run_id: Current training run ID
            model: Output model name
        """
        status_path = self.status_dir / "training.json"
        status_data = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "model": model,
        }

        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)

    # =========================================================================
    # Security Delegation
    # =========================================================================

    def should_use_bashbros_security(self) -> bool:
        """Check if bashgym should defer to bashbros for security.

        Returns:
            True if bashbros security should be primary
        """
        settings = self.get_settings()
        return settings.enabled and settings.bashbros_primary

    def get_bashbros_policy_path(self) -> Optional[Path]:
        """Get path to bashbros security policy file.

        Returns:
            Path to .bashbros.yml or None
        """
        settings = self.get_settings()
        if settings.policy_path:
            policy_path = Path(settings.policy_path).expanduser()
            if policy_path.exists():
                return policy_path

        # Check default location
        default_policy = Path.home() / ".bashbros.yml"
        if default_policy.exists():
            return default_policy

        return None


# Singleton instance
_integration: Optional[BashbrosIntegration] = None


def get_integration() -> BashbrosIntegration:
    """Get the singleton integration instance."""
    global _integration
    if _integration is None:
        _integration = BashbrosIntegration()
        _integration.setup()
    return _integration


def reset_integration() -> None:
    """Reset the singleton instance (for testing)."""
    global _integration
    if _integration:
        _integration.stop_watching()
    _integration = None
