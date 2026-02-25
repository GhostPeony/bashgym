"""Filesystem watcher for auto-importing Claude Code sessions."""

import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import PipelineConfig


class _SessionFileHandler(FileSystemEventHandler):
    """Watchdog handler that debounces .jsonl file events."""

    def __init__(self, debounce_seconds: int, on_stable: Callable[[Path], None]):
        super().__init__()
        self._debounce = debounce_seconds
        self._on_stable = on_stable
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _handle(self, path_str: str) -> None:
        path = Path(path_str)
        if path.suffix != ".jsonl":
            return

        with self._lock:
            if path_str in self._timers:
                self._timers[path_str].cancel()

            timer = threading.Timer(self._debounce, self._fire, args=[path])
            timer.daemon = True
            self._timers[path_str] = timer
            timer.start()

    def _fire(self, path: Path) -> None:
        with self._lock:
            self._timers.pop(str(path), None)
        if path.exists():
            self._on_stable(path)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)


class ImportWatcher:
    """Watches ~/.claude/projects/ for new session files and triggers import."""

    def __init__(
        self,
        config: PipelineConfig,
        watch_dir: Path,
        on_import: Callable[[Path], None],
    ):
        self.config = config
        self._watch_dir = watch_dir
        self._on_import = on_import
        self._observer: Optional[Observer] = None
        self._running = False

    def start(self) -> None:
        if not self.config.watch_enabled:
            return

        handler = _SessionFileHandler(
            debounce_seconds=self.config.watch_debounce_seconds,
            on_stable=self._on_import,
        )

        self._observer = Observer()
        self._observer.daemon = True

        if self._watch_dir.exists():
            self._observer.schedule(handler, str(self._watch_dir), recursive=True)
        else:
            wait_thread = threading.Thread(
                target=self._wait_for_dir, args=(handler,), daemon=True
            )
            wait_thread.start()

        self._observer.start()
        self._running = True

    def _wait_for_dir(self, handler: _SessionFileHandler) -> None:
        while self._running and not self._watch_dir.exists():
            time.sleep(5)
        if self._running and self._observer and self._watch_dir.exists():
            self._observer.schedule(handler, str(self._watch_dir), recursive=True)

    def stop(self) -> None:
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def reload_config(self, config: PipelineConfig) -> None:
        self.config = config
