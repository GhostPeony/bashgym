"""
Orphaned training process monitor.

When the backend restarts while a training subprocess is still running,
we can't re-attach to its stdout pipe. Instead, this monitor polls the
output directory for checkpoint progress and reads HuggingFace's
trainer_state.json to extract metrics.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from bashgym.api.training_state import (
    TrainingRunState, is_process_alive, update_run_state, MAX_LOG_LINES,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5


class OrphanedTrainingMonitor:
    """Monitors an orphaned training process by watching its output directory."""

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}

    def start_monitoring(
        self,
        state: TrainingRunState,
        progress_callback,
        log_callback=None,
    ) -> asyncio.Task:
        """Begin polling for a specific orphaned run.

        Args:
            state: Persisted run state.
            progress_callback: async callable(metrics_dict) — broadcasts progress.
            log_callback: optional async callable(log_line) — broadcasts log lines.
        """
        if state.run_id in self._tasks:
            logger.warning(f"Already monitoring {state.run_id}")
            return self._tasks[state.run_id]

        task = asyncio.create_task(
            self._monitor_loop(state, progress_callback, log_callback),
            name=f"monitor-{state.run_id}",
        )
        self._tasks[state.run_id] = task
        return task

    def stop_monitoring(self, run_id: str):
        task = self._tasks.pop(run_id, None)
        if task and not task.done():
            task.cancel()

    def stop_all(self):
        for run_id in list(self._tasks):
            self.stop_monitoring(run_id)

    # ------------------------------------------------------------------

    async def _monitor_loop(
        self,
        state: TrainingRunState,
        progress_callback,
        log_callback,
    ):
        """Poll output directory until process exits."""
        output_dir = Path(state.output_dir)
        seen_checkpoints: set = set()
        last_step = 0
        last_loss: Optional[float] = None

        # Seed seen checkpoints from what already exists
        for cp in output_dir.glob("checkpoint-*"):
            seen_checkpoints.add(cp.name)

        logger.info(
            f"Monitoring orphaned run {state.run_id} (PID {state.pid}), "
            f"output_dir={output_dir}, existing checkpoints={len(seen_checkpoints)}"
        )

        if log_callback:
            await log_callback(
                f"[Monitor] Reconnected to training run {state.run_id} (PID {state.pid})"
            )

        try:
            while is_process_alive(state.pid):
                metrics = self._scrape_metrics(output_dir, seen_checkpoints)
                if metrics:
                    step = metrics.get("step", 0)
                    loss = metrics.get("loss")
                    if step > last_step or (loss is not None and loss != last_loss):
                        last_step = step
                        last_loss = loss
                        await progress_callback(metrics)

                        # Persist latest metrics to disk
                        update_run_state(
                            state.output_dir,
                            last_metrics=metrics,
                        )

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

            # Process exited — determine outcome
            final_dir = output_dir / "final"
            merged_dir = output_dir / "merged"
            if final_dir.exists() or merged_dir.exists():
                logger.info(f"Training {state.run_id} completed (final/ found)")
                update_run_state(state.output_dir, status="completed")
                # Send one final progress so frontend can update
                final_metrics = self._scrape_metrics(output_dir, seen_checkpoints)
                if final_metrics and progress_callback:
                    await progress_callback(final_metrics)
            else:
                logger.warning(f"Training {state.run_id} process exited without final/ dir — marking failed")
                update_run_state(state.output_dir, status="failed")

        except asyncio.CancelledError:
            logger.info(f"Monitor for {state.run_id} cancelled")
        except Exception as e:
            logger.error(f"Monitor error for {state.run_id}: {e}", exc_info=True)
        finally:
            self._tasks.pop(state.run_id, None)

    def _scrape_metrics(
        self, output_dir: Path, seen_checkpoints: set
    ) -> Optional[Dict[str, Any]]:
        """Extract metrics from the latest checkpoint's trainer_state.json."""
        # Find all checkpoints
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: self._checkpoint_step(p),
        )
        if not checkpoints:
            return None

        latest_cp = checkpoints[-1]

        # Track newly seen checkpoints
        for cp in checkpoints:
            seen_checkpoints.add(cp.name)

        trainer_state_path = latest_cp / "trainer_state.json"
        if not trainer_state_path.exists():
            return None

        try:
            data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        # HuggingFace trainer_state.json has log_history with metrics per step
        log_history = data.get("log_history", [])
        if not log_history:
            return {
                "step": self._checkpoint_step(latest_cp),
                "total_steps": data.get("max_steps", 0),
            }

        latest_entry = log_history[-1]
        return {
            "epoch": latest_entry.get("epoch", 0),
            "step": latest_entry.get("step", self._checkpoint_step(latest_cp)),
            "total_steps": data.get("max_steps", 0),
            "loss": latest_entry.get("loss"),
            "learning_rate": latest_entry.get("learning_rate"),
            "grad_norm": latest_entry.get("grad_norm"),
        }

    @staticmethod
    def _checkpoint_step(cp_path: Path) -> int:
        """Extract step number from checkpoint-N directory name."""
        name = cp_path.name  # e.g. "checkpoint-100"
        try:
            return int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            return 0
