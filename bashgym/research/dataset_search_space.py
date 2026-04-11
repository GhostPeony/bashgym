"""DatasetSearchSpace — plug HF dataset scanner output into autoresearch.

This is a SearchSpace that iterates through a pre-scored list of candidate
datasets (from bashgym.research.scoring.ScoredDataset), materializes each
via DataDesignerPipeline.from_dataset() + export_nemo(), runs a short SFT
training run, and returns eval_loss. The AutoResearcher orchestrator ranks
candidates by empirical training impact rather than by static rule-based
scoring.

Scope: SFT format only. Deterministic enumeration — no mutation, no
crossover, no random sampling. The ABC's mutate() method is implemented as
a cursor advance.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bashgym.gym.autoresearch import SearchSpace

logger = logging.getLogger(__name__)


@dataclass
class DatasetCandidate:
    """One candidate dataset in an empirical ranking run.

    Populated in stages:
    - Construction: repo_id, hf_score, bashgym_format
    - After materialization: train_path, val_path, num_rows_generated
    - After evaluation: eval_loss, final_loss (or error)
    """
    repo_id: str
    hf_score: float
    bashgym_format: str
    train_path: Path | None = None
    val_path: Path | None = None
    num_rows_generated: int | None = None
    eval_loss: float | None = None
    final_loss: float | None = None
    error: str | None = None


def _safe_dirname(repo_id: str) -> str:
    """Convert 'org/name' to a filesystem-safe directory name."""
    return repo_id.replace("/", "__").replace(" ", "_")


def _get_data_designer():
    """Lazy import of DataDesignerPipeline. Raises ImportError if missing."""
    try:
        from bashgym.factory.data_designer import DataDesignerPipeline
        return DataDesignerPipeline
    except ImportError as e:
        raise ImportError(
            "DataDesignerPipeline unavailable. Install data-designer: "
            "pip install data-designer>=0.5.0"
        ) from e


def _get_trainer():
    """Lazy import of Trainer. Kept out of module top-level to avoid pulling
    in Unsloth / PyTorch at import time."""
    from bashgym.gym.trainer import Trainer
    return Trainer


class DatasetSearchSpace(SearchSpace):
    """A SearchSpace that enumerates datasets instead of mutating hyperparameters.

    The AutoResearcher contract expects mutate() -> evaluate() -> snapshot. Since
    datasets are discrete choices (not continuous parameters), mutate() advances
    an internal cursor and tags the returned config with the candidate index;
    evaluate() reads that tag and materializes / trains the corresponding
    dataset. Deterministic and idempotent.
    """

    _INDEX_ATTR = "_research_dataset_index"

    def __init__(
        self,
        candidates: list[DatasetCandidate],
        base_trainer_config: Any,
        work_dir: Path,
        num_records: int = 500,
        train_steps: int = 100,
        mode: Literal["simulate", "real"] = "simulate",
        hf_token: str | None = None,
    ):
        if not candidates:
            raise ValueError("candidates must be non-empty")
        self.candidates = list(candidates)
        self._base_config = base_trainer_config
        self._work_dir = work_dir
        self._num_records = num_records
        self._train_steps = train_steps
        self._mode = mode
        self._hf_token = hf_token
        self._cursor = 0

    # ------------------------------------------------------------------
    # SearchSpace ABC
    # ------------------------------------------------------------------

    def mutate(self, config: Any) -> Any:
        """Advance the cursor and return a config tagged with the next index."""
        idx = min(self._cursor, len(self.candidates) - 1)
        self._cursor += 1
        candidate_config = copy.deepcopy(self._base_config)
        setattr(candidate_config, self._INDEX_ATTR, idx)
        return candidate_config

    def evaluate(
        self, config: Any, experiment_number: int, total_experiments: int
    ) -> float:
        raise NotImplementedError("Task 3 implements simulate, Task 4 implements real")

    def get_config_snapshot(self, config: Any) -> dict[str, Any]:
        idx = getattr(config, self._INDEX_ATTR, None)
        if idx is None or not (0 <= idx < len(self.candidates)):
            return {
                "repo_id": None,
                "hf_score": None,
                "bashgym_format": None,
                "num_rows_generated": None,
                "eval_loss": None,
                "final_loss": None,
                "error": None,
            }
        c = self.candidates[idx]
        return {
            "repo_id": c.repo_id,
            "hf_score": c.hf_score,
            "bashgym_format": c.bashgym_format,
            "num_rows_generated": c.num_rows_generated,
            "eval_loss": c.eval_loss,
            "final_loss": c.final_loss,
            "error": c.error,
        }
