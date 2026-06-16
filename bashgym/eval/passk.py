"""pass@k for episode-level (end-to-end) agent eval.

Step-level tool-call scoring (``heldout.py``) measures imitation — did the model
pick the same call the human did. pass@k measures whether the agent actually
SOLVES the task: sample k attempts, run each through the verifier (the Docker
sandbox / ``verify.sh``), and estimate the probability that at least one of k
passes. Uses the unbiased estimator from the Codex/HumanEval paper (Chen et al.
2021) — the same metric NeMo Gym and Terminal-Bench report — so a model that
solves a task on 1 of 10 tries isn't scored the same as one that never does.

This module is verifier-agnostic: the caller injects ``run_episode`` (which runs
one attempt in the sandbox and returns pass/fail), keeping the math hermetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: P(at least one of k sampled attempts passes), given ``c``
    of ``n`` attempts passed.

    ``1 - C(n-c, k) / C(n, k)`` (Chen et al. 2021). Returns 1.0 when ``n-c < k``
    (fewer failures than k, so any draw of k must include a pass), 0.0 when
    ``c == 0`` or inputs are degenerate.
    """
    if k <= 0 or n <= 0 or c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


@dataclass
class EpisodeResult:
    task_id: str
    passes: list[bool]  # one pass/fail per sampled attempt

    @property
    def n(self) -> int:
        return len(self.passes)

    @property
    def c(self) -> int:
        return sum(1 for p in self.passes if p)


@dataclass
class PassKReport:
    k: int
    per_task: dict[str, float]  # task_id -> pass@k

    @property
    def n_tasks(self) -> int:
        return len(self.per_task)

    @property
    def mean_pass_at_k(self) -> float:
        return sum(self.per_task.values()) / self.n_tasks if self.per_task else 0.0

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n_tasks": self.n_tasks,
            "mean_pass_at_k": self.mean_pass_at_k,
            "per_task": self.per_task,
        }


def compute_pass_at_k(results: list[EpisodeResult], k: int) -> PassKReport:
    """Estimate pass@k per task from already-collected attempt outcomes."""
    per_task = {r.task_id: pass_at_k(r.n, r.c, k) for r in results}
    return PassKReport(k=k, per_task=per_task)


def _default_task_id(task: Any) -> str:
    if isinstance(task, dict):
        return str(task.get("id") or task.get("task_id") or task.get("name") or id(task))
    return str(task)


def evaluate_pass_at_k(
    tasks: list,
    run_episode: Callable[[Any, int], bool],
    *,
    n_samples: int,
    k: int,
    task_id: Callable[[Any], str] = _default_task_id,
) -> PassKReport:
    """Sample ``n_samples`` attempts per task via ``run_episode`` and estimate pass@k.

    ``run_episode(task, attempt_index) -> bool`` runs one attempt in the sandbox
    and returns whether it passed the verifier — the seam where a caller plugs in
    the Docker sandbox / ``verify.sh``. ``n_samples`` should be >= ``k``.
    """
    if n_samples < k:
        raise ValueError(f"n_samples ({n_samples}) must be >= k ({k})")
    results: list[EpisodeResult] = []
    for task in tasks:
        passes = [bool(run_episode(task, i)) for i in range(n_samples)]
        results.append(EpisodeResult(task_id=task_id(task), passes=passes))
    return compute_pass_at_k(results, k)
