"""Held-out split for trace eval: freeze a contamination-free evaluation set.

Splits by whole SESSION (in-distribution holdout) or whole REPO (hard
generalization), never by individual example — segments from one session share
context and would leak. Persists a manifest of holdout example hashes so the
training export can assert it never trains on the eval set.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


def example_hash(example: dict) -> str:
    """Stable content hash of a training example."""
    payload = json.dumps(example, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _group_key(example: dict, by: str) -> str:
    meta = example.get("metadata") if isinstance(example.get("metadata"), dict) else {}
    meta = meta or {}
    if by == "repo":
        repo = example.get("primary_repo") or meta.get("primary_repo") or {}
        if isinstance(repo, dict):
            return str(repo.get("name") or "_unknown")
        return str(repo or "_unknown")
    return str(
        example.get("session_id") or meta.get("session_id") or meta.get("trace_id") or "_unknown"
    )


@dataclass
class HoldoutSplit:
    train: list[dict]
    holdout: list[dict]
    holdout_hashes: set[str]
    by: str

    def manifest(self) -> dict:
        return {
            "by": self.by,
            "n_train": len(self.train),
            "n_holdout": len(self.holdout),
            "holdout_hashes": sorted(self.holdout_hashes),
        }


def make_holdout_split(examples, *, by: str = "session", frac: float = 0.1, seed: int = 0):
    """Freeze a holdout split at the group level (whole sessions or repos)."""
    if by not in ("session", "repo"):
        raise ValueError("by must be 'session' or 'repo'")
    groups: dict = {}
    for ex in examples:
        groups.setdefault(_group_key(ex, by), []).append(ex)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_holdout_groups = max(1, round(len(keys) * frac)) if keys else 0
    holdout_keys = set(keys[:n_holdout_groups])

    train: list[dict] = []
    holdout: list[dict] = []
    for k, exs in groups.items():
        (holdout if k in holdout_keys else train).extend(exs)
    return HoldoutSplit(
        train=train,
        holdout=holdout,
        holdout_hashes={example_hash(e) for e in holdout},
        by=by,
    )


def contamination(train_examples, holdout_hashes) -> list[str]:
    """Holdout hashes that ALSO appear in the training set — must be empty before export."""
    train_hashes = {example_hash(e) for e in train_examples}
    return sorted(train_hashes & set(holdout_hashes))
