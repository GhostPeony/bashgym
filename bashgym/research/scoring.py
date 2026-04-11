"""Pure rule-based scoring for HF dataset candidates. No I/O, no network."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from bashgym.research.contracts import (
    BLOCKED_LICENSE_PREFIXES,
    SIZE_MAX_HARD,
    SIZE_MIN_HARD,
)


@dataclass
class DatasetMetadata:
    """Plain-data summary of an HF dataset. All fields optional — HF Hub is inconsistent."""
    repo_id: str
    tags: list[str] = field(default_factory=list)
    license: str | None = None
    num_rows: int | None = None
    download_size_bytes: int | None = None
    features: dict[str, str] = field(default_factory=dict)
    last_modified: datetime | None = None
    downloads: int = 0
    gated: bool = False
    description: str = ""


@dataclass
class ScoredDataset:
    repo_id: str
    score: float
    reasons: list[str]
    warnings: list[str]
    bashgym_format: str | None
    download_command: str
    metadata: DatasetMetadata
    rejected: bool = False
    rejection_reason: str | None = None


def _license_blocked(license: str | None) -> bool:
    """Unknown or non-commercial licenses are hard-rejected."""
    if license is None:
        return True
    lic = license.lower().strip()
    return any(lic.startswith(p) for p in BLOCKED_LICENSE_PREFIXES)


def score_dataset(meta: DatasetMetadata) -> ScoredDataset:
    """Score an HF dataset against bashgym's needs. Returns a ScoredDataset.

    Hard filters (gated, non-commercial license, too small, too large) cause
    immediate rejection with `rejected=True` and `rejection_reason` populated.
    Otherwise, returns a placeholder ScoredDataset — full weighted scoring
    lives in Task 6.
    """
    empty = ScoredDataset(
        repo_id=meta.repo_id,
        score=0.0,
        reasons=[],
        warnings=[],
        bashgym_format=None,
        download_command="",
        metadata=meta,
    )

    if meta.gated:
        empty.rejected = True
        empty.rejection_reason = "dataset is gated behind approval form"
        return empty

    if _license_blocked(meta.license):
        empty.rejected = True
        empty.rejection_reason = f"license '{meta.license}' is non-commercial or unknown"
        return empty

    if meta.num_rows is not None and meta.num_rows < SIZE_MIN_HARD:
        empty.rejected = True
        empty.rejection_reason = f"too small: {meta.num_rows} rows (min {SIZE_MIN_HARD})"
        return empty

    if meta.num_rows is not None and meta.num_rows > SIZE_MAX_HARD:
        empty.rejected = True
        empty.rejection_reason = f"too large: {meta.num_rows} rows (max {SIZE_MAX_HARD})"
        return empty

    return empty
