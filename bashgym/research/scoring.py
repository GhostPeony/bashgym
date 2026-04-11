"""Pure rule-based scoring for HF dataset candidates. No I/O, no network."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from bashgym.research.contracts import (
    BLOCKED_LICENSE_PREFIXES,
    COLUMN_MAP_HINTS,
    FRESHNESS_FULL_DAYS,
    FRESHNESS_STALE_DAYS,
    PERMISSIVE_LICENSES,
    POPULARITY_SATURATION_DOWNLOADS,
    SCHEMA_PATTERNS,
    SIZE_IDEAL_MAX,
    SIZE_IDEAL_MIN,
    SIZE_MAX_HARD,
    SIZE_MIN_HARD,
    TASK_TAGS,
    WARN_LICENSES,
    WEIGHTS,
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


def _score_task_match(tags: list[str], description: str) -> tuple[float, str]:
    """Return (0-1 score, reason_suffix)."""
    # Normalize tags — HF often wraps them as "task_categories:code-generation".
    flat = {t.split(":")[-1].lower().strip() for t in tags}
    desc_low = description.lower()
    best = 0.0
    matched: str | None = None
    for tag, weight in TASK_TAGS.items():
        if tag in flat:
            if weight > best:
                best = weight
                matched = tag
        elif tag in desc_low:
            # Description match is half-credit vs a direct tag.
            if weight * 0.5 > best:
                best = weight * 0.5
                matched = f"{tag} (description)"
    if matched:
        return best, f"task match: '{matched}'"
    return 0.0, "task match: no code/tool-use signal"


def _score_license(license: str | None) -> tuple[float, str, list[str]]:
    """Return (0-1 score, reason, warnings)."""
    if license is None:
        return 0.0, "license: unknown", []
    lic = license.lower().strip()
    if lic in PERMISSIVE_LICENSES:
        return 1.0, f"license: {lic}", []
    if lic in WARN_LICENSES or lic.startswith("cc-by-sa"):
        return 0.6, f"license: {lic} (share-alike)", [f"share-alike license: {lic}"]
    return 0.0, f"license: {lic} (uncertain)", []


def _score_size(num_rows: int | None) -> tuple[float, str]:
    if num_rows is None:
        return 0.3, "size: unknown row count"
    if SIZE_IDEAL_MIN <= num_rows <= SIZE_IDEAL_MAX:
        return 1.0, f"size: {num_rows:,} rows (ideal window)"
    if num_rows < SIZE_IDEAL_MIN:
        # Between SIZE_MIN_HARD and SIZE_IDEAL_MIN — linear ramp.
        frac = (num_rows - SIZE_MIN_HARD) / (SIZE_IDEAL_MIN - SIZE_MIN_HARD)
        return max(0.2, frac), f"size: {num_rows:,} rows (below ideal)"
    # Above SIZE_IDEAL_MAX — decay toward hard cap.
    if num_rows >= SIZE_MAX_HARD:
        return 0.0, f"size: {num_rows:,} rows (above hard cap)"
    frac = 1.0 - (num_rows - SIZE_IDEAL_MAX) / (SIZE_MAX_HARD - SIZE_IDEAL_MAX)
    return max(0.1, frac), f"size: {num_rows:,} rows (oversized, downweighted)"


def _score_freshness(last_modified: datetime | None) -> tuple[float, str]:
    if last_modified is None:
        return 0.3, "freshness: unknown last_modified"
    now = datetime.now(timezone.utc) if last_modified.tzinfo else datetime.now()
    age_days = (now - last_modified).days
    if age_days <= FRESHNESS_FULL_DAYS:
        return 1.0, f"freshness: updated {age_days}d ago"
    if age_days >= FRESHNESS_STALE_DAYS:
        return 0.1, f"freshness: stale ({age_days}d old)"
    frac = 1.0 - (age_days - FRESHNESS_FULL_DAYS) / (FRESHNESS_STALE_DAYS - FRESHNESS_FULL_DAYS)
    return frac, f"freshness: updated {age_days}d ago"


def _score_popularity(downloads: int) -> tuple[float, str]:
    frac = min(1.0, downloads / POPULARITY_SATURATION_DOWNLOADS)
    return frac, f"popularity: {downloads:,} downloads"


def _infer_format(features: dict[str, str]) -> str | None:
    cols = frozenset(features.keys())
    for required, format_name in SCHEMA_PATTERNS:
        if required.issubset(cols):
            return format_name
    return None


def _score_schema(features: dict[str, str], bashgym_format: str | None) -> tuple[float, str]:
    if bashgym_format is None:
        visible = sorted(features.keys())[:6]
        return 0.0, f"schema: no match (cols={visible})"
    return 1.0, f"schema: maps to {bashgym_format.upper()}"


def _build_download_command(meta: DatasetMetadata, bashgym_format: str | None) -> str:
    """Emit a Python expression string that ingests this dataset via DataDesignerPipeline."""
    pipeline_name = {
        "sft": "coding_agent_sft",
        "dpo": "coding_agent_dpo",
        "grpo": "coding_agent_sft",
    }.get(bashgym_format or "", "coding_agent_sft")

    col_map: dict[str, str] = {}
    for hf_col in meta.features.keys():
        if hf_col in COLUMN_MAP_HINTS:
            col_map[hf_col] = COLUMN_MAP_HINTS[hf_col]

    col_map_repr = repr(col_map) if col_map else "None"
    return (
        f"DataDesignerPipeline(PipelineConfig(pipeline='{pipeline_name}', "
        f"num_records=1000)).from_dataset("
        f"source='{meta.repo_id}', split='train', column_mapping={col_map_repr})"
    )


def score_dataset(meta: DatasetMetadata) -> ScoredDataset:
    """Score an HF dataset against bashgym's needs.

    Hard filters (gated, non-commercial license, size out of bounds) cause
    immediate rejection. Otherwise runs the full weighted scoring pipeline.
    """
    # --- Hard filters ---
    if meta.gated:
        return ScoredDataset(
            repo_id=meta.repo_id, score=0.0, reasons=[], warnings=[],
            bashgym_format=None, download_command="", metadata=meta,
            rejected=True, rejection_reason="dataset is gated behind approval form",
        )
    if _license_blocked(meta.license):
        return ScoredDataset(
            repo_id=meta.repo_id, score=0.0, reasons=[], warnings=[],
            bashgym_format=None, download_command="", metadata=meta,
            rejected=True,
            rejection_reason=f"license '{meta.license}' is non-commercial or unknown",
        )
    if meta.num_rows is not None and meta.num_rows < SIZE_MIN_HARD:
        return ScoredDataset(
            repo_id=meta.repo_id, score=0.0, reasons=[], warnings=[],
            bashgym_format=None, download_command="", metadata=meta,
            rejected=True,
            rejection_reason=f"too small: {meta.num_rows} rows (min {SIZE_MIN_HARD})",
        )
    if meta.num_rows is not None and meta.num_rows > SIZE_MAX_HARD:
        return ScoredDataset(
            repo_id=meta.repo_id, score=0.0, reasons=[], warnings=[],
            bashgym_format=None, download_command="", metadata=meta,
            rejected=True,
            rejection_reason=f"too large: {meta.num_rows} rows (max {SIZE_MAX_HARD})",
        )

    # --- Weighted dimensions ---
    reasons: list[str] = []
    warnings: list[str] = []

    task_score, task_reason = _score_task_match(meta.tags, meta.description)
    lic_score, lic_reason, lic_warnings = _score_license(meta.license)
    warnings.extend(lic_warnings)
    size_score, size_reason = _score_size(meta.num_rows)
    fresh_score, fresh_reason = _score_freshness(meta.last_modified)
    pop_score, pop_reason = _score_popularity(meta.downloads)

    bashgym_format = _infer_format(meta.features)
    schema_score, schema_reason = _score_schema(meta.features, bashgym_format)

    raw = (
        task_score * WEIGHTS["task_match"]
        + lic_score * WEIGHTS["license"]
        + size_score * WEIGHTS["size"]
        + schema_score * WEIGHTS["schema"]
        + fresh_score * WEIGHTS["freshness"]
        + pop_score * WEIGHTS["popularity"]
    )
    final_score = round(raw * 10, 2)

    def _fmt(weight_key: str, subscore: float, reason: str) -> str:
        contrib = subscore * WEIGHTS[weight_key] * 10
        return f"+{contrib:.2f} {reason}"

    reasons.append(_fmt("task_match", task_score, task_reason))
    reasons.append(_fmt("license", lic_score, lic_reason))
    reasons.append(_fmt("size", size_score, size_reason))
    reasons.append(_fmt("schema", schema_score, schema_reason))
    reasons.append(_fmt("freshness", fresh_score, fresh_reason))
    reasons.append(_fmt("popularity", pop_score, pop_reason))

    download_command = _build_download_command(meta, bashgym_format)

    return ScoredDataset(
        repo_id=meta.repo_id,
        score=final_score,
        reasons=reasons,
        warnings=warnings,
        bashgym_format=bashgym_format,
        download_command=download_command,
        metadata=meta,
    )
