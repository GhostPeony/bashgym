"""Stable contracts for Hugging Face evidence sent through the BashGym canvas."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from bashgym._compat import UTC

SCHEMA_VERSION = "2"
SCORING_RULE_VERSION = "hf-context-rank-v1"


def utc_now() -> datetime:
    return datetime.now(UTC)


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FrozenContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceKind(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    EVALUATION = "evaluation"


class Visibility(str, Enum):
    PUBLIC = "public"
    WORKSPACE_PRIVATE = "workspace_private"


class Lifecycle(str, Enum):
    COLLECTING = "collecting"
    READY = "ready"


class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"


class CompletionOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class Comparability(str, Enum):
    COMPARABLE = "comparable"
    PARTIAL = "partial"
    ORIENTATION_ONLY = "orientation_only"
    UNKNOWN = "unknown"


_COMPARABILITY_SCORE = {
    Comparability.COMPARABLE: 3,
    Comparability.PARTIAL: 2,
    Comparability.ORIENTATION_ONLY: 1,
    Comparability.UNKNOWN: 0,
}


class Provenance(FrozenContractModel):
    source: str = Field(min_length=1, max_length=160)
    retrieved_at: datetime = Field(default_factory=utc_now)
    source_url: str | None = Field(default=None, max_length=2048)
    source_revision: str | None = Field(default=None, max_length=200)

    @field_validator("source_url")
    @classmethod
    def source_url_must_be_https(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://"):
            raise ValueError("source_url must use https")
        return value


class EvalSettings(FrozenContractModel):
    benchmark_id: str = Field(min_length=1, max_length=200)
    task_revision: str | None = Field(default=None, max_length=200)
    metric: str = Field(min_length=1, max_length=160)
    prompt_template: str | None = Field(default=None, max_length=200)
    few_shot: int | None = Field(default=None, ge=0, le=10_000)
    harness: str | None = Field(default=None, max_length=160)
    harness_version: str | None = Field(default=None, max_length=160)
    backend: str | None = Field(default=None, max_length=160)
    sampling: dict[str, Any] | None = None


class EvidenceAssessment(FrozenContractModel):
    task_relevance: int = Field(default=0, ge=0, le=3)
    compatibility: int = Field(default=0, ge=0, le=3)
    constraint_passes: int = Field(default=0, ge=0, le=16)
    constraint_violations: tuple[str, ...] = ()
    comparability: Comparability = Comparability.UNKNOWN
    confidence: float = Field(default=0.0, ge=0, le=1)
    rationale: str | None = Field(default=None, max_length=1000)
    rule_version: str = SCORING_RULE_VERSION


class EvidenceRecord(FrozenContractModel):
    evidence_id: Identifier
    kind: EvidenceKind
    resource_id: str = Field(min_length=1, max_length=300)
    revision: str | None = Field(default=None, max_length=200)
    canonical_url: str = Field(min_length=1, max_length=2048)
    summary: str = Field(default="", max_length=2000)
    facts: dict[str, Any] = Field(default_factory=dict)
    excerpt: str | None = Field(default=None, max_length=20_000)
    excerpt_kind: Literal["verbatim", "generated_summary"] = "verbatim"
    visibility: Visibility = Visibility.PUBLIC
    provenance: Provenance
    assessment: EvidenceAssessment = Field(default_factory=EvidenceAssessment)
    eval_settings: EvalSettings | None = None
    cautions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reject_private_excerpt_persistence(self) -> EvidenceRecord:
        if self.visibility is Visibility.WORKSPACE_PRIVATE and self.excerpt is not None:
            raise ValueError(
                "workspace-private evidence excerpts are blocked until revocable encrypted payloads are available"
            )
        return self

    @field_validator("canonical_url")
    @classmethod
    def canonical_url_must_be_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("canonical_url must use https")
        return value


class SourceStatus(FrozenContractModel):
    source: str = Field(min_length=1, max_length=160)
    status: Literal["pending", "complete", "partial", "failed"]
    result_count: int = Field(default=0, ge=0)
    error_code: str | None = Field(default=None, max_length=160)
    safe_message: str | None = Field(default=None, max_length=1000)


class HFContextBundle(FrozenContractModel):
    schema_version: str = SCHEMA_VERSION
    scoring_rule_version: str = SCORING_RULE_VERSION
    bundle_id: Identifier
    version: int = Field(ge=1)
    workspace_id: Identifier
    lifecycle: Lifecycle = Lifecycle.COLLECTING
    freshness: Freshness = Freshness.FRESH
    completion_outcome: CompletionOutcome | None = None
    intent: str = Field(min_length=1, max_length=2000)
    task: str | None = Field(default=None, max_length=200)
    target: dict[str, Any] = Field(default_factory=dict)
    evidence: tuple[EvidenceRecord, ...] = Field(default=(), max_length=12)
    selected_evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=12)
    warnings: tuple[str, ...] = Field(default=(), max_length=50)
    source_status: tuple[SourceStatus, ...] = ()
    correlation_id: Identifier | None = None
    origin: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    ready_at: datetime | None = None
    content_hash: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def validate_identity_and_hash(self) -> HFContextBundle:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique")
        if not set(self.selected_evidence_ids).issubset(evidence_ids):
            raise ValueError("selected evidence must exist in the bundle")
        if self.lifecycle is Lifecycle.COLLECTING and self.completion_outcome is not None:
            raise ValueError("collecting bundles cannot have a completion outcome")

        expected = canonical_hash(self.content_payload())
        if self.content_hash and self.content_hash != expected:
            raise ValueError("content_hash does not match bundle content")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", expected)
        return self

    def content_payload(self) -> dict[str, Any]:
        excluded = {"content_hash"}
        if self.schema_version != "1":
            excluded.add("freshness")
        return self.model_dump(mode="json", exclude=excluded)


def classify_comparability(candidate: EvalSettings, reference: EvalSettings) -> Comparability:
    """Conservatively compare published eval settings with a reference recipe."""

    if candidate.benchmark_id != reference.benchmark_id or candidate.metric != reference.metric:
        return Comparability.ORIENTATION_ONLY
    if candidate.harness != reference.harness:
        return Comparability.ORIENTATION_ONLY

    critical = ("task_revision", "prompt_template", "few_shot", "sampling")
    for name in critical:
        candidate_value = getattr(candidate, name)
        reference_value = getattr(reference, name)
        if candidate_value is None or reference_value is None or candidate_value != reference_value:
            return Comparability.ORIENTATION_ONLY

    noncritical_unknown = 0
    noncritical_difference = 0
    for name in ("harness_version", "backend"):
        candidate_value = getattr(candidate, name)
        reference_value = getattr(reference, name)
        if candidate_value is None or reference_value is None:
            noncritical_unknown += 1
        elif candidate_value != reference_value:
            noncritical_difference += 1
    if noncritical_unknown >= 2:
        return Comparability.ORIENTATION_ONLY
    if noncritical_unknown or noncritical_difference:
        return Comparability.PARTIAL
    return Comparability.COMPARABLE


def rank_evidence(items: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Return the deterministic release-one rank, excluding known constraint violations."""

    eligible = [item for item in items if not item.assessment.constraint_violations]

    def key(item: EvidenceRecord) -> tuple[Any, ...]:
        assessment = item.assessment
        popularity = int(item.facts.get("downloads") or item.facts.get("likes") or 0)
        return (
            -assessment.task_relevance,
            -assessment.compatibility,
            -assessment.constraint_passes,
            -_COMPARABILITY_SCORE[assessment.comparability],
            -assessment.confidence,
            -popularity,
            item.resource_id,
            item.revision or "",
        )

    return sorted(eligible, key=key)


__all__ = [
    "Comparability",
    "CompletionOutcome",
    "ContractModel",
    "EvalSettings",
    "EvidenceAssessment",
    "EvidenceKind",
    "EvidenceRecord",
    "Freshness",
    "HFContextBundle",
    "Identifier",
    "Lifecycle",
    "Provenance",
    "SCHEMA_VERSION",
    "SCORING_RULE_VERSION",
    "SourceStatus",
    "Visibility",
    "canonical_hash",
    "classify_comparability",
    "rank_evidence",
    "utc_now",
]
