"""Leakage-safe development evaluation contracts and deterministic comparison gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import FrozenContractModel, canonical_hash, utc_now


class DevelopmentDataContractError(ValueError):
    code = "campaign_development_data_contract_invalid"


DEVELOPMENT_DATA_CONTRACT_ERROR = DevelopmentDataContractError.code


class ComparisonVerdict(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class ValidatedDevelopmentDataset(FrozenContractModel):
    schema_version: str = "campaign_validated_dev_dataset.v1"
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    video_count: int = Field(ge=1)
    eval_ids: tuple[str, ...]
    characterization_only: bool


class DevelopmentDatasetContract(FrozenContractModel):
    """Pinned physical dev-only input checked before importing a model runtime."""

    schema_version: str = "campaign_dev_dataset_contract.v1"
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_hashes: frozenset[str] = frozenset()
    protected_path_fragments: tuple[str, ...] = ()
    minimum_queries: int = Field(default=300, ge=1)
    minimum_videos: int = Field(default=30, ge=1)

    def validate_file(self, path: Path) -> ValidatedDevelopmentDataset:
        normalized_path = str(path.resolve()).casefold()
        if any(
            fragment.casefold() in normalized_path for fragment in self.protected_path_fragments
        ):
            raise DevelopmentDataContractError(f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: protected path")
        if not path.is_file():
            raise DevelopmentDataContractError(
                f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: dev file unavailable"
            )
        actual_hash = sha256_file(path)
        if actual_hash in self.protected_hashes:
            raise DevelopmentDataContractError(f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: protected hash")
        if actual_hash != self.expected_sha256:
            raise DevelopmentDataContractError(f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: hash mismatch")
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DevelopmentDataContractError(
                        f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: malformed JSONL line {line_number}"
                    ) from exc
                if row.get("split") != "dev":
                    raise DevelopmentDataContractError(
                        f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: non-dev row {line_number}"
                    )
                rows.append(row)
        if not rows:
            raise DevelopmentDataContractError(f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: empty dev set")
        eval_ids = tuple(str(row.get("eval_id") or row.get("query_id") or "") for row in rows)
        if any(not value for value in eval_ids) or len(set(eval_ids)) != len(eval_ids):
            raise DevelopmentDataContractError(
                f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: eval IDs missing or duplicated"
            )
        video_ids = {str(row.get("positive_video_id") or row.get("video_id") or "") for row in rows}
        video_ids.discard("")
        if not video_ids:
            raise DevelopmentDataContractError(
                f"{DEVELOPMENT_DATA_CONTRACT_ERROR}: video IDs missing"
            )
        return ValidatedDevelopmentDataset(
            path=path.resolve(),
            sha256=actual_hash,
            row_count=len(rows),
            video_count=len(video_ids),
            eval_ids=eval_ids,
            characterization_only=(
                len(rows) < self.minimum_queries or len(video_ids) < self.minimum_videos
            ),
        )


class RetrievalEvaluationRow(FrozenContractModel):
    schema_version: str = "campaign_retrieval_eval_row.v1"
    eval_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    exact_rank: int = Field(ge=1)
    local_rank: int = Field(ge=1)
    top_video_correct: bool
    slices: tuple[str, ...] = ()

    @field_validator("slices")
    @classmethod
    def validate_slices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("slice labels must be sorted and unique")
        return value


class RetrievalEvaluationArtifact(FrozenContractModel):
    schema_version: str = "campaign_retrieval_evaluation.v1"
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representation_contract: dict[str, Any]
    rows: tuple[RetrievalEvaluationRow, ...]
    median_latency_ms: float | None = Field(default=None, gt=0)
    model_footprint_bytes: int | None = Field(default=None, gt=0)

    @field_validator("rows")
    @classmethod
    def validate_rows(
        cls, value: tuple[RetrievalEvaluationRow, ...]
    ) -> tuple[RetrievalEvaluationRow, ...]:
        if not value:
            raise ValueError("evaluation artifact requires rows")
        ids = [row.eval_id for row in value]
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation IDs must be unique")
        return value


class DevelopmentGateContract(FrozenContractModel):
    schema_version: str = "campaign_development_gate.v1"
    minimum_queries: int = Field(default=300, ge=1)
    minimum_videos: int = Field(default=30, ge=1)
    minimum_slice_size: int = Field(default=25, ge=1)
    local_ndcg_at_10_delta_min: float = 0.020
    local_mrr_delta_min: float = 0.020
    exact_mrr_delta_min: float = -0.005
    exact_recall_at_1_delta_min: float = -0.020
    wrong_top_video_delta_max: float = 0.020
    slice_exact_mrr_regression_max: float = 0.020
    latency_ratio_max: float = 1.25
    footprint_ratio_max: float = 1.10
    bootstrap_samples: int = Field(default=10_000, ge=100)
    bootstrap_seed: int = 20260712


class DevelopmentComparison(FrozenContractModel):
    schema_version: str = "campaign_development_comparison.v1"
    champion_digest: str
    candidate_digest: str
    sample_count: int
    video_count: int
    metrics: dict[str, float | int | None]
    slice_metrics: dict[str, dict[str, float | int]]
    verdict: ComparisonVerdict
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    gate_contract: DevelopmentGateContract
    created_at: datetime = Field(default_factory=utc_now)
    comparison_digest: str = ""

    @model_validator(mode="after")
    def verify_digest(self) -> DevelopmentComparison:
        payload = self.model_dump(mode="json", exclude={"comparison_digest", "created_at"})
        expected = canonical_hash(payload)
        if self.comparison_digest and self.comparison_digest != expected:
            raise ValueError("comparison_digest does not match comparison payload")
        if not self.comparison_digest:
            object.__setattr__(self, "comparison_digest", expected)
        return self


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _metrics(rows: list[RetrievalEvaluationRow]) -> dict[str, float]:
    return {
        "local_ndcg_at_10": _mean(
            [1.0 / math.log2(row.local_rank + 1) if row.local_rank <= 10 else 0.0 for row in rows]
        ),
        "local_mrr": _mean([1.0 / row.local_rank for row in rows]),
        "exact_mrr": _mean([1.0 / row.exact_rank for row in rows]),
        "exact_recall_at_1": _mean([1.0 if row.exact_rank == 1 else 0.0 for row in rows]),
        "wrong_top_video_rate": _mean([0.0 if row.top_video_correct else 1.0 for row in rows]),
    }


def _paired_ci_low(differences: list[float], *, samples: int, seed: int) -> float:
    rng = random.Random(seed)
    count = len(differences)
    bootstrapped = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    ]
    bootstrapped.sort()
    return bootstrapped[max(0, int(samples * 0.025) - 1)]


def compare_development_evaluations(
    champion: RetrievalEvaluationArtifact,
    candidate: RetrievalEvaluationArtifact,
    gate: DevelopmentGateContract,
) -> DevelopmentComparison:
    """Compare sealed rows deterministically; missing power or metrics never passes."""

    if (
        champion.corpus_sha256 != candidate.corpus_sha256
        or champion.development_sha256 != candidate.development_sha256
        or canonical_hash(champion.representation_contract)
        != canonical_hash(candidate.representation_contract)
    ):
        raise ValueError("campaign_evaluation_contract_mismatch")
    champion_by_id = {row.eval_id: row for row in champion.rows}
    candidate_by_id = {row.eval_id: row for row in candidate.rows}
    if set(champion_by_id) != set(candidate_by_id):
        raise ValueError("campaign_evaluation_id_mismatch")
    eval_ids = sorted(champion_by_id)
    champion_rows = [champion_by_id[value] for value in eval_ids]
    candidate_rows = [candidate_by_id[value] for value in eval_ids]
    if any(champion_by_id[value].video_id != candidate_by_id[value].video_id for value in eval_ids):
        raise ValueError("campaign_evaluation_video_mismatch")
    champion_metrics = _metrics(champion_rows)
    candidate_metrics = _metrics(candidate_rows)
    deltas = {key: candidate_metrics[key] - champion_metrics[key] for key in champion_metrics}
    ndcg_differences = [
        (
            1.0 / math.log2(candidate_by_id[value].local_rank + 1)
            if candidate_by_id[value].local_rank <= 10
            else 0.0
        )
        - (
            1.0 / math.log2(champion_by_id[value].local_rank + 1)
            if champion_by_id[value].local_rank <= 10
            else 0.0
        )
        for value in eval_ids
    ]
    ndcg_ci_low = _paired_ci_low(
        ndcg_differences,
        samples=gate.bootstrap_samples,
        seed=gate.bootstrap_seed,
    )
    video_count = len({row.video_id for row in candidate_rows})
    blocking: list[str] = []
    warnings: list[str] = []
    insufficient = False
    if len(eval_ids) < gate.minimum_queries:
        insufficient = True
        blocking.append("development_query_count_below_minimum")
    if video_count < gate.minimum_videos:
        insufficient = True
        blocking.append("development_video_count_below_minimum")
    if champion.median_latency_ms is None or candidate.median_latency_ms is None:
        insufficient = True
        blocking.append("latency_evidence_missing")
        latency_ratio = None
    else:
        latency_ratio = candidate.median_latency_ms / champion.median_latency_ms
    if champion.model_footprint_bytes is None or candidate.model_footprint_bytes is None:
        insufficient = True
        blocking.append("footprint_evidence_missing")
        footprint_ratio = None
    else:
        footprint_ratio = candidate.model_footprint_bytes / champion.model_footprint_bytes

    slice_names = sorted({label for row in champion_rows + candidate_rows for label in row.slices})
    slice_metrics: dict[str, dict[str, float | int]] = {}
    for name in slice_names:
        champion_ids = {value for value in eval_ids if name in champion_by_id[value].slices}
        candidate_ids = {value for value in eval_ids if name in candidate_by_id[value].slices}
        if champion_ids != candidate_ids:
            raise ValueError("campaign_evaluation_slice_mismatch")
        ids = sorted(champion_ids)
        champion_slice = _mean([1.0 / champion_by_id[value].exact_rank for value in ids])
        candidate_slice = _mean([1.0 / candidate_by_id[value].exact_rank for value in ids])
        delta = candidate_slice - champion_slice
        slice_metrics[name] = {"sample_count": len(ids), "exact_mrr_delta": delta}
        if len(ids) >= gate.minimum_slice_size and delta < -gate.slice_exact_mrr_regression_max:
            blocking.append(f"slice_exact_mrr_regression:{name}")

    clauses = {
        "local_ndcg_at_10_delta": deltas["local_ndcg_at_10"] >= gate.local_ndcg_at_10_delta_min,
        "local_ndcg_at_10_ci_low": ndcg_ci_low > 0,
        "local_mrr_delta": deltas["local_mrr"] >= gate.local_mrr_delta_min,
        "exact_mrr_delta": deltas["exact_mrr"] >= gate.exact_mrr_delta_min,
        "exact_recall_at_1_delta": deltas["exact_recall_at_1"] >= gate.exact_recall_at_1_delta_min,
        "wrong_top_video_delta": deltas["wrong_top_video_rate"] <= gate.wrong_top_video_delta_max,
        "latency_ratio": latency_ratio is not None and latency_ratio <= gate.latency_ratio_max,
        "footprint_ratio": footprint_ratio is not None
        and footprint_ratio <= gate.footprint_ratio_max,
    }
    if not insufficient:
        blocking.extend(name for name, passed in clauses.items() if not passed)
    else:
        warnings.extend(
            f"observed_gate_failure:{name}" for name, passed in clauses.items() if not passed
        )
    verdict = (
        ComparisonVerdict.INSUFFICIENT_EVIDENCE
        if insufficient
        else ComparisonVerdict.FAILED if blocking else ComparisonVerdict.PASSED
    )
    metrics: dict[str, float | int | None] = {
        **{f"champion_{key}": value for key, value in champion_metrics.items()},
        **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
        **{f"delta_{key}": value for key, value in deltas.items()},
        "local_ndcg_at_10_delta_ci_low": ndcg_ci_low,
        "latency_ratio": latency_ratio,
        "footprint_ratio": footprint_ratio,
    }
    return DevelopmentComparison(
        champion_digest=champion.candidate_digest,
        candidate_digest=candidate.candidate_digest,
        sample_count=len(eval_ids),
        video_count=video_count,
        metrics=metrics,
        slice_metrics=slice_metrics,
        verdict=verdict,
        blocking_reasons=tuple(sorted(set(blocking))),
        warnings=tuple(warnings),
        gate_contract=gate,
    )


def champion_evaluation_cache_key(
    *,
    champion_digest: str,
    corpus_sha256: str,
    development_sha256: str,
    representation_contract: dict[str, Any],
    evaluator_revision: str,
) -> str:
    return canonical_hash(
        {
            "champion_digest": champion_digest,
            "corpus_sha256": corpus_sha256,
            "development_sha256": development_sha256,
            "representation_contract": representation_contract,
            "evaluator_revision": evaluator_revision,
        }
    )


def load_retrieval_evaluation_artifact(
    rows_path: Path,
    *,
    candidate_digest: str,
    corpus_sha256: str,
    development_sha256: str,
    representation_contract: dict[str, Any],
    median_latency_ms: float | None,
    model_footprint_bytes: int | None,
) -> RetrievalEvaluationArtifact:
    """Normalize scorer rows into the sealed, comparison-ready contract."""

    rows: list[RetrievalEvaluationRow] = []
    with rows_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if payload.get("split") != "dev":
                    raise ValueError("non-dev evaluation row")
                exact_rank = int(payload["positive_rank_exact"])
                local_rank = int(payload["positive_rank_local_window"])
                top_video_id = str(
                    payload.get("top_retrieved_video_id")
                    or str(payload["top_retrieved_chunk_ids"][0]).split(":", 1)[0]
                )
                positive_video_id = str(payload["positive_video_id"])
                slices = tuple(
                    sorted(
                        {
                            f"query_type:{payload['query_type']}",
                            f"channel:{payload['channel']}",
                            f"source_set:{payload['source_set']}",
                        }
                    )
                )
                rows.append(
                    RetrievalEvaluationRow(
                        eval_id=str(payload["eval_id"]),
                        video_id=positive_video_id,
                        exact_rank=exact_rank,
                        local_rank=local_rank,
                        top_video_correct=top_video_id == positive_video_id,
                        slices=slices,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"campaign_retrieval_evaluation_row_invalid:{line_number}"
                ) from exc
    return RetrievalEvaluationArtifact(
        candidate_digest=candidate_digest,
        corpus_sha256=corpus_sha256,
        development_sha256=development_sha256,
        representation_contract=representation_contract,
        rows=tuple(rows),
        median_latency_ms=median_latency_ms,
        model_footprint_bytes=model_footprint_bytes,
    )


__all__ = [
    "ComparisonVerdict",
    "DevelopmentComparison",
    "DevelopmentDataContractError",
    "DevelopmentDatasetContract",
    "DevelopmentGateContract",
    "RetrievalEvaluationArtifact",
    "RetrievalEvaluationRow",
    "ValidatedDevelopmentDataset",
    "champion_evaluation_cache_key",
    "compare_development_evaluations",
    "load_retrieval_evaluation_artifact",
    "sha256_file",
]
