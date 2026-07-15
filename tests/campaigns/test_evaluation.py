"""Leakage, determinism, and promotion-gate tests for development evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bashgym.campaigns.evaluation import (
    ComparisonVerdict,
    DevelopmentDataContractError,
    DevelopmentDatasetContract,
    DevelopmentGateContract,
    RetrievalEvaluationArtifact,
    RetrievalEvaluationRow,
    champion_evaluation_cache_key,
    compare_development_evaluations,
    load_retrieval_evaluation_artifact,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_bytes(payload.encode())
    return hashlib.sha256(payload.encode()).hexdigest()


def _artifact(
    *,
    digest: str,
    rows: tuple[RetrievalEvaluationRow, ...],
    latency: float | None = 10.0,
    footprint: int | None = 1_000,
    corpus: str = "c" * 64,
    development: str = "d" * 64,
    representation: dict[str, object] | None = None,
) -> RetrievalEvaluationArtifact:
    return RetrievalEvaluationArtifact(
        candidate_digest=digest,
        corpus_sha256=corpus,
        development_sha256=development,
        representation_contract=representation or {"query_prefix": "query: "},
        rows=rows,
        median_latency_ms=latency,
        model_footprint_bytes=footprint,
    )


def _rows(*, count: int, videos: int, rank: int, reverse: bool = False):
    values = [
        RetrievalEvaluationRow(
            eval_id=f"eval-{index:04d}",
            video_id=f"video-{index % videos:03d}",
            exact_rank=rank,
            local_rank=rank,
            top_video_correct=rank == 1,
            slices=("long-query",) if index < 25 else (),
        )
        for index in range(count)
    ]
    if reverse:
        values.reverse()
    return tuple(values)


def test_dev_contract_rejects_protected_paths_hashes_and_non_dev_rows(tmp_path):
    protected = tmp_path / "heldout-test.jsonl"
    protected_hash = _write_jsonl(
        protected,
        [{"eval_id": "test-1", "video_id": "video-1", "split": "test"}],
    )
    contract = DevelopmentDatasetContract(
        expected_sha256=protected_hash,
        protected_hashes=frozenset({protected_hash}),
        protected_path_fragments=("heldout-test",),
    )
    with pytest.raises(DevelopmentDataContractError, match="protected path"):
        contract.validate_file(protected)

    disguised = tmp_path / "dev.jsonl"
    disguised.write_bytes(protected.read_bytes())
    contract = contract.model_copy(update={"protected_path_fragments": ()})
    with pytest.raises(DevelopmentDataContractError, match="protected hash"):
        contract.validate_file(disguised)

    mixed_hash = _write_jsonl(
        disguised,
        [{"eval_id": "dev-1", "video_id": "video-1", "split": "test"}],
    )
    contract = contract.model_copy(
        update={"expected_sha256": mixed_hash, "protected_hashes": frozenset()}
    )
    with pytest.raises(DevelopmentDataContractError, match="non-dev row"):
        contract.validate_file(disguised)


def test_small_physical_dev_set_is_valid_but_characterization_only(tmp_path):
    path = tmp_path / "development-only.jsonl"
    digest = _write_jsonl(
        path,
        [
            {
                "eval_id": f"dev-{index}",
                "positive_video_id": f"video-{index % 3}",
                "split": "dev",
            }
            for index in range(18)
        ],
    )
    validated = DevelopmentDatasetContract(expected_sha256=digest).validate_file(path)
    assert validated.row_count == 18
    assert validated.video_count == 3
    assert validated.characterization_only is True


def test_underpowered_or_missing_operational_evidence_never_passes():
    rows = _rows(count=18, videos=3, rank=1)
    comparison = compare_development_evaluations(
        _artifact(digest="a" * 64, rows=rows, latency=None),
        _artifact(digest="b" * 64, rows=rows, footprint=None),
        DevelopmentGateContract(bootstrap_samples=100),
    )
    assert comparison.verdict == ComparisonVerdict.INSUFFICIENT_EVIDENCE
    assert set(comparison.blocking_reasons) >= {
        "development_query_count_below_minimum",
        "development_video_count_below_minimum",
        "latency_evidence_missing",
        "footprint_evidence_missing",
    }


def test_powered_improvement_passes_and_is_order_independent():
    gate = DevelopmentGateContract(bootstrap_samples=500)
    champion_rows = _rows(count=300, videos=30, rank=2)
    candidate_rows = _rows(count=300, videos=30, rank=1, reverse=True)
    first = compare_development_evaluations(
        _artifact(digest="a" * 64, rows=champion_rows),
        _artifact(digest="b" * 64, rows=candidate_rows),
        gate,
    )
    second = compare_development_evaluations(
        _artifact(digest="a" * 64, rows=tuple(reversed(champion_rows))),
        _artifact(digest="b" * 64, rows=tuple(reversed(candidate_rows))),
        gate,
    )
    assert first.verdict == ComparisonVerdict.PASSED
    assert first.comparison_digest == second.comparison_digest
    assert first.metrics == second.metrics


def test_regression_and_slice_contracts_fail_closed():
    gate = DevelopmentGateContract(bootstrap_samples=100)
    champion_rows = _rows(count=300, videos=30, rank=1)
    candidate_rows = _rows(count=300, videos=30, rank=2)
    comparison = compare_development_evaluations(
        _artifact(digest="a" * 64, rows=champion_rows),
        _artifact(digest="b" * 64, rows=candidate_rows),
        gate,
    )
    assert comparison.verdict == ComparisonVerdict.FAILED
    assert "local_mrr_delta" in comparison.blocking_reasons
    assert "slice_exact_mrr_regression:long-query" in comparison.blocking_reasons

    mismatched = list(candidate_rows)
    mismatched[0] = mismatched[0].model_copy(update={"slices": ()})
    with pytest.raises(ValueError, match="slice_mismatch"):
        compare_development_evaluations(
            _artifact(digest="a" * 64, rows=champion_rows),
            _artifact(digest="b" * 64, rows=tuple(mismatched)),
            gate,
        )


@pytest.mark.parametrize(
    "field,changed",
    [
        ("champion_digest", "f" * 64),
        ("corpus_sha256", "e" * 64),
        ("development_sha256", "d" * 64),
        ("representation_contract", {"query_prefix": "search: "}),
        ("evaluator_revision", "revision-2"),
    ],
)
def test_champion_cache_key_covers_every_evaluation_input(field, changed):
    inputs = {
        "champion_digest": "a" * 64,
        "corpus_sha256": "b" * 64,
        "development_sha256": "c" * 64,
        "representation_contract": {"query_prefix": "query: "},
        "evaluator_revision": "revision-1",
    }
    original = champion_evaluation_cache_key(**inputs)
    inputs[field] = changed
    assert champion_evaluation_cache_key(**inputs) != original


def test_scorer_rows_normalize_wrong_video_and_slices(tmp_path):
    rows_path = tmp_path / "scored-dev.jsonl"
    _write_jsonl(
        rows_path,
        [
            {
                "eval_id": "dev-1",
                "split": "dev",
                "positive_video_id": "video-a",
                "positive_rank_exact": 4,
                "positive_rank_local_window": 2,
                "top_retrieved_video_id": "video-b",
                "query_type": "natural_question",
                "channel": "Channel A",
                "source_set": "local",
            }
        ],
    )
    artifact = load_retrieval_evaluation_artifact(
        rows_path,
        candidate_digest="a" * 64,
        corpus_sha256="b" * 64,
        development_sha256="c" * 64,
        representation_contract={"prefix": "memexai_youtube"},
        median_latency_ms=12.5,
        model_footprint_bytes=1000,
    )
    assert artifact.rows[0].top_video_correct is False
    assert artifact.rows[0].slices == (
        "channel:Channel A",
        "query_type:natural_question",
        "source_set:local",
    )
