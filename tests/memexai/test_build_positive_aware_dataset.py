from __future__ import annotations

from scripts.memexai import build_positive_aware_campaign_report as report_builder
from scripts.memexai import build_positive_aware_dataset as builder


def _corpus() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for video in ("a", "b", "c"):
        for index in range(6):
            topic = "retrieval ranking local windows" if index < 3 else "cooking garden travel"
            rows.append(
                {
                    "chunk_id": f"{video}:{index:04d}",
                    "youtube_video_id": video,
                    "chunk_index": index,
                    "start_seconds": index * 60,
                    "end_seconds": index * 60 + 72,
                    "title": f"Video {video}",
                    "text": f"{topic} unique {video} {index}",
                }
            )
    return rows


def test_candidate_filter_excludes_positive_adjacent_overlap_and_same_video() -> None:
    corpus = _corpus()
    by_id = {str(row["chunk_id"]): row for row in corpus}
    positives = {"a:0002", "a:0003"}

    reasons = {
        candidate: builder.exclusion_reason(
            by_id["a:0002"], by_id[candidate], positives, near_duplicate_threshold=0.85
        )
        for candidate in ("a:0002", "a:0001", "a:0003", "a:0005", "b:0002")
    }

    assert reasons["a:0002"] == "positive_or_equivalent"
    assert reasons["a:0001"] == "same_video_adjacent"
    assert reasons["a:0003"] == "positive_or_equivalent"
    assert reasons["a:0005"] == "same_video_unreviewed"
    assert reasons["b:0002"] is None


def test_builder_selects_safe_wrong_video_negatives_deterministically() -> None:
    queries = [
        {
            "eval_id": "q-1",
            "split": "train",
            "query": "retrieval ranking local windows",
            "query_type": "natural_question",
            "positive_chunk_id": "a:0002",
            "positive_video_id": "a",
            "local_window_chunk_ids": ["a:0001", "a:0002", "a:0003"],
            "hard_negative_chunk_ids": ["b:0000", "b:0001"],
        }
    ]

    first = builder.build_grouped_examples(queries, _corpus(), min_negatives=3, max_negatives=5)
    second = builder.build_grouped_examples(queries, _corpus(), min_negatives=3, max_negatives=5)

    assert first == second
    row = first[0]
    assert row["status"] == "ready"
    assert row["positive_chunk_ids"] == ["a:0001", "a:0002", "a:0003"]
    assert 3 <= len(row["hard_negative_chunk_ids"]) <= 5
    assert all(not chunk_id.startswith("a:") for chunk_id in row["hard_negative_chunk_ids"])
    assert row["passage_representation"] == "embedding_text_or_title_double_newline_text:v1"


def test_builder_marks_rows_with_too_few_safe_negatives() -> None:
    corpus = [row for row in _corpus() if row["youtube_video_id"] == "a"]
    queries = [
        {
            "eval_id": "q-1",
            "split": "train",
            "query": "retrieval ranking local windows",
            "query_type": "natural_question",
            "positive_chunk_id": "a:0002",
            "positive_video_id": "a",
        }
    ]

    rows = builder.build_grouped_examples(queries, corpus, min_negatives=3, max_negatives=5)

    assert rows[0]["status"] == "insufficient_safe_negatives"
    assert rows[0]["hard_negative_chunk_ids"] == []


def test_validate_grouped_examples_rejects_positive_negative_overlap() -> None:
    row = {
        "eval_id": "q-1",
        "positive_chunk_ids": ["a"],
        "hard_negative_chunk_ids": ["a", "b", "c"],
        "status": "ready",
    }

    try:
        builder.validate_grouped_examples([row], corpus_ids={"a", "b", "c"}, min_negatives=3)
    except ValueError as exc:
        assert "positive/negative overlap" in str(exc)
    else:
        raise AssertionError("expected overlap validation failure")


def test_report_facts_reconcile_dataset_counts() -> None:
    manifest = {
        "statistics": {
            "rows": 702,
            "ready_rows": 696,
            "insufficient_safe_negatives": 6,
            "negative_sources": {"bm25": 4048, "labeled": 798},
            "positive_group_sizes": {"1": 588, "5": 114},
            "exclusion_reasons": {"same_video_adjacent": 876, "same_video_unreviewed": 4311},
        }
    }

    facts = report_builder.report_facts(manifest)

    assert facts["coverage"] == 696 / 702
    assert facts["selected_negatives"] == 4846
    assert facts["excluded_candidates"] == 5187
    assert facts["grouped_positive_rows"] == 114


def test_rrf_fusion_rewards_candidates_found_by_both_lanes() -> None:
    fused = builder.rrf_fuse(
        {"dense": ["dense-only", "shared"], "bm25": ["shared", "lexical-only"]},
        k=60,
    )

    assert fused[0][0] == "shared"
    assert fused[0][2] == {"bm25": 1, "dense": 2}
