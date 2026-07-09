from __future__ import annotations

import pytest

from scripts.memexai import run_dd_retrieval_preview as dd_preview
from scripts.memexai import score_embedding_query_ablation as query_ablation
from scripts.memexai import train_embedding_retriever


def _corpus_rows() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk-0",
            "youtube_video_id": "video-1",
            "chunk_index": 0,
            "title": "Video One",
            "channel": "Channel A",
            "url": "https://example.test/v1",
            "text": "The guest explains retrieval eval design and why local windows matter.",
        },
        {
            "chunk_id": "chunk-1",
            "youtube_video_id": "video-1",
            "chunk_index": 1,
            "title": "Video One",
            "channel": "Channel A",
            "url": "https://example.test/v1",
            "text": "The guest explains reward models and training data quality.",
        },
    ]


def _dd_row() -> dict[str, object]:
    return {
        "seed_chunk_id": "chunk-1",
        "seed_video_id": "video-1",
        "seed_chunk_index": 1,
        "seed_title": "Video One",
        "seed_channel": "Channel A",
        "seed_source_set": "seed",
        "seed_url": "https://example.test/v1",
        "seed_upload_date": "2026-07-01",
        "seed_start_seconds": 10,
        "seed_end_seconds": 20,
        "query_variants": {
            "natural_question": "What does the guest say about eval design?",
            "keyword_query": "eval design metrics",
            "semantic_paraphrase": "Find the section discussing evaluation setup.",
        },
        "query_judgement": {
            "natural_question_pass": True,
            "natural_question_reason": "Grounded in the chunk.",
            "keyword_query_pass": False,
            "keyword_query_reason": "Too short and ambiguous.",
            "semantic_paraphrase_pass": True,
            "semantic_paraphrase_reason": "Grounded in the chunk.",
        },
    }


def test_flatten_eval_rows_filters_judge_rejections() -> None:
    rows, failures = dd_preview.flatten_eval_rows(
        [_dd_row()],
        _corpus_rows(),
        "local-model",
        require_judge=True,
    )

    assert [row["query_type"] for row in rows] == [
        "natural_question",
        "semantic_paraphrase",
    ]
    assert all(row["judge_pass"] is True for row in rows)
    assert failures["keyword_query:judge_rejected"] == 1


def test_judge_gate_fails_closed_below_minimum_keep_ratio() -> None:
    gate = dd_preview.judge_gate_stats(
        raw_rows=[_dd_row(), _dd_row()],
        eval_rows=[{"query_type": "natural_question"}],
        validation_failures={"keyword_query:judge_rejected": 5},
        judge_enabled=True,
        min_keep_ratio=0.70,
    )

    assert gate["expected_query_rows"] == 6
    assert gate["kept_query_rows"] == 1
    assert gate["passed"] is False


def test_query_prefix_formatting_defaults_to_memexai_youtube_shape() -> None:
    formatted = dd_preview.format_query_for_embedding(
        "What does the transcript say about reward models?",
        "memexai_youtube",
    )

    assert formatted.startswith("Instruct: Given a YouTube transcript search query")
    assert formatted.endswith("What does the transcript say about reward models?")


def test_query_ablation_rejects_matrix_id_length_mismatch() -> None:
    with pytest.raises(ValueError, match="row count does not match chunk ids"):
        query_ablation.normalize_corpus_ids(["chunk-1"], matrix_rows=2)


def test_query_ablation_metrics_include_split_breakdown() -> None:
    rows = [
        {
            "split": "dev",
            "query_type": "natural_question",
            "channel": "Channel A",
            "positive_rank_exact": 1,
            "positive_rank_local_window": 1,
            "positive_rank_same_video": 1,
        },
        {
            "split": "test",
            "query_type": "natural_question",
            "channel": "Channel A",
            "positive_rank_exact": 20,
            "positive_rank_local_window": 2,
            "positive_rank_same_video": 2,
        },
    ]

    result = query_ablation.metrics(rows)

    assert result["overall"]["exact_chunk_recall_at_1"] == 0.5
    assert result["by_split"]["dev"]["exact_chunk_recall_at_1"] == 1.0
    assert result["by_split"]["test"]["exact_chunk_recall_at_1"] == 0.0


def test_training_pair_builder_uses_train_split_and_query_prefix() -> None:
    query_rows = [
        {
            "eval_id": "eval-train",
            "split": "train",
            "query_type": "natural_question",
            "query": "What does the guest say about reward models?",
            "positive_chunk_id": "chunk-1",
            "positive_video_id": "video-1",
        },
        {
            "eval_id": "eval-dev",
            "split": "dev",
            "query_type": "natural_question",
            "query": "What does the guest say about eval design?",
            "positive_chunk_id": "chunk-0",
            "positive_video_id": "video-1",
        },
    ]

    pairs = train_embedding_retriever.build_pairs(
        query_rows,
        _corpus_rows(),
        train_splits={"train"},
        query_prefix_mode="memexai_youtube",
        max_pairs=None,
        seed=1,
    )

    assert len(pairs) == 1
    assert pairs[0]["eval_id"] == "eval-train"
    assert pairs[0]["query_for_embedding"].startswith(
        "Instruct: Given a YouTube transcript search query"
    )
    assert pairs[0]["positive_text"] == (
        "The guest explains reward models and training data quality."
    )
