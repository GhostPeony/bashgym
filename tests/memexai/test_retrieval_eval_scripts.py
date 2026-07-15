from __future__ import annotations

import sys
from collections import Counter

import pytest

from scripts.memexai import embed_corpus_with_model, train_embedding_retriever
from scripts.memexai import prepare_embedding_training_bundle as bundle_builder
from scripts.memexai import run_dd_retrieval_preview as dd_preview
from scripts.memexai import score_embedding_query_ablation as query_ablation


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


def test_corpus_passage_format_matches_product_title_and_content() -> None:
    row = _corpus_rows()[0]

    assert embed_corpus_with_model.format_passage_for_embedding(row) == (
        "Video One\n\nThe guest explains retrieval eval design and why local windows matter."
    )
    assert embed_corpus_with_model.format_passage_for_embedding(
        {**row, "embedding_text": "Pinned title\n\nPinned content"}
    ) == "Pinned title\n\nPinned content"


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


def test_query_ablation_rejects_combined_file_before_model_import(tmp_path, monkeypatch) -> None:
    queries = tmp_path / "heldout-dev-test.jsonl"
    queries.write_text(
        '{"eval_id":"dev-1","split":"dev"}\n'
        '{"eval_id":"test-1","split":"test"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_embedding_query_ablation.py",
            "--queries-jsonl",
            str(queries),
            "--corpus-jsonl",
            str(tmp_path / "unused.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--embedding-model-path",
            str(tmp_path / "model"),
            "--corpus-embedding-matrix",
            str(tmp_path / "unused.npy"),
            "--corpus-embedding-chunk-ids",
            str(tmp_path / "unused.json"),
            "--splits",
            "dev",
            "--require-exclusive-split",
        ],
    )
    sys.modules.pop("sentence_transformers", None)
    with pytest.raises(ValueError, match="not physically exclusive"):
        query_ablation.main()
    assert "sentence_transformers" not in sys.modules


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
        "Video One\n\nThe guest explains reward models and training data quality."
    )


def test_unique_positive_batches_cover_rows_without_collisions() -> None:
    pairs = [
        {"positive_chunk_id": positive_id}
        for positive_id in ("a", "a", "a", "b", "b", "c", "c", "d", "d")
    ]

    batches = train_embedding_retriever.build_unique_positive_batches(
        pairs, batch_size=3, seed=7
    )

    assert sorted(index for batch in batches for index in batch) == list(range(len(pairs)))
    assert all(len(batch) >= 2 for batch in batches)
    for batch in batches:
        positives = [pairs[index]["positive_chunk_id"] for index in batch]
        assert len(positives) == len(set(positives))


def test_unique_positive_batches_are_seed_deterministic() -> None:
    pairs = [{"positive_chunk_id": str(index // 2)} for index in range(20)]

    first = train_embedding_retriever.build_unique_positive_batches(
        pairs, batch_size=4, seed=11
    )
    second = train_embedding_retriever.build_unique_positive_batches(
        pairs, batch_size=4, seed=11
    )

    assert first == second


def test_grouped_pair_builder_adds_explicit_negatives_and_video_collision_group() -> None:
    corpus = _corpus_rows() + [
        {
            "chunk_id": "chunk-negative",
            "youtube_video_id": "video-2",
            "chunk_index": 0,
            "title": "Other Video",
            "channel": "Channel B",
            "text": "An unrelated but difficult retrieval candidate.",
        }
    ]
    grouped = [
        {
            "eval_id": "eval-train",
            "split": "train",
            "status": "ready",
            "query_type": "natural_question",
            "query": "What does the guest say about reward models?",
            "positive_video_id": "video-1",
            "primary_positive_chunk_id": "chunk-1",
            "positive_chunk_ids": ["chunk-0", "chunk-1"],
            "hard_negative_chunk_ids": ["chunk-negative"],
        }
    ]

    pairs = train_embedding_retriever.build_grouped_pairs(
        grouped, corpus, query_prefix_mode="memexai_youtube", max_pairs=None, seed=7
    )

    assert len(pairs) == 1
    assert pairs[0]["positive_collision_group"] == "video:video-1"
    assert pairs[0]["positive_chunk_ids"] == ["chunk-0", "chunk-1"]
    assert pairs[0]["negative_chunk_ids"] == ["chunk-negative"]
    assert pairs[0]["negative_texts"] == [
        "Other Video\n\nAn unrelated but difficult retrieval candidate."
    ]


def test_collision_safe_batches_separate_rows_from_same_positive_video() -> None:
    pairs = [
        {
            "positive_chunk_id": f"chunk-{index}",
            "positive_collision_group": f"video:{video}",
        }
        for index, video in enumerate(("a", "a", "b", "b", "c", "c", "d", "d"))
    ]

    batches = train_embedding_retriever.build_unique_positive_batches(
        pairs, batch_size=4, seed=9
    )

    for batch in batches:
        groups = [pairs[index]["positive_collision_group"] for index in batch]
        assert len(groups) == len(set(groups))


def test_expanded_batch_budget_counts_queries_positives_and_negatives() -> None:
    pairs = [
        {"negative_texts": [f"negative-{index}" for index in range(7)]}
        for _ in range(16)
    ]

    assert train_embedding_retriever.expanded_sequences_per_batch_upper_bound(
        pairs, batch_size=16
    ) == 144
    with pytest.raises(ValueError, match="upper bound 144 sequences exceeds limit 64"):
        train_embedding_retriever.validate_expanded_batch_budget(
            pairs,
            batch_size=16,
            resolved_loss="explicit_mnrl",
            max_expanded_sequences=64,
        )


def test_cached_mnrl_allows_large_logical_batch_under_minibatch_control() -> None:
    pairs = [{"negative_texts": ["a", "b", "c"]} for _ in range(16)]

    assert train_embedding_retriever.validate_expanded_batch_budget(
        pairs,
        batch_size=16,
        resolved_loss="cached_mnrl",
        max_expanded_sequences=64,
    ) == 80


def test_collision_safe_batch_profile_reports_realized_not_configured_batch() -> None:
    pairs = [
        {
            "positive_chunk_id": f"chunk-{index}",
            "positive_collision_group": f"video:{index % 3}",
            "negative_texts": ["negative-a", "negative-b", "negative-c"],
        }
        for index in range(12)
    ]

    profile = train_embedding_retriever.collision_safe_batch_profile(
        pairs, batch_size=10, seed=7
    )

    assert profile == {
        "batch_size_requested": 10,
        "batch_size_realized_min": 3,
        "batch_size_realized_max": 3,
        "batch_size_realized_mean": 3.0,
        "batches_per_epoch": 4,
        "distinct_positive_collision_groups": 3,
        "expanded_sequences_per_batch_realized_max": 15,
    }


def test_explicit_negative_cardinality_is_fixed_and_auditable() -> None:
    pairs = [
        {
            "query_for_embedding": "query",
            "positive_text": "positive",
            "negative_chunk_ids": ["a", "b", "c", "d"],
            "negative_texts": ["A", "B", "C", "D"],
        }
    ]

    selected = train_embedding_retriever.select_explicit_negative_cardinality(pairs, 3)

    assert selected[0]["available_negative_count"] == 4
    assert selected[0]["negative_chunk_ids"] == ["a", "b", "c"]
    assert selected[0]["negative_texts"] == ["A", "B", "C"]


def test_sentence_transformer_columns_keep_every_fixed_negative_lane() -> None:
    pairs = [
        {
            "query_for_embedding": "query-1",
            "positive_text": "positive-1",
            "negative_texts": ["negative-1a", "negative-1b", "negative-1c"],
        },
        {
            "query_for_embedding": "query-2",
            "positive_text": "positive-2",
            "negative_texts": ["negative-2a", "negative-2b", "negative-2c"],
        },
    ]

    columns = train_embedding_retriever.build_sentence_transformer_columns(pairs)

    assert list(columns) == ["anchor", "positive", "negative_1", "negative_2", "negative_3"]
    assert columns["negative_3"] == ["negative-1c", "negative-2c"]


def test_sentence_transformer_columns_reject_variable_negative_lanes() -> None:
    pairs = [
        {"query_for_embedding": "q1", "positive_text": "p1", "negative_texts": ["a"]},
        {"query_for_embedding": "q2", "positive_text": "p2", "negative_texts": ["a", "b"]},
    ]

    with pytest.raises(ValueError, match="same explicit negative count"):
        train_embedding_retriever.build_sentence_transformer_columns(pairs)


def test_capped_synthetic_selection_is_balanced_and_order_independent() -> None:
    rows = [
        {
            "eval_id": f"eval-{positive}-{query_type}",
            "positive_chunk_id": positive,
            "query_type": query_type,
        }
        for positive in ("a", "b", "c", "d")
        for query_type in ("natural_question", "keyword_query", "semantic_paraphrase")
    ]

    selected = bundle_builder.select_capped_synthetic(rows, target=8, seed="fixture")
    reversed_selected = bundle_builder.select_capped_synthetic(
        list(reversed(rows)), target=8, seed="fixture"
    )

    assert [row["eval_id"] for row in selected] == [
        row["eval_id"] for row in reversed_selected
    ]
    assert len({row["positive_chunk_id"] for row in selected}) == 4
    assert max(Counter(row["positive_chunk_id"] for row in selected).values()) == 2


def test_bundle_rejects_training_video_in_heldout() -> None:
    train = [{"positive_video_id": "video-a", "positive_chunk_id": "chunk-a"}]
    heldout = [{"positive_video_id": "video-a", "positive_chunk_id": "chunk-b"}]

    with pytest.raises(ValueError, match="overlap"):
        bundle_builder.validate_heldout_disjoint(train, heldout)
