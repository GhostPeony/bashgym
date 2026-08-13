from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts.memexai import evaluate_retrieval_system_ablation as ablation


def _corpus() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "a:0",
            "youtube_video_id": "video-a",
            "title": "Orchid retrieval",
            "text": "The primary transcript passage.",
        },
        {
            "chunk_id": "a:1",
            "youtube_video_id": "video-a",
            "title": "Orchid transcript",
            "text": "An answer-equivalent local passage.",
        },
        {
            "chunk_id": "b:0",
            "youtube_video_id": "video-b",
            "title": "Retrieval systems",
            "text": "A labeled hard negative.",
        },
        {
            "chunk_id": "c:0",
            "youtube_video_id": "video-c",
            "title": "Garden notes",
            "text": "An unrelated passage.",
        },
    ]


def _query() -> dict[str, object]:
    return {
        "eval_id": "dev-1",
        "split": "dev",
        "query": "orchid retrieval",
        "query_type": "natural_question",
        "positive_chunk_id": "a:0",
        "positive_video_id": "video-a",
        "local_window_chunk_ids": ["a:0", "a:1"],
        "hard_negative_chunk_ids": ["b:0"],
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_query_split_rejects_physically_combined_protected_file() -> None:
    rows = [_query(), {**_query(), "eval_id": "test-1", "split": "test"}]

    with pytest.raises(ValueError, match="not physically exclusive"):
        ablation.validate_query_split(
            rows,
            selected_splits={"dev"},
            require_exclusive_split=True,
        )


def test_full_bm25_ranking_is_complete_and_deterministic_for_zero_score_ties() -> None:
    corpus = _corpus()
    index = ablation.build_lexical_index(corpus)

    first = ablation._full_bm25_ranking("orchid", corpus, lexical_index=index)
    second = ablation._full_bm25_ranking("orchid", corpus, lexical_index=index)

    assert first == second
    assert [chunk_id for chunk_id, _score in first[:2]] == ["a:0", "a:1"]
    assert [chunk_id for chunk_id, _score in first[2:]] == ["b:0", "c:0"]
    assert all(score == 0.0 for _chunk_id, score in first[2:])


def test_shared_metric_contract_scores_rank_and_hard_negative_failures() -> None:
    rankings = [[("b:0", 0.9), ("a:1", 0.8), ("a:0", 0.7), ("c:0", 0.1)]]

    metrics, evidence = ablation.evaluate_rankings(
        [_query()], _corpus(), rankings, artifact_depth=3
    )

    assert metrics["exact_chunk_mrr"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["local_window_mrr"] == 0.5
    assert metrics["same_video_mrr"] == 0.5
    assert metrics["wrong_top_video_rate"] == 1.0
    assert metrics["hard_negative_win_rate"] == 0.0
    assert evidence[0]["positive_rank_exact_chunk"] == 3
    assert evidence[0]["positive_rank_local_window"] == 2
    assert evidence[0]["hard_negative_win"] is False


def test_cli_writes_dense_bm25_rrf_and_explicit_reranker_status(tmp_path, monkeypatch) -> None:
    queries_path = tmp_path / "heldout-dev.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    ids_path = tmp_path / "chunk-ids.json"
    query_matrix_path = tmp_path / "query.npy"
    corpus_matrix_path = tmp_path / "corpus.npy"
    output_dir = tmp_path / "output"
    _write_jsonl(queries_path, [_query()])
    _write_jsonl(corpus_path, _corpus())
    ids_path.write_text(json.dumps(["a:0", "a:1", "b:0", "c:0"]), encoding="utf-8")
    np.save(query_matrix_path, np.asarray([[1.0, 0.0]], dtype="float32"))
    np.save(
        corpus_matrix_path,
        np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype="float32",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_retrieval_system_ablation.py",
            "--queries-jsonl",
            str(queries_path),
            "--corpus-jsonl",
            str(corpus_path),
            "--dense-query-matrix",
            str(query_matrix_path),
            "--dense-corpus-matrix",
            str(corpus_matrix_path),
            "--dense-corpus-chunk-ids",
            str(ids_path),
            "--output-dir",
            str(output_dir),
            "--dense-model-id",
            "fixture/model",
            "--dense-model-revision",
            "fixture-revision",
            "--dimensions",
            "2",
            "--dense-candidate-depth",
            "3",
            "--bm25-candidate-depth",
            "3",
            "--artifact-depth",
            "3",
            "--splits",
            "dev",
            "--require-exclusive-split",
        ],
    )

    assert ablation.main() == 0

    manifest = json.loads(
        (output_dir / "retrieval_system_ablation_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["systems"]) == {"dense", "bm25", "rrf"}
    assert manifest["systems"]["dense"]["metrics"]["exact_chunk_recall_at_1"] == 1.0
    assert manifest["protocol"]["rrf"]["k"] == 60
    assert manifest["protocol"]["dense"]["matrix_normalization"]["policy"] == (
        "validate_only_no_silent_renormalization"
    )
    assert manifest["reranker"]["status"] == "not_run"
    assert manifest["reranker"]["supported_by_this_evaluator"] is False
    assert (output_dir / "retrieval_system_ablation_queries.jsonl").is_file()
