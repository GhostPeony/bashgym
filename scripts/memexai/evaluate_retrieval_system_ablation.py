#!/usr/bin/env python3
"""Evaluate dense, BM25, and reciprocal-rank-fusion retrieval on one protected split.

The evaluator consumes precomputed normalized dense query/corpus matrices so every
encoder is scored against its own corpus vectors.  It intentionally does not load
or infer with a reranker: a reranker result requires a separately approved model
revision, input template, truncation policy, and candidate-depth protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.memexai.build_positive_aware_dataset import (
        TOKEN_RE,
        build_lexical_index,
        lexical_rank,
        load_jsonl,
        rrf_fuse,
        sha256_file,
        stable_hash,
    )
except ModuleNotFoundError:  # Support campaign-local staging beside the dataset builder.
    from build_positive_aware_dataset import (  # type: ignore[no-redef]
        TOKEN_RE,
        build_lexical_index,
        lexical_rank,
        load_jsonl,
        rrf_fuse,
        sha256_file,
        stable_hash,
    )

SCHEMA_VERSION = "memexai.retrieval-system-ablation.v1"
CUTOFFS = (1, 3, 5, 10)


def normalize_corpus_ids(corpus_ids: Any, matrix_rows: int) -> list[str]:
    if isinstance(corpus_ids, dict):
        normalized = [str(corpus_ids[str(index)]) for index in range(matrix_rows)]
    elif isinstance(corpus_ids, list):
        normalized = [str(chunk_id) for chunk_id in corpus_ids]
    else:
        raise ValueError("dense corpus chunk IDs must be a JSON list or index-keyed object")
    if len(normalized) != matrix_rows:
        raise ValueError(
            "dense corpus matrix row count does not match chunk IDs: "
            f"{matrix_rows} rows vs {len(normalized)} IDs"
        )
    return normalized


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_query_split(
    rows: list[dict[str, Any]],
    *,
    selected_splits: set[str],
    require_exclusive_split: bool,
) -> list[dict[str, Any]]:
    """Enforce physical protected-split isolation before loading model artifacts."""
    if require_exclusive_split and not selected_splits:
        raise ValueError("--require-exclusive-split requires --splits")
    observed = {str(row.get("split") or "") for row in rows}
    if require_exclusive_split and observed != selected_splits:
        raise ValueError(
            "query file is not physically exclusive to requested splits: "
            f"expected {sorted(selected_splits)}, found {sorted(observed)}"
        )
    selected = (
        [row for row in rows if str(row.get("split") or "") in selected_splits]
        if selected_splits
        else list(rows)
    )
    if not selected:
        raise ValueError(f"no query rows left after split filter: {sorted(selected_splits)}")
    eval_ids = [str(row.get("eval_id") or "") for row in selected]
    if any(not eval_id for eval_id in eval_ids) or len(eval_ids) != len(set(eval_ids)):
        raise ValueError("query rows must have unique non-empty eval_id values")
    return selected


def _full_bm25_ranking(
    query: str,
    corpus_rows: list[dict[str, Any]],
    *,
    lexical_index: dict[str, Any],
) -> list[tuple[str, float]]:
    """Rank every corpus row, appending zero-score ties by stable chunk ID."""
    positive_scores = lexical_rank(
        query,
        corpus_rows,
        limit=len(corpus_rows),
        index=lexical_index,
    )
    present = {chunk_id for chunk_id, _score in positive_scores}
    zero_scores = sorted(
        (str(row["chunk_id"]), 0.0) for row in corpus_rows if str(row["chunk_id"]) not in present
    )
    return [*positive_scores, *zero_scores]


def _first_rank(ranked_ids: list[str], relevant_ids: set[str]) -> int | None:
    return next(
        (rank for rank, chunk_id in enumerate(ranked_ids, 1) if chunk_id in relevant_ids),
        None,
    )


def _average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(ranked_ids, 1):
        if chunk_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def _ndcg(ranked_ids: list[str], grades: dict[str, int], cutoff: int) -> float:
    def gain(grade: int, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    dcg = sum(
        gain(grades.get(chunk_id, 0), rank) for rank, chunk_id in enumerate(ranked_ids[:cutoff], 1)
    )
    ideal = sorted(grades.values(), reverse=True)[:cutoff]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _relevance(
    query_row: dict[str, Any],
    *,
    corpus_ids: set[str],
    chunk_ids_by_video: dict[str, set[str]],
) -> dict[str, tuple[set[str], dict[str, int]]]:
    primary = str(query_row.get("positive_chunk_id") or "")
    if primary not in corpus_ids:
        raise ValueError(
            f"query {query_row.get('eval_id')} references missing positive chunk {primary!r}"
        )
    declared = query_row.get("local_window_chunk_ids") or query_row.get("relevant_chunk_ids") or []
    local = {str(value) for value in declared if str(value)} | {primary}
    missing_local = local - corpus_ids
    if missing_local:
        raise ValueError(
            f"query {query_row.get('eval_id')} references missing local chunks: "
            f"{sorted(missing_local)[:5]}"
        )
    video_id = str(query_row.get("positive_video_id") or "")
    same_video = set(chunk_ids_by_video.get(video_id, set()))
    if not same_video:
        raise ValueError(
            f"query {query_row.get('eval_id')} has no corpus chunks for video {video_id!r}"
        )
    return {
        "exact_chunk": ({primary}, {primary: 3}),
        "local_window": (
            local,
            {chunk_id: (3 if chunk_id == primary else 2) for chunk_id in local},
        ),
        "same_video": (same_video, {chunk_id: 1 for chunk_id in same_video}),
    }


def evaluate_rankings(
    query_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    rankings: list[list[tuple[str, float]]],
    *,
    artifact_depth: int,
    lane_ranks: list[dict[str, dict[str, int]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the shared L1/L2 metric contract and compact per-query evidence."""
    if len(query_rows) != len(rankings):
        raise ValueError("ranking count does not match query count")
    corpus_by_id = {str(row["chunk_id"]): row for row in corpus_rows}
    if len(corpus_by_id) != len(corpus_rows):
        raise ValueError("corpus chunk IDs must be unique")
    chunk_ids_by_video: dict[str, set[str]] = {}
    for chunk_id, row in corpus_by_id.items():
        chunk_ids_by_video.setdefault(str(row.get("youtube_video_id") or ""), set()).add(chunk_id)

    per_target: dict[str, dict[str, list[float | int | None]]] = {
        target: {"ranks": [], "map": [], "ndcg_5": [], "ndcg_10": []}
        for target in ("exact_chunk", "local_window", "same_video")
    }
    wrong_top_video = 0
    hard_negative_wins: list[bool] = []
    evidence_rows: list[dict[str, Any]] = []
    for index, (query_row, ranked) in enumerate(zip(query_rows, rankings, strict=True)):
        ranked_ids = [chunk_id for chunk_id, _score in ranked]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(f"ranking {index} contains duplicate chunk IDs")
        unknown = set(ranked_ids) - set(corpus_by_id)
        if unknown:
            raise ValueError(f"ranking {index} contains unknown chunks: {sorted(unknown)[:5]}")
        relevance = _relevance(
            query_row,
            corpus_ids=set(corpus_by_id),
            chunk_ids_by_video=chunk_ids_by_video,
        )
        ranks_by_chunk = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, 1)}
        target_evidence: dict[str, Any] = {}
        for target, (relevant_ids, grades) in relevance.items():
            rank = _first_rank(ranked_ids, relevant_ids)
            per_target[target]["ranks"].append(rank)
            per_target[target]["map"].append(_average_precision(ranked_ids, relevant_ids))
            per_target[target]["ndcg_5"].append(_ndcg(ranked_ids, grades, 5))
            per_target[target]["ndcg_10"].append(_ndcg(ranked_ids, grades, 10))
            target_evidence[f"positive_rank_{target}"] = rank

        positive_video = str(query_row.get("positive_video_id") or "")
        top_video = (
            str(corpus_by_id[ranked_ids[0]].get("youtube_video_id") or "") if ranked_ids else None
        )
        wrong_top_video += int(top_video != positive_video)
        hard_negatives = {
            str(value)
            for value in query_row.get("hard_negative_chunk_ids") or []
            if str(value) in corpus_by_id
        }
        hard_negative_win: bool | None = None
        if hard_negatives:
            local_ids = relevance["local_window"][0]
            positive_ranks = [ranks_by_chunk.get(chunk_id, math.inf) for chunk_id in local_ids]
            negative_ranks = [ranks_by_chunk.get(chunk_id, math.inf) for chunk_id in hard_negatives]
            hard_negative_win = max(positive_ranks) < min(negative_ranks)
            hard_negative_wins.append(hard_negative_win)

        top = ranked[:artifact_depth]
        evidence: dict[str, Any] = {
            "eval_id": str(query_row["eval_id"]),
            **target_evidence,
            "top_retrieved_chunk_ids": [chunk_id for chunk_id, _score in top],
            "top_retrieved_scores": [round(float(score), 10) for _chunk_id, score in top],
            "top_retrieved_video_id": top_video,
            "hard_negative_win": hard_negative_win,
        }
        if lane_ranks is not None:
            evidence["top_retrieved_lane_ranks"] = [
                lane_ranks[index].get(chunk_id, {}) for chunk_id, _score in top
            ]
        evidence_rows.append(evidence)

    count = len(query_rows)
    metrics: dict[str, Any] = {"count": count}
    for target, values in per_target.items():
        ranks = values["ranks"]
        for cutoff in CUTOFFS:
            metrics[f"{target}_recall_at_{cutoff}"] = round(
                sum(rank is not None and int(rank) <= cutoff for rank in ranks) / count,
                6,
            )
        metrics[f"{target}_mrr"] = round(
            sum(0.0 if rank is None else 1.0 / int(rank) for rank in ranks) / count,
            6,
        )
        metrics[f"{target}_map"] = round(sum(values["map"]) / count, 6)
        metrics[f"{target}_ndcg_at_5"] = round(sum(values["ndcg_5"]) / count, 6)
        metrics[f"{target}_ndcg_at_10"] = round(sum(values["ndcg_10"]) / count, 6)
    metrics["wrong_top_video_rate"] = round(wrong_top_video / count, 6)
    metrics["hard_negative_queries"] = len(hard_negative_wins)
    metrics["hard_negative_win_rate"] = (
        round(sum(hard_negative_wins) / len(hard_negative_wins), 6) if hard_negative_wins else None
    )
    return metrics, evidence_rows


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        key: round(float(right[key]) - float(left[key]), 6)
        for key in sorted(set(left) & set(right))
        if key != "count"
        and isinstance(left[key], (int, float))
        and not isinstance(left[key], bool)
        and isinstance(right[key], (int, float))
        and not isinstance(right[key], bool)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--dense-query-matrix", type=Path, required=True)
    parser.add_argument("--dense-corpus-matrix", type=Path, required=True)
    parser.add_argument("--dense-corpus-chunk-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-model-id", required=True)
    parser.add_argument("--dense-model-revision", default="unknown")
    parser.add_argument("--query-prefix-mode", default="memexai_youtube")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--dense-candidate-depth", type=int, default=50)
    parser.add_argument("--bm25-candidate-depth", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--artifact-depth", type=int, default=50)
    parser.add_argument("--normalization-tolerance", type=float, default=5e-3)
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--require-exclusive-split", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        min(
            args.dimensions,
            args.dense_candidate_depth,
            args.bm25_candidate_depth,
            args.rrf_k,
            args.artifact_depth,
        )
        < 1
    ):
        raise ValueError("dimensions, depths, and RRF k must be positive")
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    query_rows = validate_query_split(
        load_jsonl(args.queries_jsonl),
        selected_splits=selected_splits,
        require_exclusive_split=args.require_exclusive_split,
    )

    import numpy as np

    corpus_rows = load_jsonl(args.corpus_jsonl)
    corpus_matrix = np.load(args.dense_corpus_matrix)
    query_matrix = np.load(args.dense_query_matrix)
    corpus_ids = normalize_corpus_ids(
        json.loads(args.dense_corpus_chunk_ids.read_text(encoding="utf-8")),
        len(corpus_matrix),
    )
    if len(corpus_rows) != len(corpus_ids):
        raise ValueError("corpus JSONL row count does not match dense corpus IDs")
    corpus_json_ids = {str(row["chunk_id"]) for row in corpus_rows}
    if corpus_json_ids != set(corpus_ids):
        raise ValueError("corpus JSONL IDs do not match dense corpus IDs")
    if query_matrix.shape != (len(query_rows), args.dimensions):
        raise ValueError(
            "dense query matrix shape mismatch: "
            f"expected {(len(query_rows), args.dimensions)}, got {query_matrix.shape}"
        )
    if corpus_matrix.shape != (len(corpus_ids), args.dimensions):
        raise ValueError(
            "dense corpus matrix shape mismatch: "
            f"expected {(len(corpus_ids), args.dimensions)}, got {corpus_matrix.shape}"
        )
    normalization: dict[str, dict[str, float]] = {}
    for label, matrix in (("query", query_matrix), ("corpus", corpus_matrix)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"dense {label} matrix contains non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        deviation = float(np.max(np.abs(norms - 1.0)))
        normalization[label] = {
            "minimum_norm": float(np.min(norms)),
            "maximum_norm": float(np.max(norms)),
            "maximum_absolute_deviation": deviation,
        }
        if deviation > args.normalization_tolerance:
            raise ValueError(
                f"dense {label} matrix is not normalized: max norm deviation {deviation:.8f}"
            )

    scores = query_matrix @ corpus_matrix.T
    corpus_id_array = np.asarray(corpus_ids)
    dense_rankings: list[list[tuple[str, float]]] = []
    for row_scores in scores:
        order = np.lexsort((corpus_id_array, -row_scores))
        dense_rankings.append(
            [(corpus_ids[int(index)], float(row_scores[int(index)])) for index in order]
        )

    lexical_index = build_lexical_index(corpus_rows)
    bm25_rankings = [
        _full_bm25_ranking(str(row.get("query") or ""), corpus_rows, lexical_index=lexical_index)
        for row in query_rows
    ]
    rrf_rankings: list[list[tuple[str, float]]] = []
    rrf_lane_ranks: list[dict[str, dict[str, int]]] = []
    for dense, bm25 in zip(dense_rankings, bm25_rankings, strict=True):
        fused = rrf_fuse(
            {
                "dense": [chunk_id for chunk_id, _score in dense[: args.dense_candidate_depth]],
                "bm25": [chunk_id for chunk_id, _score in bm25[: args.bm25_candidate_depth]],
            },
            k=args.rrf_k,
        )
        rrf_rankings.append([(chunk_id, score) for chunk_id, score, _ranks in fused])
        rrf_lane_ranks.append({chunk_id: ranks for chunk_id, _score, ranks in fused})

    system_rankings = {
        "dense": (dense_rankings, None),
        "bm25": (bm25_rankings, None),
        "rrf": (rrf_rankings, rrf_lane_ranks),
    }
    systems: dict[str, Any] = {}
    evidence_by_eval_id = {
        str(row["eval_id"]): {
            "eval_id": str(row["eval_id"]),
            "split": str(row.get("split") or ""),
            "query_type": str(row.get("query_type") or "unknown"),
            "positive_chunk_id": str(row.get("positive_chunk_id") or ""),
            "positive_video_id": str(row.get("positive_video_id") or ""),
            "systems": {},
        }
        for row in query_rows
    }
    for name, (rankings, lane_ranks) in system_rankings.items():
        metric_values, evidence = evaluate_rankings(
            query_rows,
            corpus_rows,
            rankings,
            artifact_depth=args.artifact_depth,
            lane_ranks=lane_ranks,
        )
        systems[name] = {"metrics": metric_values}
        for item in evidence:
            evidence_by_eval_id[item["eval_id"]]["systems"][name] = {
                key: value for key, value in item.items() if key != "eval_id"
            }

    rows_path = args.output_dir / "retrieval_system_ablation_queries.jsonl"
    write_jsonl(rows_path, evidence_by_eval_id.values())
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "split_policy": "physical_exclusive_file" if args.require_exclusive_split else "filtered",
        "selected_splits": sorted(selected_splits) if selected_splits else "all",
        "dense": {
            "scorer": "normalized_dot_product_exact",
            "model_id": args.dense_model_id,
            "model_revision": args.dense_model_revision,
            "dimensions": args.dimensions,
            "query_prefix_mode": args.query_prefix_mode,
            "candidate_depth": args.dense_candidate_depth,
            "matrix_normalization": {
                "policy": "validate_only_no_silent_renormalization",
                "tolerance": args.normalization_tolerance,
                **normalization,
            },
        },
        "bm25": {
            "tokenizer_regex": TOKEN_RE.pattern,
            "lowercase": "unicode_casefold",
            "k1": 1.2,
            "b": 0.75,
            "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
            "zero_score_tie_break": "chunk_id_ascending",
            "candidate_depth": args.bm25_candidate_depth,
        },
        "rrf": {
            "k": args.rrf_k,
            "lanes": ["dense", "bm25"],
            "score": "sum(1 / (k + rank))",
        },
        "metrics": {
            "cutoffs": list(CUTOFFS),
            "exact_relevance": "primary_chunk_grade_3",
            "local_relevance": "primary_grade_3_declared_equivalents_grade_2",
            "same_video_relevance": "all_corpus_chunks_from_positive_video_grade_1",
            "hard_negative_win": "every_local_positive_above_every_labeled_negative",
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "protocol_sha256": stable_hash(protocol),
        "sources": {
            "queries": {"path": str(args.queries_jsonl), "sha256": sha256_file(args.queries_jsonl)},
            "corpus": {"path": str(args.corpus_jsonl), "sha256": sha256_file(args.corpus_jsonl)},
            "dense_query_matrix": {
                "path": str(args.dense_query_matrix),
                "sha256": sha256_file(args.dense_query_matrix),
            },
            "dense_corpus_matrix": {
                "path": str(args.dense_corpus_matrix),
                "sha256": sha256_file(args.dense_corpus_matrix),
            },
            "dense_corpus_chunk_ids": {
                "path": str(args.dense_corpus_chunk_ids),
                "sha256": sha256_file(args.dense_corpus_chunk_ids),
            },
        },
        "counts": {"queries": len(query_rows), "corpus_chunks": len(corpus_rows)},
        "systems": systems,
        "contribution_deltas": {
            "rrf_minus_dense": _metric_deltas(
                systems["dense"]["metrics"], systems["rrf"]["metrics"]
            ),
            "rrf_minus_bm25": _metric_deltas(systems["bm25"]["metrics"], systems["rrf"]["metrics"]),
        },
        "reranker": {
            "status": "not_run",
            "supported_by_this_evaluator": False,
            "reason": "no_approved_pinned_model_revision_input_template_and_truncation_protocol",
            "required_before_run": [
                "model_id_and_immutable_revision",
                "query_passage_input_template",
                "candidate_depths",
                "truncation_policy",
                "hardware_and_batch_one_timing_protocol",
            ],
        },
        "output": {
            "queries_path": rows_path.name,
            "queries_sha256": sha256_file(rows_path),
        },
    }
    manifest_path = args.output_dir / "retrieval_system_ablation_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
