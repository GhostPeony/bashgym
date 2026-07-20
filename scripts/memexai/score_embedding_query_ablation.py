#!/usr/bin/env python3
"""Compare query formatting strategies against a fixed corpus embedding matrix."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFIXES = {
    "raw": "",
    "qwen_retrieval": "Instruct: Given a search query, retrieve relevant passages that answer the query.\nQuery: ",
    "memexai_youtube": (
        "Instruct: Given a YouTube transcript search query, retrieve the transcript chunk "
        "that best answers it.\nQuery: "
    ),
}


def normalize_corpus_ids(corpus_ids: Any, matrix_rows: int) -> list[str]:
    if isinstance(corpus_ids, dict):
        normalized = [corpus_ids[str(idx)] for idx in range(matrix_rows)]
    elif isinstance(corpus_ids, list):
        normalized = [str(chunk_id) for chunk_id in corpus_ids]
    else:
        raise ValueError("corpus embedding chunk ids must be a JSON list or index-keyed object")
    if len(normalized) != matrix_rows:
        raise ValueError(
            "corpus embedding matrix row count does not match chunk ids: "
            f"{matrix_rows} matrix rows vs {len(normalized)} ids"
        )
    return normalized


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is invalid JSON") from exc
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def model_footprint_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def rank_of_any(ranked_ids: list[str], acceptable_ids: set[str]) -> int | None:
    for idx, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in acceptable_ids:
            return idx
    return None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cutoffs = (1, 3, 5, 10)

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {"count": 0}
        out: dict[str, Any] = {"count": len(group_rows)}
        for rank_key, prefix in (
            ("positive_rank_exact", "exact_chunk"),
            ("positive_rank_local_window", "local_window"),
            ("positive_rank_same_video", "same_video"),
        ):
            ranks = [int(row[rank_key]) for row in group_rows if row.get(rank_key)]
            if not ranks:
                continue
            for k in cutoffs:
                out[f"{prefix}_recall_at_{k}"] = round(
                    sum(1 for rank in ranks if rank <= k) / len(group_rows), 6
                )
            out[f"{prefix}_mrr"] = round(sum(1.0 / rank for rank in ranks) / len(group_rows), 6)
            out[f"{prefix}_mean_rank"] = round(sum(ranks) / len(ranks), 3)
        return out

    result: dict[str, Any] = {
        "overall": summarize(rows),
        "by_split": {},
        "by_query_type": {},
        "by_channel": {},
    }
    for split in sorted({row.get("split") or "" for row in rows}):
        result["by_split"][split] = summarize(
            [row for row in rows if (row.get("split") or "") == split]
        )
    for query_type in sorted({row["query_type"] for row in rows}):
        result["by_query_type"][query_type] = summarize(
            [row for row in rows if row["query_type"] == query_type]
        )
    for channel in sorted({row["channel"] for row in rows}):
        result["by_channel"][channel] = summarize(
            [row for row in rows if row["channel"] == channel]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--corpus-embedding-matrix", type=Path, required=True)
    parser.add_argument("--corpus-embedding-chunk-ids", type=Path, required=True)
    parser.add_argument("--truncate-dim", type=int, default=768)
    parser.add_argument(
        "--splits",
        default="",
        help="Optional comma-separated query splits to score, e.g. dev,test.",
    )
    parser.add_argument(
        "--require-exclusive-split",
        action="store_true",
        help="Reject a query file containing rows outside --splits before model imports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.latency_repetitions < 1:
        raise ValueError("--latency-repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = load_jsonl(args.queries_jsonl)
    selected_splits = {split.strip() for split in args.splits.split(",") if split.strip()}
    if args.require_exclusive_split:
        if not selected_splits:
            raise ValueError("--require-exclusive-split requires --splits")
        observed_splits = {str(row.get("split") or "") for row in base_rows}
        if observed_splits != selected_splits:
            raise ValueError(
                "query file is not physically exclusive to requested splits: "
                f"expected {sorted(selected_splits)}, found {sorted(observed_splits)}"
            )
    if selected_splits:
        base_rows = [row for row in base_rows if row.get("split") in selected_splits]
    if not base_rows:
        raise ValueError(f"no query rows left after split filter: {sorted(selected_splits)}")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    corpus_rows = load_jsonl(args.corpus_jsonl)
    corpus_matrix = np.load(args.corpus_embedding_matrix)
    corpus_ids = normalize_corpus_ids(
        json.loads(args.corpus_embedding_chunk_ids.read_text(encoding="utf-8")),
        len(corpus_matrix),
    )
    by_chunk_id = {row["chunk_id"]: row for row in corpus_rows}
    corpus_index_by_id = {chunk_id: idx for idx, chunk_id in enumerate(corpus_ids)}
    missing_positive_ids = sorted(
        {row["positive_chunk_id"] for row in base_rows} - set(corpus_index_by_id)
    )
    if missing_positive_ids:
        preview = ", ".join(missing_positive_ids[:5])
        raise ValueError(
            f"{len(missing_positive_ids)} positive chunk ids missing from matrix: {preview}"
        )

    model = SentenceTransformer(str(args.embedding_model_path), device=args.embedding_device)
    runs: dict[str, Any] = {}
    for mode, prefix in PREFIXES.items():
        rows = copy.deepcopy(base_rows)
        texts = [prefix + row["query"] for row in rows]
        elapsed_per_query_ms = []
        query_embeddings = None
        for repetition in range(args.latency_repetitions):
            started = time.perf_counter()
            encoded = model.encode(
                texts,
                batch_size=args.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=args.truncate_dim,
                show_progress_bar=repetition == 0,
            )
            elapsed_per_query_ms.append((time.perf_counter() - started) * 1000 / len(texts))
            if query_embeddings is None:
                query_embeddings = encoded
        if query_embeddings.shape[1] != args.truncate_dim:
            raise ValueError(
                f"expected {args.truncate_dim}d query embeddings, got {query_embeddings.shape[1]}"
            )
        scores = query_embeddings @ corpus_matrix.T
        for row_idx, row in enumerate(rows):
            positive_id = row["positive_chunk_id"]
            positive_video = row["positive_video_id"]
            ranked = np.argsort(-scores[row_idx])
            ranked_ids = [corpus_ids[int(corpus_idx)] for corpus_idx in ranked]
            positive_corpus_idx = corpus_index_by_id[positive_id]
            same_video_ids = {
                chunk_id
                for chunk_id, corpus_row in by_chunk_id.items()
                if corpus_row.get("youtube_video_id") == positive_video
            }
            row["positive_rank_exact"] = int(np.where(ranked == positive_corpus_idx)[0][0]) + 1
            row["positive_rank_local_window"] = rank_of_any(
                ranked_ids,
                set(
                    row.get("local_window_chunk_ids")
                    or row.get("relevant_chunk_ids")
                    or [positive_id]
                ),
            )
            row["positive_rank_same_video"] = rank_of_any(ranked_ids, same_video_ids)
            row["top_retrieved_chunk_ids"] = ranked_ids[:10]
            row["top_retrieved_video_id"] = by_chunk_id[ranked_ids[0]].get("youtube_video_id")
            row["top_retrieved_scores"] = [
                round(float(scores[row_idx, int(corpus_idx)]), 6) for corpus_idx in ranked[:10]
            ]
            row["positive_score"] = round(float(scores[row_idx, positive_corpus_idx]), 6)

        npy_path = (
            args.output_dir
            / f"{mode}-qwen3-embedding-0.6b-{args.truncate_dim}d-query-embeddings.npy"
        )
        np.save(npy_path, query_embeddings.astype("float32"))
        rows_path = args.output_dir / f"{mode}-retrieval_eval_queries.jsonl"
        write_jsonl(rows_path, rows)
        runs[mode] = {
            "prefix": prefix,
            "query_embedding_path": str(npy_path),
            "rows_path": str(rows_path),
            "median_query_latency_ms": statistics.median(elapsed_per_query_ms),
            "metrics": metrics(rows),
        }

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "queries_jsonl": str(args.queries_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "rows": len(base_rows),
        "splits": sorted(selected_splits) if selected_splits else "all",
        "query_types": dict(sorted(Counter(row["query_type"] for row in base_rows).items())),
        "embedding_model_path": str(args.embedding_model_path),
        "model_footprint_bytes": model_footprint_bytes(args.embedding_model_path),
        "truncate_dim": args.truncate_dim,
        "runs": runs,
    }
    out_path = args.output_dir / "query_format_ablation_manifest.json"
    write_json(out_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
