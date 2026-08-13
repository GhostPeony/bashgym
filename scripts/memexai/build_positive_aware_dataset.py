#!/usr/bin/env python3
"""Build deterministic grouped-positive MemexAI training data with safe negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "memexai.positive-aware-training.v1"
PASSAGE_REPRESENTATION = "embedding_text_or_title_double_newline_text:v1"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def passage_text(row: dict[str, Any]) -> str:
    embedded = str(row.get("embedding_text") or "").strip()
    if embedded:
        return embedded
    title = clean_text(row.get("title"))
    text = clean_text(row.get("text"))
    return f"{title}\n\n{text}" if title else text


def tokens(value: Any) -> list[str]:
    return TOKEN_RE.findall(clean_text(value).casefold())


def token_jaccard(left: Any, right: Any) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _chunk_index(row: dict[str, Any]) -> int | None:
    value = row.get("chunk_index")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        return float(left["start_seconds"]) < float(right["end_seconds"]) and float(
            left["end_seconds"]
        ) > float(right["start_seconds"])
    except (KeyError, TypeError, ValueError):
        return False


def exclusion_reason(
    positive: dict[str, Any],
    candidate: dict[str, Any],
    positive_ids: set[str],
    *,
    near_duplicate_threshold: float,
) -> str | None:
    candidate_id = str(candidate.get("chunk_id") or "")
    if candidate_id in positive_ids:
        return "positive_or_equivalent"
    same_video = str(candidate.get("youtube_video_id") or "") == str(
        positive.get("youtube_video_id") or ""
    )
    if same_video:
        left, right = _chunk_index(positive), _chunk_index(candidate)
        if left is not None and right is not None and abs(left - right) <= 2:
            return "same_video_adjacent"
        if _overlaps(positive, candidate):
            return "same_video_overlap"
        return "same_video_unreviewed"
    if token_jaccard(passage_text(positive), passage_text(candidate)) >= near_duplicate_threshold:
        return "near_duplicate"
    return None


def build_lexical_index(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    documents = [tokens(passage_text(row)) for row in corpus]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    count = max(len(documents), 1)
    return {
        "documents": documents,
        "document_frequency": document_frequency,
        "count": count,
        "average_length": sum(map(len, documents)) / count or 1.0,
    }


def lexical_rank(
    query: str,
    corpus: list[dict[str, Any]],
    limit: int = 50,
    *,
    index: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    """Return deterministic BM25-style candidates without adding a runtime dependency."""
    index = index or build_lexical_index(corpus)
    documents = index["documents"]
    document_frequency = index["document_frequency"]
    count = index["count"]
    average_length = index["average_length"]
    query_terms = tokens(query)
    scored: list[tuple[str, float]] = []
    for row, document in zip(corpus, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            doc_frequency = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (count - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = frequency + 1.2 * (0.25 + 0.75 * len(document) / average_length)
            score += inverse_frequency * frequency * 2.2 / denominator
        if score > 0:
            scored.append((str(row["chunk_id"]), round(score, 8)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def rrf_fuse(
    lanes: dict[str, list[str]], *, k: int = 60
) -> list[tuple[str, float, dict[str, int]]]:
    scores: Counter[str] = Counter()
    ranks: dict[str, dict[str, int]] = {}
    for lane, chunk_ids in sorted(lanes.items()):
        seen: set[str] = set()
        for rank, chunk_id in enumerate(chunk_ids, 1):
            normalized = str(chunk_id)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            scores[normalized] += 1.0 / (k + rank)
            ranks.setdefault(normalized, {})[lane] = rank
    return [
        (chunk_id, round(score, 10), dict(sorted(ranks[chunk_id].items())))
        for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _positive_ids(row: dict[str, Any], corpus_ids: set[str]) -> list[str]:
    primary = str(row.get("positive_chunk_id") or "")
    declared = row.get("local_window_chunk_ids") or row.get("relevant_chunk_ids") or []
    ordered: list[str] = []
    for chunk_id in [*declared, primary]:
        normalized = str(chunk_id or "")
        if normalized and normalized in corpus_ids and normalized not in ordered:
            ordered.append(normalized)
    if primary and primary in ordered:
        ordered.remove(primary)
        insert_at = min(len(ordered), max(0, len(ordered) // 2))
        ordered.insert(insert_at, primary)
    return ordered


def build_grouped_examples(
    query_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    *,
    min_negatives: int = 3,
    max_negatives: int = 7,
    candidate_depth: int = 50,
    near_duplicate_threshold: float = 0.85,
    dense_rankings: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if min_negatives < 1 or max_negatives < min_negatives:
        raise ValueError("negative limits must satisfy 1 <= min <= max")
    corpus_by_id = {str(row["chunk_id"]): row for row in corpus_rows}
    corpus_ids = set(corpus_by_id)
    lexical_index = build_lexical_index(corpus_rows)
    output: list[dict[str, Any]] = []
    for query_row in sorted(query_rows, key=lambda row: str(row.get("eval_id") or "")):
        primary_id = str(query_row.get("positive_chunk_id") or "")
        if primary_id not in corpus_by_id:
            raise ValueError(f"query {query_row.get('eval_id')} missing positive {primary_id}")
        positive_ids = _positive_ids(query_row, corpus_ids)
        positive_set = set(positive_ids)
        eval_id = str(query_row.get("eval_id") or "")
        labeled = [str(value) for value in query_row.get("hard_negative_chunk_ids") or []]
        dense = (dense_rankings or {}).get(eval_id) or [
            str(value) for value in query_row.get("top_retrieved_chunk_ids") or []
        ]
        bm25 = [
            chunk_id
            for chunk_id, _score in lexical_rank(
                str(query_row.get("query") or ""),
                corpus_rows,
                candidate_depth,
                index=lexical_index,
            )
        ]
        candidates: list[tuple[str, str, float | None, dict[str, int]]] = [
            (chunk_id, "labeled", 1.0 / rank, {"labeled": rank})
            for rank, chunk_id in enumerate(labeled, 1)
        ]
        candidates.extend(
            (
                chunk_id,
                "+".join(sorted(lane_ranks)) if len(lane_ranks) > 1 else next(iter(lane_ranks)),
                score,
                lane_ranks,
            )
            for chunk_id, score, lane_ranks in rrf_fuse(
                {"dense": dense[:candidate_depth], "bm25": bm25[:candidate_depth]}, k=60
            )
        )
        safe: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        seen: set[str] = set()
        for chunk_id, source, score, lane_ranks in candidates:
            if chunk_id in seen or chunk_id not in corpus_by_id:
                continue
            seen.add(chunk_id)
            reason = exclusion_reason(
                corpus_by_id[primary_id],
                corpus_by_id[chunk_id],
                positive_set,
                near_duplicate_threshold=near_duplicate_threshold,
            )
            if reason:
                exclusions.append({"chunk_id": chunk_id, "reason": reason, "source": source})
                continue
            safe.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "score": score,
                    "lane_ranks": lane_ranks,
                }
            )
            if len(safe) >= max_negatives:
                break
        status = "ready" if len(safe) >= min_negatives else "insufficient_safe_negatives"
        selected = safe if status == "ready" else []
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_id": str(query_row.get("eval_id") or ""),
                "split": str(query_row.get("split") or "train"),
                "query": clean_text(query_row.get("query")),
                "query_type": str(query_row.get("query_type") or "unknown"),
                "positive_video_id": str(query_row.get("positive_video_id") or ""),
                "primary_positive_chunk_id": primary_id,
                "positive_chunk_ids": positive_ids,
                "hard_negative_chunk_ids": [item["chunk_id"] for item in selected],
                "hard_negatives": selected,
                "excluded_candidates": exclusions,
                "status": status,
                "passage_representation": PASSAGE_REPRESENTATION,
                "record_hash": stable_hash(
                    {
                        "query": clean_text(query_row.get("query")),
                        "positive_ids": positive_ids,
                        "negative_ids": [item["chunk_id"] for item in selected],
                        "representation": PASSAGE_REPRESENTATION,
                    }
                ),
            }
        )
    return output


def validate_grouped_examples(
    rows: list[dict[str, Any]], *, corpus_ids: set[str], min_negatives: int
) -> None:
    seen_ids: set[str] = set()
    for row in rows:
        eval_id = str(row.get("eval_id") or "")
        if not eval_id or eval_id in seen_ids:
            raise ValueError(f"missing or duplicate eval_id: {eval_id!r}")
        seen_ids.add(eval_id)
        positives = {str(value) for value in row.get("positive_chunk_ids") or []}
        negatives = {str(value) for value in row.get("hard_negative_chunk_ids") or []}
        if positives & negatives:
            raise ValueError(f"{eval_id} has positive/negative overlap")
        missing = (positives | negatives) - corpus_ids
        if missing:
            raise ValueError(f"{eval_id} references missing chunks: {sorted(missing)[:5]}")
        if row.get("status") == "ready" and len(negatives) < min_negatives:
            raise ValueError(f"{eval_id} is ready with fewer than {min_negatives} negatives")


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [row for row in rows if row["status"] == "ready"]
    exclusion_reasons = Counter(
        item["reason"] for row in rows for item in row.get("excluded_candidates") or []
    )
    negative_sources = Counter(
        item["source"] for row in ready for item in row.get("hard_negatives") or []
    )
    return {
        "rows": len(rows),
        "ready_rows": len(ready),
        "insufficient_safe_negatives": len(rows) - len(ready),
        "positive_group_sizes": dict(
            sorted(Counter(str(len(row["positive_chunk_ids"])) for row in rows).items())
        ),
        "negative_counts": dict(
            sorted(Counter(str(len(row["hard_negative_chunk_ids"])) for row in rows).items())
        ),
        "negative_sources": dict(sorted(negative_sources.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
    }


def load_dense_rankings(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    rankings: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        eval_id = str(row.get("eval_id") or row.get("id") or "")
        values = row.get("top_retrieved_chunk_ids") or row.get("chunk_ids") or []
        if not eval_id or eval_id in rankings:
            raise ValueError(f"dense rankings contain missing or duplicate eval_id: {eval_id!r}")
        rankings[eval_id] = [str(value) for value in values]
    return rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-negatives", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=7)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--dense-rankings-jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries = load_jsonl(args.queries_jsonl)
    corpus = load_jsonl(args.corpus_jsonl)
    rows = build_grouped_examples(
        queries,
        corpus,
        min_negatives=args.min_negatives,
        max_negatives=args.max_negatives,
        candidate_depth=args.candidate_depth,
        near_duplicate_threshold=args.near_duplicate_threshold,
        dense_rankings=load_dense_rankings(args.dense_rankings_jsonl),
    )
    validate_grouped_examples(
        rows,
        corpus_ids={str(row["chunk_id"]) for row in corpus},
        min_negatives=args.min_negatives,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "positive-aware-train.jsonl"
    manifest_path = args.output_dir / "positive-aware-manifest.json"
    write_jsonl(output_path, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "queries": {"path": str(args.queries_jsonl), "sha256": sha256_file(args.queries_jsonl)},
            "corpus": {"path": str(args.corpus_jsonl), "sha256": sha256_file(args.corpus_jsonl)},
        },
        "passage_representation": PASSAGE_REPRESENTATION,
        "policy": {
            "min_negatives": args.min_negatives,
            "max_negatives": args.max_negatives,
            "candidate_depth_per_lane": args.candidate_depth,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "same_video_policy": "exclude_without_non_equivalence_review",
            "fusion": "labeled_first_then_rrf_k60_dense_bm25",
        },
        "statistics": summary(rows),
        "output": {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "canonical_records_sha256": stable_hash(rows),
        },
    }
    if args.dense_rankings_jsonl:
        manifest["sources"]["dense_rankings"] = {
            "path": str(args.dense_rankings_jsonl),
            "sha256": sha256_file(args.dense_rankings_jsonl),
        }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
