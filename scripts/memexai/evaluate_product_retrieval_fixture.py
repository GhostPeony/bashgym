#!/usr/bin/env python3
"""Evaluate a local SentenceTransformer on the frozen SearchTube product fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

QUERY_PREFIX = (
    "Instruct: Given a YouTube transcript search query, retrieve the transcript chunk "
    "that best answers it.\nQuery: "
)


def document_text(doc: dict[str, Any]) -> str:
    aliases = doc.get("aliases") if isinstance(doc.get("aliases"), list) else []
    parts = [
        str(doc.get("title") or "").strip(),
        "Aliases: " + ", ".join(str(alias) for alias in aliases if str(alias).strip()),
        str(doc.get("text") or "").strip(),
    ]
    return "\n".join(part for part in parts if part.strip())


def reciprocal_rank(ranked: list[str], expected: set[str]) -> float:
    return next((1 / rank for rank, doc_id in enumerate(ranked, 1) if doc_id in expected), 0.0)


def ndcg_at_5(ranked: list[str], grades: dict[str, float]) -> float:
    def dcg(ids: list[str]) -> float:
        return sum(
            (2 ** grades.get(doc_id, 0.0) - 1) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ids[:5], 1)
        )

    ideal = sorted(grades, key=grades.get, reverse=True)
    denominator = dcg(ideal)
    return dcg(ranked) / denominator if denominator else 0.0


def evaluate_rankings(fixture: dict[str, Any], rankings: dict[str, list[str]]) -> dict[str, Any]:
    totals = {
        "recallAt1": 0,
        "recallAt3": 0,
        "recallAt5": 0,
        "mrr": 0.0,
        "ndcgAt5": 0.0,
        "wrongTopRate": 0,
        "hardNegativeAbovePositiveRate": 0,
    }
    rows = []
    for query in fixture["queries"]:
        ranked = rankings[str(query["id"])]
        expected = set(map(str, query.get("expectedDocIds") or []))
        hard_negatives = set(map(str, query.get("hardNegativeDocIds") or []))
        grades = {str(key): float(value) for key, value in (query.get("relevance") or {}).items()}
        for doc_id in expected:
            grades.setdefault(doc_id, 1.0)
        expected_rank = min(ranked.index(doc_id) for doc_id in expected if doc_id in ranked)
        hard_rank = min(
            (ranked.index(doc_id) for doc_id in hard_negatives if doc_id in ranked),
            default=len(ranked),
        )
        row = {
            "id": query["id"],
            "topDocIds": ranked[:5],
            "expectedRank": expected_rank + 1,
            "hardNegativeAbovePositive": hard_rank < expected_rank,
        }
        rows.append(row)
        totals["recallAt1"] += int(bool(expected.intersection(ranked[:1])))
        totals["recallAt3"] += int(bool(expected.intersection(ranked[:3])))
        totals["recallAt5"] += int(bool(expected.intersection(ranked[:5])))
        totals["mrr"] += reciprocal_rank(ranked, expected)
        totals["ndcgAt5"] += ndcg_at_5(ranked, grades)
        totals["wrongTopRate"] += int(ranked[0] not in expected)
        totals["hardNegativeAbovePositiveRate"] += int(hard_rank < expected_rank)
    count = len(rows)
    metrics = {key: round(value / count, 6) for key, value in totals.items()}
    bar = fixture.get("passingBar") or {}
    passed = (
        metrics["recallAt3"] >= float(bar.get("minRecallAt3", 0))
        and metrics["mrr"] >= float(bar.get("minMrr", 0))
        and metrics["ndcgAt5"] >= float(bar.get("minNdcgAt5", 0))
        and metrics["wrongTopRate"] <= float(bar.get("maxWrongTopRate", 1))
        and metrics["hardNegativeAbovePositiveRate"]
        <= float(bar.get("maxHardNegativeAbovePositiveRate", 1))
    )
    return {"passed": passed, "metrics": metrics, "queries": rows, "passingBar": bar}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--truncate-dim", type=int, default=768)
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    model = SentenceTransformer(str(args.model_path), device=args.device, local_files_only=True)
    documents = fixture["documents"]
    queries = fixture["queries"]
    doc_vectors = model.encode(
        [document_text(doc) for doc in documents],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        truncate_dim=args.truncate_dim,
        convert_to_numpy=True,
    )
    query_vectors = model.encode(
        [QUERY_PREFIX + str(query["query"]) for query in queries],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        truncate_dim=args.truncate_dim,
        convert_to_numpy=True,
    )
    scores = np.asarray(query_vectors) @ np.asarray(doc_vectors).T
    doc_ids = [str(doc["id"]) for doc in documents]
    rankings = {
        str(query["id"]): [doc_ids[index] for index in np.argsort(-scores[row_index])]
        for row_index, query in enumerate(queries)
    }
    report = {
        "fixture": fixture["name"],
        "candidate": args.candidate_name,
        "dimensions": args.truncate_dim,
        "queryPrefix": QUERY_PREFIX,
        **evaluate_rankings(fixture, rankings),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
