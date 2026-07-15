#!/usr/bin/env python3
"""Freeze deterministic MemexAI real and capped-mixed embedding train bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(seed: str, row: dict[str, Any]) -> str:
    eval_id = str(row.get("eval_id") or "")
    return hashlib.sha256(f"{seed}\0{eval_id}".encode()).hexdigest()


def largest_remainder_quotas(rows: list[dict[str, Any]], target: int) -> dict[str, int]:
    counts = Counter(str(row.get("query_type") or "unknown") for row in rows)
    exact = {key: target * count / len(rows) for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_capped_synthetic(
    rows: list[dict[str, Any]], *, target: int, seed: str
) -> list[dict[str, Any]]:
    if target < 0 or target > len(rows):
        raise ValueError(f"synthetic target {target} is outside 0..{len(rows)}")
    quotas = largest_remainder_quotas(rows, target)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    positive_counts: Counter[str] = Counter()
    while len(selected) < target:
        candidates = [
            row
            for row in rows
            if str(row.get("eval_id") or "") not in selected_ids
            and quotas.get(str(row.get("query_type") or "unknown"), 0) > 0
        ]
        if not candidates:
            raise ValueError("unable to satisfy deterministic synthetic query-type quotas")
        choice = min(
            candidates,
            key=lambda row: (
                positive_counts[str(row.get("positive_chunk_id") or "")],
                stable_key(seed, row),
                str(row.get("eval_id") or ""),
            ),
        )
        selected.append(choice)
        selected_ids.add(str(choice["eval_id"]))
        positive_counts[str(choice["positive_chunk_id"])] += 1
        quotas[str(choice.get("query_type") or "unknown")] -= 1
    return sorted(selected, key=lambda row: str(row["eval_id"]))


def validate_queries(rows: list[dict[str, Any]], corpus: list[dict[str, Any]]) -> None:
    corpus_ids = {str(row["chunk_id"]) for row in corpus}
    eval_ids: set[str] = set()
    normalized_queries: set[str] = set()
    for row in rows:
        eval_id = str(row.get("eval_id") or "")
        query = " ".join(str(row.get("query") or "").split()).casefold()
        positive_id = str(row.get("positive_chunk_id") or "")
        if not eval_id or eval_id in eval_ids:
            raise ValueError(f"missing or duplicate eval_id: {eval_id!r}")
        if not query or query in normalized_queries:
            raise ValueError(f"missing or duplicate normalized query: {query!r}")
        if positive_id not in corpus_ids:
            raise ValueError(f"query {eval_id} references missing positive {positive_id}")
        eval_ids.add(eval_id)
        normalized_queries.add(query)


def validate_heldout_disjoint(
    train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]]
) -> None:
    train_videos = {str(row.get("positive_video_id") or "") for row in train_rows}
    heldout_videos = {str(row.get("positive_video_id") or "") for row in heldout_rows}
    train_chunks = {str(row.get("positive_chunk_id") or "") for row in train_rows}
    heldout_chunks = {str(row.get("positive_chunk_id") or "") for row in heldout_rows}
    video_overlap = sorted((train_videos & heldout_videos) - {""})
    chunk_overlap = sorted((train_chunks & heldout_chunks) - {""})
    if video_overlap or chunk_overlap:
        raise ValueError(
            "training and held-out rows overlap: "
            f"videos={video_overlap[:5]}, chunks={chunk_overlap[:5]}"
        )


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "query_types": dict(sorted(Counter(str(row.get("query_type")) for row in rows).items())),
        "positive_chunks": len({str(row.get("positive_chunk_id")) for row in rows}),
        "videos": len({str(row.get("positive_video_id")) for row in rows}),
        "generators": dict(sorted(Counter(str(row.get("generator")) for row in rows).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-queries-jsonl", type=Path, required=True)
    parser.add_argument("--real-queries-jsonl", type=Path, required=True)
    parser.add_argument("--synthetic-queries-jsonl", type=Path, required=True)
    parser.add_argument("--real-corpus-jsonl", type=Path, required=True)
    parser.add_argument("--synthetic-corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-target", type=int, default=196)
    parser.add_argument("--seed", default="memexai-mixed-25-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {
        "template_queries": args.template_queries_jsonl,
        "real_queries": args.real_queries_jsonl,
        "synthetic_queries": args.synthetic_queries_jsonl,
        "real_corpus": args.real_corpus_jsonl,
        "synthetic_corpus": args.synthetic_corpus_jsonl,
    }
    template = load_jsonl(args.template_queries_jsonl)
    original_train = [row for row in template if row.get("split") == "train"]
    heldout = [row for row in template if row.get("split") in {"dev", "test"}]
    heldout_dev = [row for row in template if row.get("split") == "dev"]
    heldout_test = [row for row in template if row.get("split") == "test"]
    real = load_jsonl(args.real_queries_jsonl)
    synthetic = load_jsonl(args.synthetic_queries_jsonl)
    real_corpus = load_jsonl(args.real_corpus_jsonl)
    synthetic_corpus = load_jsonl(args.synthetic_corpus_jsonl)
    selected_synthetic = select_capped_synthetic(
        synthetic, target=args.synthetic_target, seed=args.seed
    )

    real_queries = sorted(original_train + real, key=lambda row: str(row["eval_id"]))
    mixed_queries = sorted(real_queries + selected_synthetic, key=lambda row: str(row["eval_id"]))
    selected_synthetic_ids = {
        str(row["positive_chunk_id"]) for row in selected_synthetic
    }
    mixed_corpus = real_corpus + [
        row for row in synthetic_corpus if str(row["chunk_id"]) in selected_synthetic_ids
    ]
    validate_queries(real_queries, real_corpus)
    validate_queries(mixed_queries, mixed_corpus)
    validate_heldout_disjoint(real_queries, heldout)
    validate_heldout_disjoint(mixed_queries, heldout)

    outputs = {
        "real_queries": args.output_dir / "real702.jsonl",
        "mixed_queries": args.output_dir / f"mixed{len(mixed_queries)}.jsonl",
        "mixed_corpus": args.output_dir / "corpus-augmented.jsonl",
        "heldout_dev": args.output_dir / "heldout-dev.jsonl",
        "heldout_dev_test": args.output_dir / "heldout-dev-test.jsonl",
        "heldout_test": args.output_dir / "heldout-test.jsonl",
    }
    write_jsonl(outputs["real_queries"], real_queries)
    write_jsonl(outputs["mixed_queries"], mixed_queries)
    write_jsonl(outputs["mixed_corpus"], mixed_corpus)
    write_jsonl(outputs["heldout_dev"], heldout_dev)
    write_jsonl(outputs["heldout_dev_test"], heldout)
    write_jsonl(outputs["heldout_test"], heldout_test)

    manifest = {
        "schema_version": "memexai.embedding-training-bundle.v1",
        "policy": {
            "name": "original-plus-real-and-capped-synthetic",
            "seed": args.seed,
            "synthetic_target": args.synthetic_target,
            "synthetic_share_of_dd_augmentation": round(
                args.synthetic_target / (len(real) + args.synthetic_target), 6
            ),
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "statistics": {
            "original_train": stats(original_train),
            "real_dd": stats(real),
            "selected_synthetic_dd": stats(selected_synthetic),
            "real_arm": stats(real_queries),
            "mixed_arm": stats(mixed_queries),
            "heldout_dev": stats(heldout_dev),
            "heldout_dev_test": stats(heldout),
            "heldout_test": stats(heldout_test),
            "mixed_corpus_rows": len(mixed_corpus),
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
