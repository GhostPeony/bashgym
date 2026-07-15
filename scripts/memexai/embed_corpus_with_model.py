#!/usr/bin/env python3
"""Embed a MemexAI transcript corpus with a specific SentenceTransformer model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def format_passage_for_embedding(row: dict[str, Any]) -> str:
    existing = str(row.get("embedding_text") or "").strip()
    if existing:
        return existing
    title = clean_text(str(row.get("title") or ""))
    text = clean_text(str(row.get("text") or ""))
    return f"{title}\n\n{text}" if title else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="corpus")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--truncate-dim", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.corpus_jsonl)
    if not rows:
        raise ValueError("corpus is empty")

    ids = [str(row["chunk_id"]) for row in rows]
    texts = [format_passage_for_embedding(row) for row in rows]
    if any(not text for text in texts):
        empty_count = sum(1 for text in texts if not text)
        raise ValueError(f"{empty_count} corpus rows have empty text")

    model = SentenceTransformer(str(args.model_path), device=args.device, local_files_only=True)
    model.max_seq_length = args.max_seq_length
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncate_dim=args.truncate_dim,
        show_progress_bar=True,
    )
    if embeddings.shape != (len(rows), args.truncate_dim):
        raise ValueError(
            f"expected {(len(rows), args.truncate_dim)} embeddings, got {embeddings.shape}"
        )

    matrix_path = args.output_dir / f"{args.output_prefix}-{args.truncate_dim}d-corpus.npy"
    ids_path = args.output_dir / f"{args.output_prefix}-{args.truncate_dim}d-chunk-ids.json"
    manifest_path = (
        args.output_dir / f"{args.output_prefix}-{args.truncate_dim}d-corpus-manifest.json"
    )
    np.save(matrix_path, embeddings.astype("float32"))
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_jsonl": str(args.corpus_jsonl),
        "model_path": str(args.model_path),
        "rows": len(rows),
        "embedding_shape": list(embeddings.shape),
        "truncate_dim": args.truncate_dim,
        "max_seq_length": args.max_seq_length,
        "normalize_embeddings": True,
        "passage_format": "embedding_text_or_title_double_newline_text",
        "outputs": {
            "matrix_npy": str(matrix_path),
            "chunk_ids_json": str(ids_path),
            "manifest_json": str(manifest_path),
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
