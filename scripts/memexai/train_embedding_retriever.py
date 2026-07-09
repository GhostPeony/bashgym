#!/usr/bin/env python3
"""Fine-tune a MemexAI YouTube retrieval embedding model on query-positive pairs.

This is the guarded training entrypoint for the MemexAI embedding loop. It uses
real transcript chunks as positive passages and synthetic/user-like retrieval
queries as anchors. Keep production fine-tunes behind the judged eval/comparison
gate; use small max-step runs to smoke-test the training path.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUERY_PREFIXES = {
    "raw": "",
    "qwen_retrieval": (
        "Instruct: Given a search query, retrieve relevant passages that answer the query.\n"
        "Query: "
    ),
    "memexai_youtube": (
        "Instruct: Given a YouTube transcript search query, retrieve the transcript chunk "
        "that best answers it.\nQuery: "
    ),
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def format_query_for_embedding(query: str, prefix_mode: str) -> str:
    try:
        return f"{QUERY_PREFIXES[prefix_mode]}{query}"
    except KeyError as exc:
        raise ValueError(f"unknown query prefix mode: {prefix_mode}") from exc


def build_pairs(
    query_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    *,
    train_splits: set[str],
    query_prefix_mode: str,
    max_pairs: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    corpus_by_id = {row["chunk_id"]: row for row in corpus_rows}
    pairs: list[dict[str, Any]] = []
    missing_positive_ids: Counter[str] = Counter()
    for row in query_rows:
        if str(row.get("split") or "") not in train_splits:
            continue
        positive_id = row["positive_chunk_id"]
        positive = corpus_by_id.get(positive_id)
        if not positive:
            missing_positive_ids[positive_id] += 1
            continue
        query = clean_text(str(row.get("query") or ""))
        passage = clean_text(str(positive.get("text") or ""))
        if not query or not passage:
            continue
        pairs.append(
            {
                "eval_id": row.get("eval_id"),
                "split": row.get("split"),
                "query_type": row.get("query_type"),
                "query": query,
                "query_for_embedding": format_query_for_embedding(query, query_prefix_mode),
                "positive_chunk_id": positive_id,
                "positive_video_id": row.get("positive_video_id"),
                "positive_text": passage,
                "title": positive.get("title") or row.get("title") or "",
                "channel": positive.get("channel") or row.get("channel") or "",
            }
        )
    if missing_positive_ids:
        preview = ", ".join(chunk_id for chunk_id, _ in missing_positive_ids.most_common(5))
        raise ValueError(
            f"{sum(missing_positive_ids.values())} query rows reference missing positives: {preview}"
        )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def parse_split_set(value: str) -> set[str]:
    splits = {item.strip() for item in value.split(",") if item.strip()}
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-splits", type=parse_split_set, default={"train"})
    parser.add_argument(
        "--query-prefix-mode",
        choices=sorted(QUERY_PREFIXES),
        default="memexai_youtube",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--truncate-dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("MultipleNegativesRankingLoss needs batch-size >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = load_jsonl(args.queries_jsonl)
    corpus_rows = load_jsonl(args.corpus_jsonl)
    pairs = build_pairs(
        query_rows,
        corpus_rows,
        train_splits=args.train_splits,
        query_prefix_mode=args.query_prefix_mode,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    if len(pairs) < args.batch_size:
        raise ValueError(f"need at least {args.batch_size} train pairs, got {len(pairs)}")

    pairs_path = args.output_dir / "train_pairs.jsonl"
    manifest_path = args.output_dir / "training_manifest.json"
    final_model_path = args.output_dir / "final"
    write_jsonl(pairs_path, pairs)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "dry_run" if args.dry_run else "started",
        "run_kind": "memexai_embedding_retriever_train",
        "queries_jsonl": str(args.queries_jsonl),
        "corpus_jsonl": str(args.corpus_jsonl),
        "base_model_path": str(args.base_model_path),
        "output_dir": str(args.output_dir),
        "final_model_path": str(final_model_path),
        "train_pairs_jsonl": str(pairs_path),
        "query_prefix": {
            "mode": args.query_prefix_mode,
            "prefix": QUERY_PREFIXES[args.query_prefix_mode],
            "applied_to": "query",
            "corpus_prefix": "",
        },
        "train_splits": sorted(args.train_splits),
        "train_pairs": len(pairs),
        "query_types": dict(sorted(Counter(pair["query_type"] for pair in pairs).items())),
        "channels": dict(sorted(Counter(pair["channel"] for pair in pairs).items())),
        "training": {
            "loss": "MultipleNegativesRankingLoss",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_seq_length": args.max_seq_length,
            "truncate_dim_for_eval": args.truncate_dim,
            "device": args.device,
            "use_amp": args.use_amp,
            "seed": args.seed,
            "production_gate": (
                "Smoke/proof run only unless a judged eval-v1 and Qwen/Gemini/BM25 "
                "comparison show fine-tuning is needed."
            ),
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    model = SentenceTransformer(
        str(args.base_model_path), device=args.device, local_files_only=True
    )
    model.max_seq_length = args.max_seq_length
    examples = [
        InputExample(texts=[pair["query_for_embedding"], pair["positive_text"]]) for pair in pairs
    ]
    train_loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = MultipleNegativesRankingLoss(model)
    steps_per_epoch = args.max_steps if args.max_steps > 0 else None
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=args.warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        output_path=str(final_model_path),
        save_best_model=False,
        use_amp=args.use_amp,
        checkpoint_path=str(args.output_dir / "checkpoints"),
        checkpoint_save_steps=max(1, args.max_steps) if args.max_steps > 0 else 50,
        checkpoint_save_total_limit=2,
        show_progress_bar=True,
    )
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
