#!/usr/bin/env python3
"""Fine-tune a MemexAI YouTube retrieval embedding model on query-positive pairs.

This is the guarded training entrypoint for the MemexAI embedding loop. It uses
real transcript chunks as positive passages and synthetic/user-like retrieval
queries as anchors. Keep production fine-tunes behind the judged eval/comparison
gate; use small max-step runs to smoke-test the training path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

DEFAULT_MAX_EXPANDED_SEQUENCES_PER_BATCH = 64


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def format_passage_for_embedding(row: dict[str, Any]) -> str:
    existing = str(row.get("embedding_text") or "").strip()
    if existing:
        return existing
    title = clean_text(str(row.get("title") or ""))
    text = clean_text(str(row.get("text") or ""))
    return f"{title}\n\n{text}" if title else text


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
        passage = format_passage_for_embedding(positive)
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


def build_grouped_pairs(
    grouped_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    *,
    query_prefix_mode: str,
    max_pairs: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Resolve positive-aware records into explicit-negative training examples."""
    corpus_by_id = {str(row["chunk_id"]): row for row in corpus_rows}
    pairs: list[dict[str, Any]] = []
    for row in grouped_rows:
        if row.get("status") != "ready":
            continue
        positive_id = str(row.get("primary_positive_chunk_id") or "")
        negative_ids = [str(value) for value in row.get("hard_negative_chunk_ids") or []]
        missing = [
            chunk_id for chunk_id in [positive_id, *negative_ids] if chunk_id not in corpus_by_id
        ]
        if missing:
            raise ValueError(
                f"grouped row {row.get('eval_id')} references missing chunks: {missing[:5]}"
            )
        query = clean_text(str(row.get("query") or ""))
        positive_text = format_passage_for_embedding(corpus_by_id[positive_id])
        negative_texts = [
            format_passage_for_embedding(corpus_by_id[chunk_id]) for chunk_id in negative_ids
        ]
        if not query or not positive_text or any(not text for text in negative_texts):
            continue
        video_id = str(row.get("positive_video_id") or "")
        pairs.append(
            {
                "eval_id": row.get("eval_id"),
                "split": row.get("split"),
                "query_type": row.get("query_type"),
                "query": query,
                "query_for_embedding": format_query_for_embedding(query, query_prefix_mode),
                "positive_chunk_id": positive_id,
                "positive_chunk_ids": [
                    str(value) for value in row.get("positive_chunk_ids") or [positive_id]
                ],
                "positive_video_id": video_id,
                "positive_collision_group": f"video:{video_id}",
                "positive_text": positive_text,
                "negative_chunk_ids": negative_ids,
                "negative_texts": negative_texts,
                "title": corpus_by_id[positive_id].get("title") or "",
                "channel": corpus_by_id[positive_id].get("channel") or "",
            }
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


def build_unique_positive_batches(
    pairs: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """Build MNRL batches without treating duplicate positives as negatives.

    Data Designer intentionally creates several query variants for each passage.
    Plain shuffled batching can put two variants for the same positive in one
    MultipleNegativesRankingLoss batch, where each is then labelled as the
    other's negative. This balanced schedule includes at most one example per
    positive chunk in every batch while retaining every training row.
    """
    if batch_size < 2:
        raise ValueError("unique-positive batching needs batch-size >= 2")
    grouped: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        positive_id = str(
            pair.get("positive_collision_group") or pair.get("positive_chunk_id") or ""
        )
        if not positive_id:
            raise ValueError(f"train pair {index} has no positive_chunk_id")
        grouped.setdefault(positive_id, []).append(index)
    if len(grouped) < 2:
        raise ValueError("unique-positive batching needs at least two distinct positives")

    rng = random.Random(seed)
    total = len(pairs)
    batch_count = max(max(map(len, grouped.values())), math.ceil(total / batch_size))
    if batch_count * 2 > total:
        raise ValueError(
            "cannot retain every pair in collision-safe batches of at least two; "
            "add more distinct positives or use a different dataset"
        )
    batches: list[list[int]] = [[] for _ in range(batch_count)]
    batch_positives: list[set[str]] = [set() for _ in range(batch_count)]
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    for positive_id, indices in group_items:
        rng.shuffle(indices)
        candidates = list(range(batch_count))
        rng.shuffle(candidates)
        candidates.sort(key=lambda batch_index: len(batches[batch_index]))
        chosen = [
            batch_index
            for batch_index in candidates
            if positive_id not in batch_positives[batch_index]
            and len(batches[batch_index]) < batch_size
        ][: len(indices)]
        if len(chosen) != len(indices):
            raise ValueError("could not construct collision-safe batches for this dataset")
        for index, batch_index in zip(indices, chosen, strict=True):
            batches[batch_index].append(index)
            batch_positives[batch_index].add(positive_id)
    if any(len(batch) < 2 for batch in batches):
        raise ValueError("could not avoid a singleton collision-safe batch")
    rng.shuffle(batches)
    return batches


class UniquePositiveBatchSampler:
    """Epoch-aware deterministic batch sampler for collision-safe MNRL."""

    def __init__(self, pairs: list[dict[str, Any]], batch_size: int, seed: int) -> None:
        self.pairs = pairs
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._length = len(build_unique_positive_batches(pairs, batch_size=batch_size, seed=seed))

    def __iter__(self):
        batches = build_unique_positive_batches(
            self.pairs,
            batch_size=self.batch_size,
            seed=self.seed + self.epoch,
        )
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return self._length


def expanded_sequences_per_batch_upper_bound(pairs: list[dict[str, Any]], batch_size: int) -> int:
    """Return a conservative forward-pass sequence count for an MNRL batch.

    Each example contributes one query, one positive, and every explicit hard
    negative. This number is a more useful memory warning than query batch size
    alone: a batch of 16 rows with seven negatives expands to 144 sequences.
    """
    max_negatives = max((len(pair.get("negative_texts") or []) for pair in pairs), default=0)
    return batch_size * (2 + max_negatives)


def collision_safe_batch_profile(
    pairs: list[dict[str, Any]], *, batch_size: int, seed: int
) -> dict[str, Any]:
    """Describe the batch size the collision-safe sampler can really realize.

    ``batch_size`` is only a ceiling. Positive-aware datasets may contain many
    query variants but few independent videos, so presenting the configured
    ceiling as an effective contrastive batch would overstate the number of
    in-batch negatives. Keep the realized first-epoch shape in the manifest.
    """
    batches = build_unique_positive_batches(pairs, batch_size=batch_size, seed=seed)
    sizes = [len(batch) for batch in batches]
    collision_groups = {
        str(pair.get("positive_collision_group") or pair.get("positive_chunk_id") or "")
        for pair in pairs
    }
    max_negative_count = max((len(pair.get("negative_texts") or []) for pair in pairs), default=0)
    return {
        "batch_size_requested": batch_size,
        "batch_size_realized_min": min(sizes),
        "batch_size_realized_max": max(sizes),
        "batch_size_realized_mean": round(sum(sizes) / len(sizes), 6),
        "batches_per_epoch": len(batches),
        "distinct_positive_collision_groups": len(collision_groups),
        "expanded_sequences_per_batch_realized_max": max(sizes) * (2 + max_negative_count),
    }


def validate_expanded_batch_budget(
    pairs: list[dict[str, Any]],
    *,
    batch_size: int,
    resolved_loss: str,
    max_expanded_sequences: int,
) -> int:
    """Reject unsafe non-cached batches before loading model weights."""
    if max_expanded_sequences < 1:
        raise ValueError("max expanded sequences per batch must be >= 1")
    upper_bound = expanded_sequences_per_batch_upper_bound(pairs, batch_size)
    if resolved_loss != "cached_mnrl" and upper_bound > max_expanded_sequences:
        raise ValueError(
            "expanded MNRL batch is unsafe: "
            f"upper bound {upper_bound} sequences exceeds limit {max_expanded_sequences}; "
            "reduce --batch-size, reduce explicit negatives, or use --loss cached_mnrl"
        )
    return upper_bound


def select_explicit_negative_cardinality(
    pairs: list[dict[str, Any]], negatives_per_query: int
) -> list[dict[str, Any]]:
    """Return training pairs with a fixed, explicit negative cardinality.

    Sentence Transformers represents each text lane as a dataset column. Every
    row must therefore expose the same number of negatives; allowing variable
    lengths through the legacy ``fit`` adapter silently truncates to the
    shortest row. Preserve the available count for auditability and select the
    hardest candidates deterministically from the builder's ordered list.
    """
    if negatives_per_query < 0:
        raise ValueError("negatives per query must be >= 0")
    selected: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        texts = list(pair.get("negative_texts") or [])
        ids = list(pair.get("negative_chunk_ids") or [])
        if len(texts) < negatives_per_query:
            raise ValueError(
                f"train pair {index} has {len(texts)} negatives; " f"need {negatives_per_query}"
            )
        row = dict(pair)
        row["available_negative_count"] = len(texts)
        row["negative_texts"] = texts[:negatives_per_query]
        row["negative_chunk_ids"] = ids[:negatives_per_query]
        selected.append(row)
    return selected


def build_sentence_transformer_columns(pairs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build fixed text columns consumed by SentenceTransformerTrainer."""
    if not pairs:
        raise ValueError("cannot build training columns without pairs")
    negative_count = len(pairs[0].get("negative_texts") or [])
    if any(len(pair.get("negative_texts") or []) != negative_count for pair in pairs):
        raise ValueError("every training pair must have the same explicit negative count")
    columns = {
        "anchor": [str(pair["query_for_embedding"]) for pair in pairs],
        "positive": [str(pair["positive_text"]) for pair in pairs],
    }
    for negative_index in range(negative_count):
        columns[f"negative_{negative_index + 1}"] = [
            str(pair["negative_texts"][negative_index]) for pair in pairs
        ]
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries-jsonl", type=Path)
    parser.add_argument("--grouped-jsonl", type=Path)
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
    parser.add_argument(
        "--loss",
        choices=("auto", "mnrl", "explicit_mnrl", "cached_mnrl"),
        default="auto",
    )
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--checkpoint-save-steps",
        type=int,
        default=0,
        help="Checkpoint interval; 0 means quarter-epoch, negative disables checkpoints.",
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument(
        "--max-expanded-sequences-per-batch",
        type=int,
        default=DEFAULT_MAX_EXPANDED_SEQUENCES_PER_BATCH,
        help=(
            "Refuse non-cached MNRL batches whose query/positive/negative expansion "
            "exceeds this count (default: 64)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("MultipleNegativesRankingLoss needs batch-size >= 2")
    if args.use_amp and args.use_bf16:
        raise ValueError("choose at most one of --use-amp and --use-bf16")
    if args.logging_steps < 1:
        raise ValueError("logging steps must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_rows = load_jsonl(args.corpus_jsonl)
    if bool(args.queries_jsonl) == bool(args.grouped_jsonl):
        raise ValueError("provide exactly one of --queries-jsonl or --grouped-jsonl")
    if args.grouped_jsonl:
        pairs = build_grouped_pairs(
            load_jsonl(args.grouped_jsonl),
            corpus_rows,
            query_prefix_mode=args.query_prefix_mode,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    else:
        pairs = build_pairs(
            load_jsonl(args.queries_jsonl),
            corpus_rows,
            train_splits=args.train_splits,
            query_prefix_mode=args.query_prefix_mode,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    if len(pairs) < args.batch_size:
        raise ValueError(f"need at least {args.batch_size} train pairs, got {len(pairs)}")

    resolved_loss = args.loss
    if resolved_loss == "auto":
        resolved_loss = "explicit_mnrl" if args.grouped_jsonl else "mnrl"
    if resolved_loss != "mnrl" and not args.grouped_jsonl:
        raise ValueError(f"{resolved_loss} requires --grouped-jsonl")
    if args.grouped_jsonl:
        selected_negative_count = args.negatives_per_query if resolved_loss != "mnrl" else 0
        pairs = select_explicit_negative_cardinality(pairs, selected_negative_count)

    pairs_path = args.output_dir / "train_pairs.jsonl"
    manifest_path = args.output_dir / "training_manifest.json"
    final_model_path = args.output_dir / "final"
    write_jsonl(pairs_path, pairs)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "dry_run" if args.dry_run else "started",
        "run_kind": "memexai_embedding_retriever_train",
        "queries_jsonl": str(args.queries_jsonl) if args.queries_jsonl else None,
        "grouped_jsonl": str(args.grouped_jsonl) if args.grouped_jsonl else None,
        "source_hashes": {
            "queries_or_grouped": sha256_file(args.grouped_jsonl or args.queries_jsonl),
            "corpus": sha256_file(args.corpus_jsonl),
        },
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
            "passage_format": "embedding_text_or_title_double_newline_text",
        },
        "train_splits": sorted(args.train_splits),
        "train_pairs": len(pairs),
        "query_types": dict(sorted(Counter(pair["query_type"] for pair in pairs).items())),
        "channels": dict(sorted(Counter(pair["channel"] for pair in pairs).items())),
        "training": {
            "loss": "MultipleNegativesRankingLoss",
            "batch_size": args.batch_size,
            "batch_sampler": "unique_positive_chunk_v1",
            "false_negative_protection": (
                "No batch contains two rows from the same positive collision group; "
                "positive-aware runs use positive_video_id and pair-only runs use positive_chunk_id."
            ),
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
                "Candidate artifact only; promotion requires a judged eval-v1 and "
                "Qwen/Gemini/BM25 comparison showing the fine-tune improves retrieval."
            ),
        },
    }
    manifest["training"].update(
        {
            "loss": resolved_loss,
            "temperature": args.temperature,
            "scale": 1.0 / args.temperature,
            "mini_batch_size": args.mini_batch_size if resolved_loss == "cached_mnrl" else None,
            "negatives_per_query": (
                args.negatives_per_query if args.grouped_jsonl and resolved_loss != "mnrl" else 0
            ),
            "available_negative_count_distribution": dict(
                sorted(
                    Counter(str(pair.get("available_negative_count", 0)) for pair in pairs).items()
                )
            ),
            "explicit_negative_count_distribution": dict(
                sorted(
                    Counter(str(len(pair.get("negative_texts") or [])) for pair in pairs).items()
                )
            ),
            "positive_collision_group": (
                "positive_video_id" if args.grouped_jsonl else "positive_chunk_id"
            ),
        }
    )
    batch_profile = collision_safe_batch_profile(pairs, batch_size=args.batch_size, seed=args.seed)
    expanded_upper_bound = expanded_sequences_per_batch_upper_bound(pairs, args.batch_size)
    manifest["training"].update(
        {
            **batch_profile,
            "expanded_sequences_per_batch_upper_bound": expanded_upper_bound,
            "max_expanded_sequences_per_batch": args.max_expanded_sequences_per_batch,
            "memory_guard": "expanded_mnrl_sequence_budget_v1",
            "logging_steps": args.logging_steps,
            "metrics_jsonl": str(args.output_dir / "training_metrics.jsonl"),
            "precision": "bf16" if args.use_bf16 else "fp16" if args.use_amp else "fp32",
            "scheduler": "cosine",
            "trainer": "SentenceTransformerTrainer.direct_collision_safe_v1",
        }
    )
    try:
        validate_expanded_batch_budget(
            pairs,
            batch_size=args.batch_size,
            resolved_loss=resolved_loss,
            max_expanded_sequences=args.max_expanded_sequences_per_batch,
        )
    except ValueError:
        manifest["status"] = "rejected_unsafe_batch"
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )
    from transformers import TrainerCallback

    random.seed(args.seed)
    model = SentenceTransformer(
        str(args.base_model_path), device=args.device, local_files_only=True
    )
    model.max_seq_length = args.max_seq_length
    batch_sampler = UniquePositiveBatchSampler(pairs, args.batch_size, args.seed)
    train_dataset = Dataset.from_dict(build_sentence_transformer_columns(pairs))
    loss_kwargs = {"scale": 1.0 / args.temperature}
    if resolved_loss == "cached_mnrl":
        train_loss = CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=args.mini_batch_size, **loss_kwargs
        )
    else:
        train_loss = MultipleNegativesRankingLoss(model, **loss_kwargs)
    total_steps = args.max_steps if args.max_steps > 0 else len(batch_sampler) * args.epochs
    manifest["training"]["collision_safe_batches_per_epoch"] = len(batch_sampler)
    manifest["training"]["optimizer_steps"] = total_steps
    checkpoint_steps = (
        args.checkpoint_save_steps if args.checkpoint_save_steps > 0 else max(1, total_steps // 4)
    )
    manifest["training"]["checkpoint_save_steps"] = (
        checkpoint_steps if args.checkpoint_save_steps >= 0 else None
    )
    write_json(manifest_path, manifest)

    metrics_path = args.output_dir / "training_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    class JsonlMetricsCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }
            row.update(
                {
                    key: value
                    for key, value in logs.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
            )
            with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    class CollisionSafeTrainer(SentenceTransformerTrainer):
        def get_batch_sampler(
            self,
            dataset,
            batch_size,
            drop_last,
            valid_label_columns=None,
            generator=None,
            seed=0,
        ):
            return batch_sampler

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=1,
        max_steps=total_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        fp16=args.use_amp,
        bf16=args.use_bf16,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="no" if args.checkpoint_save_steps < 0 else "steps",
        save_steps=checkpoint_steps,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        run_name=args.output_dir.name,
    )
    trainer = CollisionSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        callbacks=[JsonlMetricsCallback()],
    )
    runtime_batch_sampler = trainer.get_train_dataloader().batch_sampler
    if not isinstance(runtime_batch_sampler, UniquePositiveBatchSampler):
        raise RuntimeError(
            "SentenceTransformerTrainer did not install the collision-safe batch sampler"
        )
    manifest["training"]["runtime_batch_sampler_class"] = type(runtime_batch_sampler).__name__
    write_json(manifest_path, manifest)
    result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    model.save(str(final_model_path))
    result_metrics = {
        key: value
        for key, value in result.metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    manifest["training"]["result_metrics"] = result_metrics
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
