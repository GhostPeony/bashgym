#!/usr/bin/env python3
"""Build a grounded YouTube transcript retrieval-accuracy seed set for MemexAI.

The dataset this writes is for retrieval evaluation and embedding fine-tune prep:
real transcript chunks stay as the corpus, while synthetic generation is limited
to user-like queries and negative labels. Metrics measure query-to-corpus
ranking accuracy, not transcript content quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUERY_TYPES = ("natural_question", "keyword_query", "semantic_paraphrase")
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
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "being",
    "between",
    "could",
    "does",
    "doing",
    "down",
    "from",
    "gets",
    "going",
    "have",
    "having",
    "into",
    "just",
    "know",
    "like",
    "more",
    "most",
    "much",
    "only",
    "other",
    "really",
    "should",
    "some",
    "than",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "think",
    "this",
    "those",
    "through",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    video_id: str
    chunk_index: int
    split: str
    title: str
    channel: str
    source_set: str
    url: str
    upload_date: str
    start_seconds: int
    end_seconds: int
    text: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stable_id(*parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)]


def keywords(text: str, title: str, limit: int = 6) -> list[str]:
    counts = Counter(w for w in words(f"{title} {text}") if w not in STOPWORDS and len(w) > 3)
    return [word for word, _ in counts.most_common(limit)]


def format_query_for_embedding(query: str, prefix_mode: str) -> str:
    try:
        prefix = QUERY_PREFIXES[prefix_mode]
    except KeyError as exc:
        raise ValueError(f"unknown query prefix mode: {prefix_mode}") from exc
    return f"{prefix}{query}"


def query_prefix_metadata(prefix_mode: str) -> dict[str, Any]:
    return {
        "mode": prefix_mode,
        "prefix": QUERY_PREFIXES[prefix_mode],
        "applied_to": "query",
        "corpus_prefix": "",
    }


def choose_video_splits(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    by_source: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        video_id = row["youtube_video_id"]
        if video_id not in seen:
            seen.add(video_id)
            by_source[row.get("source_set") or "unknown"].append(video_id)

    rng = random.Random(seed)
    splits: dict[str, str] = {}
    for source, video_ids in sorted(by_source.items()):
        shuffled = sorted(video_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_n = max(1, math.floor(n * 0.7))
        dev_n = max(1, math.floor(n * 0.15)) if n >= 4 else 0
        for idx, video_id in enumerate(shuffled):
            if idx < train_n:
                split = "train"
            elif idx < train_n + dev_n:
                split = "dev"
            else:
                split = "test"
            splits[video_id] = split
        print(
            f"split {source}: {sum(1 for v in shuffled if splits[v] == 'train')} train, "
            f"{sum(1 for v in shuffled if splits[v] == 'dev')} dev, "
            f"{sum(1 for v in shuffled if splits[v] == 'test')} test",
            file=sys.stderr,
        )
    return splits


def select_positive_chunks(
    rows: list[dict[str, Any]], sample_per_video: int, seed: int
) -> list[ChunkRef]:
    splits = choose_video_splits(rows, seed)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["youtube_video_id"]].append(row)

    selected: list[ChunkRef] = []
    for video_id in sorted(by_video):
        video_chunks = sorted(by_video[video_id], key=lambda r: int(r["chunk_index"]))
        eligible = [
            r for r in video_chunks if len(clean_text(r.get("text", ""))) >= 280
        ] or video_chunks
        if sample_per_video <= 1:
            target_positions = [0.5]
        else:
            target_positions = [
                (idx + 1) / (sample_per_video + 1) for idx in range(sample_per_video)
            ]

        picked: list[dict[str, Any]] = []
        for pos in target_positions:
            target = round(pos * (len(eligible) - 1))
            candidate = min(
                eligible,
                key=lambda r: abs(int(r["chunk_index"]) - int(eligible[target]["chunk_index"])),
            )
            if candidate not in picked:
                picked.append(candidate)

        # Fill any duplicate gaps deterministically with evenly spaced remaining rows.
        for candidate in eligible:
            if len(picked) >= sample_per_video:
                break
            if candidate not in picked:
                picked.append(candidate)

        for row in picked[:sample_per_video]:
            selected.append(
                ChunkRef(
                    chunk_id=row["chunk_id"],
                    video_id=row["youtube_video_id"],
                    chunk_index=int(row["chunk_index"]),
                    split=splits[row["youtube_video_id"]],
                    title=row["title"],
                    channel=row["channel"],
                    source_set=row.get("source_set") or "",
                    url=row["url"],
                    upload_date=row.get("upload_date") or "",
                    start_seconds=int(row.get("start_seconds") or 0),
                    end_seconds=int(row.get("end_seconds") or 0),
                    text=clean_text(row.get("text", "")),
                )
            )
    return selected


def endpoint_model(endpoint: str, requested_model: str | None) -> str | None:
    if requested_model:
        return requested_model
    if not endpoint:
        return None
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"model discovery failed for {endpoint}: {exc}", file=sys.stderr)
        return None
    for item in payload.get("data", []):
        if item.get("id"):
            return item["id"]
    for item in payload.get("models", []):
        if item.get("name"):
            return item["name"]
        if item.get("model"):
            return item["model"]
    return None


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = strip_think_blocks(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def llm_completion(
    endpoint: str, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate grounded retrieval-evaluation queries and return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 405}:
            raise
        return legacy_llm_completion(endpoint, model, prompt, temperature, max_tokens)
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return message.get("content") or message.get("reasoning_content") or choice.get("text") or ""


def legacy_llm_completion(
    endpoint: str, model: str, prompt: str, temperature: float, max_tokens: int
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = (body.get("choices") or [{}])[0]
    if choice.get("text"):
        return choice["text"]
    message = choice.get("message") or {}
    return message.get("content") or message.get("reasoning_content") or ""


def llm_queries(
    chunk: ChunkRef, endpoint: str, model: str, temperature: float, max_tokens: int
) -> dict[str, str] | None:
    prompt = f"""You generate retrieval-evaluation queries for a YouTube transcript search system.

Use only this transcript chunk. Do not invent claims beyond it. Do not quote full sentences.
Return exactly one valid JSON object with these string keys:
- natural_question: a user question answerable by this chunk
- keyword_query: a short search query with entities or concepts
- semantic_paraphrase: a lower-overlap paraphrase of the same information need

Keep each query between 4 and 18 words.

Video title: {chunk.title}
Channel: {chunk.channel}
Timestamp: {chunk.start_seconds}-{chunk.end_seconds}s
Transcript chunk:
{chunk.text[:1800]}

JSON only:"""
    for attempt in range(3):
        try:
            generated = llm_completion(endpoint, model, prompt, temperature, max_tokens)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"llm attempt {attempt + 1} failed for {chunk.chunk_id}: {exc}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
            continue
        parsed = extract_json_object(generated)
        if not parsed:
            print(
                f"llm JSON parse failed for {chunk.chunk_id} attempt {attempt + 1}", file=sys.stderr
            )
            continue
        out: dict[str, str] = {}
        for query_type in QUERY_TYPES:
            value = clean_text(str(parsed.get(query_type) or ""))
            if value:
                out[query_type] = value[:220]
        if len(out) == len(QUERY_TYPES):
            return out
    return None


def template_queries(chunk: ChunkRef) -> dict[str, str]:
    keys = keywords(chunk.text, chunk.title, limit=5)
    focus = ", ".join(keys[:3]) if keys else chunk.title
    keyword = " ".join(keys[:5]) if keys else chunk.title
    title_hint = re.sub(r"\s+[|-]\s+.*$", "", chunk.title).strip()
    return {
        "natural_question": f"What does this video say about {focus}?",
        "keyword_query": keyword,
        "semantic_paraphrase": f"Find the part explaining {focus} in {title_hint}.",
    }


def build_query_rows(
    chunks: list[ChunkRef],
    generation_mode: str,
    endpoint: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "generation_mode": generation_mode,
        "generator_endpoint": endpoint or None,
        "generator_model": model,
        "llm_success_chunks": 0,
        "template_fallback_chunks": 0,
    }
    for idx, chunk in enumerate(chunks, start=1):
        queries: dict[str, str] | None = None
        generator = "llm"
        if generation_mode in {"auto", "llm"} and endpoint and model:
            queries = llm_queries(chunk, endpoint, model, temperature, max_tokens)
        if not queries:
            if generation_mode == "llm":
                raise RuntimeError(f"LLM query generation failed for {chunk.chunk_id}")
            queries = template_queries(chunk)
            generator = "template"
            stats["template_fallback_chunks"] += 1
        else:
            stats["llm_success_chunks"] += 1

        if idx % 10 == 0 or idx == len(chunks):
            print(f"generated query variants for {idx}/{len(chunks)} chunks", file=sys.stderr)

        for query_type in QUERY_TYPES:
            query = clean_text(queries[query_type])
            rows.append(
                {
                    "eval_id": f"memexai-youtube-{stable_id(chunk.chunk_id, query_type, query)}",
                    "query": query,
                    "query_type": query_type,
                    "positive_chunk_id": chunk.chunk_id,
                    "positive_video_id": chunk.video_id,
                    "positive_chunk_index": chunk.chunk_index,
                    "split": chunk.split,
                    "title": chunk.title,
                    "channel": chunk.channel,
                    "source_set": chunk.source_set,
                    "url": chunk.url,
                    "upload_date": chunk.upload_date,
                    "start_seconds": chunk.start_seconds,
                    "end_seconds": chunk.end_seconds,
                    "relevant_chunk_ids": [chunk.chunk_id],
                    "local_window_chunk_ids": [],
                    "hard_negative_chunk_ids": [],
                    "same_channel_negative_chunk_ids": [],
                    "retrieval_accuracy_target": "rank_positive_chunk_or_local_window_above_hard_negatives",
                    "generator": generator,
                    "generator_model": model if generator == "llm" else None,
                    "data_designer_role": "synthetic_query_grounded_in_real_transcript_chunk",
                }
            )
    return rows, stats


def local_window_ids(
    corpus_rows: list[dict[str, Any]], chunk_id: str, window: int = 2
) -> list[str]:
    by_id = {row["chunk_id"]: row for row in corpus_rows}
    row = by_id[chunk_id]
    video_id = row["youtube_video_id"]
    idx = int(row["chunk_index"])
    candidates = [
        other
        for other in corpus_rows
        if other["youtube_video_id"] == video_id and abs(int(other["chunk_index"]) - idx) <= window
    ]
    return [
        other["chunk_id"]
        for other in sorted(candidates, key=lambda other: int(other["chunk_index"]))
    ][: 1 + (2 * window)]


def _rank_of_any(ranked_ids: list[str], acceptable_ids: set[str]) -> int | None:
    for idx, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in acceptable_ids:
            return idx
    return None


def _retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cutoffs = (1, 3, 5, 10)

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not group_rows:
            return {"count": 0}
        summary: dict[str, Any] = {"count": len(group_rows)}
        for rank_key, prefix in (
            ("positive_rank_exact", "exact_chunk"),
            ("positive_rank_local_window", "local_window"),
            ("positive_rank_same_video", "same_video"),
        ):
            ranks = [int(row[rank_key]) for row in group_rows if row.get(rank_key)]
            if not ranks:
                continue
            for k in cutoffs:
                summary[f"{prefix}_recall_at_{k}"] = round(
                    sum(1 for rank in ranks if rank <= k) / len(group_rows), 6
                )
            summary[f"{prefix}_mrr"] = round(sum(1.0 / rank for rank in ranks) / len(group_rows), 6)
            summary[f"{prefix}_mean_rank"] = round(sum(ranks) / len(ranks), 3)
        return summary

    metrics: dict[str, Any] = {
        "overall": summarize(rows),
        "by_split": {},
        "by_query_type": {},
        "by_channel": {},
    }
    for split in sorted({row["split"] for row in rows}):
        metrics["by_split"][split] = summarize([row for row in rows if row["split"] == split])
    for query_type in sorted({row["query_type"] for row in rows}):
        metrics["by_query_type"][query_type] = summarize(
            [row for row in rows if row["query_type"] == query_type]
        )
    for channel in sorted({row["channel"] for row in rows}):
        metrics["by_channel"][channel] = summarize(
            [row for row in rows if row["channel"] == channel]
        )
    return metrics


def attach_embedding_negatives(
    args: argparse.Namespace, rows: list[dict[str, Any]], corpus_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "query_embedding_path": None,
        "query_embedding_shape": None,
        "embedding_negatives_attached": False,
        "retrieval_metrics": None,
        "query_prefix": query_prefix_metadata(args.query_prefix_mode),
    }
    for row in rows:
        row["local_window_chunk_ids"] = local_window_ids(corpus_rows, row["positive_chunk_id"])

    if (
        not args.embedding_model_path
        or not args.corpus_embedding_matrix
        or not args.corpus_embedding_chunk_ids
    ):
        return stats

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(
            f"embedding dependencies unavailable, skipping hard negatives: {exc}", file=sys.stderr
        )
        return stats

    matrix_path = Path(args.corpus_embedding_matrix)
    ids_path = Path(args.corpus_embedding_chunk_ids)
    if not matrix_path.exists() or not ids_path.exists():
        print(
            "corpus embedding matrix or chunk id map missing, skipping hard negatives",
            file=sys.stderr,
        )
        return stats

    corpus_matrix = np.load(matrix_path)
    corpus_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if isinstance(corpus_ids, dict):
        corpus_ids = [corpus_ids[str(idx)] for idx in range(len(corpus_matrix))]
    if len(corpus_ids) != len(corpus_matrix):
        raise ValueError("corpus embedding ids length does not match matrix rows")

    by_chunk_id = {row["chunk_id"]: row for row in corpus_rows}
    corpus_index_by_id = {chunk_id: idx for idx, chunk_id in enumerate(corpus_ids)}
    model = SentenceTransformer(str(args.embedding_model_path), device=args.embedding_device)
    for row in rows:
        row["query_prefix_mode"] = args.query_prefix_mode
    queries = [format_query_for_embedding(row["query"], args.query_prefix_mode) for row in rows]
    query_embeddings = model.encode(
        queries,
        batch_size=args.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncate_dim=args.truncate_dim,
        show_progress_bar=True,
    )
    if query_embeddings.shape[1] != args.truncate_dim:
        raise ValueError(
            f"expected query embeddings dim {args.truncate_dim}, got {query_embeddings.shape[1]}"
        )

    scores = query_embeddings @ corpus_matrix.T
    for row_idx, row in enumerate(rows):
        positive_id = row["positive_chunk_id"]
        positive_video = row["positive_video_id"]
        positive_channel = row["channel"]
        ranked = np.argsort(-scores[row_idx])
        ranked_ids = [corpus_ids[int(corpus_idx)] for corpus_idx in ranked]
        positive_corpus_idx = corpus_index_by_id[positive_id]
        local_window = set(row["local_window_chunk_ids"])
        same_video_ids = {
            chunk_id
            for chunk_id, corpus_row in by_chunk_id.items()
            if corpus_row.get("youtube_video_id") == positive_video
        }
        row["positive_rank_exact"] = int(np.where(ranked == positive_corpus_idx)[0][0]) + 1
        row["positive_rank_local_window"] = _rank_of_any(ranked_ids, local_window)
        row["positive_rank_same_video"] = _rank_of_any(ranked_ids, same_video_ids)
        row["top_retrieved_chunk_ids"] = ranked_ids[:10]
        row["top_retrieved_scores"] = [
            round(float(scores[row_idx, int(corpus_idx)]), 6) for corpus_idx in ranked[:10]
        ]
        row["positive_score"] = round(float(scores[row_idx, positive_corpus_idx]), 6)
        hard: list[str] = []
        same_channel: list[str] = []
        for corpus_idx in ranked:
            candidate_id = corpus_ids[int(corpus_idx)]
            if candidate_id == positive_id:
                continue
            candidate = by_chunk_id.get(candidate_id)
            if not candidate:
                continue
            if candidate["youtube_video_id"] == positive_video:
                continue
            if len(hard) < args.hard_negatives:
                hard.append(candidate_id)
            if (
                candidate.get("channel") == positive_channel
                and len(same_channel) < args.same_channel_negatives
            ):
                same_channel.append(candidate_id)
            if (
                len(hard) >= args.hard_negatives
                and len(same_channel) >= args.same_channel_negatives
            ):
                break
        row["hard_negative_chunk_ids"] = hard
        row["same_channel_negative_chunk_ids"] = same_channel

    query_embedding_path = Path(args.output_dir) / (
        f"qwen3-embedding-0.6b-{args.truncate_dim}d-"
        f"{args.query_prefix_mode}-retrieval-queries.npy"
    )
    np.save(query_embedding_path, query_embeddings.astype("float32"))
    stats.update(
        {
            "query_embedding_path": str(query_embedding_path),
            "query_embedding_shape": list(query_embeddings.shape),
            "embedding_negatives_attached": True,
            "retrieval_metrics": _retrieval_metrics(rows),
        }
    )
    return stats


def positive_chunk_rows(chunks: list[ChunkRef]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "youtube_video_id": chunk.video_id,
            "chunk_index": chunk.chunk_index,
            "split": chunk.split,
            "title": chunk.title,
            "channel": chunk.channel,
            "source_set": chunk.source_set,
            "url": chunk.url,
            "upload_date": chunk.upload_date,
            "start_seconds": chunk.start_seconds,
            "end_seconds": chunk.end_seconds,
        }
        for chunk in chunks
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-video", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--generation-mode", choices=["auto", "llm", "template"], default="auto")
    parser.add_argument("--llm-endpoint", default=os.environ.get("MEMEXAI_LLM_ENDPOINT", ""))
    parser.add_argument("--llm-model", default=os.environ.get("MEMEXAI_LLM_MODEL", ""))
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=520)
    parser.add_argument("--embedding-model-path", type=Path, default=None)
    parser.add_argument(
        "--embedding-device", default=os.environ.get("MEMEXAI_EMBEDDING_DEVICE", "cuda")
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--corpus-embedding-matrix", type=Path, default=None)
    parser.add_argument("--corpus-embedding-chunk-ids", type=Path, default=None)
    parser.add_argument("--truncate-dim", type=int, default=768)
    parser.add_argument(
        "--query-prefix-mode",
        choices=sorted(QUERY_PREFIXES),
        default="memexai_youtube",
        help="Instruction prefix applied to query text before Qwen embedding.",
    )
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--same-channel-negatives", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = load_jsonl(args.corpus_jsonl)
    selected_chunks = select_positive_chunks(corpus_rows, args.sample_per_video, args.seed)
    model = endpoint_model(args.llm_endpoint, args.llm_model or None)
    generation_mode = args.generation_mode
    if generation_mode in {"auto", "llm"} and not (args.llm_endpoint and model):
        if generation_mode == "llm":
            raise RuntimeError("LLM generation requested but no endpoint/model is available")
        print("no LLM endpoint/model available; using template query generation", file=sys.stderr)
        generation_mode = "template"

    query_rows, generation_stats = build_query_rows(
        selected_chunks,
        generation_mode,
        args.llm_endpoint,
        model,
        args.llm_temperature,
        args.llm_max_tokens,
    )
    embedding_stats = attach_embedding_negatives(args, query_rows, corpus_rows)

    query_path = args.output_dir / "retrieval_eval_queries.jsonl"
    positives_path = args.output_dir / "retrieval_positive_chunks.jsonl"
    manifest_path = args.output_dir / "retrieval_eval_manifest.json"
    write_jsonl(query_path, query_rows)
    write_jsonl(positives_path, positive_chunk_rows(selected_chunks))

    split_counts = Counter(row["split"] for row in query_rows)
    query_type_counts = Counter(row["query_type"] for row in query_rows)
    channel_counts = Counter(row["channel"] for row in query_rows)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_jsonl": str(args.corpus_jsonl),
        "corpus_rows": len(corpus_rows),
        "positive_chunks": len(selected_chunks),
        "queries": len(query_rows),
        "query_types": dict(sorted(query_type_counts.items())),
        "splits": dict(sorted(split_counts.items())),
        "channels": dict(sorted(channel_counts.items())),
        "sample_per_video": args.sample_per_video,
        "seed": args.seed,
        "dimensions": args.truncate_dim,
        "outputs": {
            "queries_jsonl": str(query_path),
            "positive_chunks_jsonl": str(positives_path),
            "manifest_json": str(manifest_path),
        },
        "generation": generation_stats,
        "embeddings": embedding_stats,
        "data_designer_recommendation": (
            "Use Data Designer for synthetic query variants, paraphrases, judges, and hard-negative "
            "explanations over real transcript chunks. Do not use it to synthesize transcript corpus "
            "facts unless rows are clearly marked synthetic and excluded from grounded retrieval evals."
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
