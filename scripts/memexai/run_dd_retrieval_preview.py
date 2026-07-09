#!/usr/bin/env python3
"""Run a tiny Data Designer retrieval-query preview and score it with Qwen.

This is the first safe iteration loop for MemexAI YouTube retrieval:

1. sample real transcript chunks into a Data Designer seed file,
2. ask a local OpenAI-compatible endpoint to generate structured query variants,
3. judge/filter the generated variants,
4. flatten valid variants into retrieval eval rows,
5. embed queries with Qwen3-Embedding-0.6B at 768d using an explicit query prefix,
6. score exact-chunk, local-window, and same-video retrieval.

The generated eval rows intentionally exclude transcript text. Transcript text
only appears in the temporary seed file consumed by Data Designer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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


class RetrievalQueryVariants(BaseModel):
    natural_question: str = Field(
        ..., description="A natural user question answerable by the transcript chunk."
    )
    keyword_query: str = Field(
        ..., description="A short entity/concept keyword query for the same chunk."
    )
    semantic_paraphrase: str = Field(
        ..., description="A lower-lexical-overlap paraphrase of the same information need."
    )


class RetrievalQueryJudgement(BaseModel):
    natural_question_pass: bool = Field(
        ..., description="True only if the natural question is grounded in the chunk/window."
    )
    natural_question_reason: str = Field(
        "", description="Short reason for the natural question judgement."
    )
    keyword_query_pass: bool = Field(
        ..., description="True only if the keyword query is grounded in the chunk/window."
    )
    keyword_query_reason: str = Field("", description="Short reason for the keyword judgement.")
    semantic_paraphrase_pass: bool = Field(
        ..., description="True only if the semantic paraphrase is grounded in the chunk/window."
    )
    semantic_paraphrase_reason: str = Field(
        "", description="Short reason for the semantic paraphrase judgement."
    )


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
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:length]


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


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


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


def select_seed_chunks(
    corpus_rows: list[dict[str, Any]], sample_chunks: int
) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        by_video[row["youtube_video_id"]].append(row)

    video_ids = sorted(by_video)
    if sample_chunks >= len(video_ids):
        selected_videos = video_ids
    else:
        selected_videos = [
            video_ids[round(idx * (len(video_ids) - 1) / max(1, sample_chunks - 1))]
            for idx in range(sample_chunks)
        ]

    seeds: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for video_id in selected_videos:
        if video_id in seen_videos:
            continue
        seen_videos.add(video_id)
        chunks = sorted(by_video[video_id], key=lambda row: int(row["chunk_index"]))
        eligible = [row for row in chunks if len(clean_text(row.get("text", ""))) >= 300] or chunks
        row = eligible[len(eligible) // 2]
        seeds.append(row)
        if len(seeds) >= sample_chunks:
            break
    return seeds


def seed_rows_for_dd(seed_chunks: list[dict[str, Any]], excerpt_chars: int) -> list[dict[str, Any]]:
    return [
        {
            "seed_chunk_id": row["chunk_id"],
            "seed_video_id": row["youtube_video_id"],
            "seed_chunk_index": int(row["chunk_index"]),
            "seed_title": row["title"],
            "seed_channel": row["channel"],
            "seed_source_set": row.get("source_set") or "",
            "seed_url": row["url"],
            "seed_upload_date": row.get("upload_date") or "",
            "seed_start_seconds": int(row.get("start_seconds") or 0),
            "seed_end_seconds": int(row.get("end_seconds") or 0),
            "seed_text": clean_text(row.get("text", ""))[:excerpt_chars],
        }
        for row in seed_chunks
    ]


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


def dataset_to_records(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas().to_dict(orient="records")
    if hasattr(dataset, "to_dict"):
        data = dataset.to_dict()
        if isinstance(data, dict):
            keys = list(data)
            length = len(data[keys[0]]) if keys else 0
            return [{key: data[key][idx] for key in keys} for idx in range(length)]
    return [dict(row) for row in dataset]


def coerce_variants(value: Any) -> dict[str, str] | None:
    if isinstance(value, RetrievalQueryVariants):
        data = value.model_dump()
    elif isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
    else:
        return None
    out: dict[str, str] = {}
    for query_type in QUERY_TYPES:
        query = clean_text(str(data.get(query_type) or ""))
        if not query:
            return None
        out[query_type] = query
    return out


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "1"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "0"}:
            return False
    return None


def coerce_judgement(value: Any) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, RetrievalQueryJudgement):
        data = value.model_dump()
    elif isinstance(value, dict):
        data = value
    elif isinstance(value, str):
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
    else:
        return None

    out: dict[str, dict[str, Any]] = {}
    for query_type in QUERY_TYPES:
        nested = data.get(query_type)
        if isinstance(nested, dict):
            passed = coerce_bool(nested.get("pass") if "pass" in nested else nested.get("passed"))
            reason = nested.get("reason") or nested.get("notes") or ""
        else:
            passed = coerce_bool(data.get(f"{query_type}_pass"))
            reason = data.get(f"{query_type}_reason") or ""
        if passed is None:
            return None
        out[query_type] = {"pass": passed, "reason": clean_text(str(reason))}
    return out


def query_is_valid(query: str, query_type: str) -> tuple[bool, str | None]:
    lowered = query.lower()
    word_count = len(query.split())
    if word_count < 3:
        return False, "too_short"
    if word_count > 28:
        return False, "too_long"
    if any(marker in lowered for marker in ("i cannot", "i can't", "not enough information")):
        return False, "refusal"
    if query_type == "keyword_query" and word_count > 12:
        return False, "keyword_too_long"
    return True, None


def flatten_eval_rows(
    dd_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    model: str,
    *,
    require_judge: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    failures: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in dd_rows:
        variants = coerce_variants(row.get("query_variants"))
        if not variants:
            failures["missing_or_invalid_structured_output"] += 1
            continue
        judgement = coerce_judgement(row.get("query_judgement"))
        if require_judge and not judgement:
            failures["missing_or_invalid_judge_output"] += len(QUERY_TYPES)
            continue
        positive_chunk_id = row["seed_chunk_id"]
        for query_type in QUERY_TYPES:
            query = variants[query_type]
            ok, reason = query_is_valid(query, query_type)
            if not ok:
                failures[f"{query_type}:{reason}"] += 1
                continue
            judge = (judgement or {}).get(query_type)
            if judge and not judge["pass"]:
                failures[f"{query_type}:judge_rejected"] += 1
                continue
            out.append(
                {
                    "eval_id": f"memexai-dd-{stable_id(positive_chunk_id, query_type, query)}",
                    "query": query,
                    "query_type": query_type,
                    "positive_chunk_id": positive_chunk_id,
                    "positive_video_id": row["seed_video_id"],
                    "positive_chunk_index": int(row["seed_chunk_index"]),
                    "split": "preview",
                    "title": row["seed_title"],
                    "channel": row["seed_channel"],
                    "source_set": row.get("seed_source_set") or "",
                    "url": row["seed_url"],
                    "upload_date": row.get("seed_upload_date") or "",
                    "start_seconds": int(row.get("seed_start_seconds") or 0),
                    "end_seconds": int(row.get("seed_end_seconds") or 0),
                    "relevant_chunk_ids": [positive_chunk_id],
                    "local_window_chunk_ids": local_window_ids(corpus_rows, positive_chunk_id),
                    "hard_negative_chunk_ids": [],
                    "same_channel_negative_chunk_ids": [],
                    "retrieval_accuracy_target": "rank_positive_chunk_or_local_window_above_hard_negatives",
                    "generator": "data_designer",
                    "generator_model": model,
                    "judge_pass": bool(judge["pass"]) if judge else None,
                    "judge_reason": judge["reason"] if judge else "",
                    "judge_model": model if judge else None,
                    "data_designer_role": "synthetic_query_grounded_in_real_transcript_chunk",
                }
            )
    return out, failures


def judge_gate_stats(
    raw_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    validation_failures: Counter[str],
    *,
    judge_enabled: bool,
    min_keep_ratio: float,
) -> dict[str, Any]:
    expected_query_rows = len(raw_rows) * len(QUERY_TYPES)
    kept_rows = len(eval_rows)
    keep_ratio = (kept_rows / expected_query_rows) if expected_query_rows else 0.0
    rejected_rows = sum(count for reason, count in validation_failures.items() if "judge" in reason)
    return {
        "enabled": judge_enabled,
        "expected_query_rows": expected_query_rows,
        "kept_query_rows": kept_rows,
        "judge_rejected_rows": rejected_rows,
        "keep_ratio": round(keep_ratio, 6),
        "min_keep_ratio": min_keep_ratio,
        "passed": (not judge_enabled) or keep_ratio >= min_keep_ratio,
    }


def rank_of_any(ranked_ids: list[str], acceptable_ids: set[str]) -> int | None:
    for idx, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in acceptable_ids:
            return idx
    return None


def retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "by_query_type": {},
        "by_channel": {},
    }
    for query_type in sorted({row["query_type"] for row in rows}):
        metrics["by_query_type"][query_type] = summarize(
            [row for row in rows if row["query_type"] == query_type]
        )
    for channel in sorted({row["channel"] for row in rows}):
        metrics["by_channel"][channel] = summarize(
            [row for row in rows if row["channel"] == channel]
        )
    return metrics


def score_queries(
    args: argparse.Namespace, rows: list[dict[str, Any]], corpus_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    corpus_matrix = np.load(args.corpus_embedding_matrix)
    corpus_ids = json.loads(Path(args.corpus_embedding_chunk_ids).read_text(encoding="utf-8"))
    if isinstance(corpus_ids, dict):
        corpus_ids = [corpus_ids[str(idx)] for idx in range(len(corpus_matrix))]
    if len(corpus_ids) != len(corpus_matrix):
        raise ValueError("corpus embedding matrix row count does not match chunk ids")

    by_chunk_id = {row["chunk_id"]: row for row in corpus_rows}
    corpus_index_by_id = {chunk_id: idx for idx, chunk_id in enumerate(corpus_ids)}
    model = SentenceTransformer(str(args.embedding_model_path), device=args.embedding_device)
    for row in rows:
        row["query_prefix_mode"] = args.query_prefix_mode
    query_embeddings = model.encode(
        [format_query_for_embedding(row["query"], args.query_prefix_mode) for row in rows],
        batch_size=args.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncate_dim=args.truncate_dim,
        show_progress_bar=True,
    )
    if query_embeddings.shape[1] != args.truncate_dim:
        raise ValueError(
            f"expected {args.truncate_dim}d embeddings, got {query_embeddings.shape[1]}"
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
        row["positive_rank_local_window"] = rank_of_any(ranked_ids, local_window)
        row["positive_rank_same_video"] = rank_of_any(ranked_ids, same_video_ids)
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

    query_embedding_path = args.output_dir / (
        f"qwen3-embedding-0.6b-{args.truncate_dim}d-"
        f"{args.query_prefix_mode}-dd-preview-queries.npy"
    )
    np.save(query_embedding_path, query_embeddings.astype("float32"))
    return {
        "query_prefix": query_prefix_metadata(args.query_prefix_mode),
        "query_embedding_path": str(query_embedding_path),
        "query_embedding_shape": list(query_embeddings.shape),
        "retrieval_metrics": retrieval_metrics(rows),
    }


def run_data_designer(
    args: argparse.Namespace, seed_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import importlib.metadata as md

    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    provider = dd.ModelProvider(
        name="local",
        endpoint=args.llm_endpoint.rstrip("/"),
        provider_type="openai",
        api_key=args.llm_api_key,
    )
    model_config = dd.ModelConfig(
        alias="query-model",
        model=args.llm_model,
        provider="local",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_parallel_requests=args.max_parallel_requests,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="query_variants",
            model_alias="query-model",
            output_format=RetrievalQueryVariants,
            system_prompt=(
                "You create retrieval-evaluation queries for a YouTube transcript search system. "
                "Use only the provided transcript chunk. Do not invent facts. Return structured JSON only."
            ),
            prompt=(
                "Video title: {{ seed_title }}\n"
                "Channel: {{ seed_channel }}\n"
                "Timestamp: {{ seed_start_seconds }}-{{ seed_end_seconds }} seconds\n\n"
                "Transcript chunk:\n{{ seed_text }}\n\n"
                "Create three retrieval queries: a natural user question, a concise keyword/entity query, "
                "and a lower-overlap semantic paraphrase. Each query should be answerable by this chunk "
                "or its immediate local window."
            ),
        )
    )
    if not args.disable_judge:
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="query_judgement",
                model_alias="query-model",
                output_format=RetrievalQueryJudgement,
                system_prompt=(
                    "You are a strict retrieval-eval judge for a YouTube transcript search system. "
                    "Judge whether each generated query is answerable by the provided transcript "
                    "chunk or its immediate local window. Reject vague, invented, or overly broad "
                    "queries. Return structured JSON only."
                ),
                prompt=(
                    "Video title: {{ seed_title }}\n"
                    "Channel: {{ seed_channel }}\n"
                    "Timestamp: {{ seed_start_seconds }}-{{ seed_end_seconds }} seconds\n\n"
                    "Transcript chunk:\n{{ seed_text }}\n\n"
                    "Generated query variants:\n{{ query_variants }}\n\n"
                    "For each query variant, set the corresponding *_pass boolean and a short "
                    "reason. Pass only if the query can be resolved by this chunk or nearby context."
                ),
            )
        )
    designer = DataDesigner(model_providers=[provider])
    started = time.perf_counter()
    designer.validate(builder)
    preview = designer.preview(builder, num_records=args.sample_chunks)
    elapsed = time.perf_counter() - started
    rows = dataset_to_records(preview.dataset)
    return rows, {
        "data_designer_version": md.version("data-designer"),
        "elapsed_seconds": round(elapsed, 3),
        "model": args.llm_model,
        "endpoint": args.llm_endpoint,
        "max_parallel_requests": args.max_parallel_requests,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "thinking_disabled": True,
        "judge_enabled": not args.disable_judge,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-chunks", type=int, default=5)
    parser.add_argument("--excerpt-chars", type=int, default=1600)
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8889/v1")
    parser.add_argument("--llm-model", default="hermes-qwen3.6-27b-dense")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--max-parallel-requests", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--corpus-embedding-matrix", type=Path, required=True)
    parser.add_argument("--corpus-embedding-chunk-ids", type=Path, required=True)
    parser.add_argument("--truncate-dim", type=int, default=768)
    parser.add_argument(
        "--query-prefix-mode",
        choices=sorted(QUERY_PREFIXES),
        default="memexai_youtube",
        help="Instruction prefix applied to query text before Qwen embedding.",
    )
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--same-channel-negatives", type=int, default=4)
    parser.add_argument(
        "--disable-judge",
        action="store_true",
        help="Skip the Data Designer judge/filter column. Use only for debugging tiny previews.",
    )
    parser.add_argument(
        "--min-judge-keep-ratio",
        type=float,
        default=0.70,
        help="Minimum kept/generated query-row ratio when the judge is enabled.",
    )
    parser.add_argument(
        "--allow-low-judge-keep-ratio",
        action="store_true",
        help="Write artifacts even if the judge keep-ratio gate fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = load_jsonl(args.corpus_jsonl)
    seed_chunks = select_seed_chunks(corpus_rows, args.sample_chunks)
    seed_rows = seed_rows_for_dd(seed_chunks, args.excerpt_chars)
    seed_path = args.output_dir / "dd_seed_chunks.jsonl"
    write_jsonl(seed_path, seed_rows)

    raw_rows, dd_stats = run_data_designer(args, seed_path)
    raw_path = args.output_dir / "dd_preview_raw.jsonl"
    write_jsonl(raw_path, raw_rows)

    judge_enabled = not args.disable_judge
    eval_rows, validation_failures = flatten_eval_rows(
        raw_rows, corpus_rows, args.llm_model, require_judge=judge_enabled
    )
    if not eval_rows:
        raise RuntimeError(
            f"Data Designer preview produced no valid eval rows: {dict(validation_failures)}"
        )
    judge_gate = judge_gate_stats(
        raw_rows,
        eval_rows,
        validation_failures,
        judge_enabled=judge_enabled,
        min_keep_ratio=args.min_judge_keep_ratio,
    )

    scoring = score_queries(args, eval_rows, corpus_rows)
    queries_path = args.output_dir / "retrieval_eval_queries.jsonl"
    write_jsonl(queries_path, eval_rows)

    prompt_tokens_est = sum(
        approx_tokens(row["seed_text"]) + approx_tokens(row["seed_title"]) + 90 for row in seed_rows
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_kind": "data_designer_retrieval_preview",
        "corpus_jsonl": str(args.corpus_jsonl),
        "corpus_rows": len(corpus_rows),
        "seed_chunks": len(seed_rows),
        "raw_rows": len(raw_rows),
        "valid_eval_rows": len(eval_rows),
        "query_types": dict(sorted(Counter(row["query_type"] for row in eval_rows).items())),
        "validation_failures": dict(sorted(validation_failures.items())),
        "judge_gate": judge_gate,
        "outputs": {
            "seed_jsonl": str(seed_path),
            "raw_preview_jsonl": str(raw_path),
            "queries_jsonl": str(queries_path),
            "manifest_json": str(args.output_dir / "dd_retrieval_preview_manifest.json"),
            "query_embeddings_npy": scoring["query_embedding_path"],
        },
        "data_designer": dd_stats,
        "query_prefix": query_prefix_metadata(args.query_prefix_mode),
        "ledger": {
            "provider": "local",
            "records_requested": args.sample_chunks,
            "records_created": len(raw_rows),
            "records_kept": len(eval_rows),
            "llm_columns": 1,
            "judge_columns": 1 if judge_enabled else 0,
            "prompt_tokens_est": prompt_tokens_est,
            "completion_tokens_est": args.sample_chunks * args.max_tokens,
            "estimated_cost_usd": 0.0,
            "max_parallel_requests": args.max_parallel_requests,
            "timeouts": {"chat_completion_seconds": args.timeout},
            "retry_count": None,
            "error_count": sum(validation_failures.values()),
        },
        "embeddings": scoring,
    }
    manifest_path = args.output_dir / "dd_retrieval_preview_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if judge_enabled and not judge_gate["passed"] and not args.allow_low_judge_keep_ratio:
        raise RuntimeError(
            "Data Designer judge gate failed: "
            f"kept {judge_gate['kept_query_rows']}/{judge_gate['expected_query_rows']} "
            f"rows ({judge_gate['keep_ratio']}); "
            f"minimum is {judge_gate['min_keep_ratio']}. "
            "Rerun with --allow-low-judge-keep-ratio only for debugging."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
