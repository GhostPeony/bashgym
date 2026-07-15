#!/usr/bin/env python3
"""Generate MemexAI retrieval training pairs with Data Designer.

Two arms, both judge-gated and seeded exclusively from train-split videos:

- ``real_chunks``: synthetic queries grounded in real transcript chunks that
  the current training set has not used as positives yet.
- ``fake_transcripts``: synthetic transcript-style chunks written in the style
  of real train-video excerpts, then synthetic queries grounded in them.

Outputs per arm: ``train_queries.jsonl`` (trainer-compatible rows with
``split="train"``), ``synthetic_chunks.jsonl`` for the fake arm (training
corpus only — synthetic chunks must never enter the eval corpus), raw Data
Designer rows, seed files, and a manifest. Generation runs in batches through
``DataDesigner.preview`` against a local OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

QUERY_TYPES = ("natural_question", "keyword_query", "semantic_paraphrase")

SYNTHETIC_CHANNEL = "Synthetic Corpus"
SYNTHETIC_ANGLES = (
    "a concrete technical deep-dive",
    "a disagreement with a common claim",
    "a historical comparison",
    "a prediction about the next two years",
    "a worked example or anecdote",
    "an infrastructure or hardware constraint",
    "an open research question",
    "a practical adoption trade-off",
)


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
        ..., description="True only if the natural question is grounded in the chunk."
    )
    natural_question_reason: str = Field(
        "", description="Short reason for the natural question judgement."
    )
    keyword_query_pass: bool = Field(
        ..., description="True only if the keyword query is grounded in the chunk."
    )
    keyword_query_reason: str = Field("", description="Short reason for the keyword judgement.")
    semantic_paraphrase_pass: bool = Field(
        ..., description="True only if the semantic paraphrase is grounded in the chunk."
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


def train_video_ids(template_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["positive_video_id"])
        for row in template_rows
        if str(row.get("split") or "") == "train"
    }


def used_positive_chunk_ids(template_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["positive_chunk_id"]) for row in template_rows}


def select_real_seed_chunks(
    corpus_rows: list[dict[str, Any]],
    train_videos: set[str],
    used_chunk_ids: set[str],
    *,
    num_seeds: int,
    min_chunk_chars: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Round-robin unused, long-enough chunks across train videos."""
    rng = random.Random(seed)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        video_id = str(row["youtube_video_id"])
        if video_id not in train_videos:
            continue
        if str(row["chunk_id"]) in used_chunk_ids:
            continue
        if len(clean_text(str(row.get("text") or ""))) < min_chunk_chars:
            continue
        by_video[video_id].append(row)

    queues: list[list[dict[str, Any]]] = []
    for video_id in sorted(by_video):
        chunks = sorted(by_video[video_id], key=lambda row: int(row["chunk_index"]))
        rng.shuffle(chunks)
        queues.append(chunks)
    rng.shuffle(queues)

    seeds: list[dict[str, Any]] = []
    while queues and len(seeds) < num_seeds:
        for queue in list(queues):
            if len(seeds) >= num_seeds:
                break
            seeds.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return seeds


def real_seed_rows(seed_chunks: list[dict[str, Any]], excerpt_chars: int) -> list[dict[str, Any]]:
    return [
        {
            "seed_chunk_id": row["chunk_id"],
            "seed_video_id": row["youtube_video_id"],
            "seed_chunk_index": int(row["chunk_index"]),
            "seed_title": row["title"],
            "seed_channel": row["channel"],
            "seed_url": row.get("url") or "",
            "seed_text": clean_text(str(row.get("text") or ""))[:excerpt_chars],
        }
        for row in seed_chunks
    ]


def fake_seed_rows(
    exemplar_chunks: list[dict[str, Any]],
    *,
    num_seeds: int,
    style_excerpt_chars: int,
) -> list[dict[str, Any]]:
    if not exemplar_chunks:
        raise ValueError("fake_transcripts arm needs at least one style exemplar chunk")
    rows: list[dict[str, Any]] = []
    for idx in range(num_seeds):
        exemplar = exemplar_chunks[idx % len(exemplar_chunks)]
        angle = SYNTHETIC_ANGLES[idx % len(SYNTHETIC_ANGLES)]
        rows.append(
            {
                "seed_index": idx,
                "seed_angle": angle,
                "seed_style_title": exemplar["title"],
                "seed_style_channel": exemplar["channel"],
                "seed_style_video_id": exemplar["youtube_video_id"],
                "seed_style_chunk_id": exemplar["chunk_id"],
                "seed_style_text": clean_text(str(exemplar.get("text") or ""))[
                    :style_excerpt_chars
                ],
            }
        )
    return rows


def synthetic_chunk_is_valid(text: str, style_text: str = "") -> tuple[bool, str | None]:
    cleaned = clean_text(text)
    word_count = len(cleaned.split())
    if word_count < 120:
        return False, "too_short"
    if word_count > 900:
        return False, "too_long"
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ("as an ai", "i cannot", "language model")):
        return False, "refusal_or_ai_slop"
    if any(marker in text for marker in ("```", "\n#", "\n- ", "\n* ", "**")):
        return False, "markdown_formatting"
    if style_text and clean_text(style_text)[:200] and clean_text(style_text)[:200] in cleaned:
        return False, "copied_style_exemplar"
    return True, None


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


def coerce_structured(value: Any, keys: tuple[str, ...]) -> dict[str, Any] | None:
    if isinstance(value, BaseModel):
        data: Any = value.model_dump()
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
    if not all(key in data for key in keys):
        return None
    return data


def coerce_variants(value: Any) -> dict[str, str] | None:
    data = coerce_structured(value, QUERY_TYPES)
    if data is None:
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
    data = coerce_structured(value, ())
    if data is None:
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


def dedupe_query_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop rows whose normalized query text repeats an earlier row."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = clean_text(str(row["query"])).lower()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dropped


def query_rows_from_variants(
    *,
    variants: dict[str, str],
    judgement: dict[str, dict[str, Any]] | None,
    positive_chunk_id: str,
    positive_video_id: str,
    title: str,
    channel: str,
    model: str,
    generator_role: str,
    failures: Counter[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
                "eval_id": f"memexai-dd-train-{stable_id(positive_chunk_id, query_type, query)}",
                "query": query,
                "query_type": query_type,
                "positive_chunk_id": positive_chunk_id,
                "positive_video_id": positive_video_id,
                "split": "train",
                "title": title,
                "channel": channel,
                "generator": "data_designer",
                "generator_model": model,
                "judge_pass": bool(judge["pass"]) if judge else None,
                "judge_reason": judge["reason"] if judge else "",
                "judge_model": model if judge else None,
                "data_designer_role": generator_role,
            }
        )
    return out


def flatten_real_rows(
    dd_rows: list[dict[str, Any]], model: str, *, require_judge: bool
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
        out.extend(
            query_rows_from_variants(
                variants=variants,
                judgement=judgement,
                positive_chunk_id=str(row["seed_chunk_id"]),
                positive_video_id=str(row["seed_video_id"]),
                title=str(row.get("seed_title") or ""),
                channel=str(row.get("seed_channel") or ""),
                model=model,
                generator_role="synthetic_query_grounded_in_real_transcript_chunk",
                failures=failures,
            )
        )
    return out, failures


def flatten_fake_rows(
    dd_rows: list[dict[str, Any]],
    model: str,
    *,
    require_judge: bool,
    synthetic_video_group_size: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    failures: Counter[str] = Counter()
    query_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for row in dd_rows:
        text = clean_text(str(row.get("synthetic_transcript") or ""))
        ok, reason = synthetic_chunk_is_valid(
            str(row.get("synthetic_transcript") or ""), str(row.get("seed_style_text") or "")
        )
        if not ok:
            failures[f"synthetic_chunk:{reason}"] += 1
            continue
        variants = coerce_variants(row.get("query_variants"))
        if not variants:
            failures["missing_or_invalid_structured_output"] += 1
            continue
        judgement = coerce_judgement(row.get("query_judgement"))
        if require_judge and not judgement:
            failures["missing_or_invalid_judge_output"] += len(QUERY_TYPES)
            continue
        seed_index = int(row.get("seed_index") or 0)
        chunk_id = f"synthetic-{stable_id(str(seed_index), text)}"
        video_id = f"synthetic-video-{seed_index // synthetic_video_group_size:03d}"
        title = f"Synthetic: {row.get('seed_angle') or 'untitled'}"
        new_query_rows = query_rows_from_variants(
            variants=variants,
            judgement=judgement,
            positive_chunk_id=chunk_id,
            positive_video_id=video_id,
            title=title,
            channel=SYNTHETIC_CHANNEL,
            model=model,
            generator_role="synthetic_query_grounded_in_synthetic_transcript_chunk",
            failures=failures,
        )
        if not new_query_rows:
            failures["synthetic_chunk:no_surviving_queries"] += 1
            continue
        query_rows.extend(new_query_rows)
        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "youtube_video_id": video_id,
                "chunk_index": seed_index % synthetic_video_group_size,
                "title": title,
                "channel": SYNTHETIC_CHANNEL,
                "url": f"synthetic://data-designer/{chunk_id}",
                "source_set": "synthetic_dd",
                "start_seconds": 0,
                "end_seconds": 0,
                "text": text,
                "style_exemplar_chunk_id": row.get("seed_style_chunk_id") or "",
                "generator_model": model,
            }
        )
    return query_rows, chunk_rows, failures


def judge_gate_stats(
    raw_rows: list[dict[str, Any]],
    kept_rows: int,
    validation_failures: Counter[str],
    *,
    judge_enabled: bool,
    min_keep_ratio: float,
) -> dict[str, Any]:
    expected_query_rows = len(raw_rows) * len(QUERY_TYPES)
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


def build_dd_config(args: argparse.Namespace, seed_path: Path) -> Any:
    import data_designer.config as dd

    query_model = dd.ModelConfig(
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
    chunk_model = dd.ModelConfig(
        alias="chunk-model",
        model=args.llm_model,
        provider="local",
        skip_health_check=True,
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=args.chunk_temperature,
            top_p=0.95,
            max_tokens=args.chunk_max_tokens,
            timeout=args.timeout,
            max_parallel_requests=args.max_parallel_requests,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    builder = dd.DataDesignerConfigBuilder(model_configs=[query_model, chunk_model])
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(seed_path)))

    if args.arm == "fake_transcripts":
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="synthetic_transcript",
                model_alias="chunk-model",
                system_prompt=(
                    "You write realistic YouTube transcript excerpts for AI/ML podcast "
                    "conversations: spoken, informal, slightly rambling ASR-style prose. "
                    "Plain text only — no markdown, no speaker labels, no timestamps."
                ),
                prompt=(
                    "Here is a real transcript excerpt for style reference from "
                    '"{{ seed_style_title }}" ({{ seed_style_channel }}):\n\n'
                    "{{ seed_style_text }}\n\n"
                    "Write a NEW transcript chunk of 250-400 words in the same spoken style, "
                    "framed as {{ seed_angle }}. Pick a specific adjacent AI/ML subtopic that "
                    "the style excerpt does not cover, and stay concrete: name techniques, "
                    "systems, numbers, or trade-offs. Do not reuse sentences from the style "
                    "excerpt. Output the transcript text only."
                ),
            )
        )
        variants_context = (
            "Transcript chunk:\n{{ synthetic_transcript }}\n\n"
            "Create three retrieval queries: a natural user question, a concise "
            "keyword/entity query, and a lower-overlap semantic paraphrase. Each query "
            "must be answerable by this chunk."
        )
        judge_context = (
            "Transcript chunk:\n{{ synthetic_transcript }}\n\n"
            "Generated query variants:\n{{ query_variants }}\n\n"
            "For each query variant, set the corresponding *_pass boolean and a short "
            "reason. Pass only if the query can be resolved by this chunk."
        )
    else:
        variants_context = (
            "Video title: {{ seed_title }}\n"
            "Channel: {{ seed_channel }}\n\n"
            "Transcript chunk:\n{{ seed_text }}\n\n"
            "Create three retrieval queries: a natural user question, a concise "
            "keyword/entity query, and a lower-overlap semantic paraphrase. Each query "
            "must be answerable by this chunk."
        )
        judge_context = (
            "Video title: {{ seed_title }}\n"
            "Channel: {{ seed_channel }}\n\n"
            "Transcript chunk:\n{{ seed_text }}\n\n"
            "Generated query variants:\n{{ query_variants }}\n\n"
            "For each query variant, set the corresponding *_pass boolean and a short "
            "reason. Pass only if the query can be resolved by this chunk."
        )

    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="query_variants",
            model_alias="query-model",
            output_format=RetrievalQueryVariants,
            system_prompt=(
                "You create retrieval-training queries for a YouTube transcript search "
                "system. Use only the provided transcript chunk. Do not invent facts. "
                "Return structured JSON only."
            ),
            prompt=variants_context,
        )
    )
    if not args.disable_judge:
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="query_judgement",
                model_alias="query-model",
                output_format=RetrievalQueryJudgement,
                system_prompt=(
                    "You are a strict retrieval judge for a YouTube transcript search "
                    "system. Judge whether each generated query is answerable by the "
                    "provided transcript chunk. Reject vague, invented, or overly broad "
                    "queries. Return structured JSON only."
                ),
                prompt=judge_context,
            )
        )
    return builder


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


def run_dd_batches(
    args: argparse.Namespace, seed_rows: list[dict[str, Any]]
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
    designer = DataDesigner(model_providers=[provider])
    all_rows: list[dict[str, Any]] = []
    batch_stats: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_start in range(0, len(seed_rows), args.batch_seeds):
        batch = seed_rows[batch_start : batch_start + args.batch_seeds]
        batch_no = batch_start // args.batch_seeds + 1
        seed_path = args.output_dir / f"dd_seed_batch_{batch_no:03d}.jsonl"
        write_jsonl(seed_path, batch)
        builder = build_dd_config(args, seed_path)
        designer.validate(builder)
        batch_started = time.perf_counter()
        preview = designer.preview(builder, num_records=len(batch))
        elapsed = time.perf_counter() - batch_started
        rows = dataset_to_records(preview.dataset)
        all_rows.extend(rows)
        batch_stats.append(
            {
                "batch": batch_no,
                "seeds": len(batch),
                "rows": len(rows),
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(
            f"[batch {batch_no}] seeds={len(batch)} rows={len(rows)} "
            f"elapsed={elapsed:.1f}s total_rows={len(all_rows)}",
            flush=True,
        )
    return all_rows, {
        "data_designer_version": md.version("data-designer"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": args.llm_model,
        "endpoint": args.llm_endpoint,
        "max_parallel_requests": args.max_parallel_requests,
        "timeout": args.timeout,
        "temperature": args.temperature,
        "chunk_temperature": args.chunk_temperature,
        "thinking_disabled": True,
        "judge_enabled": not args.disable_judge,
        "batches": batch_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("real_chunks", "fake_transcripts"), required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument(
        "--template-queries-jsonl",
        type=Path,
        required=True,
        help="Existing eval query set; defines train-video ids and used positives.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=48)
    parser.add_argument("--batch-seeds", type=int, default=8)
    parser.add_argument("--excerpt-chars", type=int, default=1600)
    parser.add_argument("--style-excerpt-chars", type=int, default=900)
    parser.add_argument("--min-chunk-chars", type=int, default=300)
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8889/v1")
    parser.add_argument("--llm-model", default="hermes-qwen3.6-27b-dense")
    parser.add_argument("--llm-api-key", default="local")
    parser.add_argument("--max-parallel-requests", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--chunk-max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--chunk-temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--disable-judge", action="store_true")
    parser.add_argument("--min-judge-keep-ratio", type=float, default=0.70)
    parser.add_argument("--allow-low-judge-keep-ratio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = load_jsonl(args.corpus_jsonl)
    template_rows = load_jsonl(args.template_queries_jsonl)
    train_videos = train_video_ids(template_rows)
    if not train_videos:
        raise ValueError("template queries contain no train-split videos")
    used_ids = used_positive_chunk_ids(template_rows)

    if args.arm == "real_chunks":
        seed_chunks = select_real_seed_chunks(
            corpus_rows,
            train_videos,
            used_ids,
            num_seeds=args.num_seeds,
            min_chunk_chars=args.min_chunk_chars,
            seed=args.seed,
        )
        seed_rows = real_seed_rows(seed_chunks, args.excerpt_chars)
    else:
        exemplars = select_real_seed_chunks(
            corpus_rows,
            train_videos,
            set(),
            num_seeds=min(args.num_seeds, 16),
            min_chunk_chars=max(args.min_chunk_chars, 600),
            seed=args.seed + 1,
        )
        seed_rows = fake_seed_rows(
            exemplars, num_seeds=args.num_seeds, style_excerpt_chars=args.style_excerpt_chars
        )
    if not seed_rows:
        raise ValueError("no eligible seed rows for this arm")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_kind": f"data_designer_train_pairs_{args.arm}",
        "arm": args.arm,
        "corpus_jsonl": str(args.corpus_jsonl),
        "template_queries_jsonl": str(args.template_queries_jsonl),
        "train_video_count": len(train_videos),
        "seed_rows": len(seed_rows),
        "status": "dry_run" if args.dry_run else "started",
    }
    seeds_path = args.output_dir / "dd_seed_rows.jsonl"
    write_jsonl(seeds_path, seed_rows)
    if args.dry_run:
        write_json(args.output_dir / "dd_train_pairs_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    raw_rows, dd_stats = run_dd_batches(args, seed_rows)
    write_jsonl(args.output_dir / "dd_raw_rows.jsonl", raw_rows)

    judge_enabled = not args.disable_judge
    synthetic_chunk_rows: list[dict[str, Any]] = []
    if args.arm == "real_chunks":
        query_rows, failures = flatten_real_rows(
            raw_rows, args.llm_model, require_judge=judge_enabled
        )
    else:
        query_rows, synthetic_chunk_rows, failures = flatten_fake_rows(
            raw_rows, args.llm_model, require_judge=judge_enabled
        )
    query_rows, duplicate_count = dedupe_query_rows(query_rows)
    if duplicate_count:
        failures["duplicate_query_text"] += duplicate_count
    if not query_rows:
        raise RuntimeError(f"no valid training query rows produced: {dict(failures)}")

    queries_path = args.output_dir / "train_queries.jsonl"
    write_jsonl(queries_path, query_rows)
    if synthetic_chunk_rows:
        write_jsonl(args.output_dir / "synthetic_chunks.jsonl", synthetic_chunk_rows)

    gate = judge_gate_stats(
        raw_rows,
        len(query_rows),
        failures,
        judge_enabled=judge_enabled,
        min_keep_ratio=args.min_judge_keep_ratio,
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "raw_rows": len(raw_rows),
            "train_query_rows": len(query_rows),
            "synthetic_chunk_rows": len(synthetic_chunk_rows),
            "query_types": dict(sorted(Counter(row["query_type"] for row in query_rows).items())),
            "validation_failures": dict(sorted(failures.items())),
            "judge_gate": gate,
            "data_designer": dd_stats,
            "outputs": {
                "seed_rows_jsonl": str(seeds_path),
                "raw_rows_jsonl": str(args.output_dir / "dd_raw_rows.jsonl"),
                "train_queries_jsonl": str(queries_path),
                "synthetic_chunks_jsonl": (
                    str(args.output_dir / "synthetic_chunks.jsonl")
                    if synthetic_chunk_rows
                    else None
                ),
            },
        }
    )
    write_json(args.output_dir / "dd_train_pairs_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if judge_enabled and not gate["passed"] and not args.allow_low_judge_keep_ratio:
        raise RuntimeError(
            "Data Designer judge gate failed: "
            f"kept {gate['kept_query_rows']}/{gate['expected_query_rows']} rows "
            f"({gate['keep_ratio']}); minimum is {gate['min_keep_ratio']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
