#!/usr/bin/env python3
"""Estimate Data Designer retrieval-query iteration size and provider cost.

This is intentionally conservative and provider-agnostic. Data Designer itself
does not set the price; the endpoint does. Pass the per-1M-token prices for the
provider/model you plan to use.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

QUERY_STYLES = (
    "natural_question",
    "keyword_query",
    "semantic_paraphrase",
)


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_chunks(
    corpus_rows: list[dict[str, Any]], sample_per_video: int
) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        by_video[row["youtube_video_id"]].append(row)

    out: list[dict[str, Any]] = []
    for video_id in sorted(by_video):
        chunks = sorted(by_video[video_id], key=lambda row: int(row["chunk_index"]))
        eligible = [row for row in chunks if len(row.get("text", "")) >= 280] or chunks
        positions = [(idx + 1) / (sample_per_video + 1) for idx in range(sample_per_video)]
        picked: list[dict[str, Any]] = []
        for pos in positions:
            target = round(pos * (len(eligible) - 1))
            candidate = eligible[target]
            if candidate not in picked:
                picked.append(candidate)
        for candidate in eligible:
            if len(picked) >= sample_per_video:
                break
            if candidate not in picked:
                picked.append(candidate)
        out.extend(picked[:sample_per_video])
    return out


def money(tokens: int, price_per_mtok: float) -> float:
    return (tokens / 1_000_000.0) * price_per_mtok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--sample-per-video", type=int, default=2)
    parser.add_argument("--query-styles", type=int, default=len(QUERY_STYLES))
    parser.add_argument("--excerpt-chars", type=int, default=1800)
    parser.add_argument("--generator-output-tokens", type=int, default=160)
    parser.add_argument("--judge-output-tokens", type=int, default=80)
    parser.add_argument("--generator-input-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--generator-output-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--judge-input-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--judge-output-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--embedding-input-price-per-mtok", type=float, default=0.0)
    parser.add_argument("--max-parallel-requests", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_rows = load_jsonl(args.corpus_jsonl)
    positives = selected_chunks(corpus_rows, args.sample_per_video)
    generated_rows = len(positives) * args.query_styles

    base_instruction_tokens = approx_tokens(
        "Generate retrieval-evaluation queries for a YouTube transcript search system. "
        "Use only this transcript chunk. Return JSON query variants."
    )
    generator_prompt_tokens = 0
    judge_prompt_tokens = 0
    embedding_tokens = 0
    for row in positives:
        excerpt = (row.get("text") or "")[: args.excerpt_chars]
        title = row.get("title") or ""
        chunk_tokens = base_instruction_tokens + approx_tokens(title) + approx_tokens(excerpt)
        generator_prompt_tokens += chunk_tokens
        judge_prompt_tokens += args.query_styles * (chunk_tokens + 60)
        embedding_tokens += args.query_styles * 18

    generator_output_tokens = generated_rows * args.generator_output_tokens
    judge_output_tokens = generated_rows * args.judge_output_tokens

    estimated_cost = (
        money(generator_prompt_tokens, args.generator_input_price_per_mtok)
        + money(generator_output_tokens, args.generator_output_price_per_mtok)
        + money(judge_prompt_tokens, args.judge_input_price_per_mtok)
        + money(judge_output_tokens, args.judge_output_price_per_mtok)
        + money(embedding_tokens, args.embedding_input_price_per_mtok)
    )

    ledger = {
        "created_at": datetime.now(UTC).isoformat(),
        "corpus_jsonl": str(args.corpus_jsonl),
        "corpus_rows": len(corpus_rows),
        "positive_chunks": len(positives),
        "query_styles": args.query_styles,
        "estimated_generated_rows": generated_rows,
        "max_parallel_requests": args.max_parallel_requests,
        "estimated_tokens": {
            "generator_input": generator_prompt_tokens,
            "generator_output": generator_output_tokens,
            "judge_input": judge_prompt_tokens,
            "judge_output": judge_output_tokens,
            "embedding_input": embedding_tokens,
        },
        "prices_per_1m_tokens_usd": {
            "generator_input": args.generator_input_price_per_mtok,
            "generator_output": args.generator_output_price_per_mtok,
            "judge_input": args.judge_input_price_per_mtok,
            "judge_output": args.judge_output_price_per_mtok,
            "embedding_input": args.embedding_input_price_per_mtok,
        },
        "estimated_cost_usd": round(estimated_cost, 6),
        "notes": [
            "Token estimates use len(text)/4 and should be replaced by provider usage when available.",
            "Data Designer cost comes from provider calls; local endpoints have no per-token bill but still need timeouts.",
            "Run preview at max_parallel_requests=1 before increasing concurrency.",
        ],
    }
    text = json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
