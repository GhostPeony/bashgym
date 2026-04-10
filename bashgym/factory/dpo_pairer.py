"""
DPO Failure Pairing — Match failed traces to similar gold traces for DPO training.

Uses embedding-based similarity (via NVIDIA NIM API) to pair each failed trace
with the nearest gold trace. Produces DPOExample pairs where the gold response
is "chosen" and the failed response is "rejected".

Falls back gracefully to an empty list when NIM API is unavailable.

Module 3: Data Synthesis (The "Factory") - DPO Pairing
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .data_factory import DPOExample
from .dedup import DedupConfig, EmbeddingDeduplicator

logger = logging.getLogger(__name__)


def _extract_repo_name(trace: dict[str, Any]) -> str:
    """Extract repo name from a trace dict's metadata."""
    meta = trace.get("metadata", {})
    primary_repo = meta.get("primary_repo", {})
    if isinstance(primary_repo, dict):
        return primary_repo.get("name", "")
    return ""


def _extract_prompt(trace: dict[str, Any]) -> str:
    """Extract the user prompt from a trace."""
    meta = trace.get("metadata", {})
    prompt = meta.get("user_initial_prompt", "")
    if prompt:
        return prompt

    # Fall back to first user message
    for msg in trace.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")[:500]

    # Fall back to first step command
    steps = trace.get("trace", trace.get("steps", []))
    if steps:
        return steps[0].get("command", "coding task")[:500]

    return "coding task"


def _serialize_trace_response(trace: dict[str, Any]) -> str:
    """Serialize a trace's assistant actions into a text response for DPO."""
    parts: list[str] = []

    # Try messages format first
    for msg in trace.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                parts.append(content[:500])
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                parts.append(f"[{fn.get('name', 'tool')}] {fn.get('arguments', '')}"[:300])

    if parts:
        return "\n".join(parts)[:3000]

    # Fall back to trace/steps format
    for step in trace.get("trace", trace.get("steps", [])):
        tool = step.get("tool_name", step.get("tool", ""))
        cmd = step.get("command", step.get("input", ""))
        parts.append(f"[{tool}] {cmd}"[:300])

    return "\n".join(parts)[:3000] or "(empty response)"


def _load_traces_from_dir(trace_dir: Path, max_count: int = 500) -> list[dict[str, Any]]:
    """Load trace JSON files from a directory."""
    traces: list[dict[str, Any]] = []
    for f in sorted(trace_dir.glob("*.json"))[:max_count]:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                traces.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return traces


def pair_failures_for_dpo(
    gold_dir: Path,
    failed_dir: Path,
    similarity_threshold: float = 0.6,
    max_pairs: int = 200,
    dedup_config: DedupConfig | None = None,
) -> list[DPOExample]:
    """Match failed traces to similar gold traces and produce DPO pairs.

    For each failed trace, finds the nearest gold trace by embedding
    similarity (preferring same-repo matches). Produces a DPOExample
    with the gold response as chosen and the failed response as rejected.

    Args:
        gold_dir: Directory containing gold trace JSON files.
        failed_dir: Directory containing failed trace JSON files.
        similarity_threshold: Minimum cosine similarity to form a pair.
        max_pairs: Maximum number of DPO pairs to produce.
        dedup_config: Optional dedup config (NIM API key, model, etc).

    Returns:
        List of DPOExample objects. Empty list if embedding API unavailable.
    """
    gold_traces = _load_traces_from_dir(gold_dir)
    failed_traces = _load_traces_from_dir(failed_dir)

    if not gold_traces or not failed_traces:
        logger.info("[DPO Pairer] No traces to pair (gold=%d, failed=%d)", len(gold_traces), len(failed_traces))
        return []

    logger.info(
        "[DPO Pairer] Pairing %d failed traces against %d gold traces",
        len(failed_traces),
        len(gold_traces),
    )

    # Build embedding deduplicator for similarity computation
    deduplicator = EmbeddingDeduplicator(dedup_config)

    # Extract text representations for embedding
    gold_texts = [_serialize_trace_response(t) for t in gold_traces]
    failed_texts = [_serialize_trace_response(t) for t in failed_traces]

    try:
        gold_embeddings = deduplicator.compute_embeddings(gold_texts)
        failed_embeddings = deduplicator.compute_embeddings(failed_texts)
    except RuntimeError as e:
        logger.warning("[DPO Pairer] Embedding API unavailable, skipping: %s", e)
        return []

    # Build repo index for gold traces (prefer same-repo matches)
    gold_repos = [_extract_repo_name(t) for t in gold_traces]

    # Match each failed trace to nearest gold trace
    pairs: list[DPOExample] = []
    used_gold: set[int] = set()

    for fi, (failed_trace, failed_emb) in enumerate(zip(failed_traces, failed_embeddings)):
        if len(pairs) >= max_pairs:
            break

        failed_repo = _extract_repo_name(failed_trace)

        # Score all gold traces by similarity, with repo bonus
        best_gi = -1
        best_score = -1.0

        for gi, gold_emb in enumerate(gold_embeddings):
            if gi in used_gold:
                continue
            sim = deduplicator._cosine_similarity(failed_emb, gold_emb)
            # Repo-match bonus: +0.1 if same repo
            if failed_repo and gold_repos[gi] == failed_repo:
                sim += 0.1
            if sim > best_score:
                best_score = sim
                best_gi = gi

        if best_gi < 0 or best_score < similarity_threshold:
            continue

        used_gold.add(best_gi)
        gold_trace = gold_traces[best_gi]

        prompt = _extract_prompt(failed_trace) or _extract_prompt(gold_trace)
        chosen = _serialize_trace_response(gold_trace)
        rejected = _serialize_trace_response(failed_trace)

        example_id = hashlib.sha256(f"{prompt}{chosen}{rejected}".encode()).hexdigest()[:16]

        pairs.append(DPOExample(
            example_id=example_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={
                "gold_repo": _extract_repo_name(gold_trace),
                "failed_repo": _extract_repo_name(failed_trace),
                "similarity": round(best_score, 4),
            },
        ))

    logger.info("[DPO Pairer] Produced %d DPO pairs (threshold=%.2f)", len(pairs), similarity_threshold)
    return pairs
