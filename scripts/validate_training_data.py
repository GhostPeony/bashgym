#!/usr/bin/env python3
"""
Training Data Validator — Comprehensive quality checks for fine-tuning datasets.

Validates structural integrity, distribution health, token lengths,
and train/val overlap for the bashgym training data.
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TRAIN_PATH = Path.home() / "bashgym-training" / "data" / "train.jsonl"
VAL_PATH = Path.home() / "bashgym-training" / "data" / "val.jsonl"
REPORT_DIR = Path.home() / "bashgym" / "data" / "evaluation_results"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Line {i+1} in {path.name}: JSON parse error: {e}")
    return examples


def validate_structure(examples: list[dict], name: str) -> dict:
    """Check structural integrity of training examples."""
    issues = []
    stats = {
        "total": len(examples),
        "missing_messages": 0,
        "empty_messages": 0,
        "final_role_counts": Counter(),
        "role_sequence_errors": 0,
        "orphan_tool_calls": 0,
        "orphan_tool_responses": 0,
        "null_content_without_tools": 0,
        "has_tools_field": 0,
    }

    for idx, ex in enumerate(examples):
        msgs = ex.get("messages")
        if msgs is None:
            stats["missing_messages"] += 1
            issues.append(f"Example {idx}: missing 'messages' key")
            continue
        if len(msgs) == 0:
            stats["empty_messages"] += 1
            issues.append(f"Example {idx}: empty messages list")
            continue

        if "tools" in ex:
            stats["has_tools_field"] += 1

        # Track final role
        stats["final_role_counts"][msgs[-1].get("role", "unknown")] += 1

        # Check for null content without tool_calls
        for mi, msg in enumerate(msgs):
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if role == "assistant" and content is None and not tool_calls:
                stats["null_content_without_tools"] += 1

        # Check tool_call / tool response pairing
        pending_call_ids = set()
        for mi, msg in enumerate(msgs):
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        pending_call_ids.add(tc_id)

            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id in pending_call_ids:
                    pending_call_ids.discard(tc_id)
                elif tc_id:
                    stats["orphan_tool_responses"] += 1

        stats["orphan_tool_calls"] += len(pending_call_ids)

    stats["final_role_counts"] = dict(stats["final_role_counts"])

    # Flag critical issues
    final_roles = stats["final_role_counts"]
    if final_roles.get("tool", 0) == stats["total"] and stats["total"] > 0:
        issues.append(
            f"CRITICAL: All {stats['total']} {name} examples end with 'tool' role. "
            "The model never sees a final assistant text response during training."
        )
    if final_roles.get("assistant", 0) == 0 and stats["total"] > 0:
        issues.append(f"WARNING: Zero {name} examples end with 'assistant' role.")

    stats["issues"] = issues
    return stats


def analyze_distributions(examples: list[dict], name: str) -> dict:
    """Analyze tool usage, conversation lengths, and task types."""
    tool_counts = Counter()
    msg_lengths = []
    first_user_msgs = []
    role_counts = Counter()

    for ex in examples:
        msgs = ex.get("messages", [])
        msg_lengths.append(len(msgs))

        for msg in msgs:
            role = msg.get("role", "")
            role_counts[role] += 1

            # Count tool usage from tool_calls
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "unknown")
                    tool_counts[tool_name] += 1

            # Also count from tool role messages
            if role == "tool":
                # Tool name is in the preceding assistant's tool_call
                pass

            # Capture first user message for task classification
            if (
                role == "user"
                and not first_user_msgs
                or (first_user_msgs and first_user_msgs[-1] is None)
            ):
                if role == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 10:
                        first_user_msgs.append(content[:500])

        # Make sure we got a user msg for this example
        if len(first_user_msgs) < len(msg_lengths):
            # Find first user message
            for msg in msgs:
                if msg.get("role") == "user":
                    first_user_msgs.append((msg.get("content") or "")[:500])
                    break
            else:
                first_user_msgs.append(None)

    msg_lengths.sort()
    return {
        "name": name,
        "tool_usage": dict(tool_counts.most_common(20)),
        "role_distribution": dict(role_counts),
        "msg_length_stats": {
            "min": msg_lengths[0] if msg_lengths else 0,
            "max": msg_lengths[-1] if msg_lengths else 0,
            "median": msg_lengths[len(msg_lengths) // 2] if msg_lengths else 0,
            "mean": round(sum(msg_lengths) / len(msg_lengths), 1) if msg_lengths else 0,
        },
        "msg_length_histogram": {
            "1-10": sum(1 for length in msg_lengths if 1 <= length <= 10),
            "11-25": sum(1 for length in msg_lengths if 11 <= length <= 25),
            "26-50": sum(1 for length in msg_lengths if 26 <= length <= 50),
            "51-100": sum(1 for length in msg_lengths if 51 <= length <= 100),
            "100+": sum(1 for length in msg_lengths if length > 100),
        },
        "first_user_messages_sample": first_user_msgs[:5],
    }


def analyze_token_lengths(examples: list[dict], max_seq_length: int = 4096, tokenizer=None) -> dict:
    """Analyze token lengths using the actual model tokenizer."""
    if tokenizer is None:
        # Fall back to character-based estimation
        logger.warning("No tokenizer provided, using char/4 estimation")
        char_lengths = []
        for ex in examples:
            total = sum(len(json.dumps(m)) for m in ex.get("messages", []))
            char_lengths.append(total)

        token_estimates = [c // 4 for c in char_lengths]
    else:
        token_estimates = []
        for ex in examples:
            msgs = ex.get("messages", [])
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                tokens = tokenizer(text, return_tensors="pt")
                token_estimates.append(tokens["input_ids"].shape[1])
            except Exception:
                # Fallback for problematic examples
                total = sum(len(json.dumps(m)) for m in msgs)
                token_estimates.append(total // 4)

    token_estimates.sort()
    n = len(token_estimates)

    exceeds_max = sum(1 for t in token_estimates if t > max_seq_length)
    truncation_amounts = [max(0, t - max_seq_length) for t in token_estimates]

    return {
        "max_seq_length": max_seq_length,
        "token_stats": {
            "min": token_estimates[0] if n else 0,
            "max": token_estimates[-1] if n else 0,
            "median": token_estimates[n // 2] if n else 0,
            "mean": round(sum(token_estimates) / n, 1) if n else 0,
            "p90": token_estimates[int(n * 0.9)] if n else 0,
            "p95": token_estimates[int(n * 0.95)] if n else 0,
        },
        "truncation": {
            "exceeds_max_count": exceeds_max,
            "exceeds_max_pct": round(exceeds_max / n * 100, 1) if n else 0,
            "avg_truncation_tokens": round(sum(truncation_amounts) / n, 1) if n else 0,
            "max_truncation_tokens": max(truncation_amounts) if truncation_amounts else 0,
        },
        "length_buckets": {
            "0-1024": sum(1 for t in token_estimates if t <= 1024),
            "1025-2048": sum(1 for t in token_estimates if 1025 <= t <= 2048),
            "2049-4096": sum(1 for t in token_estimates if 2049 <= t <= 4096),
            "4097-8192": sum(1 for t in token_estimates if 4097 <= t <= 8192),
            "8193-16384": sum(1 for t in token_estimates if 8193 <= t <= 16384),
            "16384+": sum(1 for t in token_estimates if t > 16384),
        },
        "pct_seen_by_model": {
            "100%": sum(1 for t in token_estimates if t <= max_seq_length),
            "75-99%": sum(
                1 for t in token_estimates if t > max_seq_length and max_seq_length / t >= 0.75
            ),
            "50-74%": sum(
                1
                for t in token_estimates
                if max_seq_length / t >= 0.50 and max_seq_length / t < 0.75
            ),
            "<50%": sum(1 for t in token_estimates if t > 0 and max_seq_length / t < 0.50),
        },
    }


def check_overlap(train: list[dict], val: list[dict]) -> dict:
    """Check for train/val overlap at exact and semantic levels."""

    def extract_first_user(ex):
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                return (msg.get("content") or "").strip()
        return ""

    train_prompts = [extract_first_user(ex) for ex in train]
    val_prompts = [extract_first_user(ex) for ex in val]

    # Exact match
    train_set = set(train_prompts)
    exact_overlaps = [vp for vp in val_prompts if vp in train_set]

    # Jaccard similarity for near-duplicates
    def jaccard(a: str, b: str) -> float:
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    near_duplicates = []
    for vi, vp in enumerate(val_prompts):
        if not vp:
            continue
        for ti, tp in enumerate(train_prompts):
            if not tp:
                continue
            sim = jaccard(vp, tp)
            if sim > 0.8:
                near_duplicates.append(
                    {
                        "val_idx": vi,
                        "train_idx": ti,
                        "similarity": round(sim, 3),
                        "val_prompt": vp[:200],
                        "train_prompt": tp[:200],
                    }
                )

    return {
        "exact_overlaps": len(exact_overlaps),
        "near_duplicates_count": len(near_duplicates),
        "near_duplicates": near_duplicates[:10],  # Cap for report size
    }


def main():
    logger.info("=" * 60)
    logger.info("TRAINING DATA VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"Train: {TRAIN_PATH}")
    logger.info(f"Val:   {VAL_PATH}")
    logger.info("")

    # Load data
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    logger.info(f"Loaded: {len(train)} train, {len(val)} val examples")
    logger.info("")

    report = {
        "timestamp": datetime.now().isoformat(),
        "train_path": str(TRAIN_PATH),
        "val_path": str(VAL_PATH),
        "train_count": len(train),
        "val_count": len(val),
    }

    # 1. Structural validation
    logger.info("-" * 40)
    logger.info("1. STRUCTURAL INTEGRITY")
    logger.info("-" * 40)
    train_struct = validate_structure(train, "train")
    val_struct = validate_structure(val, "val")
    report["structure"] = {"train": train_struct, "val": val_struct}

    for issue in train_struct["issues"] + val_struct["issues"]:
        logger.info(f"  !! {issue}")
    if not train_struct["issues"] and not val_struct["issues"]:
        logger.info("  No structural issues found")

    logger.info(f"  Train final roles: {train_struct['final_role_counts']}")
    logger.info(f"  Val final roles:   {val_struct['final_role_counts']}")
    logger.info(f"  Train orphan tool_calls: {train_struct['orphan_tool_calls']}")
    logger.info(f"  Train orphan tool responses: {train_struct['orphan_tool_responses']}")
    logger.info(f"  Train examples with tools field: {train_struct['has_tools_field']}")
    logger.info("")

    # 2. Distribution analysis
    logger.info("-" * 40)
    logger.info("2. DISTRIBUTION ANALYSIS")
    logger.info("-" * 40)
    train_dist = analyze_distributions(train, "train")
    val_dist = analyze_distributions(val, "val")
    report["distributions"] = {"train": train_dist, "val": val_dist}

    logger.info("  Tool usage (train):")
    for tool, count in sorted(train_dist["tool_usage"].items(), key=lambda x: -x[1])[:10]:
        logger.info(f"    {tool:20s} {count:5d}")

    logger.info(f"\n  Message lengths (train): {train_dist['msg_length_stats']}")
    logger.info(f"  Message lengths (val):   {val_dist['msg_length_stats']}")
    logger.info(f"  Length histogram (train): {train_dist['msg_length_histogram']}")
    logger.info("")

    # 3. Token length analysis
    logger.info("-" * 40)
    logger.info("3. TOKEN LENGTH ANALYSIS")
    logger.info("-" * 40)

    # Try to load tokenizer for accurate counts
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer_path = Path.home() / ".unsloth/studio/outputs/unsloth_gemma-4-E4B-it_1775370273"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info("  Using actual Gemma 4 tokenizer for token counts")
        else:
            logger.info("  Tokenizer not found, using char/4 estimation")
    except ImportError:
        logger.info("  transformers not available, using char/4 estimation")

    train_tokens = analyze_token_lengths(train, max_seq_length=4096, tokenizer=tokenizer)
    val_tokens = analyze_token_lengths(val, max_seq_length=4096, tokenizer=tokenizer)
    report["token_lengths"] = {"train": train_tokens, "val": val_tokens}

    logger.info(f"  Train token stats: {train_tokens['token_stats']}")
    logger.info(f"  Val token stats:   {val_tokens['token_stats']}")
    logger.info(
        f"\n  Train truncation: {train_tokens['truncation']['exceeds_max_count']}/{len(train)} "
        f"({train_tokens['truncation']['exceeds_max_pct']}%) exceed {train_tokens['max_seq_length']} tokens"
    )
    logger.info(f"  Train length buckets: {train_tokens['length_buckets']}")
    logger.info(f"  Train % seen by model: {train_tokens['pct_seen_by_model']}")
    logger.info("")

    # 4. Overlap detection
    logger.info("-" * 40)
    logger.info("4. TRAIN/VAL OVERLAP")
    logger.info("-" * 40)
    overlap = check_overlap(train, val)
    report["overlap"] = overlap

    logger.info(f"  Exact overlaps: {overlap['exact_overlaps']}")
    logger.info(f"  Near-duplicates (Jaccard > 0.8): {overlap['near_duplicates_count']}")
    if overlap["near_duplicates"]:
        for nd in overlap["near_duplicates"][:3]:
            logger.info(
                f"    sim={nd['similarity']}: " f"val[{nd['val_idx']}] vs train[{nd['train_idx']}]"
            )
            logger.info(f"      val:   {nd['val_prompt'][:100]}...")
            logger.info(f"      train: {nd['train_prompt'][:100]}...")
    logger.info("")

    # 5. Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    critical = []
    warnings = []

    # Check for critical issues
    if train_struct["final_role_counts"].get("tool", 0) == len(train):
        critical.append(
            "All training examples end with 'tool' role — model never learns "
            "to produce final text responses"
        )

    trunc_pct = train_tokens["truncation"]["exceeds_max_pct"]
    if trunc_pct > 50:
        critical.append(
            f"{trunc_pct}% of training data exceeds max_seq_length={train_tokens['max_seq_length']} "
            "— majority of examples are truncated during training"
        )
    elif trunc_pct > 20:
        warnings.append(f"{trunc_pct}% of training data exceeds max_seq_length")

    if overlap["exact_overlaps"] > 0:
        critical.append(
            f"{overlap['exact_overlaps']} exact train/val overlaps detected — "
            "validation metrics are unreliable"
        )

    if overlap["near_duplicates_count"] > len(val) * 0.1:
        warnings.append(
            f"{overlap['near_duplicates_count']} near-duplicate prompts between train/val"
        )

    if train_struct["orphan_tool_calls"] > len(train) * 0.05:
        warnings.append(
            f"{train_struct['orphan_tool_calls']} orphan tool_calls without matching tool responses"
        )

    report["summary"] = {
        "critical_issues": critical,
        "warnings": warnings,
        "data_health": "CRITICAL" if critical else ("WARNING" if warnings else "HEALTHY"),
    }

    if critical:
        logger.info("  CRITICAL ISSUES:")
        for c in critical:
            logger.info(f"    [!!] {c}")
    if warnings:
        logger.info("  WARNINGS:")
        for w in warnings:
            logger.info(f"    [!] {w}")
    if not critical and not warnings:
        logger.info("  Data health: HEALTHY")
    logger.info("")

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    main()
