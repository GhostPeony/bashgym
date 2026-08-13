#!/usr/bin/env python3
"""
Extract DPO training pairs from real bashgym traces.

Pairs gold traces (success) with failed traces (failure) on the same task,
using exact prompt match + Jaccard similarity for near-matches.

No LLM calls — uses real human-validated good vs bad data.

Output: data/dpo_real/train.jsonl + val.jsonl
"""

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Sources — desktop has the canonical data
GOLD_DIR = Path.home() / "desktop-home" / ".bashgym" / "gold_traces"
FAILED_DIR = Path.home() / "desktop-home" / ".bashgym" / "failed_traces"
SILVER_DIR = Path.home() / "desktop-home" / ".bashgym" / "silver_traces"
BRONZE_DIR = Path.home() / "desktop-home" / ".bashgym" / "bronze_traces"

OUTPUT_DIR = Path.home() / "bashgym" / "data" / "dpo_real"

JACCARD_THRESHOLD = 0.4  # Lower = more matches but noisier


def load_traces(directory: Path) -> list[dict]:
    """Load all traces from a directory with their metadata."""
    traces = []
    if not directory.exists():
        return traces
    for f in sorted(directory.glob("*.json")):
        try:
            data = json.loads(f.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(data, dict):
            continue

        meta = data.get("metadata", {})
        prompt = ""
        if isinstance(meta, dict):
            prompt = (meta.get("user_initial_prompt") or "").strip()
        if len(prompt) < 30:
            continue

        traces.append(
            {
                "path": f,
                "data": data,
                "prompt": prompt,
            }
        )
    return traces


def serialize_trace_response(trace_data: dict, max_length: int = 6000) -> str:
    """Build a textual representation of what the trace did, for chosen/rejected."""
    summary = trace_data.get("summary", {})
    steps = trace_data.get("trace", [])
    if not isinstance(steps, list):
        return ""

    parts = []
    if isinstance(summary, dict):
        sr = summary.get("success_rate", "N/A")
        ts = summary.get("total_steps", len(steps))
        parts.append(f"[Trace summary: {ts} steps, success_rate={sr}]")

    for i, step in enumerate(steps[:30]):
        if not isinstance(step, dict):
            continue
        tool = step.get("tool_name") or step.get("tool") or "?"
        cmd = step.get("command", "")
        out = step.get("output", "")
        success = step.get("success", True)

        if isinstance(cmd, dict):
            cmd = json.dumps(cmd)
        cmd_str = str(cmd)[:300]
        out_str = str(out)[:400] if out else ""
        marker = "✓" if success else "✗"

        block = f"\nStep {i+1}: [{tool}] {marker}\n  cmd: {cmd_str}"
        if out_str:
            block += f"\n  out: {out_str}"
        parts.append(block)

        if sum(len(p) for p in parts) > max_length:
            parts.append(f"\n... ({len(steps) - i - 1} more steps truncated)")
            break

    return "\n".join(parts)


def jaccard(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def find_pairs(gold: list[dict], failed: list[dict]) -> list[tuple[dict, dict, str, float]]:
    """Find (gold, failed, match_type, similarity) pairs."""
    # Index gold by exact prompt
    gold_by_prompt = defaultdict(list)
    for g in gold:
        gold_by_prompt[g["prompt"][:500]].append(g)

    pairs = []
    matched_failed = set()

    # 1. Exact prompt matches
    for f in failed:
        key = f["prompt"][:500]
        if key in gold_by_prompt:
            for g in gold_by_prompt[key]:
                pairs.append((g, f, "exact", 1.0))
                matched_failed.add(id(f))
                break  # one per failed

    logger.info(f"  Exact matches: {len(pairs)}")

    # 2. Fuzzy matches for unmatched failed traces
    fuzzy_count = 0
    for f in failed:
        if id(f) in matched_failed:
            continue
        best_match = None
        best_sim = 0.0
        for g in gold:
            sim = jaccard(f["prompt"], g["prompt"])
            if sim > best_sim:
                best_sim = sim
                best_match = g
        if best_match and best_sim >= JACCARD_THRESHOLD:
            pairs.append((best_match, f, "fuzzy", best_sim))
            fuzzy_count += 1

    logger.info(f"  Fuzzy matches (Jaccard >= {JACCARD_THRESHOLD}): {fuzzy_count}")
    return pairs


def main():
    logger.info("=" * 60)
    logger.info("REAL DPO PAIR EXTRACTION FROM TRACES")
    logger.info("=" * 60)

    logger.info(f"Loading gold traces from {GOLD_DIR}...")
    gold = load_traces(GOLD_DIR)
    logger.info(f"  Loaded {len(gold)} gold traces with prompts")

    logger.info(f"Loading failed traces from {FAILED_DIR}...")
    failed = load_traces(FAILED_DIR)
    logger.info(f"  Loaded {len(failed)} failed traces with prompts")

    logger.info(f"Loading bronze traces (additional rejected pool) from {BRONZE_DIR}...")
    bronze = load_traces(BRONZE_DIR)
    logger.info(f"  Loaded {len(bronze)} bronze traces with prompts")

    rejected_pool = failed + bronze
    logger.info(f"  Total rejected pool: {len(rejected_pool)}")

    logger.info("\nFinding pairs...")
    pairs = find_pairs(gold, rejected_pool)
    logger.info(f"\nTotal pairs: {len(pairs)}")

    if not pairs:
        logger.error("No pairs found")
        sys.exit(1)

    # Convert to DPO format
    dpo_examples = []
    for gold_t, fail_t, match_type, sim in pairs:
        chosen = serialize_trace_response(gold_t["data"])
        rejected = serialize_trace_response(fail_t["data"])
        if not chosen or not rejected:
            continue
        dpo_examples.append(
            {
                "prompt": gold_t["prompt"][:4000],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "match_type": match_type,
                    "similarity": round(sim, 3),
                    "gold_path": str(gold_t["path"].name),
                    "rejected_path": str(fail_t["path"].name),
                },
            }
        )

    logger.info(f"After serialization: {len(dpo_examples)} valid pairs")

    # Train/val split
    random.seed(42)
    random.shuffle(dpo_examples)
    split = max(1, int(len(dpo_examples) * 0.9))
    train = dpo_examples[:split]
    val = dpo_examples[split:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"\nTrain: {len(train)} → {train_path}")
    logger.info(f"Val:   {len(val)} → {val_path}")

    # Match type distribution
    from collections import Counter

    types = Counter(ex["metadata"]["match_type"] for ex in dpo_examples)
    logger.info(f"\nMatch types: {dict(types)}")

    # Validate against DPO contract
    from bashgym.datasets.validator import print_validation_report, validate_dataset

    result = validate_dataset(train_path, format="dpo", quiet=True)
    print_validation_report(result, max_issues=5)


if __name__ == "__main__":
    main()
