#!/usr/bin/env python3
"""
Post-process Data Designer DPO output into clean train/val DPO JSONL.

Reads /home/ponyo/bashgym/data/dpo_synthetic/raw_designer_output.parquet
and produces train.jsonl + val.jsonl in DPO format.
"""

import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.home() / "bashgym" / "data" / "dpo_synthetic"
RAW_PATH = OUTPUT_DIR / "raw_designer_output.parquet"


def extract_score(judge_value, key: str = "quality") -> int:
    """Pull integer score out of {<key>: {'reasoning': '...', 'score': N}}."""
    if not isinstance(judge_value, dict):
        return 0
    val = judge_value.get(key)
    if isinstance(val, dict):
        return int(val.get("score", 0) or 0)
    if isinstance(val, (int, float)):
        return int(val)
    return 0


def main():
    if not RAW_PATH.exists():
        logger.error(f"Raw output not found: {RAW_PATH}")
        sys.exit(1)

    from datasets import Dataset
    ds = Dataset.from_parquet(str(RAW_PATH))
    logger.info(f"Loaded {len(ds)} raw records from {RAW_PATH}")

    dpo_examples = []
    skipped_ties = 0
    skipped_missing = 0

    columns = set(ds.column_names)
    pairwise_mode = "pairwise_judgment" in columns
    independent_mode = "judge_a" in columns and "judge_b" in columns

    for row in ds:
        prompt = row.get("task_prompt", "").strip()
        sol_a = row.get("solution_a", "")
        sol_b = row.get("solution_b", "")

        if not prompt or not sol_a or not sol_b:
            skipped_missing += 1
            continue

        if pairwise_mode:
            # Pairwise judgment: score 1-2 = A wins, 4-5 = B wins, 3 = tie
            verdict = extract_score(row.get("pairwise_judgment"), key="preferred")
            if verdict == 0 or verdict == 3:
                skipped_ties += 1
                continue
            if verdict <= 2:
                chosen, rejected = sol_a, sol_b
                meta = {"verdict": verdict, "winner": "A", "strength": "clear" if verdict == 1 else "slight"}
            else:
                chosen, rejected = sol_b, sol_a
                meta = {"verdict": verdict, "winner": "B", "strength": "clear" if verdict == 5 else "slight"}
        elif independent_mode:
            score_a = extract_score(row.get("judge_a"))
            score_b = extract_score(row.get("judge_b"))
            if score_a == 0 and score_b == 0:
                skipped_missing += 1
                continue
            if score_a == score_b:
                skipped_ties += 1
                continue
            if score_a > score_b:
                chosen, rejected = sol_a, sol_b
                meta = {"score_chosen": score_a, "score_rejected": score_b, "score_diff": score_a - score_b}
            else:
                chosen, rejected = sol_b, sol_a
                meta = {"score_chosen": score_b, "score_rejected": score_a, "score_diff": score_b - score_a}
        else:
            logger.error(f"Unknown judge column format. Columns: {columns}")
            sys.exit(1)

        dpo_examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": meta,
        })

    logger.info(f"Total raw: {len(ds)}")
    logger.info(f"Skipped (missing data): {skipped_missing}")
    logger.info(f"Skipped (judge tied): {skipped_ties}")
    logger.info(f"Final DPO pairs: {len(dpo_examples)}")

    if not dpo_examples:
        logger.error("No valid DPO pairs produced — check the raw data")
        sys.exit(1)

    # Train/val split
    random.seed(42)
    random.shuffle(dpo_examples)
    split = max(1, int(len(dpo_examples) * 0.9))
    train = dpo_examples[:split]
    val = dpo_examples[split:]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Train: {len(train)} → {train_path}")
    logger.info(f"Val:   {len(val)} → {val_path}")

    # Distribution summary
    if pairwise_mode:
        winners = [ex["metadata"].get("winner", "?") for ex in dpo_examples]
        from collections import Counter
        logger.info(f"\nWinner distribution: {dict(Counter(winners))}")
    elif independent_mode:
        score_diffs = [ex["metadata"].get("score_diff", 0) for ex in dpo_examples]
        logger.info(f"\nScore differences (chosen - rejected):")
        for diff in sorted(set(score_diffs)):
            n = score_diffs.count(diff)
            logger.info(f"  diff={diff}: {n} pairs")

    # Validate against DPO contract
    from bashgym.datasets.validator import validate_dataset, print_validation_report
    result = validate_dataset(train_path, format="dpo", quiet=True)
    print_validation_report(result, max_issues=5)


if __name__ == "__main__":
    main()
