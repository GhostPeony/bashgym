#!/usr/bin/env python3
"""
Fix Training Data — Clean up the bashgym training dataset.

Fixes:
  1. Remove train/val overlaps and rebuild clean val split
  2. Add closing assistant messages to every example
  3. Filter or chunk examples that exceed max_seq_length
  4. Re-export to all Unsloth formats
"""

import hashlib
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

TRAIN_PATH = Path.home() / "bashgym-training" / "data-pipeline" / "train.jsonl"
VAL_PATH = Path.home() / "bashgym-training" / "data-pipeline" / "val.jsonl"
OUTPUT_DIR = Path.home() / "bashgym-training" / "data-pipeline-fixed"
MAX_SEQ_LENGTH = 8192

# Add bashgym scripts to path for export_unsloth
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(examples)} examples to {path}")


def get_user_prompt(ex: dict) -> str:
    """Extract first user message content for deduplication."""
    for msg in ex.get("messages", []):
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return ""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# =========================================================================
# Fix 1: Deduplicate and rebuild clean train/val split
# =========================================================================


def fix_overlaps(train: list[dict], val: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove duplicates and create a clean train/val split."""
    logger.info("--- Fix 1: Removing overlaps and deduplicating ---")

    # Combine all examples
    all_examples = train + val

    # Deduplicate by first user message hash
    seen_hashes = set()
    unique = []
    duplicates = 0
    artifact_count = 0

    for ex in all_examples:
        prompt = get_user_prompt(ex)

        # Filter out artifact prompts that aren't real tasks
        if prompt in (
            "[Request interrupted by user for tool use]",
            "",
            "y",
            "yes",
            "ok",
            "continue",
        ):
            artifact_count += 1
            continue

        h = prompt_hash(prompt)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(ex)
        else:
            duplicates += 1

    logger.info(f"  Total examples: {len(all_examples)}")
    logger.info(f"  Artifacts removed: {artifact_count}")
    logger.info(f"  Duplicates removed: {duplicates}")
    logger.info(f"  Unique examples: {len(unique)}")

    # Shuffle and split 90/10
    random.seed(42)
    random.shuffle(unique)
    split_idx = int(len(unique) * 0.9)
    new_train = unique[:split_idx]
    new_val = unique[split_idx:]

    # Verify no overlap
    train_hashes = {prompt_hash(get_user_prompt(ex)) for ex in new_train}
    val_hashes = {prompt_hash(get_user_prompt(ex)) for ex in new_val}
    overlap = train_hashes & val_hashes
    logger.info(f"  New train: {len(new_train)}, new val: {len(new_val)}")
    logger.info(f"  Overlap check: {len(overlap)} (should be 0)")

    return new_train, new_val


# =========================================================================
# Fix 2: Add closing assistant messages
# =========================================================================


def add_closing_assistant(examples: list[dict]) -> list[dict]:
    """Add a brief closing assistant message to examples that end with tool role."""
    logger.info("--- Fix 2: Adding closing assistant messages ---")

    fixed = 0
    for ex in examples:
        msgs = ex.get("messages", [])
        if not msgs:
            continue

        if msgs[-1].get("role") == "tool":
            # Summarize what was done based on the conversation
            tool_counts = Counter()
            files_touched = set()
            for msg in msgs:
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        tool_counts[name] += 1

                        # Track files from Read/Write/Edit
                        args_str = fn.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            if isinstance(args, dict):
                                fp = args.get("file_path", "")
                                if fp:
                                    files_touched.add(Path(fp).name)
                        except (json.JSONDecodeError, TypeError):
                            pass

            # Build a concise summary
            summary_parts = []
            if tool_counts:
                tool_str = ", ".join(
                    f"{count} {name}" for name, count in tool_counts.most_common(5)
                )
                summary_parts.append(f"Used {tool_str}")
            if files_touched:
                files_str = ", ".join(sorted(files_touched)[:5])
                if len(files_touched) > 5:
                    files_str += f" and {len(files_touched) - 5} more"
                summary_parts.append(f"touched {files_str}")

            summary = "Task completed. " + ". ".join(summary_parts) + "." if summary_parts else "Task completed."

            msgs.append({
                "role": "assistant",
                "content": summary,
            })
            fixed += 1

    logger.info(f"  Added closing messages to {fixed}/{len(examples)} examples")
    return examples


# =========================================================================
# Fix 3: Handle oversized examples
# =========================================================================


def estimate_tokens(ex: dict, tokenizer=None) -> int:
    """Estimate token count for an example."""
    if tokenizer:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            return len(tokenizer(text)["input_ids"])
        except Exception:
            pass

    # Fallback: char/4 estimation
    total_chars = sum(len(json.dumps(m)) for m in ex.get("messages", []))
    return total_chars // 4


def chunk_long_example(ex: dict, max_tokens: int, tokenizer=None) -> list[dict]:
    """Split a long example into smaller chunks that fit within max_tokens.

    Each chunk keeps the system prompt and user prompt, then takes a window
    of the conversation that fits within the token budget.
    """
    msgs = ex.get("messages", [])

    # Separate header (system + user) from body (assistant/tool turns)
    header = []
    body = []
    for msg in msgs:
        if msg["role"] in ("system", "user") and not body:
            header.append(msg)
        else:
            body.append(msg)

    if not body:
        return [ex]

    # Estimate header tokens
    header_tokens = estimate_tokens({"messages": header}, tokenizer)
    budget = max_tokens - header_tokens - 100  # 100 token margin

    if budget <= 0:
        # Header alone exceeds budget — truncate system prompt
        for msg in header:
            if msg["role"] == "system":
                msg["content"] = msg["content"][:2000]
        header_tokens = estimate_tokens({"messages": header}, tokenizer)
        budget = max_tokens - header_tokens - 100

    # Group body messages into assistant+tool pairs
    pairs = []
    current_pair = []
    for msg in body:
        current_pair.append(msg)
        if msg["role"] == "tool" or (msg["role"] == "assistant" and not msg.get("tool_calls")):
            pairs.append(current_pair)
            current_pair = []
    if current_pair:
        pairs.append(current_pair)

    # Build chunks
    chunks = []
    current_chunk_msgs = list(header)
    current_tokens = header_tokens

    for pair in pairs:
        pair_tokens = estimate_tokens({"messages": pair}, tokenizer)

        if current_tokens + pair_tokens > max_tokens and len(current_chunk_msgs) > len(header):
            # Save current chunk
            chunks.append({
                "messages": current_chunk_msgs,
                "tools": ex.get("tools"),
                "metadata": ex.get("metadata"),
            })
            current_chunk_msgs = list(header)
            current_tokens = header_tokens

        current_chunk_msgs.extend(pair)
        current_tokens += pair_tokens

    # Save last chunk
    if len(current_chunk_msgs) > len(header):
        chunks.append({
            "messages": current_chunk_msgs,
            "tools": ex.get("tools"),
            "metadata": ex.get("metadata"),
        })

    return chunks if chunks else [ex]


def fix_lengths(examples: list[dict], max_tokens: int, tokenizer=None) -> list[dict]:
    """Handle examples that exceed max token length."""
    logger.info(f"--- Fix 3: Handling examples exceeding {max_tokens} tokens ---")

    result = []
    chunked = 0
    dropped = 0

    for ex in examples:
        tokens = estimate_tokens(ex, tokenizer)

        if tokens <= max_tokens:
            result.append(ex)
        else:
            # Try to chunk it
            chunks = chunk_long_example(ex, max_tokens, tokenizer)
            if chunks and len(chunks[0].get("messages", [])) > 3:
                result.extend(chunks)
                chunked += 1
            else:
                dropped += 1

    logger.info(f"  Original: {len(examples)} examples")
    logger.info(f"  Within limit: {len(examples) - chunked - dropped}")
    logger.info(f"  Chunked into multiple: {chunked}")
    logger.info(f"  Dropped (too short after chunking): {dropped}")
    logger.info(f"  Final: {len(result)} examples")

    return result


# =========================================================================
# Re-export to Unsloth formats
# =========================================================================


def export_unsloth_formats(train_path: Path, val_path: Path, output_dir: Path):
    """Re-export to chatml, chatml_flat, and sharegpt formats."""
    logger.info("--- Exporting to Unsloth formats ---")

    unsloth_dir = output_dir / "unsloth"
    unsloth_dir.mkdir(parents=True, exist_ok=True)

    try:
        from export_unsloth import convert_file
        for jsonl_file in [train_path, val_path]:
            logger.info(f"  Converting {jsonl_file.name}:")
            convert_file(jsonl_file, unsloth_dir, ["chatml", "chatml_flat", "sharegpt"])
    except ImportError:
        logger.warning("  Could not import export_unsloth — skipping format conversion")
        logger.info("  Run manually: python scripts/export_unsloth.py --input-dir data-pipeline-fixed")


# =========================================================================
# Main
# =========================================================================


def main():
    logger.info("=" * 60)
    logger.info("TRAINING DATA FIX")
    logger.info("=" * 60)

    # Load
    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    logger.info(f"Loaded: {len(train)} train + {len(val)} val = {len(train) + len(val)} total\n")

    # Try to load tokenizer for accurate token counts
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer_path = Path.home() / ".unsloth/studio/outputs/unsloth_gemma-4-E4B-it_1775370273"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            logger.info(f"Using Gemma 4 tokenizer for accurate token counts\n")
    except ImportError:
        logger.info("transformers not available, using char/4 estimates\n")

    # Fix 1: Dedup and clean split
    train, val = fix_overlaps(train, val)
    logger.info("")

    # Fix 2: Add closing assistant messages
    train = add_closing_assistant(train)
    val = add_closing_assistant(val)
    logger.info("")

    # Fix 3: Handle oversized examples
    train = fix_lengths(train, MAX_SEQ_LENGTH, tokenizer)
    val = fix_lengths(val, MAX_SEQ_LENGTH, tokenizer)
    logger.info("")

    # Save fixed data
    logger.info("--- Saving fixed data ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    logger.info("")

    # Export Unsloth formats
    export_unsloth_formats(train_path, val_path, OUTPUT_DIR)
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"  Train examples: {len(train)}")
    logger.info(f"  Val examples: {len(val)}")
    logger.info(f"  Max seq length: {MAX_SEQ_LENGTH}")

    # Quick validation
    train_final_roles = Counter(
        ex["messages"][-1]["role"] for ex in train if ex.get("messages")
    )
    val_final_roles = Counter(
        ex["messages"][-1]["role"] for ex in val if ex.get("messages")
    )
    logger.info(f"  Train final roles: {dict(train_final_roles)}")
    logger.info(f"  Val final roles: {dict(val_final_roles)}")

    # Token length check
    if tokenizer:
        train_tokens = [estimate_tokens(ex, tokenizer) for ex in train[:50]]
        exceeds = sum(1 for t in train_tokens if t > MAX_SEQ_LENGTH)
        logger.info(
            f"  Sample token check (first 50): {exceeds}/{len(train_tokens)} exceed {MAX_SEQ_LENGTH}"
        )

    logger.info(f"\nDone! Run validate_training_data.py on the fixed data to verify.")
    logger.info(f"Fixed data at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
