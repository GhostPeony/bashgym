#!/usr/bin/env python3
"""
Full Training Data Rebuild — Import ALL Claude Code sessions and build clean training data.

Sources:
  - Desktop raw sessions: ~/desktop-home/.claude/projects/**/*.jsonl
  - DGX raw sessions: ~/.claude/projects/-home-ponyo/*.jsonl

Pipeline:
  1. Parse all raw JSONL session files directly (skip bashgym import/trace pipeline)
  2. Extract multi-turn conversations with tool calls
  3. Deduplicate by task content
  4. Add closing assistant messages
  5. Chunk oversized examples
  6. Clean train/val split
  7. Export to all formats
"""

import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sources
DESKTOP_CLAUDE = Path.home() / "desktop-home" / ".claude" / "projects"
DGX_CLAUDE = Path.home() / ".claude" / "projects" / "-home-ponyo"

# Output
OUTPUT_DIR = Path.home() / "bashgym-training" / "data-rebuilt"
MAX_SEQ_LENGTH = 8192
MIN_TOOL_CALLS = 2  # Minimum tool calls per example to be useful
MIN_SESSION_SIZE = 50_000  # Skip tiny sessions (< 50KB)

SYSTEM_PROMPT = """You are an expert software development agent. You execute tasks by running bash commands, reading files, and making edits. You think step-by-step and verify your work.

When given a task:
1. Analyze the requirements
2. Plan your approach
3. Execute commands to accomplish the task
4. Verify the results

You have access to these tools:
- Bash: Execute shell commands
- Read: Read file contents
- Write: Write to files
- Edit: Make targeted edits to files
- Grep: Search file contents
- Glob: Find files by pattern

Always explain your reasoning before taking action."""

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "Bash", "description": "Execute a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Write to a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Edit a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search file contents", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "Glob", "description": "Find files by pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]


# =========================================================================
# Stage 1: Parse raw Claude Code sessions
# =========================================================================


def find_all_sessions() -> list[Path]:
    """Find all raw Claude Code session JSONL files."""
    sessions = []

    # Desktop sessions
    if DESKTOP_CLAUDE.exists():
        for f in DESKTOP_CLAUDE.rglob("*.jsonl"):
            if f.stat().st_size >= MIN_SESSION_SIZE:
                sessions.append(f)

    # DGX sessions
    if DGX_CLAUDE.exists():
        for f in DGX_CLAUDE.glob("*.jsonl"):
            if f.stat().st_size >= MIN_SESSION_SIZE:
                sessions.append(f)

    return sorted(sessions, key=lambda f: f.stat().st_mtime, reverse=True)


def parse_session(path: Path) -> list[dict]:
    """Parse a raw Claude Code session JSONL into training messages.

    Reads the raw event stream and builds OpenAI-format message lists.
    Each conversation segment (user prompt → tool chain) becomes one example.
    """
    events = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if not events:
        return []

    # Build message sequence from events
    messages = []
    pending_tool_uses = {}  # id -> tool_use dict
    user_prompts = []

    for event in events:
        etype = event.get("type", "")

        if etype == "user":
            msg_content = event.get("message", {}).get("content", [])
            if isinstance(msg_content, str):
                # Plain text user message
                if msg_content.strip() and msg_content.strip() not in (
                    "[Request interrupted by user for tool use]",
                    "y", "yes", "ok", "continue", "go ahead",
                ):
                    messages.append({"role": "user", "content": msg_content.strip()})
                    user_prompts.append(msg_content.strip())
            elif isinstance(msg_content, list):
                # Could be tool results or text blocks
                text_parts = []
                for block in msg_content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")

                        if btype == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            content = block.get("content", "")
                            is_error = block.get("is_error", False)

                            # Extract text from content blocks
                            if isinstance(content, list):
                                content = "\n".join(
                                    b.get("text", "") for b in content
                                    if isinstance(b, dict) and b.get("type") == "text"
                                )
                            elif not isinstance(content, str):
                                content = str(content)

                            # Truncate long outputs but keep more than before
                            if len(content) > 8000:
                                content = content[:8000] + "\n... (truncated)"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            })

                        elif btype == "text":
                            text = block.get("text", "").strip()
                            if text and text not in (
                                "[Request interrupted by user for tool use]",
                                "y", "yes", "ok",
                            ):
                                text_parts.append(text)

                if text_parts:
                    full_text = "\n".join(text_parts)
                    messages.append({"role": "user", "content": full_text})
                    user_prompts.append(full_text)

        elif etype == "assistant":
            msg_content = event.get("message", {}).get("content", [])
            if isinstance(msg_content, str):
                if msg_content.strip():
                    messages.append({"role": "assistant", "content": msg_content.strip()})
            elif isinstance(msg_content, list):
                text_parts = []
                thinking_parts = []
                tool_calls = []

                for block in msg_content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")

                    if btype == "thinking":
                        thinking = block.get("thinking", "")
                        if thinking and len(thinking) > 20:
                            # Keep thinking but cap at 4KB for training
                            thinking_parts.append(thinking[:4000])

                    elif btype == "text":
                        text = block.get("text", "").strip()
                        if text:
                            text_parts.append(text)

                    elif btype == "tool_use":
                        tool_id = block.get("id", f"call_{hashlib.md5(json.dumps(block).encode()).hexdigest()[:8]}")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})

                        tool_calls.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                            },
                        })

                # Build content with optional thinking
                content_parts = []
                if thinking_parts:
                    content_parts.append(f"<thinking>\n{thinking_parts[0]}\n</thinking>")
                content_parts.extend(text_parts)
                content = "\n\n".join(content_parts) if content_parts else ""

                msg = {"role": "assistant", "content": content or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                    if msg["content"] is None:
                        msg["content"] = ""

                messages.append(msg)

    if not messages:
        return []

    # Segment into separate examples by user prompts
    examples = _segment_messages(messages, user_prompts, path)
    return examples


def _segment_messages(
    messages: list[dict], user_prompts: list[str], source_path: Path
) -> list[dict]:
    """Split a long conversation into separate training examples.

    Each segment starts with a user message and includes all subsequent
    assistant/tool messages until the next user message.
    """
    # Find user message indices
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]

    if not user_indices:
        return []

    # Determine project from path
    project = source_path.parent.name.replace("C--Users-Cade-projects-", "").replace("C--Users-Cade-", "home")

    segments = []
    for seg_idx, start in enumerate(user_indices):
        # End at next user message or end of conversation
        if seg_idx + 1 < len(user_indices):
            end = user_indices[seg_idx + 1]
        else:
            end = len(messages)

        seg_msgs = messages[start:end]

        # Count tool calls in this segment
        tool_call_count = sum(
            len(m.get("tool_calls", [])) for m in seg_msgs
            if m.get("role") == "assistant"
        )

        if tool_call_count < MIN_TOOL_CALLS:
            continue

        # Build the example with system prompt
        example_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + seg_msgs

        # Add closing assistant message if ends with tool
        if example_msgs[-1].get("role") == "tool":
            tool_counts = Counter()
            for m in example_msgs:
                for tc in m.get("tool_calls", []):
                    tool_counts[tc["function"]["name"]] += 1
            summary = "Task completed. " + ", ".join(
                f"{c} {n}" for n, c in tool_counts.most_common(5)
            ) + " operations."
            example_msgs.append({"role": "assistant", "content": summary})

        segments.append({
            "messages": example_msgs,
            "tools": TOOL_SCHEMAS,
            "metadata": {
                "source": str(source_path),
                "project": project,
                "tool_calls": tool_call_count,
                "segment_index": seg_idx,
            },
        })

    return segments


# =========================================================================
# Stage 2: Deduplicate
# =========================================================================


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove duplicate examples by user prompt content hash."""
    seen = set()
    unique = []

    for ex in examples:
        # Hash the user message content
        user_content = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                user_content = (msg.get("content") or "")[:500]
                break

        if not user_content or len(user_content) < 20:
            continue

        h = hashlib.sha256(user_content.encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            unique.append(ex)

    return unique


# =========================================================================
# Stage 3: Chunk oversized examples
# =========================================================================


def estimate_tokens(ex: dict, tokenizer=None) -> int:
    if tokenizer:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            return len(tokenizer(text)["input_ids"])
        except Exception:
            pass
    return sum(len(json.dumps(m)) for m in ex.get("messages", [])) // 4


def chunk_example(ex: dict, max_tokens: int, tokenizer=None) -> list[dict]:
    """Split oversized example into chunks that fit within max_tokens."""
    msgs = ex["messages"]
    header = [m for m in msgs if m["role"] in ("system", "user")][:2]
    body = msgs[len(header):]

    if not body:
        return [ex]

    header_tokens = estimate_tokens({"messages": header}, tokenizer)
    budget = max_tokens - header_tokens - 200

    # Group into assistant+tool pairs
    pairs = []
    current = []
    for msg in body:
        current.append(msg)
        if msg["role"] in ("tool", "assistant") and not msg.get("tool_calls"):
            if current:
                pairs.append(current)
                current = []
    if current:
        pairs.append(current)

    chunks = []
    chunk_msgs = list(header)
    chunk_tokens = header_tokens

    for pair in pairs:
        pair_tokens = estimate_tokens({"messages": pair}, tokenizer)
        if chunk_tokens + pair_tokens > max_tokens and len(chunk_msgs) > len(header):
            # Add closing message to chunk
            chunk_msgs.append({"role": "assistant", "content": "Continuing task..."})
            chunks.append({"messages": chunk_msgs, "tools": ex.get("tools"), "metadata": ex.get("metadata")})
            chunk_msgs = list(header)
            chunk_tokens = header_tokens

        chunk_msgs.extend(pair)
        chunk_tokens += pair_tokens

    if len(chunk_msgs) > len(header):
        chunks.append({"messages": chunk_msgs, "tools": ex.get("tools"), "metadata": ex.get("metadata")})

    return chunks if chunks else [ex]


# =========================================================================
# Main
# =========================================================================


def main():
    logger.info("=" * 60)
    logger.info("FULL TRAINING DATA REBUILD")
    logger.info("=" * 60)
    start_time = time.time()

    # Find all sessions
    logger.info("Scanning for sessions...")
    sessions = find_all_sessions()
    logger.info(f"Found {len(sessions)} sessions (>= {MIN_SESSION_SIZE//1000}KB)")

    # Count by source
    desktop_count = sum(1 for s in sessions if "desktop-home" in str(s))
    dgx_count = len(sessions) - desktop_count
    logger.info(f"  Desktop: {desktop_count}, DGX: {dgx_count}")

    # Count by project
    projects = Counter()
    for s in sessions:
        proj = s.parent.name.replace("C--Users-Cade-projects-", "").replace("C--Users-Cade-", "home")
        projects[proj] += 1
    logger.info("  By project:")
    for proj, count in projects.most_common(10):
        logger.info(f"    {proj}: {count}")
    logger.info("")

    # Parse all sessions
    logger.info("Parsing sessions...")
    all_examples = []
    failed = 0
    empty = 0

    for i, session_path in enumerate(sessions):
        try:
            examples = parse_session(session_path)
            if examples:
                all_examples.extend(examples)
            else:
                empty += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"  Failed to parse {session_path.name}: {e}")

        if (i + 1) % 200 == 0:
            logger.info(f"  [{i+1}/{len(sessions)}] extracted {len(all_examples)} examples so far")

    logger.info(f"\nParsed {len(sessions)} sessions → {len(all_examples)} raw examples")
    logger.info(f"  Empty: {empty}, Failed: {failed}")
    logger.info("")

    # Deduplicate
    logger.info("Deduplicating...")
    unique = deduplicate(all_examples)
    logger.info(f"  {len(all_examples)} → {len(unique)} unique examples")
    logger.info("")

    # Load tokenizer for accurate chunking
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tok_path = Path.home() / ".unsloth/studio/outputs/unsloth_gemma-4-E4B-it_1775370273"
        if tok_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
            logger.info("Using Gemma 4 tokenizer for accurate token counts")
    except ImportError:
        logger.info("Using char/4 token estimation")
    logger.info("")

    # Chunk oversized examples
    logger.info(f"Chunking examples exceeding {MAX_SEQ_LENGTH} tokens...")
    chunked = []
    for ex in unique:
        tokens = estimate_tokens(ex, tokenizer)
        if tokens <= MAX_SEQ_LENGTH:
            chunked.append(ex)
        else:
            chunks = chunk_example(ex, MAX_SEQ_LENGTH, tokenizer)
            chunked.extend(chunks)

    logger.info(f"  {len(unique)} → {len(chunked)} after chunking")
    logger.info("")

    # Filter out tiny examples (< 3 messages beyond system)
    filtered = [
        ex for ex in chunked
        if len(ex.get("messages", [])) >= 5  # system + user + at least one assistant+tool+closing
    ]
    logger.info(f"Filtered short examples: {len(chunked)} → {len(filtered)}")
    logger.info("")

    # Split train/val
    random.seed(42)
    random.shuffle(filtered)
    split = int(len(filtered) * 0.9)
    train = filtered[:split]
    val = filtered[split:]

    logger.info(f"Split: {len(train)} train / {len(val)} val")

    # Project distribution
    train_projects = Counter(
        ex.get("metadata", {}).get("project", "unknown") for ex in train
    )
    logger.info("  Train projects:")
    for proj, count in train_projects.most_common(10):
        logger.info(f"    {proj}: {count}")
    logger.info("")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {OUTPUT_DIR}/")
    logger.info(f"  train.jsonl: {train_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"  val.jsonl: {val_path.stat().st_size / 1e6:.1f} MB")
    logger.info("")

    # Export Unsloth formats
    logger.info("Exporting Unsloth formats...")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from export_unsloth import convert_file
        unsloth_dir = OUTPUT_DIR / "unsloth"
        unsloth_dir.mkdir(exist_ok=True)
        convert_file(train_path, unsloth_dir, ["chatml", "chatml_flat", "sharegpt"])
        convert_file(val_path, unsloth_dir, ["chatml", "chatml_flat", "sharegpt"])
    except Exception as e:
        logger.warning(f"Unsloth export failed: {e}")
    logger.info("")

    # Final stats
    duration = time.time() - start_time
    final_roles = Counter(
        ex["messages"][-1]["role"] for ex in train if ex.get("messages")
    )

    logger.info("=" * 60)
    logger.info("REBUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Duration: {duration:.0f}s")
    logger.info(f"  Sessions processed: {len(sessions)}")
    logger.info(f"  Train examples: {len(train)}")
    logger.info(f"  Val examples: {len(val)}")
    logger.info(f"  Train final roles: {dict(final_roles)}")
    logger.info(f"  Output: {OUTPUT_DIR}")

    if tokenizer:
        sample_tokens = [estimate_tokens(ex, tokenizer) for ex in train[:100]]
        exceeds = sum(1 for t in sample_tokens if t > MAX_SEQ_LENGTH)
        logger.info(f"  Token check (100 sample): {exceeds} exceed {MAX_SEQ_LENGTH}")

    logger.info(f"\nReady for training. Update configs to point at: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
