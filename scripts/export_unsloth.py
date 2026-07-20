#!/usr/bin/env python3
"""
Export bashgym traces to Unsloth Studio compatible formats.

Outputs:
  - ChatML (OpenAI) format: messages with role/content, tool calls flattened
  - ShareGPT format: conversations with from/value pairs

Both formats are directly uploadable to Unsloth Studio or HuggingFace datasets.
"""

import argparse
import json
from pathlib import Path


def fix_double_escaped_args(arguments: str) -> str:
    """Fix double-encoded JSON in tool call arguments.

    The trace capture sometimes produces arguments like:
      '{"file_path": "{\\"file_path\\": \\"actual/path\\"}"}'
    This unwraps them to the inner object.
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return arguments

    if isinstance(parsed, dict):
        # Check if any value is itself a JSON string containing the same keys
        for key, val in parsed.items():
            if isinstance(val, str) and val.startswith("{"):
                try:
                    inner = json.loads(val)
                    if isinstance(inner, dict):
                        # The inner object is the real payload
                        return json.dumps(inner)
                except (json.JSONDecodeError, TypeError):
                    pass
    return arguments if isinstance(arguments, str) else json.dumps(parsed)


def sanitize_arguments(arguments: str) -> dict:
    """Parse tool call arguments into a clean dict."""
    fixed = fix_double_escaped_args(arguments)
    try:
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return {"raw": fixed}


def flatten_tool_call(tool_call: dict) -> str:
    """Render a tool call as readable text for non-tool-call formats."""
    func = tool_call.get("function", {})
    name = func.get("name", "unknown")
    args = sanitize_arguments(func.get("arguments", "{}"))

    if name == "Bash":
        cmd = args.get("command", "")
        return f"```bash\n{cmd}\n```"
    elif name == "Read":
        return f"[Reading file: {args.get('file_path', '?')}]"
    elif name == "Write":
        path = args.get("file_path", "?")
        content = args.get("content", "")
        # Truncate very long file writes for training efficiency
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        return f"[Writing file: {path}]\n```\n{content}\n```"
    elif name == "Edit":
        path = args.get("file_path", "?")
        old = args.get("old_string", "")[:500]
        new = args.get("new_string", "")[:500]
        return f"[Editing file: {path}]\n- old: `{old}`\n+ new: `{new}`"
    elif name in ("Grep", "Glob"):
        return f"[{name}: {args.get('pattern', '?')}]"
    else:
        return f"[{name}({json.dumps(args)[:500]})]"


def to_chatml_clean(example: dict) -> dict:
    """Convert to clean ChatML format with sanitized tool calls.

    This preserves the OpenAI messages structure but fixes argument encoding
    and ensures content is never null.
    """
    messages = []
    for msg in example.get("messages", []):
        clean_msg = {"role": msg["role"], "content": msg.get("content") or ""}

        if msg.get("tool_calls"):
            clean_msg["tool_calls"] = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                clean_msg["tool_calls"].append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": json.dumps(
                                sanitize_arguments(func.get("arguments", "{}"))
                            ),
                        },
                    }
                )

        if msg.get("tool_call_id"):
            clean_msg["tool_call_id"] = msg["tool_call_id"]

        messages.append(clean_msg)

    result = {"messages": messages}
    if example.get("tools"):
        result["tools"] = example["tools"]
    return result


def to_chatml_flattened(example: dict) -> dict:
    """Convert to ChatML with tool calls flattened into content strings.

    This is the safest format for models that don't natively support
    function calling — the tool interactions become plain text.
    """
    messages = []
    for i, msg in enumerate(example.get("messages", [])):
        role = msg["role"]
        content = msg.get("content") or ""

        if role == "tool":
            # Tool responses are merged into the preceding assistant message
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # Build a combined message: thinking + tool calls + tool results
            parts = []
            if content:
                parts.append(content)

            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                parts.append(flatten_tool_call(tc))

                # Find the matching tool response
                for later_msg in example["messages"][i + 1 :]:
                    if later_msg["role"] == "tool" and later_msg.get("tool_call_id") == tc_id:
                        tool_output = later_msg.get("content", "")
                        if len(tool_output) > 1500:
                            tool_output = tool_output[:1500] + "\n... (truncated)"
                        parts.append(f"Output:\n{tool_output}")
                        break

            messages.append({"role": "assistant", "content": "\n\n".join(parts)})
        else:
            messages.append({"role": role, "content": content})

    return {"messages": messages}


def to_sharegpt(example: dict) -> dict:
    """Convert to ShareGPT format (conversations with from/value).

    Tool calls are flattened into the assistant's value text.
    System messages become the first turn with from='system'.
    """
    conversations = []

    for i, msg in enumerate(example.get("messages", [])):
        role = msg["role"]
        content = msg.get("content") or ""

        if role == "tool":
            continue

        role_map = {"system": "system", "user": "human", "assistant": "gpt"}
        sgpt_role = role_map.get(role, role)

        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if content:
                parts.append(content)

            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                parts.append(flatten_tool_call(tc))

                for later_msg in example["messages"][i + 1 :]:
                    if later_msg["role"] == "tool" and later_msg.get("tool_call_id") == tc_id:
                        tool_output = later_msg.get("content", "")
                        if len(tool_output) > 1500:
                            tool_output = tool_output[:1500] + "\n... (truncated)"
                        parts.append(f"Output:\n{tool_output}")
                        break

            conversations.append({"from": sgpt_role, "value": "\n\n".join(parts)})
        else:
            conversations.append({"from": sgpt_role, "value": content})

    return {"conversations": conversations}


def convert_file(input_path: Path, output_dir: Path, formats: list[str]) -> dict:
    """Convert a JSONL file to the requested formats."""
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    results = {}
    stem = input_path.stem  # train or val

    for fmt in formats:
        if fmt == "chatml":
            converter = to_chatml_clean
            suffix = "chatml"
        elif fmt == "chatml_flat":
            converter = to_chatml_flattened
            suffix = "chatml_flat"
        elif fmt == "sharegpt":
            converter = to_sharegpt
            suffix = "sharegpt"
        else:
            raise ValueError(f"Unknown format: {fmt}")

        out_path = output_dir / f"{stem}_{suffix}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                converted = converter(ex)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        results[fmt] = {"path": str(out_path), "count": len(examples), "size_mb": round(size_mb, 2)}
        print(f"  [{fmt}] {out_path.name}: {len(examples)} examples, {size_mb:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export bashgym traces for Unsloth Studio")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/user/bashgym-training/data"),
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir/unsloth)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["chatml", "chatml_flat", "sharegpt"],
        choices=["chatml", "chatml_flat", "sharegpt"],
        help="Output formats to generate",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir / "unsloth"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting bashgym traces for Unsloth Studio")
    print(f"  Input:   {args.input_dir}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Formats: {args.formats}")
    print()

    all_results = {}
    for jsonl_file in sorted(args.input_dir.glob("*.jsonl")):
        if jsonl_file.parent.name == "unsloth":
            continue
        print(f"Converting {jsonl_file.name}:")
        all_results[jsonl_file.stem] = convert_file(jsonl_file, args.output_dir, args.formats)
        print()

    # Write a manifest
    manifest = {
        "source": str(args.input_dir),
        "formats": {},
    }
    for fmt in args.formats:
        manifest["formats"][fmt] = {
            "description": {
                "chatml": "OpenAI ChatML with sanitized tool_calls (native function calling)",
                "chatml_flat": "OpenAI ChatML with tool calls flattened to text (universal compatibility)",
                "sharegpt": "ShareGPT format with tool calls flattened to text",
            }[fmt],
            "files": {stem: info[fmt] for stem, info in all_results.items() if fmt in info},
        }

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    print("\n--- Recommended for Unsloth Studio ---")
    print("  Gemma 4 with tool use:    upload *_chatml.jsonl files")
    print("  Gemma 4 without tool use: upload *_chatml_flat.jsonl or *_sharegpt.jsonl files")


if __name__ == "__main__":
    main()
