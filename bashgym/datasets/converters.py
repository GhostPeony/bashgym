"""
Dataset converters — turn raw bashgym traces into training-ready formats.

Each converter takes raw input and produces examples matching the
contract for a specific training method.
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


# =========================================================================
# Helpers
# =========================================================================


def _extract_user_prompt(trace: dict) -> str:
    """Pull the user's initial intent out of a raw trace."""
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        prompt = metadata.get("user_initial_prompt")
        if prompt and isinstance(prompt, str) and len(prompt.strip()) > 10:
            return prompt.strip()

    # Fallback: scan trace steps for the first user-ish thing
    for step in trace.get("trace", []):
        if not isinstance(step, dict):
            continue
        # Look for explicit user content
        for key in ("user_initial_prompt", "user_prompt", "prompt"):
            v = step.get(key)
            if v and isinstance(v, str) and len(v.strip()) > 10:
                return v.strip()

    # OpenAI/NeMo messages format: first user turn's text.
    for msg in trace.get("messages", []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if isinstance(content, str) and len(content.strip()) > 10:
            return content.strip()
    return ""


def _summarize_tool_usage(trace: dict, max_steps: int = 10) -> str:
    """Build a brief description of what the trace did, for fallback prompts."""
    steps = trace.get("trace", [])[:max_steps]
    tool_counts: dict[str, int] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool_name") or step.get("tool") or "?"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    # OpenAI/NeMo messages format: count tools from assistant tool_calls.
    for msg in trace.get("messages", []):
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name") or tc.get("name")
            if name:
                tool_counts[name] = tool_counts.get(name, 0) + 1
    if not tool_counts:
        return ""
    parts = ", ".join(f"{c} {n}" for n, c in sorted(tool_counts.items(), key=lambda x: -x[1]))
    return f"A coding task that used: {parts}"


def _gold_tool_sequence(trace: dict, max_steps: int = 5) -> list[str]:
    """Extract the first N tool names from a trace as the gold sequence."""
    seq = []
    for step in trace.get("trace", [])[:max_steps]:
        if isinstance(step, dict):
            tool = step.get("tool_name") or step.get("tool")
            if tool:
                seq.append(tool)
    return seq


def _quality_score(trace: dict) -> float | None:
    """Continuous quality score in [0, 1] for a trace (SERA-style soft signal).

    Carried into example metadata so downstream curriculum / loss-weighting can use a
    continuous score instead of the hard gold/failed bucket. Returns None when the
    trace has no scorable steps (e.g. public messages-format examples).
    """
    steps = trace.get("trace") or trace.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return None
    try:
        from bashgym.factory.quality_calculator import calculate_quality_breakdown

        vp = trace.get("verification_passed")
        meta = trace.get("metadata")
        breakdown = calculate_quality_breakdown(
            steps,
            verification_passed=vp,
            has_verification=vp is not None,
            metadata=meta if isinstance(meta, dict) else None,
        )
        return round(breakdown.total_score, 3)
    except Exception:
        return None


# =========================================================================
# Trace → GRPO converter
# =========================================================================


def trace_to_grpo_example(
    trace: dict,
    system_prompt: str | None = None,
    multimodal_format: bool = True,
) -> dict | None:
    """Convert a single bashgym trace into a GRPO training example.

    GRPO needs:
      - prompt: text or message list
      - tests: optional verification (we use the gold tool sequence as a hint)

    Args:
        trace: Raw bashgym trace dict
        system_prompt: Custom system prompt (or default coding agent one)
        multimodal_format: If True, wrap content as [{"type": "text", "text": ...}]
                           required for Gemma 4 / multimodal models

    Returns None if the trace doesn't have enough info to make a valid example.
    """
    user_prompt = _extract_user_prompt(trace)
    if not user_prompt:
        # Try to synthesize from tool summary
        summary = _summarize_tool_usage(trace)
        if not summary:
            return None
        user_prompt = summary

    # Build a chat-format prompt so the model gets the system context too
    sys_content = system_prompt or (
        "You are an expert software development agent. You execute tasks by "
        "running bash commands, reading files, and making edits. You think "
        "step-by-step and verify your work."
    )

    if multimodal_format:
        # Wrap content as list of typed parts for multimodal chat templates
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_content}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt[:4000]}]},
        ]
    else:
        prompt_messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_prompt[:4000]},
        ]

    gold_tools = _gold_tool_sequence(trace)
    metadata = {
        "trace_id": trace.get("session_id", "unknown"),
        "source": trace.get("source_tool", "unknown"),
        "gold_tool_sequence": gold_tools,
        "primary_repo": (trace.get("primary_repo") or {}).get("name"),
        "quality_score": _quality_score(trace),
    }

    example = {
        "prompt": prompt_messages,
        "metadata": metadata,
    }

    # If trace has verifiable tests, include them
    final_script = trace.get("final_bash_script")
    if isinstance(final_script, str) and "pytest" in final_script:
        example["tests"] = final_script[:8000]

    return example


def traces_to_grpo_dataset(traces: list[dict], system_prompt: str | None = None) -> list[dict]:
    """Batch-convert raw traces to GRPO format, dropping invalid ones."""
    out = []
    skipped = 0
    for trace in traces:
        ex = trace_to_grpo_example(trace, system_prompt=system_prompt)
        if ex is not None:
            out.append(ex)
        else:
            skipped += 1
    logger.info(f"trace→GRPO: {len(out)} converted, {skipped} skipped")
    return out


# =========================================================================
# Trace → SFT converter (already exists in example_generator, this is a thin wrapper)
# =========================================================================


def trace_to_sft_example(trace: dict, system_prompt: str | None = None) -> dict | None:
    """Convert a raw trace to an SFT example.

    For full segmentation use bashgym.factory.example_generator.ExampleGenerator
    which handles task boundaries, cognitive injection, and tool schemas.
    This is a simpler one-trace-to-one-example wrapper for quick conversions.
    """
    user_prompt = _extract_user_prompt(trace)
    if not user_prompt:
        return None

    sys_content = system_prompt or ("You are an expert software development agent.")

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_prompt[:4000]},
    ]

    # Walk the trace and build assistant + tool messages
    for i, step in enumerate(trace.get("trace", [])[:50]):  # cap depth
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool_name") or step.get("tool")
        if not tool_name:
            continue

        command = step.get("command", "")
        output = step.get("output", "")[:4000]

        tool_call_id = f"call_{hashlib.md5(f'{tool_name}_{i}'.encode()).hexdigest()[:10]}"

        # Try to construct args from command
        if isinstance(command, str):
            try:
                args_obj = json.loads(command) if command.startswith("{") else {"command": command}
            except json.JSONDecodeError:
                args_obj = {"command": command}
        elif isinstance(command, dict):
            args_obj = command
        else:
            args_obj = {"value": str(command)}

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args_obj),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": output if isinstance(output, str) else str(output),
            }
        )

    if len(messages) < 4:  # need at least system + user + 1 tool round
        return None

    # Add a closing assistant message summarizing what happened
    summary = _summarize_tool_usage(trace)
    if summary:
        messages.append({"role": "assistant", "content": f"Task completed. {summary}"})

    return {
        "messages": messages,
        "metadata": {
            "trace_id": trace.get("session_id", "unknown"),
            "source": trace.get("source_tool", "unknown"),
            "quality_score": _quality_score(trace),
        },
    }


# =========================================================================
# DPO and Distillation converters (stubs for future use)
# =========================================================================


def trace_pair_to_dpo_example(
    chosen_trace: dict, rejected_trace: dict, system_prompt: str | None = None
) -> dict | None:
    """Build a DPO example from two traces of the same task — one good, one bad.

    Caller is responsible for pairing — typically by user_prompt similarity
    where chosen is verified-passing and rejected is failing.
    """
    user_prompt = _extract_user_prompt(chosen_trace)
    if not user_prompt:
        return None

    chosen_summary = _summarize_tool_usage(chosen_trace)
    rejected_summary = _summarize_tool_usage(rejected_trace)
    if not chosen_summary or not rejected_summary:
        return None

    return {
        "prompt": [
            {
                "role": "system",
                "content": system_prompt or "You are an expert software development agent.",
            },
            {"role": "user", "content": user_prompt[:4000]},
        ],
        "chosen": chosen_summary,
        "rejected": rejected_summary,
        "metadata": {
            "chosen_trace": chosen_trace.get("session_id"),
            "rejected_trace": rejected_trace.get("session_id"),
        },
    }


# =========================================================================
# File-level helpers
# =========================================================================


def load_traces_from_dir(traces_dir: Path) -> Iterator[dict]:
    """Yield raw trace dicts from a directory of .json files."""
    traces_dir = Path(traces_dir)
    for path in sorted(traces_dir.glob("*.json")):
        try:
            yield json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {path.name}: {e}")
            continue


def write_jsonl(examples: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


# =========================================================================
# Public dataset → unified messages format
# =========================================================================

_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "bot": "assistant", "tool": "tool"}


def _sanitize_tool_calls(tool_calls: list) -> list:
    """Coerce tool_call ``arguments`` from JSON strings to dicts.

    Public datasets (e.g. SWE-rebench OpenHands trajectories) serialize tool-call
    arguments as strings; Gemma/Qwen chat templates require dicts.
    """
    fixed = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc = dict(tc)
        fn = tc.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {"raw": args}
            tc["function"] = fn
        fixed.append(tc)
    return fixed


def normalize_public_messages(example: dict) -> dict | None:
    """Normalize a public dataset example to our ``{"messages": [...]}`` format.

    Handles the common OpenHands / SWE-agent / ShareGPT shapes: the turns live
    under ``messages``/``conversations``/``trajectory``, roles may use ShareGPT
    aliases (``human``/``gpt``), content may be under ``content`` or ``value``, and
    tool-call arguments are coerced to dicts. Returns None if there is no usable
    user turn (no prompt to learn from).
    """
    turns = example.get("messages") or example.get("conversations") or example.get("trajectory")
    if not isinstance(turns, list) or not turns:
        return None

    out: list[dict] = []
    for msg in turns:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("from")
        role = _ROLE_ALIASES.get(role, role)
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = msg.get("content")
        if content is None:
            content = msg.get("value", "")
        norm = {"role": role, "content": content if isinstance(content, str) else str(content)}
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            norm["tool_calls"] = _sanitize_tool_calls(tcs)
        out.append(norm)

    if not any(m["role"] == "user" for m in out):
        return None
    return {"messages": out}
