#!/usr/bin/env python3
"""Thorough functional test of the fine-tuned merged model.

Tests:
  1. Tool calling format (does it produce proper JSON tool_calls?)
  2. Generation quality across different task types
  3. Following partial conversations (continuation)
  4. Comparison: with vs without tools schema in prompt
  5. Plain text response (no tool calls) — does the closing message work?
"""

import json
import time
from pathlib import Path

import torch

MODEL_PATH = "/home/user/.unsloth/studio/exports/unsloth_gemma-4-E4B-it_1775455644/checkpoint-153"
VAL_PATH = Path.home() / "bashgym-training" / "data-pipeline-fixed" / "val.jsonl"


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="sequential",
    )
    print(f"Loaded in {time.time() - start:.0f}s\n")
    return model, tokenizer


def generate(model, tokenizer, messages, tools=None, max_tokens=300, temp=0.1):
    """Generate from messages, optionally with tools."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools is not None:
        kwargs["tools"] = tools

    try:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        # Fallback without tools
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"   [tools template failed: {e}]")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temp, 0.01),
            do_sample=temp > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    elapsed = time.time() - start
    generated_ids = outputs[0][input_len:]
    text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text_out, elapsed, len(generated_ids)


def check_tool_format(response: str) -> dict:
    """Check if response contains valid tool_call JSON."""
    indicators = {
        "has_json_braces": "{" in response and "}" in response,
        "has_function_keyword": '"function"' in response or '"name"' in response,
        "has_tool_call_id": "tool_call_id" in response or '"id"' in response,
        "has_call_text": "call:" in response.lower(),
        "has_thought": response.strip().startswith("thought") or "<thinking>" in response,
    }

    # Try to find a JSON tool call
    parsed = None
    if indicators["has_json_braces"]:
        # Try to extract first JSON object
        start = response.find("{")
        if start >= 0:
            # Try increasingly longer substrings
            for end in range(len(response), start, -1):
                try:
                    parsed = json.loads(response[start:end])
                    break
                except json.JSONDecodeError:
                    pass

    indicators["parsed_json"] = parsed is not None
    return indicators


def main():
    model, tokenizer = load_model()

    # Load val examples
    with open(VAL_PATH) as f:
        val_examples = [json.loads(line) for line in f if line.strip()]

    # Get tools schema from a val example
    tools = val_examples[0].get("tools")
    print(f"Loaded {len(val_examples)} val examples")
    print(f"Tools schema present: {tools is not None}, {len(tools) if tools else 0} tools")
    print()

    # ====================================================================
    # TEST 1: Tool calling format with tools schema
    # ====================================================================
    print("=" * 70)
    print("TEST 1: Can the model produce proper tool_calls JSON?")
    print("=" * 70)

    test_prompts = [
        "List the Python files in the current directory.",
        "Read the contents of README.md",
        "Search for the function 'main' in all Python files.",
    ]

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": val_examples[0]["messages"][0]["content"]},
            {"role": "user", "content": prompt},
        ]

        response, elapsed, n_tokens = generate(
            model, tokenizer, messages, tools=tools, max_tokens=200
        )
        check = check_tool_format(response)

        print(f"\nPROMPT: {prompt}")
        print(f"RESPONSE ({elapsed:.1f}s, {n_tokens} tok):")
        print(f"  {response[:400]}")
        print("  Format check:")
        for k, v in check.items():
            mark = "✓" if v else "✗"
            print(f"    {mark} {k}: {v}")

    # ====================================================================
    # TEST 2: Continuation — given a tool result, can it produce next step?
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Continuation (given tool output, predict next action)")
    print("=" * 70)

    # Use a real val example, give the model the prefix up to a tool result
    ex = val_examples[5]
    msgs = ex["messages"]

    # Find first tool message
    cut_idx = None
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            cut_idx = i + 1
            break

    if cut_idx and cut_idx < len(msgs):
        prefix = msgs[:cut_idx]
        gold_next = msgs[cut_idx]

        print("PREFIX ends with tool output (showing last 200 chars):")
        last = prefix[-1].get("content", "")[:200]
        print(f"  {last}")

        response, elapsed, n_tokens = generate(
            model, tokenizer, prefix, tools=tools, max_tokens=300
        )
        print(f"\nGOLD NEXT ROLE: {gold_next.get('role')}")
        if gold_next.get("tool_calls"):
            print(f"GOLD TOOL: {gold_next['tool_calls'][0]['function']['name']}")
        print(f"\nMODEL RESPONSE ({elapsed:.1f}s):")
        print(f"  {response[:500]}")

    # ====================================================================
    # TEST 3: Out-of-distribution prompt (something not in training)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Out-of-distribution prompt")
    print("=" * 70)

    ood_prompts = [
        "What is 2 + 2?",
        "Write a haiku about debugging.",
        "Explain what async/await does in Python.",
    ]

    for prompt in ood_prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response, elapsed, n_tokens = generate(
            model, tokenizer, messages, tools=None, max_tokens=200
        )
        print(f"\nPROMPT: {prompt}")
        print(f"RESPONSE ({elapsed:.1f}s):")
        print(f"  {response[:400]}")

    # ====================================================================
    # TEST 4: Closing message (does it produce text after tool sequence?)
    # ====================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Closing message - can it produce text after task done?")
    print("=" * 70)

    # Find a val example that ends with assistant message (closing)
    ex = None
    for v in val_examples:
        if v["messages"][-1].get("role") == "assistant" and not v["messages"][-1].get("tool_calls"):
            ex = v
            break

    if ex:
        # Cut just before final assistant message
        prefix = ex["messages"][:-1]
        gold_closing = ex["messages"][-1].get("content", "")

        response, elapsed, n_tokens = generate(
            model, tokenizer, prefix, tools=tools, max_tokens=200
        )
        print(f"GOLD CLOSING: {gold_closing[:200]}")
        print(f"\nMODEL CLOSING ({elapsed:.1f}s):")
        print(f"  {response[:300]}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
