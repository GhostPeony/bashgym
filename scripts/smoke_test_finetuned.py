#!/usr/bin/env python3
"""Quick smoke test of the fine-tuned merged model."""

import json
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = "/home/ponyo/.unsloth/studio/exports/unsloth_gemma-4-E4B-it_1775455644/checkpoint-153"
VAL_PATH = Path.home() / "bashgym-training" / "data-pipeline-fixed" / "val.jsonl"


def main():
    print(f"Loading merged model from: {MODEL_PATH}")
    start = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="sequential",
    )
    print(f"Loaded in {time.time() - start:.1f}s")
    print(f"Device: {model.device}")
    print()

    # Load a few val examples
    with open(VAL_PATH) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(examples)} val examples")
    print()

    # Pick 3 examples to test
    test_indices = [0, 10, 50]
    for idx in test_indices:
        if idx >= len(examples):
            continue
        ex = examples[idx]
        msgs = ex["messages"]

        # Find the first user message after system
        prompt_msgs = []
        for msg in msgs:
            if msg["role"] in ("system", "user"):
                prompt_msgs.append(msg)
                if msg["role"] == "user":
                    break

        if len(prompt_msgs) < 2:
            continue

        print("=" * 70)
        print(f"TEST {idx}")
        print("=" * 70)
        print(f"USER PROMPT: {prompt_msgs[-1]['content'][:300]}")
        print()

        # Generate
        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_time = time.time() - gen_start

        generated = outputs[0][input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"FINE-TUNED RESPONSE ({gen_time:.1f}s, {len(generated)} tokens):")
        print(response[:1500])
        print()


if __name__ == "__main__":
    main()
