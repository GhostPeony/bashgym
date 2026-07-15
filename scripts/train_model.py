#!/usr/bin/env python3
"""Model-agnostic SFT training CLI for BashGym.

Replaces the per-model ``train_gemma4_*.py`` scripts with one parametrized
entrypoint: the model family (LoRA target modules, etc.) is resolved from
``--base-model`` via :mod:`bashgym.families`, so a new open model is just a
``--base-model <id>`` away.

Example:
    python scripts/train_model.py \\
        --base-model unsloth/gemma-4-E4B-it \\
        --train  ~/bashgym-training/data-pipeline-fixed/train.jsonl \\
        --val    ~/bashgym-training/data-pipeline-fixed/val.jsonl \\
        --output ~/bashgym-training/output-gemma4-e4b \\
        --max-seq-length 8192
"""

from __future__ import annotations

import argparse
import gc
import json
import os


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Coerce tool_call ``arguments`` to dicts.

    Gemma/Qwen chat templates require structured tool-call arguments; training
    data that stores them as JSON strings breaks ``apply_chat_template``. This
    parses string arguments back to dicts (falling back to ``{"raw": ...}``).
    """
    sanitized = []
    for msg in messages:
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        if isinstance(m.get("tool_calls"), list):
            fixed = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                if isinstance(tc.get("function"), dict):
                    fn = dict(tc["function"])
                    args = fn.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            fn["arguments"] = {"raw": args}
                    tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        sanitized.append(m)
    return sanitized


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-agnostic SFT training for BashGym")
    p.add_argument("--base-model", required=True, help="HF model id, e.g. unsloth/gemma-4-E4B-it")
    p.add_argument("--train", required=True, help="Path to train.jsonl")
    p.add_argument("--val", default=None, help="Path to val.jsonl (optional)")
    p.add_argument("--output", required=True, help="Output dir for checkpoints + merged model")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Off by default; GB10/sm_121 trains bf16 LoRA (bitsandbytes is broken there)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    from bashgym.models.artifact_capabilities import require_trainable_base

    try:
        require_trainable_base(args.base_model)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from bashgym.families import resolve_family_profile

    profile = resolve_family_profile(args.base_model)
    print(f"Model family: {profile.family} | LoRA targets: {list(profile.lora_target_modules)}")

    from unsloth import FastLanguageModel

    print(f"Loading {args.base_model} (bf16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
        device_map="sequential",
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=list(profile.lora_target_modules),
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    from datasets import load_dataset

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=os.path.expanduser(args.train), split="train")
    val_dataset = (
        load_dataset("json", data_files=os.path.expanduser(args.val), split="train")
        if args.val
        else None
    )
    print(f"Train: {len(dataset)}" + (f", Val: {len(val_dataset)}" if val_dataset else ""))

    def formatting_prompts_func(examples):
        texts = []
        for convo in examples["messages"]:
            try:
                clean = sanitize_messages(convo)
                texts.append(
                    tokenizer.apply_chat_template(
                        clean, tokenize=False, add_generation_prompt=False
                    )
                )
            except Exception:
                fallback = [
                    {"role": m.get("role", "user"), "content": m.get("content", "") or ""}
                    for m in convo
                    if m.get("role") in ("system", "user", "assistant")
                ]
                try:
                    texts.append(
                        tokenizer.apply_chat_template(
                            fallback, tokenize=False, add_generation_prompt=False
                        )
                    )
                except Exception:
                    texts.append("")
        return {"text": texts}

    print("Formatting datasets...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    if val_dataset is not None:
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    from transformers import TrainingArguments
    from trl import SFTTrainer

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print("Starting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=5,
            eval_strategy="steps" if val_dataset is not None else "no",
            eval_steps=25,
            save_strategy="steps",
            save_steps=100,
            output_dir=output_dir,
            report_to="none",
            weight_decay=0.01,
        ),
    )

    stats = trainer.train()
    print(f"\nTraining complete! Loss: {stats.training_loss:.4f}")

    print("Saving model...")
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    print("Merging LoRA weights...")
    model.save_pretrained_merged(
        os.path.join(output_dir, "merged"), tokenizer, save_method="merged_16bit"
    )
    print(f"Done! Model at {output_dir}")


if __name__ == "__main__":
    main()
