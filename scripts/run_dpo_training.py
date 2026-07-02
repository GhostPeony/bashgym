#!/usr/bin/env python3
"""
DPO Training script for bashgym — plain TRL + transformers + peft.

Mirrors the GRPO trainer pattern:
  - Plain transformers (no Unsloth, no compiled kernels)
  - Gemma4ClippableLinear → nn.Linear monkey patch
  - PEFT LoRA with vision/audio modules excluded
  - SDPA attention (Flash 2 not available on sm_121)
  - DGX Spark Triton fix env vars set in bash before launch

Usage:
  python scripts/run_dpo_training.py \\
    --dataset data/dpo_real/train.jsonl \\
    --val-dataset data/dpo_real/val.jsonl \\
    --base-model /home/user/.unsloth/studio/exports/unsloth_gemma-4-E4B-it_1775455644/checkpoint-153 \\
    --output-dir data/dpo_runs/smoke1 \\
    --max-steps 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# DGX Spark sm_121 fix — set BEFORE any torch/triton imports
if Path("/usr/local/cuda/bin/ptxas").exists():
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1a")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="DPO train .jsonl")
    parser.add_argument("--val-dataset", type=Path, default=None, help="DPO val .jsonl (optional)")
    parser.add_argument(
        "--base-model",
        type=str,
        default="/home/user/.unsloth/studio/exports/unsloth_gemma-4-E4B-it_1775455644/checkpoint-153",
        help="Path to base model (your fine-tuned Gemma 4)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="DPO uses lower LR than SFT")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL coefficient")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    # Gemma 4 PEFT compatibility patch
    try:
        from transformers.models.gemma4 import modeling_gemma4

        class _PatchedClippableLinear(nn.Linear):
            def __init__(self, config, in_features, out_features):
                nn.Linear.__init__(self, in_features, out_features, bias=False)
                self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
                if self.use_clipped_linears:
                    self.register_buffer("input_min", torch.tensor(-float("inf")))
                    self.register_buffer("input_max", torch.tensor(float("inf")))
                    self.register_buffer("output_min", torch.tensor(-float("inf")))
                    self.register_buffer("output_max", torch.tensor(float("inf")))

            def forward(self, x):
                if self.use_clipped_linears:
                    x = torch.clamp(x, self.input_min, self.input_max)
                out = nn.Linear.forward(self, x)
                if self.use_clipped_linears:
                    out = torch.clamp(out, self.output_min, self.output_max)
                return out

        modeling_gemma4.Gemma4ClippableLinear = _PatchedClippableLinear
        logger.info("Applied Gemma4ClippableLinear → nn.Linear PEFT compatibility patch")
    except ImportError:
        logger.info("Not a Gemma 4 model — patch skipped")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from trl import DPOTrainer, DPOConfig

    logger.info("=" * 60)
    logger.info("DPO TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {args.base_model}")
    logger.info(f"  Train data:    {args.dataset}")
    logger.info(f"  Val data:      {args.val_dataset}")
    logger.info(f"  Output:        {args.output_dir}")
    logger.info(f"  Max steps:     {args.max_steps}")
    logger.info(f"  Beta (KL):     {args.beta}")
    logger.info(f"  LR:            {args.learning_rate}")
    logger.info("")

    logger.info(f"Loading model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    logger.info("Adding LoRA adapter (plain peft)...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    val_dataset = None
    if args.val_dataset and args.val_dataset.exists():
        val_dataset = load_dataset("json", data_files=str(args.val_dataset), split="train")
    logger.info(f"Train: {len(dataset)}, Val: {len(val_dataset) if val_dataset else 0}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=1,
        save_steps=max(args.max_steps // 2, 1),
        warmup_steps=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # peft auto-creates ref by disabling adapters
        args=dpo_config,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"\nSaving model to {args.output_dir}/final...")
    model.save_pretrained(args.output_dir / "final")
    tokenizer.save_pretrained(args.output_dir / "final")

    logger.info("Merging LoRA into base model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir / "merged")
    tokenizer.save_pretrained(args.output_dir / "merged")

    logger.info("\n✓ DPO training complete")
    logger.info(f"  Adapter: {args.output_dir / 'final'}")
    logger.info(f"  Merged:  {args.output_dir / 'merged'}")


if __name__ == "__main__":
    main()
