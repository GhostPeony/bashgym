#!/usr/bin/env python3
"""
Run cascade RL training in REAL mode using the fine-tuned Gemma 4 E4B as base.

Pipeline:
  1. Load fine-tuned merged model as starting point
  2. Filter gold traces into 4 domains by tool usage
  3. Run GRPO sequentially across domains
  4. Each stage's checkpoint becomes the next stage's base
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Use our fine-tuned merged model as the base
FT_MODEL_PATH = "/home/ponyo/.unsloth/studio/exports/unsloth_gemma-4-E4B-it_1775455644/checkpoint-153"

# Test with Qwen to isolate Gemma 4 multimodal issues from bashgym pipeline issues
QWEN_TEST_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
GOLD_TRACES_DIR = Path.home() / "bashgym" / "data" / "gold_traces"
OUTPUT_DIR = Path.home() / "bashgym" / "data" / "cascade_runs"


async def main():
    from bashgym.gym.cascade_scheduler import CascadeConfig, CascadeScheduler

    config = CascadeConfig(
        # Domain order — start with simpler tasks, build up
        domains=[
            "file_operations",
            "bash_commands",
            "search_and_navigate",
            "multi_step_reasoning",
        ],
        # Use our fine-tuned Gemma 4 E4B as the cascade RL starting point
        base_model=FT_MODEL_PATH,
        dataset_path=GOLD_TRACES_DIR,
        output_dir=OUTPUT_DIR,
        # GRPO settings
        grpo_num_generations=4,
        grpo_temperature=0.7,
        train_steps_per_stage=10,  # Smoke test — verify pipeline works end-to-end
        learning_rate=5e-5,  # Lower than SFT (5e-5 vs 2e-4) for RL stability
        # LoRA matching what worked
        lora_r=16,
        lora_alpha=16,
        load_in_4bit=False,  # bf16, not QLoRA (matches our SFT approach)
        # Filtering
        min_domain_examples=10,
        skip_empty_domains=True,
        # REAL mode — actual GRPO training
        mode="real",
    )

    logger.info("=" * 60)
    logger.info("CASCADE RL — REAL MODE")
    logger.info("=" * 60)
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Dataset:    {config.dataset_path}")
    logger.info(f"Output:     {config.output_dir}")
    logger.info(f"Stages:     {len(config.domains)}")
    logger.info(f"Steps/stage: {config.train_steps_per_stage}")
    logger.info("")

    scheduler = CascadeScheduler(config)

    async def callback(event_type: str, stage_or_data):
        if hasattr(stage_or_data, "domain"):
            logger.info(f"[{event_type}] {stage_or_data.domain.name} (stage {stage_or_data.stage_number})")
            if stage_or_data.metrics:
                logger.info(f"  metrics: {stage_or_data.metrics}")
        else:
            logger.info(f"[{event_type}] {stage_or_data}")

    result = await scheduler.run_cascade(callback=callback)

    logger.info("\n" + "=" * 60)
    logger.info("CASCADE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Status: {result.status}")
    logger.info(f"Duration: {result.total_duration_seconds:.0f}s")
    logger.info(f"Stages completed: {sum(1 for s in result.stages if s.status == 'completed')}")
    for stage in result.stages:
        logger.info(f"  {stage.domain.name}: {stage.status} examples={stage.examples_count}")
        if stage.checkpoint_path:
            logger.info(f"    checkpoint: {stage.checkpoint_path}")
    if result.best_checkpoints:
        logger.info("\nBest checkpoints by domain:")
        for dom, ckpt in result.best_checkpoints.items():
            logger.info(f"  {dom}: {ckpt}")


if __name__ == "__main__":
    asyncio.run(main())
