#!/usr/bin/env python3
"""
Generate DPO training pairs using NVIDIA NeMo Data Designer (direct API).

Pipeline:
  - Load gold trace prompts as seeds
  - For each seed: generate solution_a (temp 0.9) and solution_b (temp 0.5)
  - LLM judge scores both
  - Auto-assign chosen/rejected based on judge scores
  - Filter pairs where scores differ
  - Output DPO-format JSONL

Uses Data Designer 0.5.5 client API directly (no bashgym wrapper).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_nvidia_key() -> str:
    """Load NVIDIA API key from desktop .env."""
    env_file = Path.home() / "desktop-home" / "Projects" / "ghostwork" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("NVIDIA_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("NVIDIA_API_KEY", "")


def _load_anthropic_key() -> str:
    """Load Anthropic API key from env or desktop .env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for env_file in [
        Path.home() / "desktop-home" / "Projects" / "ghostwork" / ".env",
        Path.home() / ".bashgym" / ".env",
    ]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def _extract_user_prompt(trace: dict) -> str:
    """Pull the user's initial intent from a raw bashgym trace."""
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict):
        prompt = metadata.get("user_initial_prompt", "")
        if prompt and isinstance(prompt, str) and len(prompt.strip()) > 30:
            return prompt.strip()
    return ""


def load_seed_prompts(traces_dir: Path, max_seeds: int) -> list[dict]:
    """Load gold trace prompts as Data Designer seeds."""
    seeds = []
    for f in sorted(traces_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(errors="replace"))
            prompt = _extract_user_prompt(data)
            if prompt and len(prompt) < 4000:
                seeds.append({"task_prompt": prompt})
                if len(seeds) >= max_seeds:
                    break
        except (json.JSONDecodeError, OSError):
            continue
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-records", type=int, default=10, help="Number of DPO pairs to generate (small to start)")
    parser.add_argument(
        "--gold-traces",
        type=Path,
        default=Path.home() / "desktop-home" / ".bashgym" / "gold_traces",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "bashgym" / "data" / "dpo_synthetic",
    )
    parser.add_argument("--code-model-a", default="qwen/qwen3-next-80b-a3b-instruct",
                        help="Model for solution A (the strong one)")
    parser.add_argument("--code-model-b", default="claude-haiku-4-5-20251001",
                        help="Model for solution B (intentionally weaker for clear pairs)")
    parser.add_argument("--code-provider-b", default="anthropic",
                        choices=["nvidia", "anthropic"],
                        help="Provider for code-model-b")
    parser.add_argument("--judge-model", default="mistralai/mistral-large-3-675b-instruct-2512")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    # Load NVIDIA API key (always required for code-model-a + judge)
    api_key = _load_nvidia_key()
    if not api_key:
        logger.error("NVIDIA_API_KEY not found")
        sys.exit(1)
    os.environ["NVIDIA_API_KEY"] = api_key
    logger.info(f"NVIDIA_API_KEY loaded ({api_key[:12]}...)")

    # Load Anthropic key if we're using it
    if args.code_provider_b == "anthropic":
        anth_key = _load_anthropic_key()
        if not anth_key:
            logger.error("ANTHROPIC_API_KEY required for --code-provider-b=anthropic")
            sys.exit(1)
        os.environ["ANTHROPIC_API_KEY"] = anth_key
        logger.info(f"ANTHROPIC_API_KEY loaded ({anth_key[:14]}...)")

    # Load gold trace seeds
    seeds = load_seed_prompts(args.gold_traces, max_seeds=args.num_records * 2)
    logger.info(f"Loaded {len(seeds)} seed prompts from gold traces")
    if not seeds:
        logger.error("No usable seeds found")
        sys.exit(1)

    import pandas as pd
    seed_df = pd.DataFrame(seeds[:args.num_records])

    # Build pipeline
    import data_designer.config as dd
    from data_designer.interface import DataDesigner

    # Build the DataDesigner with custom providers if needed
    custom_providers = None
    if args.code_provider_b == "anthropic":
        from data_designer.config.models import ModelProvider
        # Anthropic OpenAI-compatible endpoint:
        # https://docs.anthropic.com/en/api/openai-sdk
        custom_providers = [
            ModelProvider(
                name="nvidia",
                endpoint="https://integrate.api.nvidia.com/v1",
                provider_type="openai",
                api_key=os.environ["NVIDIA_API_KEY"],
            ),
            ModelProvider(
                name="anthropic",
                endpoint="https://api.anthropic.com/v1",
                provider_type="openai",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            ),
        ]

    designer = DataDesigner(model_providers=custom_providers) if custom_providers else DataDesigner()
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="solution-a",
                provider="nvidia",
                model=args.code_model_a,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.3,
                    top_p=0.95,
                    max_tokens=2048,
                ),
            ),
            dd.ModelConfig(
                alias="solution-b",
                provider=args.code_provider_b,
                model=args.code_model_b,
                # Anthropic API rejects sending both temperature AND top_p — pick one.
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.3,
                    max_tokens=2048,
                ),
            ),
            dd.ModelConfig(
                alias="judge",
                provider="nvidia",
                model=args.judge_model,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.1,
                    max_tokens=512,
                ),
            ),
        ]
    )

    # Use the seed dataframe as the input
    builder.with_seed_dataset(
        dd.DataFrameSeedSource(
            df=seed_df,
            sampling_strategy=dd.SamplingStrategy.SHUFFLE,
        )
    )

    # Column 1: solution A (creative)
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution_a",
            model_alias="solution-a",
            prompt=(
                "You are an expert software engineer. Solve this task with a "
                "concrete code-based response.\n\nTask: {{ task_prompt }}\n\n"
                "Provide your solution as a brief explanation followed by code."
            ),
        )
    )

    # Column 2: solution B (conservative)
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="solution_b",
            model_alias="solution-b",
            prompt=(
                "You are an expert software engineer. Solve this task with a "
                "concrete code-based response.\n\nTask: {{ task_prompt }}\n\n"
                "Provide your solution as a brief explanation followed by code."
            ),
        )
    )

    # Column 3: pairwise judge — show BOTH solutions, force a preference
    # This dramatically reduces ties vs scoring each independently.
    preference_score = dd.Score(
        name="preferred",
        description="Which solution is better overall (correctness, completeness, code quality)",
        options={
            1: "Solution A is clearly better",
            2: "Solution A is slightly better",
            3: "Both solutions are equivalent",
            4: "Solution B is slightly better",
            5: "Solution B is clearly better",
        },
    )

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="pairwise_judgment",
            model_alias="judge",
            prompt=(
                "You are comparing two solutions to the same coding task. "
                "Pick which solution is better. Be critical — even if both look "
                "reasonable, identify which one is genuinely stronger.\n\n"
                "TASK:\n{{ task_prompt }}\n\n"
                "SOLUTION A:\n{{ solution_a }}\n\n"
                "SOLUTION B:\n{{ solution_b }}\n\n"
                "Consider: correctness, completeness, code quality, clarity, "
                "and how well it actually addresses the task."
            ),
            scores=[preference_score],
        )
    )

    logger.info("=" * 60)
    logger.info("DATA DESIGNER DPO GENERATION")
    logger.info("=" * 60)
    logger.info(f"  Code model A (chosen candidate):   {args.code_model_a}")
    logger.info(f"  Code model B (rejected candidate): {args.code_model_b}")
    logger.info(f"  Judge model:                       {args.judge_model}")
    logger.info(f"  Records:                           {args.num_records}")
    logger.info(f"  Output:                            {args.output_dir}")
    logger.info("")

    if args.preview:
        logger.info("Running preview (5 records)...")
        result = designer.preview(config_builder=builder)
        logger.info(f"\nPreview type: {type(result).__name__}")
        if hasattr(result, "display_sample_record"):
            result.display_sample_record()
        return

    logger.info("Generating dataset...")
    result = designer.create(config_builder=builder, num_records=args.num_records)
    logger.info(f"\nResult type: {type(result).__name__}")

    # DatasetCreationResults exposes load_dataset() that returns a HuggingFace
    # Dataset object. Convert that to a pandas DataFrame.
    if hasattr(result, "load_dataset"):
        hf_dataset = result.load_dataset()
        df = hf_dataset.to_pandas() if hasattr(hf_dataset, "to_pandas") else pd.DataFrame(hf_dataset)
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    elif hasattr(result, "dataset"):
        df = result.dataset
    else:
        df = result

    logger.info(f"Generated {len(df)} records")
    logger.info(f"Columns: {list(df.columns)}")

    # Save raw output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_designer_output.parquet"
    df.to_parquet(raw_path)
    logger.info(f"Raw saved: {raw_path}")

    # Convert to DPO format using judge_a / judge_b score columns
    def _extract_score(val):
        """Pull numeric quality score from judge column output."""
        if isinstance(val, dict):
            return int(val.get("quality", 0)) if val.get("quality") else 0
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            try:
                obj = json.loads(val)
                return int(obj.get("quality", 0)) if isinstance(obj, dict) else 0
            except (json.JSONDecodeError, ValueError):
                return 0
        return 0

    dpo_examples = []
    for _, row in df.iterrows():
        score_a = _extract_score(row.get("judge_a"))
        score_b = _extract_score(row.get("judge_b"))

        if score_a == score_b:
            continue  # Skip ties

        if score_a > score_b:
            chosen, rejected = row["solution_a"], row["solution_b"]
        else:
            chosen, rejected = row["solution_b"], row["solution_a"]

        dpo_examples.append({
            "prompt": row["task_prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "score_chosen": max(score_a, score_b),
                "score_rejected": min(score_a, score_b),
            },
        })

    logger.info(f"\nFiltered to {len(dpo_examples)} DPO pairs (skipped ties)")

    # Save train/val
    import random
    random.seed(42)
    random.shuffle(dpo_examples)
    split = max(1, int(len(dpo_examples) * 0.9))
    train = dpo_examples[:split]
    val = dpo_examples[split:]

    with open(args.output_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.output_dir / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Train: {len(train)} → {args.output_dir / 'train.jsonl'}")
    logger.info(f"Val:   {len(val)} → {args.output_dir / 'val.jsonl'}")

    # Validate against our DPO contract
    from bashgym.datasets.validator import validate_dataset, print_validation_report
    result = validate_dataset(args.output_dir / "train.jsonl", format="dpo", quiet=True)
    print_validation_report(result, max_issues=5)


if __name__ == "__main__":
    main()
