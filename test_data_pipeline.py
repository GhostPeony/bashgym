"""
Test Data Pipeline - End-to-end test of DataFactory + Data Designer

Exercises the full data pipeline using real imported traces:
  Stage 1: Trace processing (no API needed)
  Stage 2: LLM augmentation (needs ANTHROPIC_API_KEY)
  Stage 3: Data Designer synthetic generation (sampler + expression columns only)
  Stage 4: Save & summary
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bashgym.factory.data_factory import DataFactory, DataFactoryConfig, SynthesisStrategy
from bashgym.factory.schema_builder import SchemaBuilder, DataDesignerClient


TRACE_DIR = Path.home() / ".bashgym" / "traces"


def stage1_trace_processing():
    """Stage 1: Load and process traces (no API needed)."""
    print("=" * 60)
    print("STAGE 1: Trace Processing")
    print("=" * 60)

    if not TRACE_DIR.exists():
        print(f"ERROR: Trace directory not found: {TRACE_DIR}")
        return []

    trace_files = list(TRACE_DIR.glob("*.json"))
    print(f"Found {len(trace_files)} trace files in {TRACE_DIR}")

    # Use DIRECT strategy so no augmentation API calls are made
    config = DataFactoryConfig(
        strategy=SynthesisStrategy.DIRECT,
        require_successful_verification=True,
        min_trace_steps=2,
        max_trace_steps=200,
    )
    factory = DataFactory(config)

    examples = []
    validation_passed = 0
    validation_failed = 0

    for trace_path in trace_files:
        example = factory.process_gold_trace(trace_path)
        if example:
            examples.append(example)
            validation_passed += 1
        else:
            validation_failed += 1

    print(f"\nResults:")
    print(f"  Traces loaded:     {len(trace_files)}")
    print(f"  Passed validation: {validation_passed}")
    print(f"  Failed validation: {validation_failed}")

    if examples:
        step_counts = [e.metadata.get("total_steps", 0) for e in examples]
        avg_steps = sum(step_counts) / len(step_counts)
        print(f"  Avg steps/trace:   {avg_steps:.1f}")
        print(f"  Min steps:         {min(step_counts)}")
        print(f"  Max steps:         {max(step_counts)}")

        print(f"\nSample user prompts:")
        for ex in examples[:5]:
            prompt_preview = ex.user_prompt[:80].replace("\n", " ")
            print(f"  - [{ex.metadata.get('total_steps', '?')} steps] {prompt_preview}...")

    return examples


async def stage2_llm_augmentation(examples):
    """Stage 2: LLM augmentation (needs ANTHROPIC_API_KEY)."""
    print("\n" + "=" * 60)
    print("STAGE 2: LLM Augmentation")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("SKIPPED: No ANTHROPIC_API_KEY set in environment")
        return []

    if len(examples) < 2:
        print("SKIPPED: Need at least 2 examples from Stage 1")
        return []

    # Pick 2 examples with different step counts for variety
    sorted_examples = sorted(examples, key=lambda e: e.metadata.get("total_steps", 0))
    picks = [sorted_examples[0], sorted_examples[len(sorted_examples) // 2]]

    config = DataFactoryConfig(
        strategy=SynthesisStrategy.AUGMENTED,
        augmentation_factor=2,
    )
    factory = DataFactory(config)

    all_variations = []
    try:
        for i, example in enumerate(picks):
            print(f"\nAugmenting example {i+1}/{len(picks)}: "
                  f"{example.user_prompt[:60].replace(chr(10), ' ')}...")

            variations = await factory.augment_example(example, num_variations=2)
            all_variations.extend(variations)

            print(f"  Original + {len(variations) - 1} variations generated")
            for j, var in enumerate(variations):
                label = "ORIGINAL" if j == 0 else f"VAR {j}"
                prompt_preview = var.user_prompt[:100].replace("\n", " ")
                print(f"    [{label}] {prompt_preview}")
    finally:
        await factory.close()

    print(f"\nTotal augmented examples: {len(all_variations)}")
    return all_variations


async def stage3_data_designer():
    """Stage 3: Data Designer synthetic generation (sampler + expression columns only)."""
    print("\n" + "=" * 60)
    print("STAGE 3: Data Designer Synthetic Generation")
    print("=" * 60)

    # Build a coding task schema using ONLY sampler + expression columns
    # (no LLM columns - avoids needing an external LLM endpoint)
    schema = (
        SchemaBuilder("coding_tasks_sampler")
        .uuid("task_id", description="Unique task identifier")
        .category("language", [
            "python", "javascript", "typescript", "rust", "go", "bash"
        ], description="Programming language")
        .category("difficulty", [
            "easy", "medium", "hard"
        ], weights=[0.4, 0.4, 0.2], description="Task difficulty")
        .category("task_type", [
            "function_implementation",
            "bug_fix",
            "code_optimization",
            "test_writing",
            "refactoring",
            "cli_tool",
            "api_endpoint",
            "data_processing"
        ], description="Type of coding task")
        .category("context", [
            "greenfield project",
            "legacy codebase",
            "open source contribution",
            "microservice",
            "monorepo module"
        ], description="Project context")
        .bernoulli("has_tests", p=0.7, description="Whether task includes tests")
        .gaussian("estimated_complexity", mean=5.0, std=2.0,
                  description="Estimated complexity 1-10")
        .person("requester", description="Person requesting the task")
        # Expression columns that compose sampler values
        .expression(
            "task_summary",
            expression="{{ difficulty }} {{ language }} {{ task_type }} task in {{ context }}",
            depends_on=["difficulty", "language", "task_type", "context"],
            description="Human-readable task summary"
        )
        .expression(
            "test_note",
            expression="{% if has_tests %}Includes unit tests{% else %}No tests required{% endif %}",
            depends_on=["has_tests"],
            description="Test requirement note"
        )
        .with_rows(10)
        .with_seed(42)
        .build()
    )

    print(f"Schema: {schema.name}")
    print(f"Columns: {len(schema.columns)}")
    for col in schema.columns:
        print(f"  - {col.name} ({col.column_type.value})")

    # Generate records using the local DataDesignerClient
    # (sampler + expression columns don't need an external service)
    client = DataDesignerClient()

    try:
        records = await client.generate(schema)
        print(f"\nGenerated {len(records)} records:")
        print("-" * 40)

        for i, record in enumerate(records):
            print(f"\nRecord {i+1}:")
            print(f"  ID:         {record.get('task_id', 'N/A')[:8]}...")
            print(f"  Language:   {record.get('language')}")
            print(f"  Difficulty: {record.get('difficulty')}")
            print(f"  Task Type:  {record.get('task_type')}")
            print(f"  Context:    {record.get('context')}")
            print(f"  Has Tests:  {record.get('has_tests')}")
            print(f"  Complexity: {record.get('estimated_complexity', 0):.1f}")
            print(f"  Summary:    {record.get('task_summary')}")
            print(f"  Test Note:  {record.get('test_note')}")

        return records
    finally:
        await client.close()


def stage4_save_and_summary(examples, augmented, synthetic_records):
    """Stage 4: Save and print summary."""
    print("\n" + "=" * 60)
    print("STAGE 4: Save & Summary")
    print("=" * 60)

    # Combine all SFT examples
    all_examples = list(examples)
    # Add augmented (skip duplicates of originals already in examples)
    augmented_only = [e for e in augmented if e.metadata.get("augmented")]
    all_examples.extend(augmented_only)

    if not all_examples:
        print("No examples to save.")
        return

    # Save to JSONL
    config = DataFactoryConfig(output_dir="data/training_batches")
    factory = DataFactory(config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = factory.save_training_batch(all_examples, f"test_sft_{timestamp}")

    # Print summary
    print(f"\nPipeline Summary:")
    print(f"  Stage 1 examples (from traces):  {len(examples)}")
    print(f"  Stage 2 augmented variations:    {len(augmented_only)}")
    print(f"  Stage 3 synthetic records:       {len(synthetic_records)}")
    print(f"  Total SFT examples saved:        {len(all_examples)}")
    print(f"  Output file:                     {output_path}")

    # Print step count stats
    step_counts = [e.metadata.get("total_steps", e.metadata.get("step_count", 0))
                   for e in all_examples]
    if step_counts:
        print(f"  Avg steps per example:           {sum(step_counts) / len(step_counts):.1f}")

    # Print a sample NeMo-format record
    if all_examples:
        sample = all_examples[0].to_dict()
        print(f"\nSample NeMo format record:")
        sample_str = json.dumps(sample, indent=2)
        # Truncate long content for display
        lines = sample_str.split("\n")
        for line in lines[:20]:
            print(f"  {line}")
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")


async def main():
    print("=" * 60)
    print("  Test Data Pipeline")
    print("  DataFactory + Data Designer End-to-End Test")
    print("=" * 60)
    print()

    # Stage 1: Trace processing
    examples = stage1_trace_processing()

    # Stage 2: LLM augmentation
    augmented = await stage2_llm_augmentation(examples)

    # Stage 3: Data Designer
    synthetic_records = await stage3_data_designer()

    # Stage 4: Save & summary
    stage4_save_and_summary(examples, augmented, synthetic_records)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
