#!/usr/bin/env python3
"""
Full pipeline rebuild using bashgym's existing pipeline.

1. Import ALL sessions via ClaudeSessionImporter
2. Process via TraceProcessor
3. Generate examples via ExampleGenerator
4. Export via export_unsloth
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DESKTOP_CLAUDE = Path.home() / "desktop-home" / ".claude"
DGX_CLAUDE = Path.home() / ".claude"
OUTPUT_DIR = Path.home() / "bashgym-training" / "data-pipeline"


def main():
    start = time.time()
    logger.info("=" * 60)
    logger.info("FULL PIPELINE REBUILD — Using bashgym pipeline")
    logger.info("=" * 60)

    # ---- Step 1: Import all sessions ----
    logger.info("\n--- Step 1: Importing ALL sessions ---")
    from bashgym.trace_capture.importers.claude_history import ClaudeSessionImporter

    importer = ClaudeSessionImporter()

    # Collect all session files from both sources
    all_sessions = []

    # Desktop sessions
    desktop_projects = DESKTOP_CLAUDE / "projects"
    if desktop_projects.exists():
        for project_dir in desktop_projects.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                all_sessions.append(jsonl_file)
        logger.info(f"  Desktop: {len(all_sessions)} session files")

    # DGX sessions
    dgx_count = 0
    dgx_projects = DGX_CLAUDE / "projects"
    if dgx_projects.exists():
        for project_dir in dgx_projects.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                all_sessions.append(jsonl_file)
                dgx_count += 1
    logger.info(f"  DGX: {dgx_count} session files")
    logger.info(f"  Total: {len(all_sessions)} session files")

    # Import each session
    imported = 0
    skipped = 0
    failed = 0
    for i, session_file in enumerate(all_sessions):
        try:
            result = importer.import_session(session_file, force=True)
            if result.steps_imported > 0:
                imported += 1
            elif result.skipped:
                skipped += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning(f"  Failed: {session_file.name}: {e}")

        if (i + 1) % 500 == 0:
            logger.info(
                f"  [{i+1}/{len(all_sessions)}] imported={imported} skipped={skipped} failed={failed}"
            )

    logger.info(f"  Import done: {imported} imported, {skipped} skipped, {failed} failed")

    # ---- Step 2: Process traces ----
    logger.info("\n--- Step 2: Processing traces ---")
    from bashgym.factory.trace_processor import TraceProcessor

    traces_dir = importer.trace_capture.traces_dir
    processor = TraceProcessor()
    processed = processor.process_directory(traces_dir)
    logger.info(f"  Processed {len(processed)} traces from {traces_dir}")

    # Also process gold traces if they exist
    gold_dir = Path.home() / "bashgym" / "data" / "gold_traces"
    if gold_dir.exists():
        gold_processed = processor.process_directory(gold_dir)
        logger.info(f"  Processed {len(gold_processed)} gold traces from {gold_dir}")
        processed.extend(gold_processed)

    logger.info(f"  Total processed: {len(processed)}")

    # ---- Step 3: Generate examples ----
    logger.info("\n--- Step 3: Generating training examples ---")
    from bashgym.factory.example_generator import ExampleGenerator

    generator = ExampleGenerator()

    # Process all trace directories
    all_examples = []

    # Process from traces dir
    examples, stats = generator.process_directory(traces_dir)
    logger.info(f"  From traces: {len(examples)} examples ({stats})")
    all_examples.extend(examples)

    # Process from gold traces
    if gold_dir.exists():
        examples, stats = generator.process_directory(gold_dir)
        logger.info(f"  From gold: {len(examples)} examples ({stats})")
        all_examples.extend(examples)

    logger.info(f"  Total examples: {len(all_examples)}")

    # ---- Step 4: Export ----
    logger.info("\n--- Step 4: Exporting ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_result = generator.export_for_nemo(
        all_examples,
        OUTPUT_DIR,
        train_split=0.9,
    )
    logger.info(f"  Exported to: {export_result}")

    # ---- Step 5: Export Unsloth formats ----
    logger.info("\n--- Step 5: Exporting Unsloth formats ---")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from export_unsloth import convert_file

        unsloth_dir = OUTPUT_DIR / "unsloth"
        unsloth_dir.mkdir(exist_ok=True)

        train_path = OUTPUT_DIR / "train.jsonl"
        val_path = OUTPUT_DIR / "val.jsonl"

        if train_path.exists():
            convert_file(train_path, unsloth_dir, ["chatml", "chatml_flat", "sharegpt"])
        if val_path.exists():
            convert_file(val_path, unsloth_dir, ["chatml", "chatml_flat", "sharegpt"])
    except Exception as e:
        logger.warning(f"  Unsloth export: {e}")

    duration = time.time() - start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"DONE in {duration:.0f}s")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
