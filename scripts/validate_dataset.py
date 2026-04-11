#!/usr/bin/env python3
"""
CLI tool to validate any bashgym training dataset against its format contract.

Usage:
  python scripts/validate_dataset.py path/to/dataset.jsonl --type sft
  python scripts/validate_dataset.py path/to/dataset.jsonl --type grpo
  python scripts/validate_dataset.py path/to/dataset.jsonl --type dpo --sample 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bashgym.datasets.contracts import DatasetFormat
from bashgym.datasets.validator import (
    print_validation_report,
    validate_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Validate a bashgym training dataset")
    parser.add_argument("path", type=Path, help="Path to .jsonl file")
    parser.add_argument(
        "--type",
        "--format",
        dest="format",
        required=True,
        choices=[f.value for f in DatasetFormat],
        help="Dataset format to validate against",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only validate first N examples (faster for huge files)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help="Max issues to print in the report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON instead of formatted text",
    )
    args = parser.parse_args()

    result = validate_dataset(
        path=args.path,
        format=args.format,
        sample_only=args.sample,
        quiet=args.json,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print_validation_report(result, max_issues=args.max_issues)

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
