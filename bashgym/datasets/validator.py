"""
Dataset validators — verify a dataset matches its declared format contract
before training starts.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bashgym.datasets.contracts import (
    CONTRACTS,
    DatasetFormat,
    FormatContract,
)


@dataclass
class ValidationIssue:
    """A single validation problem."""

    line: int
    severity: str  # "error" | "warning"
    field: str
    message: str
    sample: str | None = None  # First N chars of the offending value


@dataclass
class ValidationResult:
    """Outcome of validating a dataset file."""

    path: Path
    format: DatasetFormat
    total_examples: int
    valid_examples: int
    issues: list[ValidationIssue] = field(default_factory=list)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.total_examples > 0 and self.valid_examples == self.total_examples

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "format": self.format.value,
            "total_examples": self.total_examples,
            "valid_examples": self.valid_examples,
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "line": i.line,
                    "severity": i.severity,
                    "field": i.field,
                    "message": i.message,
                    "sample": i.sample,
                }
                for i in self.issues[:50]  # cap for readability
            ],
            "summary_stats": self.summary_stats,
        }


def _check_field(example: dict, spec, line: int, issues: list[ValidationIssue]) -> bool:
    """Check a single field against its spec. Returns True if valid."""
    if spec.name not in example:
        if spec.required:
            issues.append(
                ValidationIssue(
                    line=line,
                    severity="error",
                    field=spec.name,
                    message=f"Required field '{spec.name}' is missing",
                )
            )
            return False
        return True

    value = example[spec.name]

    # Type check
    if not isinstance(value, spec.type):
        type_name = (
            spec.type.__name__
            if isinstance(spec.type, type)
            else str([t.__name__ for t in spec.type])
        )
        issues.append(
            ValidationIssue(
                line=line,
                severity="error",
                field=spec.name,
                message=f"Field '{spec.name}' has wrong type — got {type(value).__name__}, expected {type_name}",
                sample=str(value)[:100],
            )
        )
        return False

    # Custom validator
    if spec.validator is not None:
        if not spec.validator(value):
            issues.append(
                ValidationIssue(
                    line=line,
                    severity="error",
                    field=spec.name,
                    message=f"Field '{spec.name}' failed format check",
                    sample=json.dumps(value)[:200] if not isinstance(value, str) else value[:200],
                )
            )
            return False

    return True


def validate_example(
    example: dict, contract: FormatContract, line: int = 0
) -> tuple[bool, list[ValidationIssue]]:
    """Validate a single example against a contract."""
    issues: list[ValidationIssue] = []
    valid = True

    # Check unknown top-level keys (warning only)
    known = {f.name for f in contract.fields}
    for key in example.keys():
        if key not in known:
            issues.append(
                ValidationIssue(
                    line=line,
                    severity="warning",
                    field=key,
                    message=f"Unknown field '{key}' (not in {contract.format.value} contract)",
                )
            )

    # Check each contract field
    for spec in contract.fields:
        if not _check_field(example, spec, line, issues):
            valid = False

    return valid, issues


def validate_dataset(
    path: Path | str,
    format: DatasetFormat | str,
    sample_only: int | None = None,
    quiet: bool = False,
) -> ValidationResult:
    """Validate every example in a JSONL dataset against a format contract.

    Args:
        path: Path to .jsonl file
        format: Format to validate against (string or DatasetFormat)
        sample_only: If set, only validate first N examples (faster for huge files)
        quiet: If False, prints progress dots

    Returns:
        ValidationResult with all issues found
    """
    path = Path(path)
    if isinstance(format, str):
        format = DatasetFormat(format.lower())
    contract = CONTRACTS[format]

    result = ValidationResult(path=path, format=format, total_examples=0, valid_examples=0)

    if not path.exists():
        result.issues.append(
            ValidationIssue(
                line=0,
                severity="error",
                field="<file>",
                message=f"File does not exist: {path}",
            )
        )
        return result

    # Per-format stats
    field_role_counts: dict[str, int] = {}
    final_role_counts: dict[str, int] = {}
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []

    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if sample_only and result.total_examples >= sample_only:
                break

            result.total_examples += 1

            try:
                example = json.loads(raw_line)
            except json.JSONDecodeError as e:
                result.issues.append(
                    ValidationIssue(
                        line=line_num,
                        severity="error",
                        field="<json>",
                        message=f"Invalid JSON: {e}",
                        sample=raw_line[:200],
                    )
                )
                continue

            if not isinstance(example, dict):
                result.issues.append(
                    ValidationIssue(
                        line=line_num,
                        severity="error",
                        field="<root>",
                        message=f"Top-level value must be an object, got {type(example).__name__}",
                    )
                )
                continue

            valid, issues = validate_example(example, contract, line=line_num)
            result.issues.extend(issues)
            if valid:
                result.valid_examples += 1

            # Format-specific stats
            if format == DatasetFormat.SFT:
                msgs = example.get("messages", [])
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict):
                            r = m.get("role", "?")
                            field_role_counts[r] = field_role_counts.get(r, 0) + 1
                    if msgs:
                        last = msgs[-1].get("role", "?") if isinstance(msgs[-1], dict) else "?"
                        final_role_counts[last] = final_role_counts.get(last, 0) + 1

            elif format == DatasetFormat.GRPO:
                p = example.get("prompt")
                if isinstance(p, str):
                    prompt_lengths.append(len(p))
                elif isinstance(p, list):
                    total = sum(len(m.get("content", "") or "") for m in p if isinstance(m, dict))
                    prompt_lengths.append(total)

            elif format == DatasetFormat.DPO:
                for key in ("chosen", "rejected"):
                    v = example.get(key)
                    if isinstance(v, str):
                        completion_lengths.append(len(v))

            if not quiet and result.total_examples % 500 == 0:
                print(f"  ... validated {result.total_examples} examples")

    # Build summary stats
    if format == DatasetFormat.SFT:
        result.summary_stats = {
            "role_counts": field_role_counts,
            "final_role_counts": final_role_counts,
        }
    elif format == DatasetFormat.GRPO and prompt_lengths:
        prompt_lengths.sort()
        n = len(prompt_lengths)
        result.summary_stats = {
            "prompt_length_chars": {
                "min": prompt_lengths[0],
                "max": prompt_lengths[-1],
                "median": prompt_lengths[n // 2],
                "mean": round(sum(prompt_lengths) / n, 1),
            },
        }
    elif format == DatasetFormat.DPO and completion_lengths:
        completion_lengths.sort()
        n = len(completion_lengths)
        result.summary_stats = {
            "completion_length_chars": {
                "min": completion_lengths[0],
                "max": completion_lengths[-1],
                "median": completion_lengths[n // 2],
            },
        }

    return result


def print_validation_report(result: ValidationResult, max_issues: int = 10) -> None:
    """Pretty-print a validation result."""
    print("=" * 70)
    print("Dataset Validation Report")
    print("=" * 70)
    print(f"  Path:    {result.path}")
    print(f"  Format:  {result.format.value}")
    print(f"  Total:   {result.total_examples}")
    print(f"  Valid:   {result.valid_examples}")
    print(f"  Errors:  {result.error_count}")
    print(f"  Warnings: {result.warning_count}")
    print(f"  Status:  {'✓ VALID' if result.is_valid else '✗ INVALID'}")

    if result.summary_stats:
        print()
        print("  Summary stats:")
        for k, v in result.summary_stats.items():
            print(f"    {k}: {v}")

    if result.issues:
        print()
        print(f"  First {min(max_issues, len(result.issues))} issues:")
        for issue in result.issues[:max_issues]:
            marker = "[ERROR]" if issue.severity == "error" else "[WARN] "
            print(f"    {marker} line {issue.line}: {issue.field} — {issue.message}")
            if issue.sample:
                print(f"             sample: {issue.sample[:120]}")
    print("=" * 70)
