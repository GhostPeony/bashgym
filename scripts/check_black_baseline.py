"""Enforce Black while ratcheting pre-existing formatting debt downward."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / ".github" / "black-baseline.txt"


def _load_baseline() -> dict[str, str]:
    baseline: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        BASELINE_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"invalid Black baseline entry on line {line_number}")
        baseline[relative] = digest
    return baseline


def main() -> int:
    baseline = _load_baseline()
    failures: list[str] = []
    for relative, expected_digest in baseline.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"baseline file is missing: {relative}")
            continue
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            failures.append(
                f"baseline file changed: {relative}; format it and remove the entry, "
                "or explicitly review and replace its hash"
            )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    python_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "bashgym").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    candidates = [relative for relative in python_files if relative not in baseline]
    return subprocess.run(
        [sys.executable, "-m", "black", "--check", *candidates],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
