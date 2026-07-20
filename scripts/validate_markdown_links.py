"""Validate local Markdown links without making network requests."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
HTML_TAG = re.compile(r"<[^>]+>")
NON_SLUG = re.compile(r"[^\w\s-]")
WHITESPACE = re.compile(r"\s+")


def _slugify(heading: str) -> str:
    text = HTML_TAG.sub("", heading).replace("`", "").lower()
    text = NON_SLUG.sub("", text)
    return WHITESPACE.sub("-", text.strip())


def _anchors(path: Path) -> set[str]:
    counts: dict[str, int] = {}
    anchors: set[str] = set()
    for heading in HEADING.findall(path.read_text(encoding="utf-8")):
        base = _slugify(heading)
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _local_destination(destination: str) -> tuple[str, str] | None:
    destination = destination.strip()
    if destination.startswith("<") and destination.endswith(">"):
        destination = destination[1:-1]
    elif " " in destination:
        destination = destination.split(" ", 1)[0]

    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or destination.startswith("//"):
        return None
    return unquote(parsed.path), unquote(parsed.fragment)


def _is_within_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True


def validate_markdown_links(paths: list[Path]) -> list[str]:
    """Return one human-readable issue for every invalid local Markdown link."""
    issues: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}

    for source in paths:
        text = source.read_text(encoding="utf-8")
        for match in INLINE_LINK.finditer(text):
            local = _local_destination(match.group(1))
            if local is None:
                continue
            raw_target, anchor = local
            target = source.resolve() if not raw_target else (source.parent / raw_target).resolve()
            line = text.count("\n", 0, match.start()) + 1

            if not _is_within_repository(target):
                issues.append(f"{source}:{line}: target escapes repository root: {raw_target}")
                continue
            if not target.exists():
                issues.append(f"{source}:{line}: missing local target: {raw_target}")
                continue
            if anchor and target.is_file() and target.suffix.lower() == ".md":
                anchors = anchor_cache.setdefault(target, _anchors(target))
                if anchor not in anchors:
                    issues.append(f"{source}:{line}: missing anchor: {raw_target}#{anchor}")
    return issues


def _tracked_markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / path for path in result.stdout.decode().split("\0") if path]


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    paths = [Path(path).resolve() for path in arguments] if arguments else _tracked_markdown_files()
    issues = validate_markdown_links(paths)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
