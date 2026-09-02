"""Value-content scan for credential-shaped strings and unresolved placeholders."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

_CREDENTIAL = re.compile(
    r"(?:"
    r"gh[pousr]_[A-Za-z0-9]{30,}"
    r"|github_pat_[A-Za-z0-9_]{30,}"
    r"|hf_[A-Za-z0-9]{30,}"
    r"|sk-(?:(?:proj|live|test)-)?[A-Za-z0-9_-]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bBearer [A-Za-z0-9_.~+/=-]{20,}"
    r")"
)
_PLACEHOLDER = re.compile(
    r"(?:REPLACE_ME|<ASK_USER\b|TODO_FILL_IN|\bCHANGEME\b|YOUR_[A-Z0-9_]+_HERE)"
)
MAX_SCAN_DEPTH = 32
MAX_SCAN_NODES = 10_000

FindingKind = Literal["credential", "placeholder", "unscannable"]


@dataclass(frozen=True)
class SecretScanFinding:
    """Where a problem was found and what kind it is; never the matched text."""

    path: str
    kind: FindingKind


def scan_values(value: Any) -> tuple[SecretScanFinding, ...]:
    """Walk mappings, sequences, and strings; report credential shapes and placeholders."""

    findings: list[SecretScanFinding] = []
    remaining = MAX_SCAN_NODES
    exhausted = False

    def walk(item: Any, path: str, depth: int) -> None:
        nonlocal remaining, exhausted
        if exhausted:
            return
        if depth > MAX_SCAN_DEPTH or remaining <= 0:
            findings.append(SecretScanFinding(path=path or "$", kind="unscannable"))
            exhausted = True
            return
        remaining -= 1
        if isinstance(item, str):
            if _CREDENTIAL.search(item):
                findings.append(SecretScanFinding(path=path, kind="credential"))
            elif _PLACEHOLDER.search(item):
                findings.append(SecretScanFinding(path=path, kind="placeholder"))
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                child_path = f"{path}.{key}" if path else str(key)
                walk(item[key], child_path, depth + 1)
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]", depth + 1)

    walk(value, "", 0)
    return tuple(findings)


__all__ = ["MAX_SCAN_DEPTH", "MAX_SCAN_NODES", "SecretScanFinding", "scan_values"]
