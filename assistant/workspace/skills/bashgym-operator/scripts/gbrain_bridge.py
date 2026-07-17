#!/usr/bin/env python3
"""Publish bounded BashGym handoffs to an authoritative remote GBrain over SSH."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_SCHEMA = "bashgym.gbrain.bridge-profile.v1"
RESULT_SCHEMA = "bashgym.gbrain.bridge-result.v1"
SAFE_TARGET = re.compile(r"^[A-Za-z0-9._@:-]{1,160}$")
SAFE_SOURCE = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
MAX_DOCUMENT_BYTES = 256 * 1024

REMOTE_RECEIVER = r"""
import os, pathlib, sys, uuid
root = pathlib.Path(os.path.expanduser(sys.argv[1])).resolve()
relative = pathlib.PurePosixPath(sys.argv[2])
destination = (root / pathlib.Path(*relative.parts)).resolve()
if root != destination and root not in destination.parents:
    raise SystemExit('destination escaped activity root')
payload = sys.stdin.buffer.read()
if len(payload) > 262144:
    raise SystemExit('document is too large')
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
temporary.write_bytes(payload)
os.replace(temporary, destination)
print(destination.relative_to(root).as_posix())
""".strip()

REMOTE_GBRAIN = r"""
import json, os, pathlib, subprocess, sys
binary = pathlib.Path(os.path.expanduser(sys.argv[1]))
source = sys.argv[2]
mode = sys.argv[3]
if mode == 'status':
    result = subprocess.run([str(binary), 'sources', 'list', '--json'], capture_output=True, text=True)
else:
    # Curated source repos may intentionally hold reviewed-but-uncommitted receipts while
    # Git publication is paused. Incremental sync follows the Git head and can miss those
    # files, so this bounded activity source must be rescanned after an atomic publish.
    result = subprocess.run(
        [str(binary), 'sync', '--source', source, '--full', '--yes', '--json'],
        capture_output=True,
        text=True,
    )
print(json.dumps({'returncode': result.returncode, 'stdout': result.stdout[-12000:], 'stderr': result.stderr[-4000:]}))
raise SystemExit(result.returncode)
""".strip()

REMOTE_VERIFY = r"""
import os, pathlib, subprocess, sys
binary = pathlib.Path(os.path.expanduser(sys.argv[1]))
source = sys.argv[2]
query = sys.argv[3]
relative = sys.argv[4]
result = subprocess.run(
    [str(binary), 'search', query, '--source', source, '--limit', '10', '--json'],
    capture_output=True,
    text=True,
)
output = result.stdout + result.stderr
if result.returncode != 0 or relative not in output:
    print(output[-12000:])
    raise SystemExit(3)
print(relative)
""".strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile must be a JSON object")
    return value


def _profile(path: Path) -> dict[str, str]:
    value = _load_json(path)
    if value.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"profile schema_version must be {PROFILE_SCHEMA}")
    required = ("ssh_target", "activity_root", "gbrain_bin", "source_id")
    missing = [key for key in required if not str(value.get(key) or "").strip()]
    if missing:
        raise ValueError(f"profile is missing: {', '.join(missing)}")
    target = str(value["ssh_target"])
    source = str(value["source_id"])
    if not SAFE_TARGET.fullmatch(target):
        raise ValueError("ssh_target contains unsupported characters")
    if not SAFE_SOURCE.fullmatch(source):
        raise ValueError("source_id contains unsupported characters")
    for key in ("activity_root", "gbrain_bin"):
        if any(char in str(value[key]) for char in ("\x00", "\r", "\n")):
            raise ValueError(f"{key} contains unsupported characters")
    return {key: str(value[key]) for key in required}


def _relative_document(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative path must stay inside the activity source")
    if path.suffix.casefold() != ".md":
        raise ValueError("only rendered Markdown receipts can be published")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", part) for part in path.parts):
        raise ValueError("relative path contains unsupported characters")
    return path.as_posix()


def _document(path: Path) -> bytes:
    payload = path.read_bytes()
    if len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError("document exceeds the 256 KiB bridge limit")
    text = payload.decode("utf-8")
    if 'schema_version: "bashgym.activity.v1"' not in text or 'privacy: "full-tier-only"' not in text:
        raise ValueError("document is not a rendered BashGym activity receipt")
    return payload


def _remote_python(target: str, script: str, *arguments: str, input_bytes: bytes | None = None):
    command = "python3 -c " + shlex.quote(script)
    if arguments:
        command += " " + " ".join(shlex.quote(item) for item in arguments)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, command],
        input=input_bytes,
        capture_output=True,
        timeout=45,
        check=False,
    )


def _status(profile: dict[str, str]) -> dict[str, Any]:
    result = _remote_python(
        profile["ssh_target"],
        REMOTE_GBRAIN,
        profile["gbrain_bin"],
        profile["source_id"],
        "status",
    )
    output = result.stdout.decode("utf-8", errors="replace").strip()
    detail = json.loads(output) if output else {}
    return {
        "schema_version": RESULT_SCHEMA,
        "connected": result.returncode == 0,
        "source_id": profile["source_id"],
        "operation": "status",
        "remote": detail,
    }


def _publish(
    profile: dict[str, str],
    *,
    document: Path,
    relative: str,
    execute: bool,
    sync: bool,
) -> dict[str, Any]:
    payload = _document(document)
    relative = _relative_document(relative)
    base = {
        "schema_version": RESULT_SCHEMA,
        "operation": "publish",
        "source_id": profile["source_id"],
        "relative_path": relative,
        "size_bytes": len(payload),
        "executed": execute,
        "synced": False,
        "indexed": False,
    }
    if not execute:
        return base
    receive = _remote_python(
        profile["ssh_target"],
        REMOTE_RECEIVER,
        profile["activity_root"],
        relative,
        input_bytes=payload,
    )
    if receive.returncode != 0:
        error = receive.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"remote atomic publish failed: {error}")
    if sync:
        synced = _remote_python(
            profile["ssh_target"],
            REMOTE_GBRAIN,
            profile["gbrain_bin"],
            profile["source_id"],
            "sync",
        )
        if synced.returncode != 0:
            error = synced.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"GBrain source sync failed after publish: {error}")
        base["synced"] = True
        indexed_reference = PurePosixPath(relative).with_suffix("").as_posix()
        verified = _remote_python(
            profile["ssh_target"],
            REMOTE_VERIFY,
            profile["gbrain_bin"],
            profile["source_id"],
            PurePosixPath(relative).stem,
            indexed_reference,
        )
        if verified.returncode != 0:
            raise RuntimeError(
                "GBrain sync completed but the published receipt is not searchable by its stable ID"
            )
        base["indexed"] = True
        base["indexed_reference"] = indexed_reference
    return base


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Read the configured remote GBrain source status.")
    publish = subparsers.add_parser("publish", help="Validate or atomically publish one receipt.")
    publish.add_argument("--file", required=True, type=Path)
    publish.add_argument("--relative", required=True)
    publish.add_argument("--execute", action="store_true")
    publish.add_argument("--sync", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = _profile(args.profile)
        if args.command == "status":
            result = _status(profile)
        else:
            result = _publish(
                profile,
                document=args.file,
                relative=args.relative,
                execute=args.execute,
                sync=args.sync,
            )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema_version": RESULT_SCHEMA, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
