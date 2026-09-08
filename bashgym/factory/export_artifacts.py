"""Resolve only complete, unchanged personal trace exports for download."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def resolve_training_export(directory: Path, split: str, export_id: str | None = None) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    if export_id is not None and re.fullmatch(r"[0-9a-f]{64}", export_id) is None:
        raise ValueError("Invalid export identity")
    root = directory.resolve()
    if export_id is not None:
        manifest = root / f"personal_{export_id}_manifest.json"
    else:
        manifests = sorted(
            root.glob("personal_*_manifest.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        if not manifests:
            # Preserve older direct exports when no versioned personal export exists.
            legacy = root / f"{split}.jsonl"
            if legacy.is_symlink() or not legacy.resolve().is_relative_to(root):
                raise ValueError("Export path is not a regular local artifact")
            if not legacy.is_file():
                raise FileNotFoundError("No completed export found. Run export first.")
            return legacy
        manifest = manifests[0]
    if manifest.is_symlink() or not manifest.resolve().is_relative_to(root):
        raise ValueError("Export manifest is not a regular local artifact")
    if not manifest.is_file():
        raise FileNotFoundError("Requested completed export was not found")
    if manifest.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Export manifest exceeds the supported size")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
        identity = record["export_id"]
        kind = "train" if split == "train" else "validation"
        artifact = record["files"][kind]
        name, digest = artifact["name"], artifact["sha256"]
        identity_payload = {
            key: value for key, value in record.items() if key not in {"export_id", "files"}
        }
        expected_identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            record.get("schema_version") != "bashgym.personal_trace_split.v1"
            or not isinstance(identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", identity) is None
            or identity != expected_identity
            or manifest.name != f"personal_{identity}_manifest.json"
            or (export_id is not None and identity != export_id)
            or name != f"personal_{identity}_{kind}.jsonl"
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or digest != record["file_hashes"][kind]
        ):
            raise ValueError("Export manifest identity is invalid")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Export manifest is invalid") from exc
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Export file is not a regular local artifact")
    if not path.is_file():
        raise ValueError("Completed export is missing its data file")
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        observed = checksum.hexdigest()
    if observed != digest:
        raise ValueError("Export data changed after its split was recorded")
    return path
