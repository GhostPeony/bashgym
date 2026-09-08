"""Deterministic authored coding/recovery bundles with separate confirmation data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from bashgym.environments.contracts import BuildSpec, EnvironmentSpec, RolloutSpec, VerifierSpec
from bashgym.environments.personal_coding_fixtures import (
    TASKS,
    TEST_SUPPORT_SOURCE,
    VERIFIER_SOURCE,
)

PERSONAL_CODING_SCHEMA = "bashgym.personal_coding.v1"
PERSONAL_CODING_SPLITS = ("train", "dev", "confirmation")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def safe_task_path(value: str) -> str:
    """Accept portable relative file paths, without normalization aliases."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("unsafe task file path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("unsafe task file path")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in path.parts):
        raise ValueError("unsafe task file path")
    if any(
        part.endswith(".")
        or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        for part in path.parts
    ):
        raise ValueError("unsafe reserved task file path")
    return value


def environment_content_digest(spec: EnvironmentSpec) -> str:
    payload = spec.to_dict()
    payload["metadata"] = {
        key: value for key, value in payload["metadata"].items() if key != "content_sha256"
    }
    return canonical_digest(payload)


def personal_coding_environment_specs(*, split: str | None = None) -> list[EnvironmentSpec]:
    if split is not None and split not in PERSONAL_CODING_SPLITS:
        raise ValueError("unknown personal coding split")
    specs = []
    for task in TASKS:
        if split is not None and task["split"] != split:
            continue
        spec = EnvironmentSpec(
            id=task["id"],
            instruction=task["instruction"],
            source="authored_synthetic",
            domain="coding",
            skills=[task["task_family"]],
            license="MIT",
            build=BuildSpec(
                dockerfile="", network_disabled=True, metadata={"requires_pinned_local_image": True}
            ),
            rollout=RolloutSpec(
                harness="bashgym-docker-coding-v1",
                max_steps=12,
                max_tool_calls=12,
                timeout_sec=120,
                bash_timeout_sec=10,
            ),
            verifier=VerifierSpec(
                kind="coding_unittest",
                command="python -I /workspace/verify.py",
                path="verify.py",
                timeout_sec=15,
                metadata={"test_count": task["test_count"]},
            ),
            files={
                **task["files"],
                "tests/test_solution.py": task["tests"],
                "verify.py": VERIFIER_SOURCE,
                "coding_test_support.py": TEST_SUPPORT_SOURCE,
            },
            metadata={
                "suite_version": PERSONAL_CODING_SCHEMA,
                "split": task["split"],
                "task_family": task["task_family"],
                "origin": "authored_synthetic",
                "protected_paths": ["coding_test_support.py", *task.get("protected_paths", [])],
                **(
                    {"initial_command": task["initial_command"]}
                    if task.get("initial_command")
                    else {}
                ),
            },
        )
        spec.metadata["content_sha256"] = environment_content_digest(spec)
        specs.append(spec)
    return specs


def build_personal_coding_bundle(output_directory: str | Path) -> dict[str, Any]:
    root = Path(output_directory)
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise FileExistsError(
            "personal coding bundle destination must be an empty regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    inventory = []
    for split in PERSONAL_CODING_SPLITS:
        path = root / f"{split}.jsonl"
        payload = "".join(
            json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
            for spec in personal_coding_environment_specs(split=split)
        ).encode("utf-8")
        path.write_bytes(payload)
        inventory.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    identity = {
        "schema_version": PERSONAL_CODING_SCHEMA,
        "provenance": {"origin": "authored_synthetic", "license": "MIT", "user_data": False},
        "splits": {split: 2 for split in PERSONAL_CODING_SPLITS},
        "files": sorted(inventory, key=lambda item: item["path"]),
    }
    manifest = {**identity, "dataset_digest": canonical_digest(identity)}
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_personal_coding_bundle(
    directory: str | Path, *, split: str | None = None
) -> list[EnvironmentSpec]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("personal coding bundle must be a regular directory")
    if split is not None and split not in PERSONAL_CODING_SPLITS:
        raise ValueError("unknown personal coding split")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("personal coding bundle has no regular manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {key: value for key, value in manifest.items() if key != "dataset_digest"}
    if manifest.get("schema_version") != PERSONAL_CODING_SCHEMA or canonical_digest(
        identity
    ) != manifest.get("dataset_digest"):
        raise ValueError("personal coding manifest digest mismatch")
    expected = {f"{name}.jsonl" for name in PERSONAL_CODING_SPLITS}
    inventory = manifest.get("files", [])
    if {item.get("path") for item in inventory} != expected or len(inventory) != len(expected):
        raise ValueError("personal coding file inventory mismatch")
    if {path.name for path in root.iterdir()} != expected | {"manifest.json"}:
        raise ValueError("personal coding file inventory mismatch")
    specs, ids = [], set()
    for item in inventory:
        name = safe_task_path(item["path"])
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("personal coding inventory file is not regular")
        payload = path.read_bytes()
        if (
            len(payload) != item["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError("personal coding inventory digest mismatch")
        expected_split = Path(name).stem
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != manifest["splits"].get(expected_split):
            raise ValueError("personal coding split count mismatch")
        for row in rows:
            spec = EnvironmentSpec.from_dict(row)
            if spec.id in ids or spec.metadata.get("split") != expected_split:
                raise ValueError("personal coding split identity mismatch")
            ids.add(spec.id)
            if spec.validation_errors() or environment_content_digest(spec) != spec.metadata.get(
                "content_sha256"
            ):
                raise ValueError("personal coding environment content digest mismatch")
            for file_path in spec.files:
                safe_task_path(file_path)
            if split is None or expected_split == split:
                specs.append(spec)
    return sorted(specs, key=lambda spec: spec.id)
