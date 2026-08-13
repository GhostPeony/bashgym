"""Bounded metadata for datasets generated and retained on private compute."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bashgym.campaigns.contracts import (
    ActionAttempt,
    FrozenContractModel,
    HexDigest,
    Identifier,
    StageKind,
    canonical_hash,
)
from bashgym.ledger.contracts import DatasetSpec, DatasetVersionSpec, stable_ledger_id

AUTORESEARCH_DATASET_RECEIPT_FILENAME = "autoresearch_dataset_receipt.json"
AUTORESEARCH_DATASET_RECEIPT_SCHEMA = "autoresearch_dataset_receipt.v1"
AUTORESEARCH_DATASET_FILE_SCHEMA = "autoresearch_dataset_file.v1"
MAX_AUTORESEARCH_DATASET_RECEIPT_BYTES = 1024 * 1024


class AutoResearchDatasetFile(FrozenContractModel):
    """One generated dataset shard that remains on the training target."""

    schema_version: Literal["autoresearch_dataset_file.v1"] = AUTORESEARCH_DATASET_FILE_SCHEMA
    path: str = Field(min_length=9, max_length=4096)
    sha256: HexDigest
    size_bytes: int = Field(ge=0)
    split: Identifier
    row_count: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def confined_dataset_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or not value.startswith("dataset/")
            or len(path.parts) < 2
            or "\\" in value
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("generated dataset files must stay under dataset/")
        return value


class AutoResearchDatasetReceipt(FrozenContractModel):
    """Small, typed description of generated rows that never copies those rows."""

    schema_version: Literal["autoresearch_dataset_receipt.v1"] = AUTORESEARCH_DATASET_RECEIPT_SCHEMA
    files: tuple[AutoResearchDatasetFile, ...] = Field(min_length=1, max_length=1000)
    row_counts: dict[Identifier, int] = Field(default_factory=dict)
    split_manifest: dict[Identifier, list[str]] = Field(default_factory=dict)
    generator: dict[str, Any] = Field(default_factory=dict)
    content_digest: HexDigest = ""

    @model_validator(mode="after")
    def canonical_summary(self) -> AutoResearchDatasetReceipt:
        paths = tuple(item.path for item in self.files)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("generated dataset files must be sorted and unique")
        generator_bytes = json.dumps(
            self.generator,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(generator_bytes) > 64 * 1024:
            raise ValueError("dataset generator metadata exceeds the bounded limit")
        expected_counts: dict[str, int] = {}
        expected_manifest: dict[str, list[str]] = {}
        for item in self.files:
            expected_counts[item.split] = expected_counts.get(item.split, 0) + item.row_count
            expected_manifest.setdefault(item.split, []).append(item.path)
        if sum(expected_counts.values()) < 1:
            raise ValueError("generated dataset receipt must contain at least one row")
        if self.row_counts and self.row_counts != expected_counts:
            raise ValueError("dataset row counts do not match the file manifest")
        if self.split_manifest and self.split_manifest != expected_manifest:
            raise ValueError("dataset split manifest does not match the file manifest")
        expected_digest = canonical_hash(
            [
                [item.path, item.sha256, item.size_bytes, item.split, item.row_count]
                for item in self.files
            ]
        )
        if self.content_digest and self.content_digest != expected_digest:
            raise ValueError("generated dataset content digest mismatch")
        if not self.row_counts:
            object.__setattr__(self, "row_counts", expected_counts)
        if not self.split_manifest:
            object.__setattr__(self, "split_manifest", expected_manifest)
        if not self.content_digest:
            object.__setattr__(self, "content_digest", expected_digest)
        return self


def build_dataset_ledger_specs(
    attempt: ActionAttempt,
    receipt: AutoResearchDatasetReceipt,
    *,
    project_id: str,
    task_type: str,
    created_at: datetime,
) -> tuple[DatasetSpec, DatasetVersionSpec]:
    """Project remote rows to opaque, deterministic ledger identities."""

    if attempt.stage != StageKind.DATA_BUILD:
        raise ValueError("generated dataset projection requires a data-build attempt")
    dataset_id = stable_ledger_id(
        "autoresearch-generated-dataset",
        attempt.workspace_id,
        attempt.campaign_id,
    )
    version_id = stable_ledger_id(
        "autoresearch-generated-dataset-version",
        attempt.workspace_id,
        attempt.campaign_id,
        attempt.attempt_id,
        receipt.content_digest,
    )
    dataset = DatasetSpec(
        workspace_id=attempt.workspace_id,
        project_id=project_id,
        dataset_id=dataset_id,
        display_name=f"AutoResearch generated data for {attempt.campaign_id}",
        task_type=task_type,
        metadata={"source_kind": "remote_data_build"},
        created_at=created_at,
    )
    version = DatasetVersionSpec(
        workspace_id=attempt.workspace_id,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_version_id=version_id,
        source_uri=f"autoresearch-remote-dataset://sha256/{receipt.content_digest}",
        content_digest=receipt.content_digest,
        split_manifest=receipt.split_manifest,
        row_counts=receipt.row_counts,
        metadata={
            "source_kind": "remote_data_build",
            "producer_action_id": attempt.action_id,
            "producer_attempt_id": attempt.attempt_id,
            "producer_study_id": attempt.study_id,
            "generator": receipt.generator,
        },
        created_at=created_at,
    )
    return dataset, version


__all__ = [
    "AUTORESEARCH_DATASET_FILE_SCHEMA",
    "AUTORESEARCH_DATASET_RECEIPT_FILENAME",
    "AUTORESEARCH_DATASET_RECEIPT_SCHEMA",
    "AutoResearchDatasetFile",
    "AutoResearchDatasetReceipt",
    "MAX_AUTORESEARCH_DATASET_RECEIPT_BYTES",
    "build_dataset_ledger_specs",
]
