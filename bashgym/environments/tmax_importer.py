"""Import TMax/Harbor-style terminal tasks into BashGym environment specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bashgym.environments.contracts import EnvironmentSpec
from bashgym.environments.loader import environment_from_record

TMAX_HF_DATASETS = {
    "tmax_15k": "allenai/TMax-15K",
    "tmax_15k_open_instruct": "allenai/tmax-15k-open-instruct",
    "tmax_15k_harbor": "tmax/TMax-15K-Harbor",
}


class TMaxImporter:
    """Best-effort importer for released TMax rows and Harbor task records."""

    def __init__(self, *, preserve_raw: bool = True):
        self.preserve_raw = preserve_raw

    def from_records(
        self,
        records: list[dict[str, Any]],
        *,
        source_uri: str | None = None,
        source: str = "tmax",
    ) -> list[EnvironmentSpec]:
        return [
            environment_from_record(
                record,
                source=source,
                source_uri=source_uri,
                preserve_raw=self.preserve_raw,
            )
            for record in records
        ]

    def from_jsonl(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        source: str = "tmax",
    ) -> list[EnvironmentSpec]:
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self.from_records(records, source_uri=source_uri or str(path), source=source)

    def from_json(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        source: str = "tmax",
    ) -> list[EnvironmentSpec]:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            records = payload["data"]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = [payload]
        else:
            raise ValueError(f"Unsupported TMax JSON payload in {path}")
        return self.from_records(records, source_uri=source_uri or str(path), source=source)

    def from_huggingface(
        self,
        dataset: str = TMAX_HF_DATASETS["tmax_15k"],
        *,
        split: str = "train",
        limit: int | None = None,
        source: str = "tmax",
    ) -> list[EnvironmentSpec]:
        """Load a released TMax dataset via ``datasets`` when available.

        This is intentionally optional so unit tests and local development do not
        need network access or the HuggingFace datasets package.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required for from_huggingface(); install bashgym[training]"
            ) from exc

        ds = load_dataset(dataset, split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        records = [dict(row) for row in ds]
        return self.from_records(records, source_uri=f"hf://{dataset}/{split}", source=source)
