"""Managed fine-tuning backends — submit a job to a hosted fine-tuning API.

Together, Fireworks, and OpenAI share an upload-file -> create-job -> poll flow
with small dialect differences (paths, payload shape, status vocabulary, result
field). One backend handles all of them via a ``FineTuneDialect``; ``DIALECTS``
covers Together and the OpenAI standard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .base import TrainingBackend, TrainingJob, TrainingSpec, TrainingStatus

_OPENAI_STATUS = {
    "validating_files": TrainingStatus.PENDING,
    "queued": TrainingStatus.PENDING,
    "running": TrainingStatus.RUNNING,
    "succeeded": TrainingStatus.SUCCEEDED,
    "failed": TrainingStatus.FAILED,
    "cancelled": TrainingStatus.CANCELLED,
}
_TOGETHER_STATUS = {
    "pending": TrainingStatus.PENDING,
    "queued": TrainingStatus.PENDING,
    "running": TrainingStatus.RUNNING,
    "compressing": TrainingStatus.RUNNING,
    "uploading": TrainingStatus.RUNNING,
    "completed": TrainingStatus.SUCCEEDED,
    "error": TrainingStatus.FAILED,
    "cancelled": TrainingStatus.CANCELLED,
}


def _openai_payload(spec: TrainingSpec, file_id: str) -> dict:
    payload: dict[str, Any] = {
        "training_file": file_id,
        "model": spec.base_model,
        "hyperparameters": {"n_epochs": spec.n_epochs},
    }
    if spec.suffix:
        payload["suffix"] = spec.suffix
    payload.update(spec.extra)
    return payload


def _together_payload(spec: TrainingSpec, file_id: str) -> dict:
    payload: dict[str, Any] = {
        "training_file": file_id,
        "model": spec.base_model,
        "n_epochs": spec.n_epochs,
        "learning_rate": spec.learning_rate,
    }
    if spec.suffix:
        payload["suffix"] = spec.suffix
    payload.update(spec.extra)
    return payload


@dataclass
class FineTuneDialect:
    base_url: str
    jobs_path: str = "/fine_tuning/jobs"
    files_path: str = "/files"
    status_field: str = "status"
    output_model_field: str = "fine_tuned_model"
    status_map: dict[str, TrainingStatus] = field(default_factory=lambda: dict(_OPENAI_STATUS))
    build_payload: Callable[[TrainingSpec, str], dict] = _openai_payload


DIALECTS: dict[str, FineTuneDialect] = {
    "openai": FineTuneDialect(base_url="https://api.openai.com/v1"),
    "together": FineTuneDialect(
        base_url="https://api.together.xyz/v1",
        jobs_path="/fine-tunes",
        output_model_field="output_name",
        status_map=dict(_TOGETHER_STATUS),
        build_payload=_together_payload,
    ),
}


class ManagedFineTuneBackend(TrainingBackend):
    """Drive a hosted fine-tuning API (Together / OpenAI-standard) to completion."""

    def __init__(
        self,
        name: str,
        dialect: FineTuneDialect,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ):
        self._name = name
        self._d = dialect
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def for_platform(cls, platform: str, *, api_key: str, **kwargs: Any) -> ManagedFineTuneBackend:
        dialect = DIALECTS.get(platform)
        if not dialect:
            raise ValueError(f"unknown platform {platform!r}; known: {sorted(DIALECTS)}")
        return cls(platform, dialect, api_key, **kwargs)

    @property
    def backend_type(self) -> str:
        return self._name

    def _auth(self, json: bool = True) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._api_key}"}
        if json:
            h["Content-Type"] = "application/json"
        return h

    def _map_status(self, raw_status: Any) -> TrainingStatus:
        return self._d.status_map.get(str(raw_status or "").lower(), TrainingStatus.UNKNOWN)

    async def submit(self, spec: TrainingSpec) -> TrainingJob:
        base = self._d.base_url.rstrip("/")
        data = Path(spec.dataset_path).read_bytes()
        files = {
            "file": (Path(spec.dataset_path).name, data, "application/jsonl"),
            "purpose": (None, "fine-tune"),
        }
        fr = await self._client.post(
            f"{base}{self._d.files_path}", headers=self._auth(json=False), files=files
        )
        fr.raise_for_status()
        file_id = fr.json()["id"]

        payload = self._d.build_payload(spec, file_id)
        jr = await self._client.post(
            f"{base}{self._d.jobs_path}", headers=self._auth(), json=payload
        )
        jr.raise_for_status()
        body = jr.json()
        return TrainingJob(
            job_id=str(body["id"]),
            backend=self._name,
            status=self._map_status(body.get(self._d.status_field)),
            raw=body,
        )

    async def poll(self, job: TrainingJob) -> TrainingJob:
        base = self._d.base_url.rstrip("/")
        r = await self._client.get(
            f"{base}{self._d.jobs_path}/{job.job_id}", headers=self._auth(json=False)
        )
        r.raise_for_status()
        body = r.json()
        status = self._map_status(body.get(self._d.status_field))
        err = body.get("error")
        return TrainingJob(
            job_id=job.job_id,
            backend=self._name,
            status=status,
            output_model=body.get(self._d.output_model_field),
            error=str(err) if status == TrainingStatus.FAILED and err else None,
            raw=body,
        )

    async def cancel(self, job: TrainingJob) -> TrainingJob:
        base = self._d.base_url.rstrip("/")
        try:
            await self._client.post(
                f"{base}{self._d.jobs_path}/{job.job_id}/cancel", headers=self._auth(json=False)
            )
        except Exception:  # noqa: BLE001 - cancel is best-effort
            pass
        return TrainingJob(job_id=job.job_id, backend=self._name, status=TrainingStatus.CANCELLED)

    async def close(self) -> None:
        await self._client.aclose()
