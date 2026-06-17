"""Managed fine-tuning backends — submit a job to a hosted fine-tuning API.

Together, OpenAI, and Fireworks all expose an upload -> create-job -> poll flow,
but the shapes differ enough that one declarative ``FineTuneDialect`` captures the
differences:

- **OpenAI / Together** — multipart ``/files`` upload returns a ``training_file``
  id; the job is created at a flat ``/fine_tuning/jobs`` (OpenAI) or ``/fine-tunes``
  (Together) path with a ``status``/``fine_tuned_model`` response.
- **Fireworks** — account-scoped paths (``/v1/accounts/{account_id}/...``), a
  two-step dataset upload (create dataset -> ``:upload`` the bytes), a camelCase
  job payload, and a ``state`` (``JOB_STATE_*``) / ``outputModel`` response whose
  job id is the resource ``name``.

The dialect makes the upload step and the field names pluggable so the single
``ManagedFineTuneBackend`` drives all three.

Note: the Fireworks request/response shapes follow its published REST contract
(docs.fireworks.ai); they are exercised by hermetic mock-transport tests but a
live smoke test with real credentials + an ``account_id`` is the remaining step.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
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
# Fireworks reports states like "JOB_STATE_RUNNING" (lowercased before lookup).
_FIREWORKS_STATUS = {
    "job_state_pending": TrainingStatus.PENDING,
    "job_state_creating": TrainingStatus.PENDING,
    "job_state_validating": TrainingStatus.PENDING,
    "job_state_running": TrainingStatus.RUNNING,
    "job_state_completed": TrainingStatus.SUCCEEDED,
    "job_state_failed": TrainingStatus.FAILED,
    "job_state_cancelled": TrainingStatus.CANCELLED,
    "job_state_deleting": TrainingStatus.CANCELLED,
}


def _openai_payload(spec: TrainingSpec, file_ref: str) -> dict:
    payload: dict[str, Any] = {
        "training_file": file_ref,
        "model": spec.base_model,
        "hyperparameters": {"n_epochs": spec.n_epochs},
    }
    if spec.suffix:
        payload["suffix"] = spec.suffix
    payload.update(spec.extra)
    return payload


def _together_payload(spec: TrainingSpec, file_ref: str) -> dict:
    payload: dict[str, Any] = {
        "training_file": file_ref,
        "model": spec.base_model,
        "n_epochs": spec.n_epochs,
        "learning_rate": spec.learning_rate,
    }
    if spec.suffix:
        payload["suffix"] = spec.suffix
    payload.update(spec.extra)
    return payload


def _fireworks_payload(spec: TrainingSpec, dataset_ref: str) -> dict:
    payload: dict[str, Any] = {
        "dataset": dataset_ref,
        "baseModel": spec.base_model,
        "epochs": spec.n_epochs,
        "learningRate": spec.learning_rate,
    }
    if spec.suffix:
        payload["outputModel"] = spec.suffix  # user-chosen id for the resulting model
    payload.update(spec.extra)
    return payload


async def _default_upload(backend: ManagedFineTuneBackend, spec: TrainingSpec) -> str:
    """OpenAI/Together: multipart POST to ``/files`` -> returns the file id."""
    data = Path(spec.dataset_path).read_bytes()
    files = {
        "file": (Path(spec.dataset_path).name, data, "application/jsonl"),
        "purpose": (None, "fine-tune"),
    }
    r = await backend._client.post(
        f"{backend._base}{backend._d.files_path}", headers=backend._auth(json=False), files=files
    )
    r.raise_for_status()
    return r.json()["id"]


async def _fireworks_upload(backend: ManagedFineTuneBackend, spec: TrainingSpec) -> str:
    """Fireworks: create a dataset resource, upload the JSONL, return its resource name."""
    data = Path(spec.dataset_path).read_bytes()
    example_count = sum(1 for line in data.splitlines() if line.strip())
    dataset_id = f"{(spec.suffix or 'bashgym')[:24]}-{uuid.uuid4().hex[:8]}".lower()

    create = await backend._client.post(
        f"{backend._base}/datasets",
        headers=backend._auth(),
        json={
            "datasetId": dataset_id,
            "dataset": {"userUploaded": {}, "exampleCount": str(example_count)},
        },
    )
    create.raise_for_status()

    upload = await backend._client.post(
        f"{backend._base}/datasets/{dataset_id}:upload",
        headers=backend._auth(json=False),
        files={"file": (Path(spec.dataset_path).name, data, "application/jsonl")},
    )
    upload.raise_for_status()
    return f"accounts/{backend._account_id}/datasets/{dataset_id}"


@dataclass
class FineTuneDialect:
    base_url: str  # may contain "{account_id}" when needs_account is True
    jobs_path: str = "/fine_tuning/jobs"
    files_path: str = "/files"
    status_field: str = "status"
    output_model_field: str = "fine_tuned_model"
    job_id_field: str = "id"
    needs_account: bool = False
    status_map: dict[str, TrainingStatus] = field(default_factory=lambda: dict(_OPENAI_STATUS))
    build_payload: Callable[[TrainingSpec, str], dict] = _openai_payload
    upload: Callable[[ManagedFineTuneBackend, TrainingSpec], Awaitable[str]] = _default_upload


DIALECTS: dict[str, FineTuneDialect] = {
    "openai": FineTuneDialect(base_url="https://api.openai.com/v1"),
    "together": FineTuneDialect(
        base_url="https://api.together.xyz/v1",
        jobs_path="/fine-tunes",
        output_model_field="output_name",
        status_map=dict(_TOGETHER_STATUS),
        build_payload=_together_payload,
    ),
    "fireworks": FineTuneDialect(
        base_url="https://api.fireworks.ai/v1/accounts/{account_id}",
        jobs_path="/supervisedFineTuningJobs",
        status_field="state",
        output_model_field="outputModel",
        job_id_field="name",
        needs_account=True,
        status_map=dict(_FIREWORKS_STATUS),
        build_payload=_fireworks_payload,
        upload=_fireworks_upload,
    ),
}


class ManagedFineTuneBackend(TrainingBackend):
    """Drive a hosted fine-tuning API (OpenAI / Together / Fireworks) to completion."""

    def __init__(
        self,
        name: str,
        dialect: FineTuneDialect,
        api_key: str,
        *,
        account_id: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ):
        if dialect.needs_account and not account_id:
            raise ValueError(f"platform {name!r} requires an account_id")
        self._name = name
        self._d = dialect
        self._api_key = api_key
        self._account_id = account_id
        self._base = (
            dialect.base_url.format(account_id=account_id)
            if dialect.needs_account
            else dialect.base_url
        ).rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def for_platform(
        cls, platform: str, *, api_key: str, account_id: str | None = None, **kwargs: Any
    ) -> ManagedFineTuneBackend:
        dialect = DIALECTS.get(platform)
        if not dialect:
            raise ValueError(f"unknown platform {platform!r}; known: {sorted(DIALECTS)}")
        return cls(platform, dialect, api_key, account_id=account_id, **kwargs)

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

    @staticmethod
    def _short_id(job_id: str) -> str:
        """Fireworks job ids are resource names (accounts/.../jobs/<id>); poll by the id."""
        return job_id.rsplit("/", 1)[-1] if "/" in job_id else job_id

    async def submit(self, spec: TrainingSpec) -> TrainingJob:
        file_ref = await self._d.upload(self, spec)
        payload = self._d.build_payload(spec, file_ref)
        jr = await self._client.post(
            f"{self._base}{self._d.jobs_path}", headers=self._auth(), json=payload
        )
        jr.raise_for_status()
        body = jr.json()
        return TrainingJob(
            job_id=self._short_id(str(body[self._d.job_id_field])),
            backend=self._name,
            status=self._map_status(body.get(self._d.status_field)),
            output_model=body.get(self._d.output_model_field),
            raw=body,
        )

    async def poll(self, job: TrainingJob) -> TrainingJob:
        r = await self._client.get(
            f"{self._base}{self._d.jobs_path}/{self._short_id(job.job_id)}",
            headers=self._auth(json=False),
        )
        r.raise_for_status()
        body = r.json()
        status = self._map_status(body.get(self._d.status_field))
        err = body.get("error") or body.get("status")  # Fireworks puts detail under status.message
        if isinstance(err, dict):
            err = err.get("message")
        return TrainingJob(
            job_id=job.job_id,
            backend=self._name,
            status=status,
            output_model=body.get(self._d.output_model_field),
            error=str(err) if status == TrainingStatus.FAILED and err else None,
            raw=body,
        )

    async def cancel(self, job: TrainingJob) -> TrainingJob:
        try:
            await self._client.post(
                f"{self._base}{self._d.jobs_path}/{self._short_id(job.job_id)}/cancel",
                headers=self._auth(json=False),
            )
        except Exception:  # noqa: BLE001 - cancel is best-effort
            pass
        return TrainingJob(job_id=job.job_id, backend=self._name, status=TrainingStatus.CANCELLED)

    async def close(self) -> None:
        await self._client.aclose()
