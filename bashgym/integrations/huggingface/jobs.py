"""Honest, opt-in Hugging Face Jobs seam for cloud training.

The adapter projects BashGym training inputs onto ``HfApi.run_uv_job`` and
uses the provider's returned identity for all subsequent observation. It does
not invent job IDs, prices, URLs, status, logs, or a successful simulation.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from .client import (
    HF_HUB_AVAILABLE,
    HFJobFailedError,
    HFProRequiredError,
    HuggingFaceClient,
)

try:  # Optional dependency: keep importing BashGym useful without the Hub SDK.
    from huggingface_hub import HfApi as _ProviderHfApi
    from huggingface_hub import SpaceHardware as _ProviderSpaceHardware
except ImportError:  # pragma: no cover - exercised by availability monkeypatch tests.
    _ProviderHfApi = None
    _ProviderSpaceHardware = None


def _provider_version() -> str | None:
    try:
        return version("huggingface_hub")
    except PackageNotFoundError:
        return None


def _provider_hardware_flavors() -> tuple[str, ...]:
    if _ProviderSpaceHardware is None:
        return ()
    return tuple(item.value for item in _ProviderSpaceHardware)


_HARDWARE_FLAVORS = _provider_hardware_flavors()

# Compatibility projection for existing callers. Provider pricing and capacity
# are intentionally absent because the installed SDK does not authoritatively
# expose them. The accepted IDs are derived from its SpaceHardware enum.
HARDWARE_SPECS: dict[str, dict[str, Any]] = {
    flavor: {
        "provider_value": flavor,
        "source": "huggingface_hub.SpaceHardware",
        "gpu": None,
        "vram_gb": None,
        "memory_gb": None,
        "cost_per_hour": None,
        "pro_required": True,
    }
    for flavor in _HARDWARE_FLAVORS
}

HF_JOB_LOG_DEFAULT_TAIL_LINES = 200
HF_JOB_LOG_MAX_TAIL_LINES = 2_000
HF_JOB_LOG_MAX_BYTES = 256 * 1024
_HF_JOB_LOG_CHUNK_CHARACTERS = 4_096


def _bounded_log_tail(logs: str | Iterable[str], *, tail_lines: int) -> str:
    """Collect the final log suffix without retaining the complete response."""

    retained = bytearray()
    chunks: Iterable[str] = (logs,) if isinstance(logs, str) else logs
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError("Hugging Face Jobs logs must contain text chunks")
        for offset in range(0, len(chunk), _HF_JOB_LOG_CHUNK_CHARACTERS):
            encoded = chunk[offset : offset + _HF_JOB_LOG_CHUNK_CHARACTERS].encode("utf-8")
            retained.extend(encoded)
            overflow = len(retained) - HF_JOB_LOG_MAX_BYTES
            if overflow > 0:
                del retained[:overflow]

    text = retained.decode("utf-8", errors="ignore")
    return "".join(text.splitlines(keepends=True)[-tail_lines:])


class JobStatus(str, Enum):
    """Stable BashGym projection of Hugging Face job stages."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HFJobLogsNotReadyError(HFJobFailedError):
    """Stable refusal returned before an active provider log stream is opened."""

    reason_code = "hf_job_logs_not_ready"

    def __init__(self, job_id: str, status: JobStatus):
        super().__init__(
            f"{self.reason_code}: job status is {status.value}",
            job_id=job_id,
        )
        self.status = status


_PROVIDER_STATUS = {
    "SCHEDULING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "ERROR": JobStatus.FAILED,
    "CANCELED": JobStatus.CANCELLED,
    "CANCELLED": JobStatus.CANCELLED,
    "DELETED": JobStatus.CANCELLED,
}


@dataclass(frozen=True)
class HFJobsAvailability:
    """Local SDK capability only; it never probes credentials or the network."""

    dependency_available: bool
    api_available: bool
    provider_version: str | None
    api_method: str
    hardware_flavors: tuple[str, ...]


def detect_hf_jobs_availability() -> HFJobsAvailability:
    """Report whether the installed SDK exposes the canonical Jobs methods."""

    api_available = bool(
        HF_HUB_AVAILABLE
        and _ProviderHfApi is not None
        and callable(getattr(_ProviderHfApi, "run_uv_job", None))
        and callable(getattr(_ProviderHfApi, "inspect_job", None))
        and callable(getattr(_ProviderHfApi, "fetch_job_logs", None))
        and callable(getattr(_ProviderHfApi, "cancel_job", None))
        and callable(getattr(_ProviderHfApi, "list_jobs", None))
    )
    return HFJobsAvailability(
        dependency_available=bool(HF_HUB_AVAILABLE and _ProviderHfApi is not None),
        api_available=api_available,
        provider_version=_provider_version(),
        api_method="HfApi.run_uv_job",
        hardware_flavors=_HARDWARE_FLAVORS,
    )


@dataclass
class HFJobConfig:
    """Explicit inputs that map directly onto ``HfApi.run_uv_job``."""

    hardware: str = "a10g-small"
    timeout_minutes: int = 30
    docker_image: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict, repr=False)
    dependencies: tuple[str, ...] = ()
    python: str | None = None
    namespace: str | None = None
    requirements: str | None = None
    dataset_repo: str | None = None
    """Dataset declaration injected into the job as ``BASHGYM_DATASET_REPO``."""

    output_repo: str | None = None
    """Output declaration injected into the job as ``BASHGYM_OUTPUT_REPO``."""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if _HARDWARE_FLAVORS and self.hardware not in _HARDWARE_FLAVORS:
            errors.append(
                f"Invalid hardware {self.hardware!r}; use an installed SpaceHardware value"
            )
        if not isinstance(self.timeout_minutes, int) or self.timeout_minutes < 1:
            errors.append("timeout_minutes must be at least 1")
        if self.requirements is not None:
            errors.append(
                "requirements is not supported by HfApi.run_uv_job; use PEP 723 or dependencies"
            )
        if not all(isinstance(item, str) and item.strip() for item in self.dependencies):
            errors.append("dependencies must contain non-empty package requirements")
        if self.secrets.get("HF_TOKEN") == "$HF_TOKEN":
            errors.append(
                "literal $HF_TOKEN is MCP-only; HfApi.run_uv_job requires an injected secret value"
            )
        return errors


@dataclass(frozen=True)
class HFJobRequest:
    """Secret-redacted projection of a future ``run_uv_job`` request."""

    script: str
    script_args: tuple[str, ...]
    dependencies: tuple[str, ...]
    image: str | None
    env: dict[str, str]
    secret_names: tuple[str, ...]
    flavor: str
    timeout: str
    python: str | None
    namespace: str | None
    dataset_repo: str | None
    output_repo: str

    def to_dict(self) -> dict[str, Any]:
        projection: dict[str, Any] = {
            "api_method": "HfApi.run_uv_job",
            "script": self.script,
            "script_args": list(self.script_args),
            "dependencies": list(self.dependencies),
            "image": self.image,
            "env": dict(self.env),
            "secret_names": list(self.secret_names),
            "flavor": self.flavor,
            "timeout": self.timeout,
            "namespace": self.namespace,
            "dataset_repo": self.dataset_repo,
            "output_repo": self.output_repo,
        }
        if self.python is not None:
            projection["python"] = self.python
        return projection


@dataclass(frozen=True)
class HFJobsPreflight:
    """Fail-closed readiness result for one exact training request."""

    availability: HFJobsAvailability
    ready: bool
    reason_codes: tuple[str, ...]
    config_errors: tuple[str, ...]
    request: HFJobRequest | None


@dataclass
class HFJobInfo:
    """Provider-derived job identity and state; no fields are synthesized."""

    job_id: str
    status: JobStatus
    hardware: str
    created_at: datetime | None
    namespace: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    logs_url: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    output_repo: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "hardware": self.hardware,
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "logs_url": self.logs_url,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "output_repo": self.output_repo,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HFJobInfo:
        def parse_datetime(value: Any) -> datetime | None:
            if not isinstance(value, str):
                return None
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

        return cls(
            job_id=str(data.get("job_id", "")),
            status=JobStatus(str(data.get("status", "pending"))),
            hardware=str(data.get("hardware", "unknown")),
            created_at=parse_datetime(data.get("created_at")),
            namespace=data.get("namespace"),
            started_at=parse_datetime(data.get("started_at")),
            completed_at=parse_datetime(data.get("completed_at")),
            logs_url=data.get("logs_url"),
            error_message=data.get("error_message"),
            metrics=dict(data.get("metrics") or {}),
            output_repo=data.get("output_repo"),
        )

    @classmethod
    def from_provider(
        cls,
        job: Any,
        *,
        output_repo: str | None = None,
        namespace: str | None = None,
    ) -> HFJobInfo:
        job_id = getattr(job, "id", None)
        stage = getattr(getattr(job, "status", None), "stage", None)
        stage = getattr(stage, "value", stage)
        if not isinstance(job_id, str) or not job_id or stage not in _PROVIDER_STATUS:
            raise HFJobFailedError("Hugging Face Jobs returned an invalid job projection.")
        status = _PROVIDER_STATUS[stage]
        message = getattr(getattr(job, "status", None), "message", None)
        owner_namespace = getattr(getattr(job, "owner", None), "name", None)
        flavor = getattr(job, "flavor", None)
        flavor = getattr(flavor, "value", flavor)
        return cls(
            job_id=job_id,
            status=status,
            hardware=str(flavor or "unknown"),
            created_at=getattr(job, "created_at", None),
            namespace=(
                owner_namespace
                if isinstance(owner_namespace, str) and owner_namespace
                else namespace
            ),
            started_at=getattr(job, "started_at", None),
            completed_at=getattr(job, "finished_at", None),
            logs_url=getattr(job, "url", None),
            error_message=str(message) if status is JobStatus.FAILED and message else None,
            output_repo=output_repo,
        )


class HFJobRunner:
    """Thin adapter over an explicitly configured Hugging Face Jobs API."""

    def __init__(
        self,
        client: HuggingFaceClient | None = None,
        token: str | None = None,
        pro_enabled: bool = False,
        *,
        api: Any | None = None,
        jobs_access_confirmed: bool | None = None,
    ) -> None:
        self._client = client or (HuggingFaceClient(token=token) if token else None)
        self._api = api
        self._jobs_access_confirmed = (
            pro_enabled if jobs_access_confirmed is None else jobs_access_confirmed
        )

    @property
    def client(self) -> HuggingFaceClient:
        if self._client is None:
            raise HFJobFailedError("Hugging Face client is not configured.")
        return self._client

    @property
    def is_pro(self) -> bool:
        """Compatibility alias for explicit eligible Jobs-plan confirmation."""

        return self._jobs_access_confirmed

    def preflight(
        self,
        script_path: str | Path,
        *,
        repo_id: str | None = None,
        config: HFJobConfig | None = None,
        script_args: list[str] | None = None,
    ) -> HFJobsPreflight:
        config = config or HFJobConfig()
        availability = detect_hf_jobs_availability()
        reasons: list[str] = []
        config_errors = config.validate()
        path = Path(script_path)
        output_repo = repo_id or config.output_repo
        request_environment = dict(config.environment)

        if not availability.dependency_available:
            reasons.append("huggingface_hub_not_installed")
        elif not availability.api_available:
            reasons.append("huggingface_jobs_api_unavailable")
        if not self._jobs_access_confirmed:
            reasons.append("jobs_access_not_confirmed")
        if self._api is None and self._client is None:
            reasons.append("jobs_api_not_configured")
        if not path.is_file():
            reasons.append("training_script_not_found")
        if config_errors:
            reasons.append("job_config_invalid")
        if repo_id and config.output_repo and repo_id != config.output_repo:
            reasons.append("output_repo_mismatch")
        if not output_repo:
            reasons.append("output_repo_missing")
        token_secret = config.secrets.get("HF_TOKEN")
        if not token_secret:
            reasons.append("hf_token_secret_missing")
        declarations = {
            "BASHGYM_DATASET_REPO": config.dataset_repo,
            "BASHGYM_OUTPUT_REPO": output_repo,
        }
        for name, value in declarations.items():
            if not value:
                continue
            existing = request_environment.get(name)
            if existing is not None and existing != value:
                config_errors.append(f"{name} conflicts with the declared repository")
                if "job_config_invalid" not in reasons:
                    reasons.append("job_config_invalid")
                continue
            request_environment[name] = value

        request = None
        if not reasons and output_repo is not None:
            request = HFJobRequest(
                script=str(path.resolve()),
                script_args=tuple(script_args or ()),
                dependencies=tuple(config.dependencies),
                image=config.docker_image,
                env=request_environment,
                secret_names=tuple(sorted(config.secrets)),
                flavor=config.hardware,
                timeout=f"{config.timeout_minutes}m",
                python=config.python,
                namespace=config.namespace,
                dataset_repo=config.dataset_repo,
                output_repo=output_repo,
            )
        return HFJobsPreflight(
            availability=availability,
            ready=not reasons,
            reason_codes=tuple(reasons),
            config_errors=tuple(config_errors),
            request=request,
        )

    def _resolve_api(self) -> Any:
        if self._api is not None:
            return self._api
        if self._client is None:
            raise HFJobFailedError("jobs_api_not_configured")
        api = self._client.api
        if api is None:
            raise HFJobFailedError("jobs_api_not_authenticated")
        return api

    def _require_observation_api(self, operation: str) -> Any:
        availability = detect_hf_jobs_availability()
        if not availability.api_available:
            raise HFJobFailedError("huggingface_jobs_api_unavailable")
        if not self._jobs_access_confirmed:
            raise HFProRequiredError(
                f"{operation} requires explicit confirmation of eligible Hugging Face Jobs access"
            )
        return self._resolve_api()

    def submit_training_job(
        self,
        script_path: str | Path,
        repo_id: str | None = None,
        config: HFJobConfig | None = None,
        script_args: list[str] | None = None,
        description: str | None = None,
    ) -> HFJobInfo:
        """Submit through ``run_uv_job`` only after an exact preflight succeeds."""

        if description is not None:
            raise ValueError("HfApi.run_uv_job does not support a description field")
        config = config or HFJobConfig()
        preflight = self.preflight(
            script_path,
            repo_id=repo_id,
            config=config,
            script_args=script_args,
        )
        if preflight.config_errors:
            raise ValueError(f"Invalid job configuration: {'; '.join(preflight.config_errors)}")
        if "training_script_not_found" in preflight.reason_codes:
            raise FileNotFoundError(f"Training script not found: {script_path}")
        if not preflight.ready or preflight.request is None:
            raise HFJobFailedError(
                "HF Jobs launch preflight failed: " + ", ".join(preflight.reason_codes)
            )

        request = preflight.request
        kwargs: dict[str, Any] = {
            "script_args": list(request.script_args),
            "dependencies": list(request.dependencies),
            "env": dict(request.env),
            "secrets": dict(config.secrets),
            "flavor": request.flavor,
            "timeout": request.timeout,
        }
        if request.image is not None:
            kwargs["image"] = request.image
        if request.python is not None:
            kwargs["python"] = request.python
        if request.namespace is not None:
            kwargs["namespace"] = request.namespace
        try:
            provider_job = self._resolve_api().run_uv_job(request.script, **kwargs)
        except HFJobFailedError:
            raise
        except Exception as exc:
            raise HFJobFailedError("Hugging Face Jobs submission failed.") from exc
        return HFJobInfo.from_provider(
            provider_job,
            output_repo=request.output_repo,
            namespace=request.namespace,
        )

    @staticmethod
    def _namespace_kwargs(namespace: str | None) -> dict[str, str]:
        return {"namespace": namespace} if namespace else {}

    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        if isinstance(error, KeyError):
            return True
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) == 404

    @staticmethod
    def _supports_parameter(operation: Any, name: str) -> bool:
        try:
            parameter = signature(operation).parameters.get(name)
        except (TypeError, ValueError):
            return False
        return parameter is not None and parameter.kind is not Parameter.VAR_KEYWORD

    def get_job_status(self, job_id: str, *, namespace: str | None = None) -> HFJobInfo:
        api = self._require_observation_api("Checking job status")
        try:
            job = api.inspect_job(job_id=job_id, **self._namespace_kwargs(namespace))
            return HFJobInfo.from_provider(job, namespace=namespace)
        except HFJobFailedError:
            raise
        except Exception as exc:
            if self._is_not_found(exc):
                raise KeyError(f"Job not found: {job_id}") from exc
            raise HFJobFailedError(
                "Hugging Face Jobs status request failed.", job_id=job_id
            ) from exc

    def get_job_logs(
        self,
        job_id: str,
        tail: int | None = None,
        since: datetime | None = None,
        *,
        namespace: str | None = None,
    ) -> str:
        if since is not None:
            raise ValueError("HfApi.fetch_job_logs does not support a since filter")
        if isinstance(tail, bool) or (tail is not None and not isinstance(tail, int)):
            raise ValueError("tail must be an integer")
        if tail is not None and tail < 1:
            raise ValueError("tail must be at least 1")
        effective_tail = min(
            tail if tail is not None else HF_JOB_LOG_DEFAULT_TAIL_LINES,
            HF_JOB_LOG_MAX_TAIL_LINES,
        )
        current = self.get_job_status(job_id, namespace=namespace)
        if not current.is_terminal:
            raise HFJobLogsNotReadyError(job_id, current.status)
        api = self._require_observation_api("Retrieving job logs")
        try:
            operation = api.fetch_job_logs
            kwargs: dict[str, Any] = self._namespace_kwargs(namespace)
            if self._supports_parameter(operation, "follow"):
                kwargs["follow"] = False
            if self._supports_parameter(operation, "tail"):
                kwargs["tail"] = effective_tail
            logs = operation(job_id=job_id, **kwargs)
            return _bounded_log_tail(logs, tail_lines=effective_tail)
        except Exception as exc:
            if self._is_not_found(exc):
                raise KeyError(f"Job not found: {job_id}") from exc
            raise HFJobFailedError("Hugging Face Jobs log request failed.", job_id=job_id) from exc

    def cancel_job(self, job_id: str, *, namespace: str | None = None) -> HFJobInfo:
        api = self._require_observation_api("Cancelling jobs")
        current = self.get_job_status(job_id, namespace=namespace)
        if current.is_terminal:
            raise HFJobFailedError(
                f"Cannot cancel job in terminal state: {current.status.value}",
                job_id=job_id,
            )
        try:
            resolved_namespace = current.namespace or namespace
            namespace_kwargs = self._namespace_kwargs(resolved_namespace)
            api.cancel_job(job_id=job_id, **namespace_kwargs)
            job = api.inspect_job(job_id=job_id, **namespace_kwargs)
            return HFJobInfo.from_provider(job, namespace=resolved_namespace)
        except HFJobFailedError:
            raise
        except Exception as exc:
            raise HFJobFailedError("Hugging Face Jobs cancellation failed.", job_id=job_id) from exc

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
        *,
        namespace: str | None = None,
    ) -> list[HFJobInfo]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        api = self._require_observation_api("Listing jobs")
        try:
            jobs = [
                HFJobInfo.from_provider(item, namespace=namespace)
                for item in api.list_jobs(**self._namespace_kwargs(namespace))
            ]
        except HFJobFailedError:
            raise
        except Exception as exc:
            raise HFJobFailedError("Hugging Face Jobs listing failed.") from exc
        if status is not None:
            jobs = [job for job in jobs if job.status is status]
        return jobs[:limit]

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int | None = None,
        *,
        namespace: str | None = None,
    ) -> HFJobInfo:
        """Explicit compatibility helper; callers should prefer agent-driven observation."""

        started = datetime.now(timezone.utc)
        while True:
            job = self.get_job_status(job_id, namespace=namespace)
            if job.is_terminal:
                if job.status is JobStatus.FAILED:
                    raise HFJobFailedError(job.error_message or "Job failed", job_id=job_id)
                return job
            if timeout is not None:
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            time.sleep(poll_interval)

    def __repr__(self) -> str:
        readiness = "confirmed" if self._jobs_access_confirmed else "unconfirmed"
        return f"<HFJobRunner jobs_access={readiness}>"


def create_job_runner(
    client: HuggingFaceClient | None = None,
    token: str | None = None,
    pro_enabled: bool = False,
    *,
    api: Any | None = None,
    jobs_access_confirmed: bool | None = None,
) -> HFJobRunner:
    """Create the concrete Hugging Face Jobs adapter; no provider call occurs."""

    return HFJobRunner(
        client=client,
        token=token,
        pro_enabled=pro_enabled,
        api=api,
        jobs_access_confirmed=jobs_access_confirmed,
    )


__all__ = [
    "HARDWARE_SPECS",
    "HF_JOB_LOG_DEFAULT_TAIL_LINES",
    "HF_JOB_LOG_MAX_BYTES",
    "HF_JOB_LOG_MAX_TAIL_LINES",
    "HFJobConfig",
    "HFJobInfo",
    "HFJobLogsNotReadyError",
    "HFJobRequest",
    "HFJobRunner",
    "HFJobsAvailability",
    "HFJobsPreflight",
    "JobStatus",
    "create_job_runner",
    "detect_hf_jobs_availability",
]
