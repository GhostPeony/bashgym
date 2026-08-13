from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

import bashgym.integrations.huggingface.jobs as jobs_module
from bashgym.integrations.huggingface.client import HFJobFailedError
from bashgym.integrations.huggingface.jobs import (
    HARDWARE_SPECS,
    HFJobConfig,
    HFJobRunner,
    JobStatus,
    detect_hf_jobs_availability,
)

EXPECTED_HF_JOB_LOG_MAX_BYTES = 256 * 1024
EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES = 2_000


class RecordingJobsApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.jobs = {
            "provider-job-1": provider_job("provider-job-1", "RUNNING"),
            "provider-job-2": provider_job("provider-job-2", "COMPLETED"),
        }

    def run_uv_job(self, script: str, **kwargs: Any) -> Any:
        self.calls.append(("run_uv_job", {"script": script, **kwargs}))
        return self.jobs["provider-job-1"]

    def inspect_job(self, **kwargs: Any) -> Any:
        self.calls.append(("inspect_job", kwargs))
        return self.jobs[kwargs["job_id"]]

    def fetch_job_logs(self, **kwargs: Any) -> list[str]:
        self.calls.append(("fetch_job_logs", kwargs))
        return ["step=1 loss=0.4\n", "step=2 loss=0.2\n"]

    def cancel_job(self, **kwargs: Any) -> None:
        self.calls.append(("cancel_job", kwargs))
        job_id = kwargs["job_id"]
        self.jobs[job_id] = provider_job(
            job_id,
            "CANCELED",
            namespace=kwargs.get("namespace") or "test-user",
        )

    def list_jobs(self, **kwargs: Any) -> list[Any]:
        self.calls.append(("list_jobs", kwargs))
        return list(self.jobs.values())


class MinimumJobsApi:
    """Provider double with the exact huggingface_hub 1.3 log signature."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def inspect_job(self, **kwargs: Any) -> Any:
        self.calls.append(("inspect_job", kwargs))
        return provider_job(kwargs["job_id"], "COMPLETED", namespace="research-org")

    def fetch_job_logs(
        self,
        *,
        job_id: str,
        namespace: str | None = None,
        token: str | bool | None = None,
    ) -> list[str]:
        self.calls.append(
            (
                "fetch_job_logs",
                {"job_id": job_id, "namespace": namespace, "token": token},
            )
        )
        return ["step=1 loss=0.4\n", "step=2 loss=0.2\n"]


class LargeTerminalLogsApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def inspect_job(self, **kwargs: Any) -> Any:
        self.calls.append(("inspect_job", kwargs))
        return provider_job(kwargs["job_id"], "COMPLETED")

    def fetch_job_logs(self, *, job_id: str, tail: int, follow: bool = True):
        self.calls.append(("fetch_job_logs", {"job_id": job_id, "tail": tail, "follow": follow}))
        for index in range(EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES + 100):
            yield f"line={index:05d} " + ("x" * 1024) + "\n"


class InterruptibleNonTerminatingLogs:
    """Iterator that cannot finish unless test cleanup explicitly releases it."""

    def __init__(self) -> None:
        self.iteration_started = threading.Event()
        self.release = threading.Event()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        self.iteration_started.set()
        self.release.wait()
        raise StopIteration


class ActiveJobWithNonTerminatingLogsApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.logs = InterruptibleNonTerminatingLogs()

    def inspect_job(self, **kwargs: Any) -> Any:
        self.calls.append(("inspect_job", kwargs))
        return provider_job(kwargs["job_id"], "RUNNING")

    def fetch_job_logs(self, **kwargs: Any) -> InterruptibleNonTerminatingLogs:
        self.calls.append(("fetch_job_logs", kwargs))
        return self.logs


class ProviderRequestError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


def provider_job(
    job_id: str,
    stage: str,
    *,
    namespace: str = "test-user",
) -> SimpleNamespace:
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=job_id,
        status=SimpleNamespace(stage=stage, message=None),
        owner=SimpleNamespace(name=namespace),
        flavor="a10g-small",
        created_at=now,
        started_at=now if stage != "SCHEDULING" else None,
        finished_at=now if stage in {"COMPLETED", "ERROR", "CANCELED"} else None,
        url=f"https://huggingface.co/jobs/{namespace}/{job_id}",
    )


def ready_config() -> HFJobConfig:
    return HFJobConfig(
        hardware="a10g-small",
        timeout_minutes=60,
        docker_image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        environment={"BASHGYM_CAMPAIGN_ID": "campaign-1"},
        secrets={"HF_TOKEN": "injected-at-launch"},
        dependencies=("transformers", "trl"),
        dataset_repo="test-user/campaign-1-data",
        output_repo="test-user/campaign-1-output",
    )


def test_availability_and_hardware_are_derived_without_price_claims():
    availability = detect_hf_jobs_availability()

    assert availability.dependency_available is True
    assert availability.api_available is True
    assert availability.api_method == "HfApi.run_uv_job"
    assert "a10g-small" in availability.hardware_flavors
    assert set(HARDWARE_SPECS) == set(availability.hardware_flavors)
    assert all(
        spec["source"] == "huggingface_hub.SpaceHardware" for spec in HARDWARE_SPECS.values()
    )
    assert all(spec["cost_per_hour"] is None for spec in HARDWARE_SPECS.values())


def test_preflight_projects_canonical_request_without_secret_values(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    api = RecordingJobsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)

    preflight = runner.preflight(
        script,
        repo_id="test-user/campaign-1-output",
        config=ready_config(),
        script_args=["--seed", "7"],
    )

    assert preflight.ready is True
    assert preflight.reason_codes == ()
    assert preflight.request is not None
    projection = preflight.request.to_dict()
    assert projection == {
        "api_method": "HfApi.run_uv_job",
        "script": str(script.resolve()),
        "script_args": ["--seed", "7"],
        "dependencies": ["transformers", "trl"],
        "image": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        "env": {
            "BASHGYM_CAMPAIGN_ID": "campaign-1",
            "BASHGYM_DATASET_REPO": "test-user/campaign-1-data",
            "BASHGYM_OUTPUT_REPO": "test-user/campaign-1-output",
        },
        "secret_names": ["HF_TOKEN"],
        "flavor": "a10g-small",
        "timeout": "60m",
        "namespace": None,
        "dataset_repo": "test-user/campaign-1-data",
        "output_repo": "test-user/campaign-1-output",
    }
    assert "injected-at-launch" not in repr(preflight)
    assert api.calls == []


@pytest.mark.parametrize(
    ("runner", "config", "expected_reason"),
    [
        (HFJobRunner(api=RecordingJobsApi()), ready_config(), "jobs_access_not_confirmed"),
        (
            HFJobRunner(jobs_access_confirmed=True),
            ready_config(),
            "jobs_api_not_configured",
        ),
        (
            HFJobRunner(api=RecordingJobsApi(), jobs_access_confirmed=True),
            HFJobConfig(output_repo="test-user/output"),
            "hf_token_secret_missing",
        ),
    ],
)
def test_launch_preflight_fails_closed_without_auth_or_configuration(
    tmp_path,
    runner: HFJobRunner,
    config: HFJobConfig,
    expected_reason: str,
):
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")

    preflight = runner.preflight(script, repo_id="test-user/output", config=config)

    assert preflight.ready is False
    assert expected_reason in preflight.reason_codes
    with pytest.raises(HFJobFailedError, match=expected_reason):
        runner.submit_training_job(script, repo_id="test-user/output", config=config)
    api = getattr(runner, "_api", None)
    if isinstance(api, RecordingJobsApi):
        assert api.calls == []


def test_launch_fails_closed_when_jobs_dependency_is_unavailable(tmp_path, monkeypatch):
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    api = RecordingJobsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)
    monkeypatch.setattr(jobs_module, "HF_HUB_AVAILABLE", False)

    preflight = runner.preflight(
        script,
        repo_id="test-user/campaign-1-output",
        config=ready_config(),
    )

    assert preflight.ready is False
    assert "huggingface_hub_not_installed" in preflight.reason_codes
    with pytest.raises(HFJobFailedError, match="huggingface_hub_not_installed"):
        runner.submit_training_job(
            script,
            repo_id="test-user/campaign-1-output",
            config=ready_config(),
        )
    assert api.calls == []


def test_launch_uses_run_uv_job_and_returns_provider_identity(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('train')\n", encoding="utf-8")
    api = RecordingJobsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)
    config = ready_config()

    job = runner.submit_training_job(
        script,
        repo_id="test-user/campaign-1-output",
        config=config,
        script_args=["--seed", "7"],
    )

    assert job.job_id == "provider-job-1"
    assert job.logs_url == "https://huggingface.co/jobs/test-user/provider-job-1"
    assert job.status is JobStatus.RUNNING
    assert api.calls == [
        (
            "run_uv_job",
            {
                "script": str(script.resolve()),
                "script_args": ["--seed", "7"],
                "dependencies": ["transformers", "trl"],
                "image": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
                "env": {
                    "BASHGYM_CAMPAIGN_ID": "campaign-1",
                    "BASHGYM_DATASET_REPO": "test-user/campaign-1-data",
                    "BASHGYM_OUTPUT_REPO": "test-user/campaign-1-output",
                },
                "secrets": {"HF_TOKEN": "injected-at-launch"},
                "flavor": "a10g-small",
                "timeout": "60m",
            },
        )
    ]


def test_observation_methods_use_canonical_jobs_api():
    api = RecordingJobsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)

    status = runner.get_job_status("provider-job-1")
    logs = runner.get_job_logs("provider-job-2", tail=2)
    jobs = runner.list_jobs(status=JobStatus.COMPLETED, limit=1)
    cancelled = runner.cancel_job("provider-job-1")

    assert status.status is JobStatus.RUNNING
    assert logs == "step=1 loss=0.4\nstep=2 loss=0.2\n"
    assert [job.job_id for job in jobs] == ["provider-job-2"]
    assert cancelled.status is JobStatus.CANCELLED
    assert [call[0] for call in api.calls] == [
        "inspect_job",
        "inspect_job",
        "fetch_job_logs",
        "list_jobs",
        "inspect_job",
        "cancel_job",
        "inspect_job",
    ]


def test_active_job_logs_refuse_before_consuming_nonterminating_iterator():
    """Removing the status gate would block forever while joining an active log stream."""

    api = ActiveJobWithNonTerminatingLogsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)
    observed: list[BaseException] = []

    def retrieve_logs() -> None:
        try:
            runner.get_job_logs("provider-job-1")
        except BaseException as error:  # pragma: no branch - result asserted below
            observed.append(error)

    thread = threading.Thread(target=retrieve_logs, daemon=True)
    thread.start()
    thread.join(timeout=0.5)
    try:
        assert not thread.is_alive(), "active log retrieval consumed a nonterminating iterator"
        assert len(observed) == 1
        error = observed[0]
        assert isinstance(error, jobs_module.HFJobLogsNotReadyError)
        assert error.job_id == "provider-job-1"
        assert error.status is JobStatus.RUNNING
        assert error.reason_code == "hf_job_logs_not_ready"
        assert api.calls == [("inspect_job", {"job_id": "provider-job-1"})]
        assert not api.logs.iteration_started.is_set()
    finally:
        api.logs.release.set()
        thread.join(timeout=2)


def test_logs_support_minimum_provider_signature_and_apply_tail_locally():
    api = MinimumJobsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)

    logs = runner.get_job_logs("provider-job-1", tail=1, namespace="research-org")

    assert logs == "step=2 loss=0.2\n"
    assert api.calls == [
        (
            "inspect_job",
            {"job_id": "provider-job-1", "namespace": "research-org"},
        ),
        (
            "fetch_job_logs",
            {"job_id": "provider-job-1", "namespace": "research-org", "token": None},
        ),
    ]


def test_terminal_job_logs_are_bounded_by_server_tail_and_byte_caps():
    api = LargeTerminalLogsApi()
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)

    logs = runner.get_job_logs(
        "provider-job-1",
        tail=EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES + 100,
    )

    assert len(logs.encode("utf-8")) <= EXPECTED_HF_JOB_LOG_MAX_BYTES
    assert len(logs.splitlines()) <= EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES
    assert logs.endswith(
        f"line={EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES + 99:05d} " + ("x" * 1024) + "\n"
    )
    assert api.calls == [
        ("inspect_job", {"job_id": "provider-job-1"}),
        (
            "fetch_job_logs",
            {
                "job_id": "provider-job-1",
                "tail": EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES,
                "follow": False,
            },
        ),
    ]


def test_org_namespace_is_passed_through_every_job_lifecycle_call():
    api = RecordingJobsApi()
    api.jobs["provider-job-1"] = provider_job(
        "provider-job-1",
        "RUNNING",
        namespace="research-org",
    )
    api.jobs["provider-job-2"] = provider_job(
        "provider-job-2",
        "COMPLETED",
        namespace="research-org",
    )
    runner = HFJobRunner(api=api, jobs_access_confirmed=True)

    status = runner.get_job_status("provider-job-1", namespace="research-org")
    runner.get_job_logs("provider-job-2", namespace="research-org")
    runner.list_jobs(namespace="research-org")
    cancelled = runner.cancel_job("provider-job-1", namespace="research-org")

    assert status.namespace == "research-org"
    assert cancelled.namespace == "research-org"
    assert [call for call in api.calls if call[0] != "fetch_job_logs"] == [
        ("inspect_job", {"job_id": "provider-job-1", "namespace": "research-org"}),
        ("inspect_job", {"job_id": "provider-job-2", "namespace": "research-org"}),
        ("list_jobs", {"namespace": "research-org"}),
        ("inspect_job", {"job_id": "provider-job-1", "namespace": "research-org"}),
        ("cancel_job", {"job_id": "provider-job-1", "namespace": "research-org"}),
        ("inspect_job", {"job_id": "provider-job-1", "namespace": "research-org"}),
    ]
    assert api.calls[2] == (
        "fetch_job_logs",
        {"job_id": "provider-job-2", "namespace": "research-org"},
    )


def test_only_provider_404_is_reported_as_job_not_found():
    class FailingJobsApi:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def inspect_job(self, **_kwargs: Any) -> Any:
            raise ProviderRequestError(self.status_code)

    missing = HFJobRunner(api=FailingJobsApi(404), jobs_access_confirmed=True)
    unavailable = HFJobRunner(api=FailingJobsApi(503), jobs_access_confirmed=True)

    with pytest.raises(KeyError, match="provider-job-1"):
        missing.get_job_status("provider-job-1")
    with pytest.raises(HFJobFailedError, match="status request failed"):
        unavailable.get_job_status("provider-job-1")
