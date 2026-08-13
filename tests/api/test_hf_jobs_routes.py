import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.integrations.huggingface.jobs import (
    HFJobInfo,
    HFJobsAvailability,
    JobStatus,
)

EXPECTED_HF_JOB_LOG_DEFAULT_TAIL_LINES = 200
EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES = 2_000

_HF_ROUTES_PATH = Path(__file__).parents[2] / "bashgym" / "api" / "hf_routes.py"
_HF_ROUTES_SPEC = importlib.util.spec_from_file_location(
    "_bashgym_hf_routes_under_test", _HF_ROUTES_PATH
)
assert _HF_ROUTES_SPEC and _HF_ROUTES_SPEC.loader
hf_routes = importlib.util.module_from_spec(_HF_ROUTES_SPEC)
sys.modules[_HF_ROUTES_SPEC.name] = hf_routes
_HF_ROUTES_SPEC.loader.exec_module(hf_routes)

app = FastAPI()
app.include_router(hf_routes.router)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/api/hf/jobs",
        "/api/hf/jobs/provider-job-123",
        "/api/hf/jobs/provider-job-123/logs",
    ],
)
def test_job_observation_requires_explicit_jobs_access_confirmation(client, monkeypatch, path):
    provider_calls = []
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda: provider_calls.append("client") or SimpleNamespace(),
    )

    response = client.get(path)

    assert response.status_code == 412
    assert response.json()["detail"]["reason_code"] == "jobs_access_not_confirmed"
    assert provider_calls == []


def test_job_list_uses_confirmed_adapter_and_returns_provider_projection(client, monkeypatch):
    runner_calls = []

    class ConfirmedClient:
        def require_pro(self):
            raise AssertionError(
                "explicit Jobs access confirmation must not reuse legacy Pro inference"
            )

    class ReadOnlyRunner:
        def list_jobs(self, *, namespace=None):
            assert namespace is None
            return [
                HFJobInfo(
                    job_id="provider-job-123",
                    status=JobStatus.RUNNING,
                    hardware="provider-flavor",
                    created_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
                    logs_url="https://huggingface.co/jobs/provider-job-123",
                    namespace="research-org",
                )
            ]

    confirmed_client = ConfirmedClient()
    monkeypatch.setattr("bashgym.integrations.huggingface.get_hf_client", lambda: confirmed_client)

    def create_runner(received_client, *, jobs_access_confirmed=False, **_kwargs):
        runner_calls.append((received_client, jobs_access_confirmed))
        return ReadOnlyRunner()

    monkeypatch.setattr("bashgym.integrations.huggingface.jobs.create_job_runner", create_runner)

    response = client.get("/api/hf/jobs", params={"jobs_access_confirmed": "true"})

    assert response.status_code == 200
    assert response.json() == [
        {
            "job_id": "provider-job-123",
            "status": "running",
            "hardware": "provider-flavor",
            "created_at": "2026-08-11T00:00:00+00:00",
            "logs_url": "https://huggingface.co/jobs/provider-job-123",
            "error_message": None,
            "namespace": "research-org",
        }
    ]
    assert runner_calls == [(confirmed_client, True)]


def test_job_observation_passes_org_namespace_to_provider_adapter(client, monkeypatch):
    observed: list[tuple[str, str, str | None]] = []

    class ReadOnlyRunner:
        def get_job_status(self, job_id, *, namespace=None):
            observed.append(("status", job_id, namespace))
            return HFJobInfo(
                job_id=job_id,
                status=JobStatus.RUNNING,
                hardware="provider-flavor",
                created_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
                namespace=namespace,
            )

        def get_job_logs(self, job_id, tail=None, *, namespace=None):
            observed.append(("logs", job_id, namespace, tail))
            return "training\n"

    monkeypatch.setattr(hf_routes, "_confirmed_job_runner", lambda: ReadOnlyRunner())
    params = {"jobs_access_confirmed": "true", "namespace": "research-org"}

    status = client.get("/api/hf/jobs/provider-job-123", params=params)
    logs = client.get("/api/hf/jobs/provider-job-123/logs", params=params)

    assert status.status_code == 200
    assert status.json()["namespace"] == "research-org"
    assert logs.status_code == 200
    assert logs.json() == {"logs": "training\n"}
    assert observed == [
        ("status", "provider-job-123", "research-org"),
        (
            "logs",
            "provider-job-123",
            "research-org",
            EXPECTED_HF_JOB_LOG_DEFAULT_TAIL_LINES,
        ),
    ]


def test_job_logs_route_rejects_tail_above_server_cap(client, monkeypatch):
    provider_calls: list[str] = []
    monkeypatch.setattr(
        hf_routes,
        "_confirmed_job_runner",
        lambda: provider_calls.append("runner") or SimpleNamespace(),
    )

    response = client.get(
        "/api/hf/jobs/provider-job-123/logs",
        params={
            "jobs_access_confirmed": "true",
            "tail": EXPECTED_HF_JOB_LOG_MAX_TAIL_LINES + 1,
        },
    )

    assert response.status_code == 422
    assert provider_calls == []


def test_provider_outage_is_not_misreported_as_missing_job(client, monkeypatch):
    from bashgym.integrations.huggingface.client import HFJobFailedError

    class UnavailableRunner:
        def get_job_status(self, _job_id, *, namespace=None):
            del namespace
            raise HFJobFailedError("provider unavailable")

    monkeypatch.setattr(hf_routes, "_confirmed_job_runner", lambda: UnavailableRunner())

    response = client.get(
        "/api/hf/jobs/provider-job-123",
        params={"jobs_access_confirmed": "true"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "provider unavailable"


@pytest.mark.parametrize(
    ("method", "path", "json_body", "reason_code"),
    [
        (
            "post",
            "/api/hf/jobs",
            {
                "dataset_repo": "example/data",
                "output_repo": "example/model",
                "base_model": "example/base",
            },
            "hf_job_submission_unavailable",
        ),
        (
            "delete",
            "/api/hf/jobs/provider-job-123",
            None,
            "hf_job_cancellation_unavailable",
        ),
    ],
)
def test_direct_job_mutations_are_blocked_before_provider_resolution(
    client, monkeypatch, method, path, json_body, reason_code
):
    provider_calls = []
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda: provider_calls.append("client") or SimpleNamespace(),
    )

    response = client.request(method, path, json=json_body)

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason_code"] == reason_code
    assert detail["direct_mutations_enabled"] is False
    assert "launch_authority" not in detail
    assert "campaign" not in detail["message"].lower()
    assert provider_calls == []


def test_direct_job_mutation_openapi_contract_only_advertises_unavailable_response(client):
    paths = client.get("/openapi.json").json()["paths"]

    for path, method in (
        ("/api/hf/jobs", "post"),
        ("/api/hf/jobs/{job_id}", "delete"),
    ):
        responses = paths[path][method]["responses"]
        assert "409" in responses
        assert "200" not in responses


def test_jobs_capabilities_are_local_only_and_do_not_invent_launch_authority(client, monkeypatch):
    provider_calls = []
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.get_hf_client",
        lambda: provider_calls.append("client") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.jobs.detect_hf_jobs_availability",
        lambda: HFJobsAvailability(
            dependency_available=True,
            api_available=True,
            provider_version="1.16.1",
            api_method="HfApi.run_uv_job",
            hardware_flavors=("provider-flavor",),
        ),
    )

    response = client.get("/api/hf/jobs/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "bashgym.hf_jobs_capabilities.v1",
        "dependency_available": True,
        "api_available": True,
        "provider_version": "1.16.1",
        "api_method": "HfApi.run_uv_job",
        "hardware_flavors": ["provider-flavor"],
        "direct_mutations_enabled": False,
    }
    assert provider_calls == []


def test_hardware_route_returns_provider_flavors_without_pricing_or_capacity_claims(
    client, monkeypatch
):
    monkeypatch.setattr(
        "bashgym.integrations.huggingface.jobs.HARDWARE_SPECS",
        {
            "provider-flavor": {
                "provider_value": "provider-flavor",
                "source": "huggingface_hub.SpaceHardware",
                "gpu": "unverified-gpu",
                "vram_gb": 999,
                "cost_per_hour": 42,
                "pro_required": True,
            }
        },
    )

    response = client.get("/api/hf/jobs/hardware")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "provider-flavor",
            "provider_value": "provider-flavor",
            "source": "huggingface_hub.SpaceHardware",
        }
    ]
