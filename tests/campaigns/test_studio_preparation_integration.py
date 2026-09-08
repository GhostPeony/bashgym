"""Concrete registered preparation to READY, with fixture transport boundaries.

No SSH, model acquisition, training, or OS service operation runs. The test uses
real local activation, registry sync, API authentication, setup validation/create,
and persisted onboarding/session/campaign records.
"""

from __future__ import annotations

import io
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.campaigns import onboarding, remote, worker_service
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.autoresearch import AutoResearchRepository
from bashgym.campaigns.contracts import CampaignStatus
from bashgym.campaigns.studio_preparation import build_registered_preparation
from bashgym.campaigns.worker import scheduler_lease_key
from bashgym.config import state_root_digest
from tests.campaigns.test_studio_preparation import _registered


def test_registered_preparation_real_local_services_and_api_reaches_ready_then_resumes(
    tmp_path,
    monkeypatch,
):
    from bashgym.api.campaign_routes import campaign_auth_router, campaign_router
    from bashgym.api.campaign_setup_routes import campaign_setup_router

    original, definition, activation, kwargs, secrets = _registered(tmp_path)
    monkeypatch.setenv("BASHGYM_DIR", str(tmp_path))
    monkeypatch.setattr("bashgym.secrets.get_secret", secrets.get)
    monkeypatch.setattr("bashgym.secrets.set_secret", secrets.__setitem__)
    compiled = build_registered_preparation(tmp_path, **kwargs)
    compiled.write_inputs()
    contract = compiled.contract
    model_request = remote.RemoteModelRegistrationRequest.model_validate_json(
        contract.model_request_file.read_text()
    )
    assert model_request.operation == "register"
    expected_model_command = remote._remote_model_registration_command(model_request)
    database = tmp_path / "campaigns" / "campaigns.sqlite3"
    repository = AutoResearchRepository(database)
    repository.initialize()
    approved = activation.executor_profile.registered_base_model
    heldout = activation.executor_profile.registered_evaluation_dataset
    assert approved is not None and approved.artifact_receipt is not None
    assert heldout is not None
    transport_calls = []
    supervisor_calls = []
    api_calls = []

    class Session:
        async def run(self, command, **options):
            transport_calls.append(command)
            if command == 'printf %s "$HOME"':
                output = "/fixture-home"
            elif "MemAvailable:" in command:
                output = "256\t2048\t\n"
            elif "sha256sum" in command and heldout.remote_dataset_path in command:
                assert heldout.content_digest in command
                output = ""
            elif command == expected_model_command:
                # Return the exact fixture's existing model receipt. Real parser
                # and registered-source comparison still execute in production.
                assert "snapshot_download" not in command
                output = approved.artifact_receipt.model_dump_json()
            else:
                raise AssertionError(f"unexpected transport command: {command}")
            return SimpleNamespace(exit_status=0, stdout=output, stderr="")

    @asynccontextmanager
    async def session(adapter):
        assert adapter.compute_profile_id == activation.executor_profile.compute_profile_id
        yield Session()

    monkeypatch.setattr(remote.RemoteTrainingAdapter, "_session", session)

    # Keep real service-definition construction inside the fixture home. Only
    # supervisor operations and the loopback HTTP health transport are replaced.
    original_api_definition = worker_service.build_api_service_definition
    original_worker_definition = worker_service.build_service_definition
    monkeypatch.setattr(
        worker_service,
        "build_api_service_definition",
        lambda **kw: original_api_definition(home=tmp_path / "service-home", **kw),
    )
    monkeypatch.setattr(
        worker_service,
        "build_service_definition",
        lambda *args, **kw: original_worker_definition(*args, home=tmp_path / "service-home", **kw),
    )

    class Manager:
        def install(self, service_definition, config=None):
            supervisor_calls.append("worker" if config is not None else "api")
            assert service_definition.definition_path.is_relative_to(tmp_path)
            service_definition.definition_path.parent.mkdir(parents=True, exist_ok=True)
            service_definition.definition_path.write_bytes(service_definition.definition_payload)
            if config is not None:
                # Simulate the resident worker's lease heartbeat, without
                # starting a scheduler or executing a campaign attempt.
                repository.acquire_lease(
                    scheduler_lease_key(tmp_path),
                    contract.controller_owner_id,
                    ttl=timedelta(minutes=10),
                )

        replace = install

        def status(self, *args):
            return {"supervisor_state": "available"}

    monkeypatch.setattr(worker_service, "ApiServiceManager", Manager)
    monkeypatch.setattr(worker_service, "WorkerServiceManager", Manager)

    class HealthResponse(io.BytesIO):
        status = 200

    def health(url, **options):
        assert url == str(contract.api_base).rstrip("/") + "/health"
        return HealthResponse(
            json.dumps({"state_root_digest": state_root_digest(tmp_path)}).encode()
        )

    monkeypatch.setattr(onboarding.urllib.request, "urlopen", health)

    def application():
        api_repository = AutoResearchRepository(database)
        api_repository.initialize()
        app = FastAPI()
        app.state.campaign_repository = api_repository
        app.state.campaign_auth_service = CampaignAuthService(api_repository)
        app.state.campaign_worker_config_path = tmp_path / "campaigns" / "worker-config.v1.json"
        app.state.campaign_autoresearch_template_directory = (
            tmp_path / "campaigns" / "autoresearch-templates"
        )
        app.state.campaign_authority_seal_key = secrets["BASHGYM_CAMPAIGN_SEAL_KEY"].encode()
        app.include_router(campaign_auth_router)
        app.include_router(campaign_router)
        app.include_router(campaign_setup_router)
        return app

    with TestClient(application()) as http:
        exchanged = http.post(
            "/api/campaign-auth/exchange",
            headers={"Authorization": "Bearer " + secrets[contract.credential_ref]},
        )
        assert exchanged.status_code == 200, exchanged.text
        access = exchanged.json()["raw_token"]

        class ApiAdapter:
            def request_json(self, method, path, *, query=None, payload=None, headers=None):
                api_calls.append((method, path, payload))
                assert "/start" not in path
                response = http.request(
                    method,
                    "/api" + path,
                    params=query,
                    json=payload,
                    headers={"Authorization": "Bearer " + access, **(headers or {})},
                )
                assert response.status_code == 200, (path, response.status_code, response.text)
                return response.json()

        monkeypatch.setattr(onboarding, "_campaign_client", lambda _: ApiAdapter())
        services = onboarding.LocalAutoResearchOnboardingServices(contract)
        receipt = onboarding.AutoResearchOnboardingCoordinator(services).apply(contract)
        assert receipt.applied is True
        assert receipt.campaign_status == "ready"
        assert receipt.next_action == "explicit_start_confirmation_required"
        assert tuple(item.step for item in receipt.completed_steps) == onboarding.ONBOARDING_STEPS
        context = services.client.request_json(
            "GET", "/campaigns/setup/context", query={"workspace_id": contract.workspace_id}
        )
        assert context["session"]["session_id"] == kwargs["session_id"]
        assert context["session"]["version"] == 6
        assert context["session"]["completed_steps"] == [
            "template",
            "installation",
            "model",
            "data",
            "compute",
            "evaluation",
        ]
        assert context["session"]["ready_for_validation"] is True
        assert context["session"]["latest_receipt"]["actor_id"] == "studio-codex"
        assert (
            repository.get_campaign(contract.workspace_id, contract.campaign_id).status
            == CampaignStatus.READY
        )
        first_calls = list(api_calls)
        assert [
            payload["step"]
            for method, path, payload in first_calls
            if path == "/campaigns/setup/session"
        ] == ["model", "data", "compute", "evaluation"]
        assert any(path == "/campaigns/setup/validate" for _, path, _ in first_calls)
        assert any(path == "/campaigns/setup/create" for _, path, _ in first_calls)

        # A new concrete service/coordinator reopens durable receipt state. It
        # rechecks the fixture's model/data evidence but does not recreate READY.
        with TestClient(application()) as restarted_http:
            # The adapter uses a new API application with no cached setup,
            # service, or auth instances; the same token is resolved from disk.
            http = restarted_http
            restarted = onboarding.LocalAutoResearchOnboardingServices(contract)
            replay = onboarding.AutoResearchOnboardingCoordinator(restarted).apply(contract)
            assert replay.replayed is True
            assert replay.campaign_status == "ready"
            assert replay.next_action == "explicit_start_confirmation_required"
            assert all(method == "GET" for method, _, _ in api_calls[len(first_calls) :])
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM campaign_attempts").fetchone()[0] == 0
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM campaign_guided_setup_step_receipts"
                ).fetchone()[0]
                == 6
            )
        assert supervisor_calls == ["api", "worker"]
        assert len([command for command in transport_calls if "sha256sum" in command]) == 2
        assert original.installation_id == contract.installation_id
        assert definition.target_model == services.definition.target_model
