"""Live HTTP proof for the campaign CLI and launch-scoped MCP surface."""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import pytest
import uvicorn
from fastapi import FastAPI

from bashgym.api.campaign_routes import campaign_auth_router, campaign_router
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.client import CampaignApiClient
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    CampaignTrigger,
    CredentialKind,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import CampaignWorker
from bashgym.cli import main as cli_main
from bashgym.ledger.contracts import (
    ArtifactSpec,
    DecisionSpec,
    EvaluationResultSpec,
    EvaluationSuiteSpec,
    ExperimentSpec,
    ProjectSpec,
    RunSpec,
    RunStatus,
)
from bashgym.ledger.persistence import ExperimentLedgerRepository
from bashgym.mcp.campaign_server import build_server
from tests.campaigns.test_dry_campaign import dry_proposal
from tests.campaigns.test_persistence import campaign, create, manifest
from tests.campaigns.test_proposals import principal
from tests.campaigns.test_worker import START


@contextmanager
def live_campaign_api(repository: CampaignRuntimeRepository):
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.campaign_auth_service = CampaignAuthService(repository)
    app.state.campaign_service = CampaignService(repository)
    app.include_router(campaign_auth_router)
    app.include_router(campaign_router)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        raise RuntimeError("live campaign API did not start")
    try:
        yield f"http://127.0.0.1:{port}/api"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


@pytest.mark.asyncio
async def test_cli_and_mcp_use_the_authenticated_live_http_contract(tmp_path, monkeypatch, capsys):
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    create(repository)
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="surface-proof",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    credential_ref = "BASHGYM_TEST_CAMPAIGN_REFRESH"
    monkeypatch.setenv(credential_ref, refresh.raw_token)

    with live_campaign_api(repository) as api_base:
        exit_code = cli_main(
            [
                "campaign",
                "studies",
                "--workspace-id",
                "workspace-a",
                "--api-base",
                api_base,
                "--credential-ref",
                credential_ref,
                "--campaign",
                "campaign-1",
                "--json",
            ]
        )
        assert exit_code == 0
        assert json.loads(capsys.readouterr().out)["studies"] == []

        mcp = build_server(
            workspace_id="workspace-a",
            credential_ref=credential_ref,
            agent="codex",
            api_base=api_base,
        )
        result = await mcp._tool_manager.call_tool(
            "campaign_studies", {"campaign_id": "campaign-1"}, convert_result=False
        )
        assert result == {
            "ok": True,
            "studies": [],
            "count": 0,
            "truncated": False,
        }


@pytest.mark.asyncio
async def test_restart_safe_dry_campaign_is_consistent_across_rest_cli_and_mcp(
    tmp_path, monkeypatch, capsys
):
    database = tmp_path / "campaigns.sqlite3"
    repository = CampaignRuntimeRepository(database)
    repository.initialize()
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="surface-restart-proof",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    credential_ref = "BASHGYM_TEST_CAMPAIGN_RESTART_REFRESH"
    monkeypatch.setenv(credential_ref, refresh.raw_token)

    with live_campaign_api(repository) as api_base:
        rest = CampaignApiClient(api_base=api_base, credential_ref=credential_ref)
        fixture_campaign = campaign()
        created_mutation = rest.request_json(
            "POST",
            "/campaigns",
            payload={
                "workspace_id": "workspace-a",
                "campaign_id": "campaign-1",
                "title": fixture_campaign.title,
                "kind": fixture_campaign.kind.value,
                "objective": fixture_campaign.objective,
                "target_model": fixture_campaign.target_model.model_dump(mode="json"),
                "manifest": manifest().model_dump(mode="json"),
            },
            headers={
                "Idempotency-Key": "surface-rest-create",
                "X-Correlation-ID": "surface-dry-proof",
            },
        )
        assert created_mutation["event"]["event_type"] == "campaign:created"

        version = 1
        for trigger, key in (
            (CampaignTrigger.VALIDATE, "surface-validate"),
            (CampaignTrigger.VALIDATION_PASSED, "surface-ready"),
            (CampaignTrigger.START, "surface-start"),
        ):
            transitioned = repository.transition_campaign(
                "workspace-a",
                "campaign-1",
                trigger,
                expected_version=version,
                actor_id="campaign-controller",
                credential_kind=CredentialKind.CONTROLLER,
                correlation_id=key,
                idempotency_key=key,
            )
            version = transitioned.campaign.version

        service = CampaignService(repository)
        actor = principal(repository)
        for value in (
            dry_proposal("surface-high-cheap", priority=90, cost=1),
            dry_proposal("surface-high-costly", priority=90, cost=2),
            dry_proposal("surface-lower", priority=50, cost=1),
        ):
            submitted = service.submit_proposal(
                value,
                expected_version=version,
                principal=actor,
                correlation_id=f"surface-submit-{value.proposal_id}",
                idempotency_key=f"surface-submit-{value.proposal_id}",
            )
            version = submitted.campaign.version

        created = rest.request_json(
            "GET",
            "/campaigns/campaign-1",
            query={"workspace_id": "workspace-a"},
        )
        assert created["campaign_id"] == "campaign-1"
        assert created["status"] == "active"
        assert created["version"] == version

        connection = [
            "--workspace-id",
            "workspace-a",
            "--api-base",
            api_base,
            "--credential-ref",
            credential_ref,
            "--json",
        ]
        assert (
            cli_main(
                [
                    "campaign",
                    "advance",
                    *connection,
                    "--campaign",
                    "campaign-1",
                    "--expected-version",
                    str(version),
                    "--idempotency-key",
                    "surface-dry-advance",
                    "--correlation-id",
                    "surface-dry-proof",
                ]
            )
            == 0
        )
        advanced = json.loads(capsys.readouterr().out)
        assert advanced["event"]["event_type"] == "campaign:advance-requested"

        results = []
        for index in range(6):
            reopened = CampaignRuntimeRepository(database)
            reopened.initialize()
            worker = CampaignWorker(
                reopened,
                tmp_path / "artifacts",
                ArtifactSealer(b"s" * 32, key_version="surface-proof-v1"),
                data_directory=tmp_path / "data-root",
                worker_id=f"surface-worker-restart-{index}",
            )
            results.append(worker.run_once(now=START + timedelta(seconds=20 * (index + 1))))
        assert results == ["completed"] * 6

        rest_studies = rest.request_json(
            "GET",
            "/campaigns/campaign-1/studies",
            query={"workspace_id": "workspace-a"},
        )["studies"]
        rest_attempts = rest.request_json(
            "GET",
            "/campaigns/campaign-1/attempts",
            query={"workspace_id": "workspace-a"},
        )["attempts"]
        assert len(rest_studies) == 3
        assert len(rest_attempts) == 6
        assert {item["status"] for item in rest_studies} == {"completed"}
        assert {item["status"] for item in rest_attempts} == {"completed"}
        assert all("sealed_result_uri" not in item for item in rest_attempts)
        study_ids = [item["study_id"] for item in rest_studies]
        attempt_ids = [item["attempt_id"] for item in rest_attempts]

        ledger = ExperimentLedgerRepository(database)
        ledger.initialize()
        ledger.register_project(
            ProjectSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                display_name="Dry campaign proof",
                owner_actor_id="surface-proof",
            )
        )
        ledger.register_experiment(
            ExperimentSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                experiment_id="surface-experiment",
                campaign_id="campaign-1",
                name="Restart-safe orchestration",
                objective="Verify a bounded campaign without launching compute.",
            )
        )
        ledger.register_run(
            RunSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                experiment_id="surface-experiment",
                run_id="surface-run",
                source_system="bashgym-campaign-smoke",
                source_run_id=attempt_ids[0],
                campaign_id="campaign-1",
                study_id=study_ids[0],
                run_kind="training",
                task_type="orchestration-smoke",
                training_method="fake",
                status=RunStatus.COMPLETED,
                recipe_digest="a" * 64,
                correlation_id="surface-dry-proof",
                is_simulation=True,
            )
        )
        ledger.register_evaluation_suite(
            EvaluationSuiteSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                evaluation_suite_id="surface-suite",
                name="Dry orchestration gate",
                task_type="orchestration-smoke",
                metric_contract={"completed_iterations": {"direction": "higher"}},
                code_digest="b" * 64,
            )
        )
        ledger.record_evaluation_result(
            EvaluationResultSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                evaluation_result_id="surface-eval",
                evaluation_suite_id="surface-suite",
                run_id="surface-run",
                status=RunStatus.COMPLETED,
                metrics={"completed_iterations": 6.0},
            )
        )
        ledger.record_artifact(
            ArtifactSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                artifact_id="surface-report",
                run_id="surface-run",
                kind="report",
                uri="file://private/dry-campaign-report.json",
                sha256="c" * 64,
                size_bytes=42,
                media_type="application/json",
            )
        )
        ledger.record_decision(
            DecisionSpec(
                workspace_id="workspace-a",
                project_id="surface-project",
                decision_id="surface-decision",
                experiment_id="surface-experiment",
                run_id="surface-run",
                decision_type="smoke-gate",
                outcome="Dry campaign orchestration passed.",
                rationale="All fake stages completed and emitted loss metrics.",
                evidence_refs=("surface-eval",),
                actor_id="surface-proof",
            )
        )

        rest_ledger = rest.request_json(
            "GET",
            "/campaigns/campaign-1/ledger",
            query={"workspace_id": "workspace-a"},
        )
        assert rest_ledger["linked"] is True
        assert rest_ledger["projects"][0]["evidence"] == {
            "experiment_ids": ["surface-experiment"],
            "run_ids": ["surface-run"],
            "evaluation_result_ids": ["surface-eval"],
            "artifact_ids": ["surface-report"],
            "decision_ids": ["surface-decision"],
        }
        assert "uri" not in rest_ledger["projects"][0]["artifacts"][0]

        assert (
            cli_main(
                [
                    "campaign",
                    "ledger",
                    *connection,
                    "--campaign",
                    "campaign-1",
                ]
            )
            == 0
        )
        cli_ledger = json.loads(capsys.readouterr().out)
        assert cli_ledger["projects"][0]["evidence"] == rest_ledger["projects"][0]["evidence"]

        first_attempt_id = attempt_ids[0]
        rest_metrics = rest.request_json(
            "GET",
            f"/campaigns/campaign-1/attempts/{first_attempt_id}/metrics",
            query={
                "workspace_id": "workspace-a",
                "source": "training_metrics.jsonl",
                "metric_name": "loss",
            },
        )
        assert rest_metrics["values"]

        assert (
            cli_main(
                [
                    "campaign",
                    "attempts",
                    *connection,
                    "--campaign",
                    "campaign-1",
                ]
            )
            == 0
        )
        cli_attempts = json.loads(capsys.readouterr().out)["attempts"]
        assert [item["attempt_id"] for item in cli_attempts] == attempt_ids

        assert (
            cli_main(
                [
                    "campaign",
                    "studies",
                    *connection,
                    "--campaign",
                    "campaign-1",
                ]
            )
            == 0
        )
        cli_studies = json.loads(capsys.readouterr().out)["studies"]
        assert [item["study_id"] for item in cli_studies] == study_ids

        assert (
            cli_main(
                [
                    "campaign",
                    "metrics",
                    *connection,
                    "--campaign",
                    "campaign-1",
                    "--attempt",
                    first_attempt_id,
                    "--source",
                    "training_metrics.jsonl",
                    "--metric",
                    "loss",
                ]
            )
            == 0
        )
        cli_metrics = json.loads(capsys.readouterr().out)
        assert cli_metrics["values"] == rest_metrics["values"]

        mcp = build_server(
            workspace_id="workspace-a",
            credential_ref=credential_ref,
            agent="codex",
            api_base=api_base,
        )
        mcp_campaign = await mcp._tool_manager.call_tool(
            "campaign_inspect", {"campaign_id": "campaign-1"}, convert_result=False
        )
        mcp_studies = await mcp._tool_manager.call_tool(
            "campaign_studies", {"campaign_id": "campaign-1"}, convert_result=False
        )
        mcp_attempts = await mcp._tool_manager.call_tool(
            "campaign_attempts", {"campaign_id": "campaign-1"}, convert_result=False
        )
        mcp_metrics = await mcp._tool_manager.call_tool(
            "campaign_metrics",
            {
                "campaign_id": "campaign-1",
                "attempt_id": first_attempt_id,
                "source": "training_metrics.jsonl",
                "metric_name": "loss",
            },
            convert_result=False,
        )
        assert mcp_campaign["campaign"]["campaign_id"] == "campaign-1"
        assert [item["study_id"] for item in mcp_studies["studies"]] == study_ids
        assert [item["attempt_id"] for item in mcp_attempts["attempts"]] == attempt_ids
        assert mcp_metrics["values"] == rest_metrics["values"]
