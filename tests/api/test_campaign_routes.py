"""Fail-closed campaign REST authentication, authority, and projection tests."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bashgym.api import campaign_routes
from bashgym.api.campaign_routes import (
    CampaignTemplate,
    campaign_auth_router,
    campaign_router,
)
from bashgym.api.routes import create_app
from bashgym.campaigns.auth import CampaignAuthService
from bashgym.campaigns.autoresearch import (
    AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
    AutoResearchCampaignSpec,
    AutoResearchRepository,
    AutoResearchStopRules,
    AutoResearchTemplateDefinition,
    MetricDirection,
)
from bashgym.campaigns.contracts import (
    AutonomyProfile,
    CampaignTrigger,
    CodeLineageRecord,
    CodeLineageState,
    CodeMutationKind,
    CredentialKind,
    canonical_hash,
)
from bashgym.campaigns.runtime import CampaignRuntimeRepository
from bashgym.campaigns.service import CampaignService
from bashgym.campaigns.worker import scheduler_lease_key
from bashgym.campaigns.worker_service import (
    ControllerStatusProjection,
    DesktopWorkerStatusProjection,
)
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
from tests.campaigns.test_autoresearch_readiness import definition as readiness_definition
from tests.campaigns.test_lineage import initialized_repository, source_profile
from tests.campaigns.test_persistence import campaign, manifest
from tests.campaigns.test_proposals import proposal as study_proposal
from tests.campaigns.test_remote_persistence import _claimed_attempt
from tests.campaigns.test_worker import START


def campaign_client(tmp_path, *, profile=AutonomyProfile.CODEX_TRUSTED):
    repository = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    repository.initialize()
    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="hermes-agent" if profile == AutonomyProfile.HERMES_BOUNDED else "codex-agent",
        autonomy_profile=profile,
        workspace_ids=("workspace-a",),
    )
    template = CampaignTemplate(
        kind=campaign().kind,
        objective=campaign().objective,
        target_model=campaign().target_model,
        manifest=manifest(),
    )
    service = CampaignService(
        repository,
        approved_template_hashes={
            "memexai-approved-v1": canonical_hash(manifest().model_dump(mode="json"))
        },
    )
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.campaign_auth_service = auth
    app.state.campaign_service = service
    app.state.campaign_templates = {"memexai-approved-v1": template}
    app.state.campaign_worker_config_path = tmp_path / "worker-config.v1.json"
    app.include_router(campaign_auth_router)
    app.include_router(campaign_router)
    return TestClient(app), repository, refresh


def exchange(http: TestClient, refresh_token: str) -> str:
    response = http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {refresh_token}"},
    )
    assert response.status_code == 200
    return response.json()["raw_token"]


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_from_template(http: TestClient, token: str, campaign_id="campaign-1"):
    return http.post(
        "/api/campaigns/from-template",
        headers={**bearer(token), "Idempotency-Key": f"create-{campaign_id}"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": campaign_id,
            "title": "MemexAI embedding campaign",
            "template_id": "memexai-approved-v1",
        },
    )


def test_create_app_registers_campaign_auth_and_campaign_routes():
    route_paths = {route.path for route in create_app().routes}

    assert {
        "/api/campaign-auth/exchange",
        "/api/campaign-auth/capabilities",
        "/api/campaigns",
        "/api/campaigns/from-template",
        "/api/campaigns/templates",
        "/api/campaigns/templates/{template_id}/doctor",
        "/api/campaigns/{campaign_id}/control-room-snapshot",
        "/api/campaigns/{campaign_id}",
        "/api/campaigns/{campaign_id}/autoresearch",
        "/api/campaigns/{campaign_id}/autoresearch/baseline",
        "/api/campaigns/{campaign_id}/autoresearch/candidates",
        "/api/campaigns/{campaign_id}/autoresearch/results",
        "/api/campaigns/{campaign_id}/autoresearch/ingest-evaluation",
        "/api/campaigns/{campaign_id}/manifest/{revision}",
        "/api/campaigns/{campaign_id}/manifest/revise",
        "/api/campaigns/{campaign_id}/proposals",
        "/api/campaigns/{campaign_id}/studies",
        "/api/campaigns/{campaign_id}/studies/{study_id}",
        "/api/campaigns/{campaign_id}/proposals/{proposal_id}/withdraw",
        "/api/campaigns/{campaign_id}/proposals/{proposal_id}/code-lineage/prepare",
        "/api/campaigns/{campaign_id}/proposals/{proposal_id}/code-lineage/capture",
        "/api/campaigns/{campaign_id}/evidence",
        "/api/campaigns/{campaign_id}/advance",
        "/api/campaigns/{campaign_id}/actions/{action_id}/retry",
        "/api/campaigns/{campaign_id}/actions/{action_id}/force-stop",
        "/api/campaigns/{campaign_id}/studies/{study_id}/abandon",
        "/api/campaigns/{campaign_id}/budget/amend",
        "/api/campaigns/{campaign_id}/sources/{source_id}/approve",
        "/api/campaigns/{campaign_id}/protected-lease",
        "/api/campaigns/{campaign_id}/protected-result",
        "/api/campaigns/{campaign_id}/promotion",
        "/api/campaigns/{campaign_id}/export",
        "/api/campaigns/{campaign_id}/start",
        "/api/campaigns/{campaign_id}/pause",
        "/api/campaigns/{campaign_id}/resume",
        "/api/campaigns/{campaign_id}/cancel",
        "/api/campaigns/{campaign_id}/conclude",
        "/api/campaigns/{campaign_id}/events",
        "/api/campaigns/{campaign_id}/artifacts",
        "/api/campaigns/{campaign_id}/attempts",
        "/api/campaigns/{campaign_id}/comparisons",
        "/api/campaigns/{campaign_id}/ledger",
        "/api/campaigns/{campaign_id}/attempts/{attempt_id}/metrics",
    } <= route_paths


def test_builtin_autoresearch_template_prepares_to_authorized_start_gate(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)

    templates = http.get(
        "/api/campaigns/templates",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert templates.status_code == 200
    built_in = next(
        item
        for item in templates.json()["templates"]
        if item["template_id"] == AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID
    )
    assert built_in["manifest"]["promotion_gates"]["quality_claim_eligible"] is False
    assert "C:\\" not in json.dumps(built_in)

    created = http.post(
        "/api/campaigns/from-template",
        headers={**bearer(access), "Idempotency-Key": "create-autoresearch-smoke"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "autoresearch-smoke-1",
            "title": "AutoResearch control smoke",
            "template_id": AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
        },
    )
    assert created.status_code == 200
    assert created.json()["campaign"]["status"] == "ready"
    assert created.json()["autoresearch"]["next_action"] == "start_campaign"
    assert created.json()["autoresearch"]["baseline_verified"] is False

    doctor = http.get(
        f"/api/campaigns/templates/{AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID}/doctor",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert doctor.status_code == 200
    assert doctor.json()["materializable"] is True
    assert doctor.json()["launch_ready"] is False
    assert doctor.json()["blocking_codes"] == ["controller_offline"]

    state = http.get(
        "/api/campaigns/autoresearch-smoke-1/autoresearch",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert state.status_code == 200
    assert state.json()["spec"]["primary_metric"] == "control_path_score"
    assert state.json()["state"]["next_action"] == "start_campaign"
    assert state.json()["diagnostics"] == {
        "schema_version": "autoresearch_diagnostics.v1",
        "workspace_id": "workspace-a",
        "campaign_id": "autoresearch-smoke-1",
        "primary_metric": "control_path_score",
        "metric_direction": "maximize",
        "low_signal": False,
        "signals": [],
        "checkpoint_comparisons": [],
        "error_slices": [],
        "ranked_hypotheses": [],
    }
    event_types = [
        event.event_type
        for _cursor, event in repository.list_events("workspace-a", "autoresearch-smoke-1")
    ]
    assert event_types[-2:] == ["campaign:validation-started", "campaign:ready"]


def test_installed_real_template_fails_closed_before_campaign_creation(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    installed = readiness_definition()
    http.app.state.campaign_autoresearch_templates = {installed.template_id: installed}

    doctor = http.get(
        f"/api/campaigns/templates/{installed.template_id}/doctor",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert doctor.status_code == 200
    assert doctor.json()["materializable"] is False
    assert doctor.json()["blocking_codes"] == [
        "data_binding_unresolved",
        "evaluator_binding_unresolved",
        "compute_binding_unresolved",
        "source_repository_binding_unresolved",
        "code_lineage_execution_binding_unresolved",
        "controller_offline",
    ]

    created = http.post(
        "/api/campaigns/from-template",
        headers={**bearer(access), "Idempotency-Key": "create-unresolved-real"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "autoresearch-real-1",
            "title": "Installation-bound AutoResearch",
            "template_id": installed.template_id,
        },
    )
    assert created.status_code == 422
    assert "installation_bindings_unresolved" in created.json()["detail"]["message"]
    assert repository.list_campaigns("workspace-a") == []


def test_autoresearch_requires_explicit_role_and_accepts_bounded_baseline(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    created = http.post(
        "/api/campaigns/from-template",
        headers={**bearer(access), "Idempotency-Key": "create-autoresearch-baseline"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "autoresearch-baseline-1",
            "title": "AutoResearch baseline smoke",
            "template_id": AUTORESEARCH_CONTROL_SMOKE_TEMPLATE_ID,
        },
    )
    assert created.status_code == 200
    started = http.post(
        "/api/campaigns/autoresearch-baseline-1/start",
        headers={**bearer(access), "Idempotency-Key": "start-autoresearch-baseline"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": created.json()["campaign"]["version"],
        },
    )
    assert started.status_code == 200

    proposal_body = study_proposal("baseline-smoke", estimated_cost=0.01).model_dump(
        mode="json", exclude={"schema_version", "workspace_id", "campaign_id"}
    )
    proposal_body.update(
        {
            "workspace_id": "workspace-a",
            "expected_version": started.json()["campaign"]["version"],
            "dataset_recipe": {
                "schema_version": "recipe.v1",
                "data_scope_id": "autoresearch-control-smoke",
            },
            "training_recipe": {
                "schema_version": "recipe.v1",
                "runtime": {
                    "executor_kind": "fake",
                    "budget_unit": "gpu_hours",
                    "budget_reservation": 0.01,
                    "fake_steps": 3,
                },
            },
        }
    )
    generic = http.post(
        "/api/campaigns/autoresearch-baseline-1/proposals",
        headers={**bearer(access), "Idempotency-Key": "generic-autoresearch-proposal"},
        json=proposal_body,
    )
    assert generic.status_code == 422
    assert generic.json()["detail"]["code"] == "autoresearch_invariant_failed"

    baseline = http.post(
        "/api/campaigns/autoresearch-baseline-1/autoresearch/baseline",
        headers={**bearer(access), "Idempotency-Key": "explicit-baseline-proposal"},
        json=proposal_body,
    )
    assert baseline.status_code == 200
    assert baseline.json()["record"]["proposal"]["proposal_id"] == "baseline-smoke"
    state = http.get(
        "/api/campaigns/autoresearch-baseline-1/autoresearch",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert state.json()["state"]["next_action"] == "wait_for_result"


def test_code_lineage_routes_prepare_private_worktree_and_capture_safe_record(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    version = repository.get_campaign("workspace-a", "campaign-1").version
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "lineage-validate"),
        (CampaignTrigger.VALIDATION_PASSED, "lineage-ready"),
        (CampaignTrigger.START, "lineage-start"),
    ):
        version = repository.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=key,
            idempotency_key=key,
        ).campaign.version
    proposal_body = study_proposal("candidate-code").model_dump(
        mode="json", exclude={"schema_version", "workspace_id", "campaign_id"}
    )
    proposal_body.update({"workspace_id": "workspace-a", "expected_version": version})
    submitted = http.post(
        "/api/campaigns/campaign-1/proposals",
        headers={**bearer(access), "Idempotency-Key": "lineage-proposal"},
        json=proposal_body,
    )
    assert submitted.status_code == 200
    observed_at = datetime.now(UTC)
    repository.register_code_lineage_requirement(
        CodeLineageRecord(
            lineage_id="lineage-candidate-code",
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            proposal_id="candidate-code",
            mutation_kind=CodeMutationKind.TRAINER,
            source_repository_profile_id="bashgym-source-v1",
            state=CodeLineageState.REQUIRED,
            created_at=observed_at,
            updated_at=observed_at,
        )
    )
    source_root = tmp_path / "source-fixture"
    source_root.mkdir()
    source_repository, _base_commit = initialized_repository(source_root)
    http.app.state.campaign_source_profiles = {
        "bashgym-source-v1": source_profile(source_repository)
    }
    http.app.state.campaign_lineage_worktree_root = tmp_path / "lineage-worktrees"

    prepared = http.post(
        "/api/campaigns/campaign-1/proposals/candidate-code/code-lineage/prepare",
        headers=bearer(access),
        json={"workspace_id": "workspace-a"},
    )
    assert prepared.status_code == 200
    worktree = Path(prepared.json()["worktree_path"])
    assert worktree.is_dir()
    assert str(source_repository) not in json.dumps(prepared.json()["record"])
    (worktree / "bashgym" / "gym" / "trainer.py").write_text(
        "LEARNING_RATE = 2e-4\n", encoding="utf-8"
    )

    captured = http.post(
        "/api/campaigns/campaign-1/proposals/candidate-code/code-lineage/capture",
        headers=bearer(access),
        json={"workspace_id": "workspace-a"},
    )
    assert captured.status_code == 200
    assert captured.json()["record"]["state"] == "captured"
    assert captured.json()["record"]["changed_paths"] == ["bashgym/gym/trainer.py"]
    assert "worktree_path" not in captured.json()


def test_desktop_bootstrap_env_is_required_and_exchanges_without_renderer_identity(
    tmp_path, monkeypatch
):
    bootstrap = "bgcb.electron-launch." + "main-process-secret-" * 2
    monkeypatch.setenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET", bootstrap)
    monkeypatch.setattr(campaign_routes, "get_bashgym_dir", lambda: tmp_path / "managed")
    app = FastAPI()
    app.include_router(campaign_auth_router)
    app.include_router(campaign_router)
    http = TestClient(app)

    access = exchange(http, bootstrap)
    capabilities = http.get("/api/campaign-auth/capabilities", headers=bearer(access))
    assert capabilities.status_code == 200
    assert capabilities.json()["actor_id"] == "desktop-user"
    assert capabilities.json()["autonomy_profile"] == "desktop_user"
    assert capabilities.json()["workspace_ids"] == ["desktop-local"]

    monkeypatch.delenv("BASHGYM_DESKTOP_BOOTSTRAP_SECRET")
    monkeypatch.setattr(campaign_routes, "get_bashgym_dir", lambda: tmp_path / "standalone")
    standalone = FastAPI()
    standalone.include_router(campaign_auth_router)
    standalone_http = TestClient(standalone)
    denied = standalone_http.post(
        "/api/campaign-auth/exchange",
        headers={"Authorization": f"Bearer {bootstrap}"},
    )
    assert denied.status_code == 401
    assert denied.json()["detail"]["code"] == "campaign_auth_required"


def test_authenticated_capabilities_expose_only_safe_worker_readiness(tmp_path) -> None:
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    observed_at = datetime.now(UTC)

    class SafeSupervisor:
        def status(self, *, repository=None):
            assert repository is not None
            return DesktopWorkerStatusProjection(
                managed=True,
                online=False,
                state="restarting",
                code="worker_restarting",
                observed_at=observed_at,
                thread_alive=True,
                restart_count=2,
                controller=campaign_routes.project_controller_status(
                    repository,
                    tmp_path / "managed",
                    now=observed_at,
                ),
                guidance="The desktop worker is restarting; campaigns remain paused.",
            )

    http.app.state.campaign_worker_supervisor = SafeSupervisor()
    http.app.state.campaign_worker_managed = True
    http.app.state.campaign_worker_bootstrap_failure_code = None
    response = http.get("/api/campaign-auth/capabilities", headers=bearer(access))

    assert response.status_code == 200
    worker = response.json()["worker"]
    assert worker["state"] == "restarting"
    assert worker["code"] == "worker_restarting"
    assert worker["restart_count"] == 2
    rendered = json.dumps(worker)
    assert str(tmp_path) not in rendered
    assert "seal" not in rendered.casefold()


def test_every_campaign_read_requires_campaign_bearer_even_in_desktop_app(tmp_path):
    http, _repository, _refresh = campaign_client(tmp_path)
    response = http.get("/api/campaigns", params={"workspace_id": "workspace-a"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "campaign_auth_required"


def test_campaign_list_projects_workspace_controller_health(tmp_path, monkeypatch):
    data_directory = tmp_path / "bashgym-data"
    monkeypatch.setattr(campaign_routes, "get_bashgym_dir", lambda: data_directory)
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)

    offline = http.get(
        "/api/campaigns",
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )
    assert offline.status_code == 200
    assert offline.json()["controller"]["state"] == "offline"
    assert offline.json()["controller"]["code"] == "controller_offline"

    now = datetime.now(UTC)
    repository.acquire_lease(
        scheduler_lease_key(data_directory),
        "resident-worker",
        ttl=timedelta(seconds=15),
        now=now,
    )
    online = http.get(
        "/api/campaigns",
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )
    assert online.status_code == 200
    assert online.json()["controller"]["state"] == "online"
    assert online.json()["controller"]["online"] is True
    assert online.json()["controller"]["owner_id"] == "resident-worker"


def _fixed_controller_observation(observed_at: datetime) -> ControllerStatusProjection:
    return ControllerStatusProjection(
        online=True,
        state="online",
        code="controller_online",
        observed_at=observed_at,
        heartbeat_age_seconds=7.5,
        owner_id="resident-worker",
        generation=3,
        heartbeat_at=observed_at - timedelta(seconds=7.5),
        expires_at=observed_at + timedelta(seconds=7.5),
    )


def _exact_control_room_definition(
    template_id: str = "control-room-exact-v1",
) -> AutoResearchTemplateDefinition:
    value = campaign()
    return AutoResearchTemplateDefinition(
        template_id=template_id,
        kind=value.kind,
        objective=value.objective,
        target_model=value.target_model,
        manifest=manifest(),
        policy=None,
    )


def test_control_room_snapshot_requires_auth_and_sets_private_cache_headers(tmp_path, monkeypatch):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    observed_at = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(
        campaign_routes,
        "project_controller_status",
        lambda *_args, **_kwargs: _fixed_controller_observation(observed_at),
    )
    route = "/api/campaigns/campaign-1/control-room-snapshot"

    denied = http.get(route, params={"workspace_id": "workspace-a"})
    response = http.get(
        route,
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )

    assert denied.status_code == 401
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["vary"] == "Authorization"
    assert response.headers["etag"].startswith('"')
    payload = response.json()
    assert tuple(payload) == (
        "schema_version",
        "workspace_id",
        "campaign_id",
        "aggregate_version",
        "manifest_revision",
        "authorization_revision",
        "snapshot_at",
        "latest_event_cursor",
        "campaign",
        "controller",
        "readiness",
        "bindings",
        "journey",
        "active_work",
        "champion",
        "candidate",
        "metrics",
        "budget",
        "human_work",
        "agents",
        "collections",
        "decision_surface",
    )
    assert payload["schema_version"] == "control_room_snapshot.v1"
    assert payload["workspace_id"] == "workspace-a"
    assert payload["campaign"]["aggregate_version"] == 1
    assert [phase["phase_id"] for phase in payload["journey"]] == [
        "setup",
        "baseline",
        "experiments",
        "human_review",
        "decision",
    ]
    assert payload["human_work"] == {
        "schema_version": "human_work_summary.v1",
        "blocking_count": 0,
        "open_count": 0,
        "newest": [],
    }
    assert payload["agents"] == []
    assert "durable_state" not in payload
    assert "controller_observation" not in payload
    assert (
        datetime.fromisoformat(payload["controller"]["observed_at"].replace("Z", "+00:00"))
        == observed_at
    )
    assert payload["controller"]["heartbeat_age_seconds"] == 7.5


def test_fresh_control_room_process_attaches_existing_ledger_without_writes(
    tmp_path, monkeypatch
):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    definition = _exact_control_room_definition()

    fresh_repository = CampaignRuntimeRepository(repository.db_path)
    fresh_repository.initialize()
    fresh_app = FastAPI()
    fresh_app.state.campaign_repository = fresh_repository
    fresh_app.state.campaign_auth_service = CampaignAuthService(fresh_repository)
    fresh_app.state.campaign_service = CampaignService(
        fresh_repository,
        approved_template_hashes={
            "memexai-approved-v1": canonical_hash(manifest().model_dump(mode="json"))
        },
    )
    fresh_app.state.campaign_templates = http.app.state.campaign_templates
    fresh_app.state.campaign_autoresearch_templates = {
        definition.template_id: definition
    }
    fresh_app.state.campaign_worker_config_path = tmp_path / "worker-config.v1.json"
    fresh_app.include_router(campaign_auth_router)
    fresh_app.include_router(campaign_router)
    fresh_http = TestClient(fresh_app)
    database_before = repository.db_path.read_bytes()

    def fail_if_initialized(_self):
        raise AssertionError("snapshot GET must not initialize or migrate the ledger")

    monkeypatch.setattr(ExperimentLedgerRepository, "initialize", fail_if_initialized)
    response = fresh_http.get(
        "/api/campaigns/campaign-1/control-room-snapshot",
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )

    assert response.status_code == 200
    assert response.json()["readiness"]["blocking_codes"] == [
        "template_policy_missing",
        "controller_offline",
    ]
    assert repository.db_path.read_bytes() == database_before


def test_control_room_snapshot_readiness_changes_etag_but_checked_at_does_not(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}

    unavailable = http.get(route, params=params, headers=bearer(access))
    definition = _exact_control_room_definition()
    http.app.state.campaign_autoresearch_templates = {definition.template_id: definition}
    changed = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": unavailable.headers["etag"]},
    )
    stable = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": changed.headers["etag"]},
    )

    assert unavailable.json()["readiness"]["blocking_codes"] == [
        "campaign_readiness_definition_unavailable"
    ]
    assert changed.status_code == 200
    assert changed.headers["etag"] != unavailable.headers["etag"]
    assert changed.json()["readiness"]["blocking_codes"] == [
        "template_policy_missing",
        "controller_offline",
    ]
    assert stable.status_code == 304


def test_control_room_snapshot_requires_one_full_identity_definition(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}

    wrong_objective = _exact_control_room_definition().model_copy(
        update={"objective": "A different campaign objective."}
    )
    http.app.state.campaign_autoresearch_templates = {wrong_objective.template_id: wrong_objective}
    unavailable = http.get(route, params=params, headers=bearer(access))

    first = _exact_control_room_definition("control-room-exact-one")
    second = _exact_control_room_definition("control-room-exact-two")
    http.app.state.campaign_autoresearch_templates = {
        first.template_id: first,
        second.template_id: second,
    }
    ambiguous = http.get(route, params=params, headers=bearer(access))

    assert unavailable.json()["readiness"]["blocking_codes"] == [
        "campaign_readiness_definition_unavailable"
    ]
    assert ambiguous.json()["readiness"]["blocking_codes"] == [
        "campaign_readiness_definition_ambiguous"
    ]


def test_control_room_definition_must_align_with_persisted_autoresearch_spec(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    autoresearch = AutoResearchRepository(repository.db_path)
    autoresearch.initialize()
    autoresearch.create_autoresearch_spec(
        AutoResearchCampaignSpec(
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            primary_metric="mrr_at_10",
            metric_direction=MetricDirection.MAXIMIZE,
            stop_rules=AutoResearchStopRules(
                max_attempts=1,
                budget_unit="gpu_hours",
                max_total_cost=1.0,
            ),
        )
    )
    definition = _exact_control_room_definition()
    http.app.state.campaign_autoresearch_templates = {definition.template_id: definition}
    http.app.state.campaign_experiment_ledger = ExperimentLedgerRepository(repository.db_path)

    response = http.get(
        "/api/campaigns/campaign-1/control-room-snapshot",
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )

    assert response.status_code == 200
    assert response.json()["readiness"]["blocking_codes"] == [
        "campaign_readiness_definition_unavailable"
    ]


def test_campaign_create_rejects_oversized_budget_with_typed_422_and_no_mutation(
    tmp_path,
):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    error_http = TestClient(http.app, raise_server_exceptions=False)
    oversized_manifest = manifest().model_dump(mode="json")
    oversized_manifest["budget_limits"] = {
        f"unit_{index:03d}": 1.0 for index in range(65)
    }
    value = campaign(campaign_id="campaign-oversized")

    response = error_http.post(
        "/api/campaigns",
        headers={**bearer(access), "Idempotency-Key": "oversized-create"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": value.campaign_id,
            "title": value.title,
            "kind": value.kind.value,
            "objective": value.objective,
            "target_model": value.target_model.model_dump(mode="json"),
            "manifest": oversized_manifest,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "campaign_budget_resource_limit_exceeded",
        "message": "Campaign bounded-resource policy rejected the operation.",
    }
    assert repository.list_campaigns("workspace-a") == []


def test_manifest_revise_rejects_oversized_budget_with_typed_422_and_no_mutation(
    tmp_path,
):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    error_http = TestClient(http.app, raise_server_exceptions=False)
    oversized_manifest = manifest().model_dump(mode="json")
    oversized_manifest["budget_limits"] = {
        f"unit_{index:03d}": 1.0 for index in range(65)
    }
    before = repository.get_campaign("workspace-a", "campaign-1")
    before_events = repository.list_events("workspace-a", "campaign-1")

    response = error_http.post(
        "/api/campaigns/campaign-1/manifest/revise",
        headers={**bearer(access), "Idempotency-Key": "oversized-revise"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": before.version,
            "reason": "Oversized revision",
            "manifest": oversized_manifest,
        },
    )

    after = repository.get_campaign("workspace-a", "campaign-1")
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "campaign_budget_resource_limit_exceeded",
        "message": "Campaign bounded-resource policy rejected the operation.",
    }
    assert after.version == before.version
    assert after.manifest_revision == before.manifest_revision
    assert repository.list_events("workspace-a", "campaign-1") == before_events


def test_control_room_snapshot_etag_is_principal_specific_and_supports_304(tmp_path, monkeypatch):
    http, _repository, refresh = campaign_client(tmp_path)
    first_access = exchange(http, refresh.raw_token)
    assert create_from_template(http, first_access).status_code == 200
    observed_at = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(
        campaign_routes,
        "project_controller_status",
        lambda *_args, **_kwargs: _fixed_controller_observation(observed_at),
    )
    auth = http.app.state.campaign_auth_service
    second_refresh = auth.issue_refresh_credential(
        actor_id="second-codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    second_access = exchange(http, second_refresh.raw_token)
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}

    first = http.get(route, params=params, headers=bearer(first_access))
    not_modified = http.get(
        route,
        params=params,
        headers={**bearer(first_access), "If-None-Match": first.headers["etag"]},
    )
    second = http.get(route, params=params, headers=bearer(second_access))

    assert not_modified.status_code == 304
    assert not not_modified.content
    assert not_modified.headers["cache-control"] == "private, no-store"
    assert not_modified.headers["vary"] == "Authorization"
    assert first.headers["etag"] != second.headers["etag"]


def test_control_room_snapshot_etag_invalidates_on_durable_authorization_revision(
    tmp_path, monkeypatch
):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    observed_at = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(
        campaign_routes,
        "project_controller_status",
        lambda *_args, **_kwargs: _fixed_controller_observation(observed_at),
    )
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}
    first = http.get(route, params=params, headers=bearer(access))

    revision = http.app.state.campaign_auth_service.revise_credential_authorization(
        refresh.credential_id,
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a", "workspace-b"),
    )
    current = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": first.headers["etag"]},
    )

    assert first.json()["authorization_revision"] == 1
    assert revision == 2
    assert current.status_code == 200
    assert current.headers["etag"] != first.headers["etag"]
    assert current.json()["authorization_revision"] == 2


def test_control_room_snapshot_etag_invalidates_on_persisted_controller_observation(
    tmp_path, monkeypatch
):
    data_directory = tmp_path / "bashgym-data"
    monkeypatch.setattr(campaign_routes, "get_bashgym_dir", lambda: data_directory)
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}
    now = datetime.now(UTC)
    lease = repository.acquire_lease(
        scheduler_lease_key(data_directory),
        "resident-worker",
        ttl=timedelta(seconds=60),
        now=now - timedelta(seconds=1),
    )

    first = http.get(route, params=params, headers=bearer(access))
    stable = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": first.headers["etag"]},
    )
    renewed = repository.heartbeat_lease(
        lease.lease_key,
        lease.owner_id,
        lease.generation,
        ttl=timedelta(seconds=60),
        now=now,
    )
    current = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": first.headers["etag"]},
    )

    assert first.json()["controller"]["controller_observation_version"] == 1
    assert stable.status_code == 304
    assert renewed.controller_observation_version == 2
    assert current.status_code == 200
    assert current.headers["etag"] != first.headers["etag"]
    assert current.json()["controller"]["controller_observation_version"] == 2


def test_control_room_snapshot_keeps_durable_and_controller_observation_times_distinct(
    tmp_path, monkeypatch
):
    http, _repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    controller_observed_at = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(
        campaign_routes,
        "project_controller_status",
        lambda *_args, **_kwargs: _fixed_controller_observation(controller_observed_at),
    )

    response = http.get(
        "/api/campaigns/campaign-1/control-room-snapshot",
        params={"workspace_id": "workspace-a"},
        headers=bearer(access),
    )
    payload = response.json()

    assert payload["snapshot_at"] != payload["controller"]["observed_at"]
    assert "transactionally_observed_at" not in payload
    assert "snapshot_observed_at" not in payload
    assert "state_observed_at" not in payload


def test_control_room_snapshot_returns_current_v1_for_an_older_etag(tmp_path, monkeypatch):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    observed_at = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(
        campaign_routes,
        "project_controller_status",
        lambda *_args, **_kwargs: _fixed_controller_observation(observed_at),
    )
    route = "/api/campaigns/campaign-1/control-room-snapshot"
    params = {"workspace_id": "workspace-a"}
    first = http.get(route, params=params, headers=bearer(access))

    repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.VALIDATE,
        expected_version=1,
        actor_id="campaign-controller",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="compatibility-transition",
        idempotency_key="compatibility-transition",
    )
    current = http.get(
        route,
        params=params,
        headers={**bearer(access), "If-None-Match": first.headers["etag"]},
    )

    assert current.status_code == 200
    assert current.headers["etag"] != first.headers["etag"]
    assert current.json()["schema_version"] == "control_room_snapshot.v1"
    assert current.json()["campaign"]["aggregate_version"] == 2


def test_campaign_ledger_projection_keeps_project_evals_artifacts_and_decisions_linked(
    tmp_path,
):
    http, repository, refresh = campaign_client(tmp_path)
    token = exchange(http, refresh.raw_token)
    assert create_from_template(http, token).status_code == 200

    ledger = ExperimentLedgerRepository(repository.db_path)
    ledger.initialize()
    ledger.register_project(
        ProjectSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            display_name="General model work",
            owner_actor_id="codex-agent",
        )
    )
    ledger.register_experiment(
        ExperimentSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-a",
            campaign_id="campaign-1",
            name="Instruction quality",
            objective="Improve instruction following.",
        )
    )
    ledger.register_run(
        RunSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            experiment_id="experiment-a",
            campaign_id="campaign-1",
            run_id="run-a",
            source_system="bashgym",
            source_run_id="run-a",
            run_kind="training",
            task_type="instruction-following",
            training_method="sft",
            status=RunStatus.COMPLETED,
            recipe_digest="a" * 64,
            correlation_id="campaign-ledger-test",
        )
    )
    ledger.register_evaluation_suite(
        EvaluationSuiteSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_suite_id="suite-a",
            name="Heldout instruction suite",
            task_type="instruction-following",
            metric_contract={"accuracy": {"direction": "higher"}},
            code_digest="b" * 64,
        )
    )
    ledger.record_evaluation_result(
        EvaluationResultSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            evaluation_result_id="eval-a",
            evaluation_suite_id="suite-a",
            run_id="run-a",
            status=RunStatus.COMPLETED,
            metrics={"accuracy": 0.83},
        )
    )
    ledger.record_artifact(
        ArtifactSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            artifact_id="artifact-a",
            run_id="run-a",
            kind="report",
            uri="file://private/report.pdf",
            sha256="c" * 64,
            size_bytes=42,
            media_type="application/pdf",
        )
    )
    ledger.record_decision(
        DecisionSpec(
            workspace_id="workspace-a",
            project_id="project-a",
            decision_id="decision-a",
            experiment_id="experiment-a",
            run_id="run-a",
            decision_type="retain",
            outcome="Retain the current champion.",
            rationale="The candidate missed the declared gate.",
            evidence_refs=("eval-a",),
            actor_id="codex-agent",
        )
    )

    response = http.get(
        "/api/campaigns/campaign-1/ledger",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
    )

    assert response.status_code == 200
    projection = response.json()
    assert projection["linked"] is True
    project = projection["projects"][0]
    assert project["project"]["project_id"] == "project-a"
    assert project["evaluations"][0]["evaluation_result_id"] == "eval-a"
    assert project["artifacts"][0]["artifact_id"] == "artifact-a"
    assert "uri" not in project["artifacts"][0]
    assert project["decisions"][0]["decision_id"] == "decision-a"


def test_campaign_feature_flag_fails_closed_before_initializing_state(monkeypatch):
    monkeypatch.setattr(
        campaign_routes,
        "get_settings",
        lambda: SimpleNamespace(campaigns_enabled=False),
    )
    app = FastAPI()
    app.include_router(campaign_auth_router)
    app.include_router(campaign_router)
    response = TestClient(app).get("/api/campaigns", params={"workspace_id": "workspace-a"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "campaigns_disabled"
    assert not hasattr(app.state, "campaign_repository")


def test_exchange_capabilities_and_workspace_scope_are_server_derived(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.HERMES_BOUNDED)
    access = exchange(http, refresh.raw_token)
    capabilities = http.get("/api/campaign-auth/capabilities", headers=bearer(access)).json()
    assert capabilities["actor_id"] == "hermes-agent"
    assert "campaign.create_from_template" in capabilities["capabilities"]
    assert "promotion.decide" not in capabilities["capabilities"]

    denied = http.get(
        "/api/campaigns",
        params={"workspace_id": "workspace-b"},
        headers=bearer(access),
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "campaign_scope_denied"


def test_hermes_template_creation_derives_owner_and_arbitrary_creation_is_denied(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.HERMES_BOUNDED)
    access = exchange(http, refresh.raw_token)
    created = create_from_template(http, access)
    assert created.status_code == 200
    assert created.json()["campaign"]["owner_actor_id"] == "hermes-agent"

    arbitrary_campaign = campaign(campaign_id="campaign-2")
    arbitrary = http.post(
        "/api/campaigns",
        headers={**bearer(access), "Idempotency-Key": "arbitrary-hermes"},
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "campaign-2",
            "title": arbitrary_campaign.title,
            "kind": arbitrary_campaign.kind.value,
            "objective": arbitrary_campaign.objective,
            "target_model": arbitrary_campaign.target_model.model_dump(mode="json"),
            "manifest": manifest().model_dump(mode="json"),
        },
    )
    assert arbitrary.status_code == 403
    assert arbitrary.json()["detail"]["code"] == "campaign_capability_required"


def test_transitions_use_versions_idempotency_and_cursor_events(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.VALIDATE,
        expected_version=1,
        actor_id="campaign-controller",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="validation",
        idempotency_key="validation",
    )
    repository.transition_campaign(
        "workspace-a",
        "campaign-1",
        CampaignTrigger.VALIDATION_PASSED,
        expected_version=2,
        actor_id="campaign-controller",
        credential_kind=CredentialKind.CONTROLLER,
        correlation_id="validation-passed",
        idempotency_key="validation-passed",
    )
    headers = {**bearer(access), "Idempotency-Key": "start-once"}
    body = {"workspace_id": "workspace-a", "expected_version": 3}
    first = http.post("/api/campaigns/campaign-1/start", headers=headers, json=body)
    replay = http.post("/api/campaigns/campaign-1/start", headers=headers, json=body)
    assert first.status_code == replay.status_code == 200
    assert replay.json()["replayed"] is True

    stale = http.post(
        "/api/campaigns/campaign-1/pause",
        headers={**bearer(access), "Idempotency-Key": "pause-stale"},
        json={"workspace_id": "workspace-a", "expected_version": 3},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "campaign_version_conflict"

    events = http.get(
        "/api/campaigns/campaign-1/events",
        headers=bearer(access),
        params={"workspace_id": "workspace-a", "after_cursor": 0, "limit": 2},
    )
    assert events.status_code == 200
    payload = events.json()
    assert len(payload["items"]) == 2
    assert payload["next_cursor"] == payload["items"][-1]["cursor"]


def test_events_endpoint_projects_fail_closed_public_events(tmp_path):
    http, repository, refresh = campaign_client(
        tmp_path, profile=AutonomyProfile.HERMES_BOUNDED
    )
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    canaries = {
        "location": str(tmp_path / "operator" / "restricted-result.json"),
        "reference": "protected-epoch-canary",
        "result": "candidate-map-canary",
        "note": "private-error-canary",
        "nested": {"ordinary": ["nested-protected-canary"]},
    }
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            UPDATE campaign_events
            SET event_type = ?, payload_json = ?, correlation_id = ?, idempotency_key = ?
            WHERE workspace_id = ? AND campaign_id = ? AND sequence = 1
            """,
            (
                "campaign:protected-evaluation-completed",
                json.dumps(canaries),
                "protected-eval-correlation-canary",
                "protected-eval-idempotency-canary",
                "workspace-a",
                "campaign-1",
            ),
        )

    response = http.get(
        "/api/campaigns/campaign-1/events",
        headers=bearer(access),
        params={"workspace_id": "workspace-a", "after_cursor": 0, "limit": 10},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert set(item) == {"cursor", "event"}
    assert set(item["event"]) == {
        "schema_version",
        "event_id",
        "workspace_id",
        "campaign_id",
        "sequence",
        "aggregate_version",
        "event_type",
        "actor_id",
        "credential_kind",
        "correlation_identity",
        "idempotency_identity",
        "created_at",
    }
    assert item["event"]["schema_version"] == "public_campaign_event.v1"
    assert "summary" not in item["event"]
    serialized = json.dumps(response.json(), sort_keys=True)
    for canary in (
        "restricted-result.json",
        "protected-epoch-canary",
        "candidate-map-canary",
        "private-error-canary",
        "nested-protected-canary",
    ):
        assert canary not in serialized
    assert "protected-eval-correlation-canary" not in serialized
    assert "protected-eval-idempotency-canary" not in serialized


def test_artifact_projection_redacts_absolute_uri(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_artifacts(
                workspace_id, campaign_id, artifact_id, producer_action_id, uri,
                sha256, size_bytes, schema_name, sealed, valid, metadata_json, created_at
            ) VALUES (?, ?, ?, NULL, ?, ?, 10, ?, 1, 1, '{}', ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "artifact-1",
                str(tmp_path / "private" / "model.bin"),
                "a" * 64,
                "huggingface_model_file.v1",
                campaign().created_at.isoformat(),
            ),
        )
    response = http.get(
        "/api/campaigns/campaign-1/artifacts",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert response.status_code == 200
    assert "uri" not in response.json()["artifacts"][0]


def test_proposal_manifest_evidence_and_advance_rest_contract(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    version = 1
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "validate-proposals"),
        (CampaignTrigger.VALIDATION_PASSED, "ready-proposals"),
        (CampaignTrigger.START, "start-proposals"),
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

    proposal_body = study_proposal("proposal-rest").model_dump(
        mode="json", exclude={"schema_version", "workspace_id", "campaign_id"}
    )
    proposal_body.update({"workspace_id": "workspace-a", "expected_version": version})
    headers = {**bearer(access), "Idempotency-Key": "submit-rest"}
    submitted = http.post(
        "/api/campaigns/campaign-1/proposals", headers=headers, json=proposal_body
    )
    replay = http.post("/api/campaigns/campaign-1/proposals", headers=headers, json=proposal_body)
    assert submitted.status_code == replay.status_code == 200
    assert submitted.json()["record"]["proposal"]["status"] == "submitted"
    assert replay.json()["replayed"] is True

    listed = http.get(
        "/api/campaigns/campaign-1/proposals",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    manifest_response = http.get(
        "/api/campaigns/campaign-1/manifest/1",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    evidence = http.get(
        "/api/campaigns/campaign-1/evidence",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert listed.status_code == manifest_response.status_code == evidence.status_code == 200
    assert listed.json()["proposals"][0]["proposal"]["proposal_id"] == "proposal-rest"
    assert manifest_response.json()["manifest_hash"]
    assert evidence.json()["snapshot_digest"]
    assert "artifact_references" in evidence.json()

    submitted_version = submitted.json()["campaign"]["version"]
    advanced = http.post(
        "/api/campaigns/campaign-1/advance",
        headers={**bearer(access), "Idempotency-Key": "advance-rest"},
        json={"workspace_id": "workspace-a", "expected_version": submitted_version},
    )
    assert advanced.status_code == 200
    assert advanced.json()["event"]["event_type"] == "campaign:advance-requested"
    assert advanced.json()["campaign"]["active_study_id"] is None
    with repository._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_studies").fetchone()[0] == 0

    withdrawn = http.post(
        "/api/campaigns/campaign-1/proposals/proposal-rest/withdraw",
        headers={**bearer(access), "Idempotency-Key": "withdraw-rest"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": advanced.json()["campaign"]["version"],
        },
    )
    assert withdrawn.status_code == 200
    assert withdrawn.json()["record"]["proposal"]["status"] == "withdrawn"

    invalid = http.post(
        "/api/campaigns/campaign-1/proposals/proposal-rest/withdraw",
        headers={**bearer(access), "Idempotency-Key": "withdraw-rest-again"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": withdrawn.json()["campaign"]["version"],
        },
    )
    assert invalid.status_code == 409
    assert invalid.json()["detail"]["code"] == "campaign_invalid_transition"


def test_operator_rest_actions_are_real_versioned_and_capability_gated(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    version = 1
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "operator-validate"),
        (CampaignTrigger.VALIDATION_PASSED, "operator-ready"),
        (CampaignTrigger.START, "operator-start"),
    ):
        result = repository.transition_campaign(
            "workspace-a",
            "campaign-1",
            trigger,
            expected_version=version,
            actor_id="campaign-controller",
            credential_kind=CredentialKind.CONTROLLER,
            correlation_id=key,
            idempotency_key=key,
        )
        version = result.campaign.version

    revised_manifest = manifest().model_copy(update={"max_proposal_rounds": 9})
    revised = http.post(
        "/api/campaigns/campaign-1/manifest/revise",
        headers={**bearer(access), "Idempotency-Key": "operator-revise"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": version,
            "manifest": revised_manifest.model_dump(mode="json"),
            "reason": "Bounded dry-campaign expansion.",
        },
    )
    assert revised.status_code == 200
    assert revised.json()["details"]["revision"] == 2
    version = revised.json()["campaign"]["version"]

    budget = http.post(
        "/api/campaigns/campaign-1/budget/amend",
        headers={**bearer(access), "Idempotency-Key": "operator-budget"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": version,
            "resource": "gpu_hours",
            "delta": 1.5,
            "reason": "Approved bounded extension.",
        },
    )
    assert budget.status_code == 200
    assert budget.json()["entry"]["limit_delta"] == 1.5
    version = budget.json()["campaign"]["version"]

    source = http.post(
        "/api/campaigns/campaign-1/sources/hf-community-evals-v1/approve",
        headers={**bearer(access), "Idempotency-Key": "operator-source"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": version,
            "evidence": {
                "provenance": "Pinned Hub dataset artifact and dataset card.",
                "license": "apache-2.0",
                "privacy_review": "Approved automated and manual review.",
                "contamination_review": "No protected-set overlap.",
                "artifact_sha256": "a" * 64,
            },
        },
    )
    assert source.status_code == 200
    assert source.json()["details"]["source_id"] == "hf-community-evals-v1"
    version = source.json()["campaign"]["version"]

    exported = http.post(
        "/api/campaigns/campaign-1/export",
        headers={**bearer(access), "Idempotency-Key": "operator-export"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": version,
            "formats": ["markdown", "json", "csv", "png", "docx", "pdf"],
        },
    )
    replay = http.post(
        "/api/campaigns/campaign-1/export",
        headers={**bearer(access), "Idempotency-Key": "operator-export"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": version,
            "formats": ["markdown", "json", "csv", "png", "docx", "pdf"],
        },
    )
    assert exported.status_code == replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert "path" not in str(exported.json()).casefold()


def test_protected_result_is_candidate_locked_replayable_and_promotable(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    version = 1
    for trigger, key in (
        (CampaignTrigger.VALIDATE, "protected-validate"),
        (CampaignTrigger.VALIDATION_PASSED, "protected-ready"),
        (CampaignTrigger.START, "protected-start"),
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
    with repository._connection(immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO campaign_gate_decisions(
                workspace_id, campaign_id, decision_id, decision_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "workspace-a",
                "campaign-1",
                "gate-protected-pass",
                json.dumps({"verdict": "passed", "candidate_digest": "c" * 64}),
                campaign().created_at.isoformat(),
            ),
        )

    leased = http.post(
        "/api/campaigns/campaign-1/protected-lease",
        headers={**bearer(access), "Idempotency-Key": "protected-lease"},
        json={"workspace_id": "workspace-a", "expected_version": version},
    )
    assert leased.status_code == 200
    result_body = {
        "workspace_id": "workspace-a",
        "expected_version": leased.json()["campaign"]["version"],
        "result": {
            "protected_epoch_id": leased.json()["details"]["protected_epoch_id"],
            "candidate_digest": "c" * 64,
            "passed": True,
            "metrics": {"recall_at_10": 0.84},
            "artifact_sha256": "d" * 64,
        },
    }
    completed = http.post(
        "/api/campaigns/campaign-1/protected-result",
        headers={**bearer(access), "Idempotency-Key": "protected-result"},
        json=result_body,
    )
    replay = http.post(
        "/api/campaigns/campaign-1/protected-result",
        headers={**bearer(access), "Idempotency-Key": "protected-result"},
        json=result_body,
    )
    assert completed.status_code == replay.status_code == 200
    assert completed.json()["details"]["passed"] is True
    assert replay.json()["replayed"] is True

    promoted = http.post(
        "/api/campaigns/campaign-1/promotion",
        headers={**bearer(access), "Idempotency-Key": "protected-promote"},
        json={
            "workspace_id": "workspace-a",
            "expected_version": completed.json()["campaign"]["version"],
        },
    )
    assert promoted.status_code == 200
    assert promoted.json()["details"]["protected_gate_passed"] is True


def test_hermes_cannot_revise_budget_approve_source_or_promote(tmp_path):
    http, _repository, refresh = campaign_client(tmp_path, profile=AutonomyProfile.HERMES_BOUNDED)
    access = exchange(http, refresh.raw_token)
    assert create_from_template(http, access).status_code == 200
    headers = {**bearer(access), "Idempotency-Key": "hermes-denied"}
    denied = http.post(
        "/api/campaigns/campaign-1/manifest/revise",
        headers=headers,
        json={
            "workspace_id": "workspace-a",
            "expected_version": 1,
            "manifest": manifest().model_dump(mode="json"),
            "reason": "Hermes must not expand authority.",
        },
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "campaign_capability_required"


def test_metric_projection_filters_exact_source_paginates_and_enforces_workspace(tmp_path):
    repository, attempt = _claimed_attempt(tmp_path)
    repository.append_remote_metrics(
        attempt,
        (
            '{"step":1,"loss":0.9}',
            '{"step":2,"loss":0.7}',
            '{"step":3,"loss":0.5}',
        ),
        source="trainer.jsonl",
        cursor_end=120,
        now=START + timedelta(seconds=2),
    )
    repository.append_remote_metrics(
        attempt,
        ('{"step":1,"loss":9.0}', '{"step":2,"loss":7.0}'),
        source="shadow.jsonl",
        cursor_end=80,
        now=START + timedelta(seconds=3),
    )

    auth = CampaignAuthService(repository)
    refresh = auth.issue_refresh_credential(
        actor_id="codex-agent",
        autonomy_profile=AutonomyProfile.CODEX_TRUSTED,
        workspace_ids=("workspace-a",),
    )
    app = FastAPI()
    app.state.campaign_repository = repository
    app.state.campaign_auth_service = auth
    app.state.campaign_service = CampaignService(repository)
    app.state.campaign_templates = {}
    app.include_router(campaign_auth_router)
    app.include_router(campaign_router)
    http = TestClient(app)
    access = exchange(http, refresh.raw_token)
    attempts = http.get(
        "/api/campaigns/campaign-1/attempts",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert attempts.status_code == 200
    assert [item["attempt_id"] for item in attempts.json()["attempts"]] == [attempt.attempt_id]
    assert "sealed_result_uri" not in attempts.json()["attempts"][0]
    comparisons = http.get(
        "/api/campaigns/campaign-1/comparisons",
        headers=bearer(access),
        params={"workspace_id": "workspace-a"},
    )
    assert comparisons.status_code == 200
    assert comparisons.json() == {"comparisons": []}
    metric_path = f"/api/campaigns/campaign-1/attempts/{attempt.attempt_id}/metrics"

    first = http.get(
        metric_path,
        headers=bearer(access),
        params={
            "workspace_id": "workspace-a",
            "metric_name": "loss",
            "source": "trainer.jsonl",
            "after_step": -1,
            "limit": 2,
        },
    )
    assert first.status_code == 200
    assert first.json() == {
        "metric_name": "loss",
        "source": "trainer.jsonl",
        "values": [
            {
                "schema_version": "campaign_metric_series_value.v1",
                "step": 1,
                "source": "trainer.jsonl",
                "value": 0.9,
                "observed_at": (START + timedelta(seconds=2)).isoformat(),
            },
            {
                "schema_version": "campaign_metric_series_value.v1",
                "step": 2,
                "source": "trainer.jsonl",
                "value": 0.7,
                "observed_at": (START + timedelta(seconds=2)).isoformat(),
            },
        ],
        "next_after_step": 2,
    }

    second = http.get(
        metric_path,
        headers=bearer(access),
        params={
            "workspace_id": "workspace-a",
            "metric_name": "loss",
            "source": "trainer.jsonl",
            "after_step": first.json()["next_after_step"],
            "limit": 2,
        },
    )
    assert second.status_code == 200
    assert [(item["step"], item["value"]) for item in second.json()["values"]] == [(3, 0.5)]
    assert all(item["source"] == "trainer.jsonl" for item in second.json()["values"])

    shadow = http.get(
        metric_path,
        headers=bearer(access),
        params={
            "workspace_id": "workspace-a",
            "metric_name": "loss",
            "source": "shadow.jsonl",
            "limit": 10,
        },
    )
    assert shadow.status_code == 200
    assert [(item["step"], item["value"]) for item in shadow.json()["values"]] == [
        (1, 9.0),
        (2, 7.0),
    ]

    denied = http.get(
        metric_path,
        headers=bearer(access),
        params={
            "workspace_id": "workspace-b",
            "metric_name": "loss",
            "source": "trainer.jsonl",
        },
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "campaign_scope_denied"
