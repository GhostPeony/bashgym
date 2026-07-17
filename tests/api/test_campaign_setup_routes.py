"""Authenticated portable guided-setup API tests."""

import sqlite3

from bashgym.api.campaign_setup_routes import campaign_setup_router
from bashgym.api.routes import create_app
from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import builtin_autoresearch_template_definitions
from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
from bashgym.campaigns.guided_setup import GuidedSetupRepository
from tests.api.test_campaign_routes import bearer, campaign_client, exchange

INSTALLATION_ID = "ins_0123456789abcdef0123456789abcdef"


def test_create_app_registers_guided_setup_routes():
    def collect(routes) -> set[str]:
        paths: set[str] = set()
        for route in routes:
            path = getattr(route, "path", None)
            if isinstance(path, str):
                paths.add(path)
            included = getattr(route, "original_router", None)
            if included is not None:
                paths.update(collect(included.routes))
        return paths

    paths = collect(create_app().routes)
    assert {
        "/api/campaigns/setup/templates",
        "/api/campaigns/setup/context",
        "/api/campaigns/setup/session",
        "/api/campaigns/setup/doctor",
        "/api/campaigns/setup/validate",
        "/api/campaigns/setup/create",
    } <= paths


def _register_setup(http, repository):
    http.app.include_router(campaign_setup_router)
    sealer = ArtifactSealer(
        b"campaign-route-human-seal-key-v1",
        key_version="campaign-seal-v1",
    )
    recovery = CampaignRecoveryRepository(repository.db_path, sealer=sealer)
    recovery.initialize()
    setup = GuidedSetupRepository(
        repository.db_path,
        sealer=sealer,
    )
    setup.initialize()
    http.app.state.campaign_recovery_repository = recovery
    http.app.state.campaign_guided_setup_repository = setup
    recovery.register_installation(
        installation_id=INSTALLATION_ID,
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    definition = builtin_autoresearch_template_definitions()[0]
    assert definition.policy is not None
    bindings = {
        "model": definition.target_model.target_contract_key,
        "data": definition.manifest.approved_data_scopes[0],
        "compute": definition.manifest.compute_profile_id,
        "evaluation": definition.policy.evaluation_suite_id,
    }
    for public_kind, storage_kind in {
        "model": "model",
        "data": "data",
        "compute": "compute",
        "evaluation": "evaluator",
    }.items():
        recovery.register_binding(
            installation_id=INSTALLATION_ID,
            kind=storage_kind,
            logical_id=bindings[public_kind],
            availability="reachable",
        )
    body = {
        "workspace_id": "workspace-a",
        "template_id": definition.template_id,
        "installation_id": INSTALLATION_ID,
        "bindings": bindings,
    }
    return recovery, definition, body


def test_guided_setup_lists_doctors_validates_and_creates_from_sealed_receipt(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    _recovery, definition, draft = _register_setup(http, repository)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)

    templates = http.get(
        "/api/campaigns/setup/templates",
        params={"workspace_id": "workspace-a"},
        headers=headers,
    )
    assert templates.status_code == 200
    selected = next(
        item
        for item in templates.json()["templates"]
        if item["template_id"] == definition.template_id
    )
    wire = str(selected).casefold()
    assert "base_model_ref" not in wire
    assert "controller-owner" not in wire
    assert "private-lease-key" not in wire

    with sqlite3.connect(repository.db_path) as connection:
        before_receipts = connection.execute(
            "SELECT COUNT(*) FROM campaign_guided_setup_receipts"
        ).fetchone()[0]
    doctor = http.post("/api/campaigns/setup/doctor", json=draft, headers=headers)
    assert doctor.status_code == 200
    assert doctor.json()["ready"] is True
    assert before_receipts == 0
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM campaign_guided_setup_receipts").fetchone()[0]
            == 0
        )

    validated = http.post(
        "/api/campaigns/setup/validate",
        json=draft,
        headers={**headers, "Idempotency-Key": "setup-validation-1"},
    )
    assert validated.status_code == 200
    assert validated.json()["ready"] is True
    replay = http.post(
        "/api/campaigns/setup/validate",
        json=draft,
        headers={**headers, "Idempotency-Key": "setup-validation-1"},
    )
    assert replay.status_code == 200
    assert replay.headers["X-BashGym-Replayed"] == "true"
    assert replay.json() == validated.json()

    create_body = {
        "workspace_id": "workspace-a",
        "campaign_id": "guided-campaign-1",
        "title": "Portable control campaign",
        "validation_receipt_id": validated.json()["receipt_id"],
    }
    created = http.post(
        "/api/campaigns/setup/create",
        json=create_body,
        headers={**headers, "Idempotency-Key": "guided-create-1"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["campaign"]["campaign_id"] == "guided-campaign-1"
    assert created.json()["setup"]["validation_receipt_id"] == validated.json()["receipt_id"]
    restarted = http.post(
        "/api/campaigns/setup/create",
        json=create_body,
        headers={**headers, "Idempotency-Key": "guided-create-1"},
    )
    assert restarted.status_code == 200


def test_guided_setup_rejects_stale_binding_and_bounded_invalid_payload(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    recovery, _definition, draft = _register_setup(http, repository)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)
    unvalidated = http.post(
        "/api/campaigns/setup/create",
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "guided-unvalidated",
            "title": "Must not create",
            "validation_receipt_id": "setuprcpt_ffffffffffffffffffffffffffffffff",
        },
        headers={**headers, "Idempotency-Key": "guided-create-unvalidated"},
    )
    assert unvalidated.status_code == 409
    validated = http.post(
        "/api/campaigns/setup/validate",
        json=draft,
        headers={**headers, "Idempotency-Key": "setup-validation-stale"},
    )
    assert validated.status_code == 200
    recovery.register_binding(
        installation_id=INSTALLATION_ID,
        kind="compute",
        logical_id=draft["bindings"]["compute"],
        availability="unknown",
    )
    stale = http.post(
        "/api/campaigns/setup/create",
        json={
            "workspace_id": "workspace-a",
            "campaign_id": "guided-stale",
            "title": "Must not create",
            "validation_receipt_id": validated.json()["receipt_id"],
        },
        headers={**headers, "Idempotency-Key": "guided-create-stale"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "campaign_guided_setup_conflict"

    oversized = {**draft, "bindings": {**draft["bindings"], "model": "m" * 161}}
    invalid = http.post("/api/campaigns/setup/doctor", json=oversized, headers=headers)
    assert invalid.status_code == 422


def test_guided_setup_transport_validation_never_echoes_invalid_canaries(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    _recovery, _definition, draft = _register_setup(http, repository)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)
    canary = "GUIDED-SETUP-SECRET-CANARY-DO-NOT-ECHO"

    invalid_query = http.get(
        "/api/campaigns/setup/templates",
        params={"workspace_id": canary + "/"},
        headers=headers,
    )
    assert invalid_query.status_code == 422
    assert canary not in invalid_query.text

    invalid_body = http.post(
        "/api/campaigns/setup/doctor",
        json={**draft, "bindings": {**draft["bindings"], "model": canary + "/"}},
        headers=headers,
    )
    assert invalid_body.status_code == 422
    assert canary not in invalid_body.text

    invalid_header = http.post(
        "/api/campaigns/setup/validate",
        json=draft,
        headers={**headers, "Idempotency-Key": canary + "/"},
    )
    assert invalid_header.status_code == 422
    assert canary not in invalid_header.text

    malformed_json = http.post(
        "/api/campaigns/setup/create",
        content=(b'{"title":"' + canary.encode() + b'"'),
        headers={**headers, "Content-Type": "application/json"},
    )
    assert malformed_json.status_code == 422
    assert canary not in malformed_json.text


def test_authenticated_context_is_read_only_and_session_steps_resume(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    _recovery, definition, draft = _register_setup(http, repository)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)
    session_id = "setupsess_0123456789abcdef0123456789abcdef"
    with sqlite3.connect(repository.db_path) as connection:
        before_sessions = connection.execute(
            "SELECT COUNT(*) FROM campaign_guided_setup_sessions"
        ).fetchone()[0]
        before_receipts = connection.execute(
            "SELECT COUNT(*) FROM campaign_guided_setup_step_receipts"
        ).fetchone()[0]

    context = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a"},
        headers=headers,
    )
    assert context.status_code == 200, context.text
    assert context.json()["session"] is None
    assert context.json()["reason_codes"] == ["setup_session_not_started"]
    wire = context.text.casefold()
    assert "controller-owner" not in wire
    assert "private-lease-key" not in wire
    with sqlite3.connect(repository.db_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM campaign_guided_setup_sessions").fetchone()[0]
            == before_sessions
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM campaign_guided_setup_step_receipts"
            ).fetchone()[0]
            == before_receipts
        )

    steps = (
        ("template", definition.template_id),
        ("installation", INSTALLATION_ID),
        ("model", draft["bindings"]["model"]),
        ("data", draft["bindings"]["data"]),
        ("compute", draft["bindings"]["compute"]),
        ("evaluation", draft["bindings"]["evaluation"]),
    )
    response = None
    for expected_version, (step, selection_id) in enumerate(steps):
        response = http.post(
            "/api/campaigns/setup/session",
            json={
                "workspace_id": "workspace-a",
                "session_id": session_id,
                "expected_version": expected_version,
                "step": step,
                "selection_id": selection_id,
            },
            headers={**headers, "Idempotency-Key": f"api-setup-step-{expected_version}"},
        )
        assert response.status_code == 200, response.text
    assert response is not None
    assert response.json()["session"]["ready_for_validation"] is True

    resumed = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a", "session_id": session_id},
        headers=headers,
    )
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["session"] == response.json()["session"]
    assert resumed.json()["reason_codes"] == []


def test_context_and_session_transport_fail_closed_without_echoing_canaries(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    _recovery, _definition, _draft_body = _register_setup(http, repository)
    token = exchange(http, refresh.raw_token)
    headers = bearer(token)
    canary = "GUIDED-SESSION-PRIVATE-CANARY-DO-NOT-ECHO"

    unauthenticated = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a"},
    )
    assert unauthenticated.status_code == 401

    invalid_query = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a", "session_id": canary + "/"},
        headers=headers,
    )
    assert invalid_query.status_code == 422
    assert canary not in invalid_query.text

    invalid_body = http.post(
        "/api/campaigns/setup/session",
        json={
            "workspace_id": "workspace-a",
            "session_id": "setupsess_0123456789abcdef0123456789abcdef",
            "expected_version": 0,
            "step": "template",
            "selection_id": canary + "/",
        },
        headers={**headers, "Idempotency-Key": "api-invalid-session-step"},
    )
    assert invalid_body.status_code == 422
    assert canary not in invalid_body.text


def test_context_get_does_not_create_setup_schema_when_only_registry_exists(tmp_path):
    http, repository, refresh = campaign_client(tmp_path)
    http.app.include_router(campaign_setup_router)
    sealer = ArtifactSealer(
        b"campaign-route-human-seal-key-v1",
        key_version="campaign-seal-v1",
    )
    recovery = CampaignRecoveryRepository(repository.db_path, sealer=sealer)
    recovery.initialize()
    recovery.register_installation(
        installation_id=INSTALLATION_ID,
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    definition = builtin_autoresearch_template_definitions()[0]
    recovery.register_binding(
        installation_id=INSTALLATION_ID,
        kind="model",
        logical_id=definition.target_model.target_contract_key,
        availability="reachable",
    )
    token = exchange(http, refresh.raw_token)
    with sqlite3.connect(repository.db_path) as connection:
        before = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "campaign_guided_setup_sessions" not in before

    context = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
    )
    assert context.status_code == 200, context.text
    assert context.json()["installations"][0]["installation_id"] == INSTALLATION_ID
    with sqlite3.connect(repository.db_path) as connection:
        after = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert after == before


def test_context_without_session_does_not_require_seal_authority(tmp_path, monkeypatch):
    http, repository, refresh = campaign_client(tmp_path)
    http.app.include_router(campaign_setup_router)
    recovery = CampaignRecoveryRepository(repository.db_path, sealer=None)
    recovery.initialize()
    recovery.register_installation(
        installation_id=INSTALLATION_ID,
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    del http.app.state.campaign_human_seal_key
    del http.app.state.campaign_worker_config_path
    monkeypatch.setattr("bashgym.api.campaign_routes.get_secret", lambda _key: "")
    token = exchange(http, refresh.raw_token)

    context = http.get(
        "/api/campaigns/setup/context",
        params={"workspace_id": "workspace-a"},
        headers=bearer(token),
    )

    assert context.status_code == 200, context.text
    assert context.json()["installations"][0]["installation_id"] == INSTALLATION_ID
    assert "PRIVATE-SEAL-CANARY" not in context.text


def test_context_with_session_fails_closed_without_seal_authority(tmp_path, monkeypatch):
    http, repository, refresh = campaign_client(tmp_path)
    http.app.include_router(campaign_setup_router)
    recovery = CampaignRecoveryRepository(repository.db_path, sealer=None)
    recovery.initialize()
    recovery.register_installation(
        installation_id=INSTALLATION_ID,
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    del http.app.state.campaign_human_seal_key
    del http.app.state.campaign_worker_config_path
    monkeypatch.setattr("bashgym.api.campaign_routes.get_secret", lambda _key: "")
    token = exchange(http, refresh.raw_token)

    resumed = http.get(
        "/api/campaigns/setup/context",
        params={
            "workspace_id": "workspace-a",
            "session_id": "setupsess_22222222222222222222222222222222",
        },
        headers=bearer(token),
    )

    assert resumed.status_code >= 400
    assert "PRIVATE-SEAL-CANARY" not in resumed.text
