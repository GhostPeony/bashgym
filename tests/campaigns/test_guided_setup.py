"""Portable, durable AutoResearch guided-setup contracts."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from bashgym.campaigns.artifacts import ArtifactSealer
from bashgym.campaigns.autoresearch import builtin_autoresearch_template_definitions
from bashgym.campaigns.campaign_recovery import CampaignRecoveryRepository
from bashgym.campaigns.contracts import Campaign, CredentialKind, ManifestRevision
from bashgym.campaigns.guided_setup import (
    GUIDED_SETUP_MAX_BINDINGS_PER_KIND,
    GUIDED_SETUP_MAX_INSTALLATIONS,
    GUIDED_SETUP_MAX_TEMPLATES,
    GuidedSetupBindings,
    GuidedSetupConflictError,
    GuidedSetupDraft,
    GuidedSetupRepository,
)
from bashgym.campaigns.persistence import CampaignRepository
from bashgym.campaigns.readiness import AutoResearchDoctorCheck, AutoResearchDoctorReport
from bashgym.campaigns.runtime import CampaignRuntimeRepository

_SEALER = ArtifactSealer(b"g" * 32, key_version="guided-setup-test-v1")


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if exists is None:
        return 0
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _definition():
    return builtin_autoresearch_template_definitions()[0]


def _bindings() -> GuidedSetupBindings:
    definition = _definition()
    assert definition.policy is not None
    return GuidedSetupBindings(
        model=definition.target_model.target_contract_key,
        data=str(
            definition.manifest.evaluation_plan.get("dataset_binding_id")
            or definition.manifest.approved_data_scopes[0]
        ),
        compute=definition.manifest.compute_profile_id,
        evaluation=definition.policy.evaluation_suite_id,
    )


def _doctor(*, materializable: bool = True) -> AutoResearchDoctorReport:
    definition = _definition()
    return AutoResearchDoctorReport(
        workspace_id="workspace-a",
        template_id=definition.template_id,
        definition_digest=definition.definition_digest,
        quality_claim_eligible=False,
        materializable=materializable,
        launch_ready=False,
        available=True,
        blocking_codes=() if materializable else ("compute_binding_unresolved",),
        checks=(
            AutoResearchDoctorCheck(
                check_id="portable-test",
                ready=materializable,
                code="portable_ready" if materializable else "compute_binding_unresolved",
            ),
        ),
    )


def _repositories(tmp_path):
    campaigns = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    campaigns.initialize()
    recovery = CampaignRecoveryRepository(campaigns.db_path)
    recovery.initialize()
    recovery.register_installation(
        installation_id="ins_0123456789abcdef0123456789abcdef",
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    for kind, logical_id in {
        "model": _bindings().model,
        "data": _bindings().data,
        "compute": _bindings().compute,
        "evaluator": _bindings().evaluation,
    }.items():
        recovery.register_binding(
            installation_id="ins_0123456789abcdef0123456789abcdef",
            kind=kind,
            logical_id=logical_id,
            availability="reachable",
        )
    setup = GuidedSetupRepository(campaigns.db_path, sealer=_SEALER)
    setup.initialize()
    return campaigns, recovery, setup


def _draft() -> GuidedSetupDraft:
    return GuidedSetupDraft(
        workspace_id="workspace-a",
        template_id=_definition().template_id,
        installation_id="ins_0123456789abcdef0123456789abcdef",
        bindings=_bindings(),
    )


def _campaign_contract(campaign_id: str) -> tuple[Campaign, ManifestRevision]:
    definition = _definition()
    value = Campaign(
        campaign_id=campaign_id,
        workspace_id="workspace-a",
        title="Portable campaign",
        kind=definition.kind,
        objective=definition.objective,
        target_model=definition.target_model,
        owner_actor_id="desktop-user",
    )
    return value, ManifestRevision(
        workspace_id="workspace-a",
        campaign_id=campaign_id,
        revision=1,
        manifest=definition.manifest,
        actor_id="desktop-user",
        correlation_id="guided-setup-test",
    )


def _atomic_create(
    setup: GuidedSetupRepository,
    campaigns: CampaignRuntimeRepository,
    receipt_id: str,
    campaign_id: str,
) -> None:
    value, revision = _campaign_contract(campaign_id)
    setup.create_campaign_atomically(
        receipt_id,
        repository=campaigns,
        campaign=value,
        manifest_revision=revision,
        definition=_definition(),
        actor_id="desktop-user",
        credential_kind=CredentialKind.ACCESS,
        correlation_id="guided-setup-test",
        idempotency_key=f"create-{campaign_id}",
    )


def test_validation_is_idempotent_durable_and_contains_only_logical_references(tmp_path):
    _campaigns, _recovery, setup = _repositories(tmp_path)
    first, replayed = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-1",
    )
    assert replayed is False
    assert first["ready"] is True
    assert first["receipt_id"].startswith("setuprcpt_")
    assert {item["kind"] for item in first["binding_references"]} == {
        "model",
        "data",
        "compute",
        "evaluation",
    }
    wire = str(first).casefold()
    assert "private-lease-key" not in wire
    assert "controller-owner" not in wire

    restarted = GuidedSetupRepository(setup.db_path, sealer=_SEALER)
    restarted.initialize()
    replay, was_replayed = restarted.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-1",
    )
    assert was_replayed is True
    assert replay == first
    changed = _draft().model_copy(
        update={"bindings": _bindings().model_copy(update={"compute": "changed-compute"})}
    )
    with pytest.raises(GuidedSetupConflictError, match="idempotency"):
        restarted.validate(
            changed,
            definition=_definition(),
            doctor=_doctor(),
            actor_id="desktop-user",
            idempotency_key="validate-setup-1",
        )


def test_invalid_and_stale_bindings_fail_closed(tmp_path):
    _campaigns, recovery, setup = _repositories(tmp_path)
    invalid = _draft().model_copy(
        update={"bindings": _bindings().model_copy(update={"model": "operator-choice"})}
    )
    report = setup.doctor(invalid, definition=_definition(), doctor=_doctor())
    assert report["ready"] is False
    assert "model_binding_contract_mismatch" in report["blocking_codes"]

    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-2",
    )
    recovery.register_binding(
        installation_id=_draft().installation_id,
        kind="compute",
        logical_id=_bindings().compute,
        availability="inaccessible",
    )
    with pytest.raises(GuidedSetupConflictError, match="stale"):
        setup.authorize_creation(
            receipt["receipt_id"],
            workspace_id="workspace-a",
            campaign_id="campaign-1",
            definition=_definition(),
            actor_id="desktop-user",
        )


def test_doctor_is_non_mutating_and_create_authority_is_one_campaign(tmp_path):
    campaigns, _recovery, setup = _repositories(tmp_path)
    before = campaigns.db_path.stat().st_size
    first = setup.doctor(_draft(), definition=_definition(), doctor=_doctor())
    second = setup.doctor(_draft(), definition=_definition(), doctor=_doctor())
    assert first == second
    assert campaigns.db_path.stat().st_size == before

    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-3",
    )
    _atomic_create(setup, campaigns, receipt["receipt_id"], "campaign-1")
    setup.record_creation(receipt["receipt_id"], "workspace-a", "campaign-1")
    setup.authorize_creation(
        receipt["receipt_id"],
        workspace_id="workspace-a",
        campaign_id="campaign-1",
        definition=_definition(),
        actor_id="desktop-user",
    )
    with pytest.raises(GuidedSetupConflictError, match="another campaign"):
        setup.authorize_creation(
            receipt["receipt_id"],
            workspace_id="workspace-a",
            campaign_id="campaign-2",
            definition=_definition(),
            actor_id="desktop-user",
        )


def test_payload_bounds_and_no_model_provider_defaults(tmp_path):
    _campaigns, _recovery, setup = _repositories(tmp_path)
    with pytest.raises(ValueError):
        GuidedSetupBindings(
            model="m" * 161,
            data="data",
            compute="compute",
            evaluation="evaluation",
        )
    summary = GuidedSetupRepository.template_summary(_definition())
    wire = str(summary).casefold()
    for forbidden in ("qwen", "gemma", "nemo", "cloud", "localhost", "c:\\"):
        assert forbidden not in wire


def test_receipt_authority_tampering_is_detected(tmp_path):
    _campaigns, _recovery, setup = _repositories(tmp_path)
    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-tamper",
    )
    with sqlite3.connect(setup.db_path) as connection:
        connection.execute(
            "UPDATE campaign_guided_setup_receipts SET installation_id=? WHERE receipt_id=?",
            ("ins_ffffffffffffffffffffffffffffffff", receipt["receipt_id"]),
        )
    with pytest.raises(GuidedSetupConflictError, match="seal"):
        setup.receipt_template_id(
            receipt["receipt_id"],
            workspace_id="workspace-a",
            actor_id="desktop-user",
        )


def test_database_readable_recovery_key_cannot_forge_guided_authority(tmp_path):
    _campaigns, _recovery, setup = _repositories(tmp_path)
    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-db-key-forgery",
    )
    with sqlite3.connect(setup.db_path) as connection:
        row = connection.execute(
            "SELECT * FROM campaign_guided_setup_receipts WHERE receipt_id=?",
            (receipt["receipt_id"],),
        ).fetchone()
        columns = [
            value[1]
            for value in connection.execute(
                "PRAGMA table_info(campaign_guided_setup_receipts)"
            ).fetchall()
        ]
        authority = dict(zip(columns, row, strict=True))
        forged_response = json.loads(authority["response_json"])
        forged_response["ready"] = False
        forged_json = json.dumps(forged_response, sort_keys=True, separators=(",", ":"))
        database_key = b"database-readable-attacker-key-01"
        connection.execute(
            "INSERT OR REPLACE INTO campaign_recovery_meta(key, value) VALUES ('seal_key', ?)",
            (database_key.hex(),),
        )
        forged_authority = {
            "receipt_id": authority["receipt_id"],
            "workspace_id": authority["workspace_id"],
            "actor_id": authority["actor_id"],
            "idempotency_key": authority["idempotency_key"],
            "request_hash": authority["request_hash"],
            "template_id": authority["template_id"],
            "definition_digest": authority["definition_digest"],
            "installation_id": authority["installation_id"],
            "bindings": json.loads(authority["bindings_json"]),
            "binding_state_digest": authority["binding_state_digest"],
            "response": forged_response,
            "ready": bool(authority["ready"]),
            "consumed_campaign_id": authority["consumed_campaign_id"],
            "created_at": authority["created_at"],
        }
        forged_seal = "sha256:" + ArtifactSealer(
            database_key,
            key_version=authority["seal_key_version"],
        ).sign_canonical_payload(
            forged_authority,
            domain="bashgym.guided-setup-receipt.v1",
        )
        connection.execute(
            """
            UPDATE campaign_guided_setup_receipts
            SET response_json=?, response_seal=?
            WHERE receipt_id=?
            """,
            (forged_json, forged_seal, receipt["receipt_id"]),
        )
    with pytest.raises(GuidedSetupConflictError, match="seal"):
        setup.receipt_template_id(
            receipt["receipt_id"],
            workspace_id="workspace-a",
            actor_id="desktop-user",
        )


def test_atomic_creation_rolls_back_campaign_receipt_and_binding_on_failure(tmp_path, monkeypatch):
    campaigns, _recovery, setup = _repositories(tmp_path)
    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-atomic-rollback",
    )

    def fail_insert(*_args, **_kwargs):
        raise RuntimeError("injected transaction failure")

    monkeypatch.setattr(CampaignRepository, "_insert_mutation", fail_insert)
    with pytest.raises(RuntimeError, match="injected transaction failure"):
        _atomic_create(setup, campaigns, receipt["receipt_id"], "campaign-failed")
    with sqlite3.connect(setup.db_path) as connection:
        assert (
            connection.execute(
                "SELECT 1 FROM campaigns WHERE workspace_id=? AND campaign_id=?",
                ("workspace-a", "campaign-failed"),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT consumed_campaign_id FROM campaign_guided_setup_receipts WHERE receipt_id=?",
                (receipt["receipt_id"],),
            ).fetchone()[0]
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM campaign_guided_setup_bindings WHERE workspace_id=?",
                ("workspace-a",),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM campaign_manifest_revisions WHERE workspace_id=? AND campaign_id=?",
                ("workspace-a", "campaign-failed"),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM campaign_events WHERE workspace_id=? AND campaign_id=?",
                ("workspace-a", "campaign-failed"),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM campaign_mutations WHERE workspace_id=? AND campaign_id=?",
                ("workspace-a", "campaign-failed"),
            ).fetchone()
            is None
        )


def test_one_receipt_concurrently_creates_exactly_one_bound_campaign(tmp_path):
    campaigns, _recovery, setup = _repositories(tmp_path)
    receipt, _ = setup.validate(
        _draft(),
        definition=_definition(),
        doctor=_doctor(),
        actor_id="desktop-user",
        idempotency_key="validate-setup-concurrent",
    )

    def attempt(campaign_id: str) -> str:
        try:
            _atomic_create(setup, campaigns, receipt["receipt_id"], campaign_id)
        except GuidedSetupConflictError:
            return "conflict"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, ("campaign-race-a", "campaign-race-b")))
    assert sorted(outcomes) == ["conflict", "created"]
    with sqlite3.connect(setup.db_path) as connection:
        campaigns_count = connection.execute(
            "SELECT COUNT(*) FROM campaigns WHERE workspace_id=?",
            ("workspace-a",),
        ).fetchone()[0]
        bindings_count = connection.execute(
            "SELECT COUNT(*) FROM campaign_guided_setup_bindings WHERE workspace_id=?",
            ("workspace-a",),
        ).fetchone()[0]
        consumed = connection.execute(
            "SELECT consumed_campaign_id FROM campaign_guided_setup_receipts WHERE receipt_id=?",
            (receipt["receipt_id"],),
        ).fetchone()[0]
    assert campaigns_count == 1
    assert bindings_count == 1
    assert consumed in {"campaign-race-a", "campaign-race-b"}


def test_read_only_doctor_does_not_create_setup_schema_or_receipts(tmp_path):
    campaigns = CampaignRuntimeRepository(tmp_path / "campaigns.sqlite3")
    campaigns.initialize()
    recovery = CampaignRecoveryRepository(campaigns.db_path)
    recovery.initialize()
    recovery.register_installation(
        installation_id=_draft().installation_id,
        controller_owner_id="controller-owner",
        controller_lease_key="private-lease-key",
    )
    for kind, logical_id in {
        "model": _bindings().model,
        "data": _bindings().data,
        "compute": _bindings().compute,
        "evaluator": _bindings().evaluation,
    }.items():
        recovery.register_binding(
            installation_id=_draft().installation_id,
            kind=kind,
            logical_id=logical_id,
            availability="reachable",
        )
    readonly = GuidedSetupRepository.open_binding_registry(campaigns.db_path)
    assert readonly.doctor(_draft(), definition=_definition(), doctor=_doctor())["ready"] is True
    with sqlite3.connect(campaigns.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "campaign_guided_setup_receipts" not in tables
    assert "campaign_guided_setup_bindings" not in tables


def test_context_discovers_only_public_registered_bindings_without_mutation(tmp_path):
    campaigns, _recovery, setup = _repositories(tmp_path)
    with sqlite3.connect(campaigns.db_path) as connection:
        before = {
            table: _table_count(connection, table)
            for table in (
                "campaign_guided_setup_receipts",
                "campaign_guided_setup_bindings",
                "campaign_guided_setup_sessions",
                "campaign_guided_setup_step_receipts",
            )
        }

    context = setup.context(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        definitions={_definition().template_id: _definition()},
    )

    assert context["schema_version"] == "guided_setup_context.v1"
    assert context["session"] is None
    assert context["reason_codes"] == ["setup_session_not_started"]
    installation = context["installations"][0]
    assert installation["installation_id"] == _draft().installation_id
    assert installation["ready"] is True
    assert set(installation["bindings"]) == {"model", "data", "compute", "evaluation"}
    assert installation["bindings"]["model"] == [
        {
            "logical_id": _bindings().model,
            "availability": "reachable",
            "selectable": True,
            "reason_codes": [],
        }
    ]
    wire = json.dumps(context, sort_keys=True).casefold()
    assert "controller-owner" not in wire
    assert "private-lease-key" not in wire
    assert "localhost" not in wire
    assert "c:\\" not in wire

    with sqlite3.connect(campaigns.db_path) as connection:
        after = {table: _table_count(connection, table) for table in before}
    assert after == before


def test_context_bounds_templates_installations_and_bindings_with_explicit_truncation(tmp_path):
    campaigns, recovery, setup = _repositories(tmp_path)
    first_installation = "ins_00000000000000000000000000000000"
    for index in range(GUIDED_SETUP_MAX_INSTALLATIONS + 1):
        installation_id = f"ins_{index:032x}"
        recovery.register_installation(
            installation_id=installation_id,
            controller_owner_id="bounded-controller",
            controller_lease_key="bounded-lease-key",
        )
    for index in range(GUIDED_SETUP_MAX_BINDINGS_PER_KIND + 1):
        recovery.register_binding(
            installation_id=first_installation,
            kind="model",
            logical_id=f"bounded-model-{index:03d}",
            availability="reachable",
        )
    definitions = {
        f"bounded-template-{index:03d}": _definition().model_copy(
            update={"template_id": f"bounded-template-{index:03d}"}
        )
        for index in range(GUIDED_SETUP_MAX_TEMPLATES + 1)
    }

    context = setup.context(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        definitions=definitions,
    )

    assert len(context["templates"]) == GUIDED_SETUP_MAX_TEMPLATES
    assert len(context["installations"]) == GUIDED_SETUP_MAX_INSTALLATIONS
    first = context["installations"][0]
    assert first["installation_id"] == first_installation
    assert len(first["bindings"]["model"]) == GUIDED_SETUP_MAX_BINDINGS_PER_KIND
    assert first["truncation"] == {
        "truncated": True,
        "reason_codes": ["bindings_truncated"],
        "limit_per_kind": GUIDED_SETUP_MAX_BINDINGS_PER_KIND,
        "kinds": ["model"],
    }
    assert context["truncation"] == {
        "truncated": True,
        "reason_codes": [
            "bindings_truncated",
            "installations_truncated",
            "templates_truncated",
        ],
        "limits": {
            "templates": GUIDED_SETUP_MAX_TEMPLATES,
            "installations": GUIDED_SETUP_MAX_INSTALLATIONS,
            "bindings_per_kind": GUIDED_SETUP_MAX_BINDINGS_PER_KIND,
        },
    }


@pytest.mark.parametrize("availability", ["unknown", "inaccessible"])
def test_session_rejects_required_binding_that_is_not_reachable(tmp_path, availability):
    _campaigns, recovery, setup = _repositories(tmp_path)
    recovery.register_binding(
        installation_id=_draft().installation_id,
        kind="model",
        logical_id=_bindings().model,
        availability=availability,
    )
    session_id = "setupsess_11111111111111111111111111111111"
    for expected_version, (step, selection_id) in enumerate(
        (
            ("template", _definition().template_id),
            ("installation", _draft().installation_id),
        )
    ):
        setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=expected_version,
            step=step,
            selection_id=selection_id,
            definitions={_definition().template_id: _definition()},
            idempotency_key=f"setup-unreachable-{availability}-{step}",
        )

    with pytest.raises(GuidedSetupConflictError, match="reachable"):
        setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=2,
            step="model",
            selection_id=_bindings().model,
            definitions={_definition().template_id: _definition()},
            idempotency_key=f"setup-unreachable-{availability}-model",
        )


def test_step_receipts_resume_an_exact_registered_session_after_restart(tmp_path):
    campaigns, _recovery, setup = _repositories(tmp_path)
    session_id = "setupsess_0123456789abcdef0123456789abcdef"
    steps = (
        ("template", _definition().template_id),
        ("installation", _draft().installation_id),
        ("model", _bindings().model),
        ("data", _bindings().data),
        ("compute", _bindings().compute),
        ("evaluation", _bindings().evaluation),
    )
    response = None
    for expected_version, (step, selection_id) in enumerate(steps):
        response, replayed = setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=expected_version,
            step=step,
            selection_id=selection_id,
            definitions={_definition().template_id: _definition()},
            idempotency_key=f"setup-step-{expected_version}",
        )
        assert replayed is False
        assert response["receipt"]["step"] == step
        assert response["receipt"]["version"] == expected_version + 1
        replay, was_replayed = setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=expected_version,
            step=step,
            selection_id=selection_id,
            definitions={_definition().template_id: _definition()},
            idempotency_key=f"setup-step-{expected_version}",
        )
        assert was_replayed is True
        assert replay == response

    assert response is not None
    assert response["session"]["ready_for_validation"] is True
    assert response["session"]["reason_codes"] == []
    assert response["session"]["selections"] == {
        "template_id": _definition().template_id,
        "installation_id": _draft().installation_id,
        "bindings": _bindings().model_dump(mode="json"),
    }

    restarted = GuidedSetupRepository(campaigns.db_path, sealer=_SEALER)
    restarted.initialize()
    context = restarted.context(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        session_id=session_id,
        definitions={_definition().template_id: _definition()},
    )
    assert context["session"] == response["session"]
    assert context["reason_codes"] == []


def test_session_rejects_unregistered_or_out_of_contract_selection_and_tampering(tmp_path):
    campaigns, recovery, setup = _repositories(tmp_path)
    recovery.register_binding(
        installation_id=_draft().installation_id,
        kind="model",
        logical_id="registered-but-not-required-model",
        availability="reachable",
    )
    session_id = "setupsess_abcdef0123456789abcdef0123456789"
    setup.advance_session(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        session_id=session_id,
        expected_version=0,
        step="template",
        selection_id=_definition().template_id,
        definitions={_definition().template_id: _definition()},
        idempotency_key="setup-exact-template",
    )
    setup.advance_session(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        session_id=session_id,
        expected_version=1,
        step="installation",
        selection_id=_draft().installation_id,
        definitions={_definition().template_id: _definition()},
        idempotency_key="setup-exact-installation",
    )
    with pytest.raises(GuidedSetupConflictError, match="contract"):
        setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=2,
            step="model",
            selection_id="registered-but-not-required-model",
            definitions={_definition().template_id: _definition()},
            idempotency_key="setup-wrong-model",
        )
    with pytest.raises(GuidedSetupConflictError, match="registered"):
        setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=2,
            step="model",
            selection_id="unregistered-model",
            definitions={_definition().template_id: _definition()},
            idempotency_key="setup-unregistered-model",
        )

    accepted, _ = setup.advance_session(
        workspace_id="workspace-a",
        actor_id="desktop-user",
        session_id=session_id,
        expected_version=2,
        step="model",
        selection_id=_bindings().model,
        definitions={_definition().template_id: _definition()},
        idempotency_key="setup-exact-model",
    )
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            "UPDATE campaign_guided_setup_step_receipts SET selection_id=? WHERE receipt_id=?",
            ("tampered-model", accepted["receipt"]["receipt_id"]),
        )
    with pytest.raises(GuidedSetupConflictError, match="seal"):
        setup.context(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            definitions={_definition().template_id: _definition()},
        )


def test_session_receipt_chain_detects_deleted_intermediate_step(tmp_path):
    campaigns, _recovery, setup = _repositories(tmp_path)
    session_id = "setupsess_fedcba9876543210fedcba9876543210"
    for expected_version, (step, selection_id) in enumerate(
        (
            ("template", _definition().template_id),
            ("installation", _draft().installation_id),
            ("model", _bindings().model),
        )
    ):
        setup.advance_session(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            expected_version=expected_version,
            step=step,
            selection_id=selection_id,
            definitions={_definition().template_id: _definition()},
            idempotency_key=f"setup-chain-{expected_version}",
        )
    with sqlite3.connect(campaigns.db_path) as connection:
        connection.execute(
            """
            DELETE FROM campaign_guided_setup_step_receipts
            WHERE workspace_id=? AND actor_id=? AND session_id=? AND version=2
            """,
            ("workspace-a", "desktop-user", session_id),
        )
    with pytest.raises(GuidedSetupConflictError, match="chain"):
        setup.context(
            workspace_id="workspace-a",
            actor_id="desktop-user",
            session_id=session_id,
            definitions={_definition().template_id: _definition()},
        )
